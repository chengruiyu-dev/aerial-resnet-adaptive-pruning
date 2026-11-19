"""
Pruning_screen.py  v3

修复历史：
v1: 初版，存在跨stage维度传递遗漏
v2: 修复了部分传递，但漏掉了关键一步
v3: 完整修复

根本问题（v3修复）：
    下采样块conv2+conv3联合剪枝后，block输出通道从128缩减为N。
    下一个恒等块接收的X（shortcut）通道数也变为N。
    恒等块forward里 Y += X 要求Y和X通道数相同。
    Y = bn2(conv2(...))，所以conv2的输出通道数也必须变为N。
    原版只更新了conv2的输入通道，没有更新conv2的输出通道，导致维度不匹配。

    修复：联合剪枝后，同步用同一个keep索引更新恒等块conv2的输出通道和对应bn2。
"""

import sys
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



import os as _os
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(_os.environ.get("PRUNING_ROOT", _Path(__file__).resolve().parents[1])).resolve()
_sys.path.insert(0, str(ROOT / "src"))

from models.resnet import get_resnet18
from data.dataset import AerialLandscapeDataset, get_transforms
from utils.split_data import load_split_from_file

# ================================================================
# 路径配置
# ================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints" / "best_resnet18_aerial.pth"))
VAL_SPLIT_FILE = os.environ.get("VAL_SPLIT_FILE", str(ROOT / "splits" / "aerial_split_val.txt"))
DATA_ROOT = os.environ.get("AERIAL_DATA_ROOT", str(ROOT / "data" / "Aerial_Landscapes"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(ROOT / "results"))
NUM_CLASSES    = 15
BATCH_SIZE     = 256

B_LIST     = [1, 5, 10]
ALPHA_LIST = [0.25, 0.30, 0.40, 0.50, 0.60]

SENSITIVITY_RAW = {
    "stage1_block1_conv1"         : 0.0917,
    "stage1_block2_conv1"         : 0.0222,
    "stage2_block1_conv1"         : 0.0361,
    "stage2_block1_conv2_shortcut": 0.3139,
    "stage2_block2_conv1"         : 0.0222,
    "stage3_block1_conv1"         : 0.0611,
    "stage3_block1_conv2_shortcut": 0.1806,
    "stage3_block2_conv1"         : 0.0472,
    "stage4_block1_conv1"         : 0.0222,
    "stage4_block1_conv2_shortcut": 0.8972,
    "stage4_block2_conv1"         : 0.2472,
}

FLOPS_RAW = {
    "stage1_block1_conv1"         : 115_605_504,
    "stage1_block2_conv1"         : 115_605_504,
    "stage2_block1_conv1"         :  57_802_752,
    "stage2_block1_conv2_shortcut": 115_605_504 + 6_422_528,
    "stage2_block2_conv1"         : 115_605_504,
    "stage3_block1_conv1"         :  57_802_752,
    "stage3_block1_conv2_shortcut": 115_605_504 + 6_422_528,
    "stage3_block2_conv1"         : 115_605_504,
    "stage4_block1_conv1"         :  57_802_752,
    "stage4_block1_conv2_shortcut": 115_605_504 + 6_422_528,
    "stage4_block2_conv1"         : 115_605_504,
}

TOTAL_FLOPS_ORIGINAL = 1_695_547_392


# ================================================================
# 归一化
# ================================================================
def minmax_normalize(d):
    vals = list(d.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 1.0 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}


def divmax_normalize(d):
    vmax = max(d.values())
    return {k: v / vmax for k, v in d.items()}


def compute_pruning_rates(alpha, b, flops_norm, sens_norm):
    return {
        layer: alpha * flops_norm[layer] / (flops_norm[layer] + b * sens_norm[layer] + 1e-8)
        for layer in flops_norm
    }


# ================================================================
# 基础工具函数
# ================================================================
def get_keep_idx(weight, prune_ratio):
    """按L1-norm降序，返回保留的输出通道索引"""
    n = weight.shape[0]
    n_keep = max(1, n - max(1, int(n * prune_ratio)))
    l1 = weight.abs().sum(dim=[1, 2, 3])
    _, sorted_idx = torch.sort(l1, descending=True)
    keep, _ = torch.sort(sorted_idx[:n_keep])
    return keep


def conv_keep_out(conv, keep):
    """新建conv，保留指定输出通道"""
    nc = nn.Conv2d(conv.in_channels, len(keep),
                   kernel_size=conv.kernel_size, stride=conv.stride,
                   padding=conv.padding, bias=(conv.bias is not None))
    nc.weight.data = conv.weight.data[keep].clone()
    if conv.bias is not None:
        nc.bias.data = conv.bias.data[keep].clone()
    return nc


def conv_keep_in(conv, keep):
    """新建conv，保留指定输入通道（输出通道数不变）"""
    nc = nn.Conv2d(len(keep), conv.out_channels,
                   kernel_size=conv.kernel_size, stride=conv.stride,
                   padding=conv.padding, bias=(conv.bias is not None))
    nc.weight.data = conv.weight.data[:, keep, :, :].clone()
    if conv.bias is not None:
        nc.bias.data = conv.bias.data.clone()
    return nc


def conv_keep_in_out(conv, keep_in, keep_out):
    """新建conv，同时裁剪输入和输出通道"""
    nc = nn.Conv2d(len(keep_in), len(keep_out),
                   kernel_size=conv.kernel_size, stride=conv.stride,
                   padding=conv.padding, bias=(conv.bias is not None))
    nc.weight.data = conv.weight.data[keep_out][:, keep_in, :, :].clone()
    if conv.bias is not None:
        nc.bias.data = conv.bias.data[keep_out].clone()
    return nc


def bn_keep(bn, keep):
    """新建BN，保留指定通道"""
    nb = nn.BatchNorm2d(len(keep))
    nb.weight.data  = bn.weight.data[keep].clone()
    nb.bias.data    = bn.bias.data[keep].clone()
    nb.running_mean = bn.running_mean[keep].clone()
    nb.running_var  = bn.running_var[keep].clone()
    nb.eps          = bn.eps
    nb.momentum     = bn.momentum
    return nb


# ================================================================
# 物理剪枝
#
# 维度约束总结：
#
# 恒等块（无conv3）：
#   conv2输出 == block输入通道（shortcut X的通道数），不可随意改变
#   conv1可自由剪输出通道，但conv2只能改输入通道，不能改输出通道
#
# 下采样块（有conv3/shortcut）：
#   conv2输出 == conv3输出（两者联合剪枝，用同一keep）
#   下一个恒等块的conv1输入 = keep
#   下一个恒等块的conv2输出 = keep（！这是之前遗漏的关键步骤）
#   下一个恒等块的conv2输入 = 由conv1剪枝决定（在conv1剪枝那步处理）
# ================================================================
def physical_prune(model, rates):
    m = copy.deepcopy(model)

    s1b0 = m[1][0]  # stage1 block1（恒等）
    s1b1 = m[1][1]  # stage1 block2（恒等）
    s2b0 = m[2][0]  # stage2 block1（下采样）
    s2b1 = m[2][1]  # stage2 block2（恒等）
    s3b0 = m[3][0]  # stage3 block1（下采样）
    s3b1 = m[3][1]  # stage3 block2（恒等）
    s4b0 = m[4][0]  # stage4 block1（下采样）
    s4b1 = m[4][1]  # stage4 block2（恒等）

    # ── Stage1 block1（恒等块）conv1 ──────────────────────────────
    # conv1剪输出通道 → conv2剪对应输入通道
    # conv2输出通道锁定=64（与X对齐），不变
    k = get_keep_idx(s1b0.conv1.weight.data, rates["stage1_block1_conv1"])
    s1b0.conv1 = conv_keep_out(s1b0.conv1, k)
    s1b0.bn1   = bn_keep(s1b0.bn1, k)
    s1b0.conv2 = conv_keep_in(s1b0.conv2, k)
    # s1b0输出=64，不变，s1b1输入无需调整

    # ── Stage1 block2（恒等块）conv1 ──────────────────────────────
    k = get_keep_idx(s1b1.conv1.weight.data, rates["stage1_block2_conv1"])
    s1b1.conv1 = conv_keep_out(s1b1.conv1, k)
    s1b1.bn1   = bn_keep(s1b1.bn1, k)
    s1b1.conv2 = conv_keep_in(s1b1.conv2, k)
    # s1b1输出=64，不变，s2b0输入无需调整

    # ── Stage2 block1（下采样块）conv1 ────────────────────────────
    k = get_keep_idx(s2b0.conv1.weight.data, rates["stage2_block1_conv1"])
    s2b0.conv1 = conv_keep_out(s2b0.conv1, k)
    s2b0.bn1   = bn_keep(s2b0.bn1, k)
    s2b0.conv2 = conv_keep_in(s2b0.conv2, k)

    # ── Stage2 block1（下采样块）conv2 + conv3 联合剪枝 ───────────
    # 排序依据：conv3(shortcut)的L1-norm
    # 影响：s2b0.conv2输出，s2b0.conv3输出
    #       s2b1.conv1输入，s2b1.conv2输出（！关键，使Y与X通道数一致）
    #       s2b1.bn2也要同步（因为conv2输出通道变了）
    k = get_keep_idx(s2b0.conv3.weight.data, rates["stage2_block1_conv2_shortcut"])
    s2b0.conv2 = conv_keep_out(s2b0.conv2, k)
    s2b0.bn2   = bn_keep(s2b0.bn2, k)
    s2b0.conv3 = conv_keep_out(s2b0.conv3, k)   # conv3无BN
    s2b1.conv1 = conv_keep_in(s2b1.conv1, k)
    s2b1.conv2 = conv_keep_out(s2b1.conv2, k)   # 锁定输出=len(k)，与X对齐
    s2b1.bn2   = bn_keep(s2b1.bn2, k)

    # ── Stage2 block2（恒等块）conv1 ──────────────────────────────
    # 此时s2b1.conv2输出已是len(k_prev)，conv1输入也已更新
    # conv1在此基础上继续剪自己的输出通道（影响conv2输入）
    k = get_keep_idx(s2b1.conv1.weight.data, rates["stage2_block2_conv1"])
    s2b1.conv1 = conv_keep_out(s2b1.conv1, k)
    s2b1.bn1   = bn_keep(s2b1.bn1, k)
    # conv2输入需同时更新（输出通道已在上一步固定）
    # 用conv_keep_in_out：输入=k（conv1输出），输出=当前conv2.out_channels（已是剪枝后的值）
    keep_out_s2b1 = torch.arange(s2b1.conv2.out_channels)
    s2b1.conv2 = conv_keep_in_out(s2b1.conv2, k, keep_out_s2b1)

    # ── 跨Stage传递：s2b1输出通道 → s3b0输入通道 ─────────────────
    # s2b1.conv2输出通道已由联合剪枝缩减，s3b0的两条输入路径都要跟着更新
    keep_cross = torch.arange(s2b1.conv2.out_channels)
    s3b0.conv1 = conv_keep_in(s3b0.conv1, keep_cross)
    s3b0.conv3 = conv_keep_in(s3b0.conv3, keep_cross)

    # ── Stage3 block1（下采样块）conv1 ────────────────────────────
    k = get_keep_idx(s3b0.conv1.weight.data, rates["stage3_block1_conv1"])
    s3b0.conv1 = conv_keep_out(s3b0.conv1, k)
    s3b0.bn1   = bn_keep(s3b0.bn1, k)
    s3b0.conv2 = conv_keep_in(s3b0.conv2, k)

    # ── Stage3 block1（下采样块）conv2 + conv3 联合剪枝 ───────────
    k = get_keep_idx(s3b0.conv3.weight.data, rates["stage3_block1_conv2_shortcut"])
    s3b0.conv2 = conv_keep_out(s3b0.conv2, k)
    s3b0.bn2   = bn_keep(s3b0.bn2, k)
    s3b0.conv3 = conv_keep_out(s3b0.conv3, k)
    s3b1.conv1 = conv_keep_in(s3b1.conv1, k)
    s3b1.conv2 = conv_keep_out(s3b1.conv2, k)
    s3b1.bn2   = bn_keep(s3b1.bn2, k)

    # ── Stage3 block2（恒等块）conv1 ──────────────────────────────
    k = get_keep_idx(s3b1.conv1.weight.data, rates["stage3_block2_conv1"])
    s3b1.conv1 = conv_keep_out(s3b1.conv1, k)
    s3b1.bn1   = bn_keep(s3b1.bn1, k)
    keep_out_s3b1 = torch.arange(s3b1.conv2.out_channels)
    s3b1.conv2 = conv_keep_in_out(s3b1.conv2, k, keep_out_s3b1)

    # ── 跨Stage传递：s3b1输出通道 → s4b0输入通道 ─────────────────
    keep_cross = torch.arange(s3b1.conv2.out_channels)
    s4b0.conv1 = conv_keep_in(s4b0.conv1, keep_cross)
    s4b0.conv3 = conv_keep_in(s4b0.conv3, keep_cross)

    # ── Stage4 block1（下采样块）conv1 ────────────────────────────
    k = get_keep_idx(s4b0.conv1.weight.data, rates["stage4_block1_conv1"])
    s4b0.conv1 = conv_keep_out(s4b0.conv1, k)
    s4b0.bn1   = bn_keep(s4b0.bn1, k)
    s4b0.conv2 = conv_keep_in(s4b0.conv2, k)

    # ── Stage4 block1（下采样块）conv2 + conv3 联合剪枝 ───────────
    k = get_keep_idx(s4b0.conv3.weight.data, rates["stage4_block1_conv2_shortcut"])
    s4b0.conv2 = conv_keep_out(s4b0.conv2, k)
    s4b0.bn2   = bn_keep(s4b0.bn2, k)
    s4b0.conv3 = conv_keep_out(s4b0.conv3, k)
    s4b1.conv1 = conv_keep_in(s4b1.conv1, k)
    s4b1.conv2 = conv_keep_out(s4b1.conv2, k)
    s4b1.bn2   = bn_keep(s4b1.bn2, k)

    # ── Stage4 block2（恒等块）conv1 ──────────────────────────────
    # stage4最后，conv2输出通道决定Linear输入维度
    # conv2输出已被上一步固定为len(k_stage4)，Linear输入也要跟着改
    k = get_keep_idx(s4b1.conv1.weight.data, rates["stage4_block2_conv1"])
    s4b1.conv1 = conv_keep_out(s4b1.conv1, k)
    s4b1.bn1   = bn_keep(s4b1.bn1, k)
    keep_out_s4b1 = torch.arange(s4b1.conv2.out_channels)
    s4b1.conv2 = conv_keep_in_out(s4b1.conv2, k, keep_out_s4b1)

    # ── Linear层输入通道跟随stage4输出 ───────────────────────────
    # stage4 block2的conv2输出通道 = s4b1.conv2.out_channels（已更新）
    final_ch = s4b1.conv2.out_channels
    old_linear = m[7]  # m[7]是nn.Linear（m[5]=AvgPool, m[6]=Flatten, m[7]=Linear）
    new_linear = nn.Linear(final_ch, old_linear.out_features)
    new_linear.weight.data = old_linear.weight.data[:, :final_ch].clone()
    new_linear.bias.data   = old_linear.bias.data.clone()
    m[7] = new_linear

    return m


# ================================================================
# FLOPs计算
# ================================================================
def calc_model_flops(model):
    total = [0]
    count = [0]
    handles = []

    def hook(mod, inp, out):
        count[0] += 1
        if count[0] == 1:
            return
        _, ic, _, _ = inp[0].shape
        kh, kw = mod.kernel_size
        oh, ow = out.shape[2], out.shape[3]
        total[0] += kh * kw * ic * mod.out_channels * oh * ow

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))

    for h in handles:
        h.remove()
    return total[0]


# ================================================================
# 验证
# ================================================================
def load_val_loader(batch_size):
    samples = load_split_from_file(VAL_SPLIT_FILE)
    _, val_transform = get_transforms()
    ds = AerialLandscapeDataset(DATA_ROOT, samples, transform=val_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, preds = torch.max(model(inputs), 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ================================================================
# 主流程
# ================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = get_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                     weights_only=False))
    model.to(device).eval()

    val_loader = load_val_loader(BATCH_SIZE)

    baseline_acc = evaluate(model, val_loader, device)
    print(f"Baseline acc:   {baseline_acc:.4f}")
    print(f"Baseline FLOPs: {TOTAL_FLOPS_ORIGINAL / 1e9:.3f} G\n")

    flops_norm = divmax_normalize(FLOPS_RAW)
    sens_norm  = minmax_normalize(SENSITIVITY_RAW)

    print("归一化FLOPs:")
    for k, v in flops_norm.items():
        print(f"  {k:<40} {v:.4f}")
    print("\n归一化敏感度:")
    for k, v in sens_norm.items():
        print(f"  {k:<40} {v:.4f}")
    print()

    results = []
    header = f"{'b':>4} {'alpha':>6} | {'Acc':>7} {'ΔAcc':>7} {'FLOPs(G)':>9} {'压缩率':>8} {'score':>8}"
    print(header)
    print("-" * len(header))

    for b in B_LIST:
        for alpha in ALPHA_LIST:
            rates = compute_pruning_rates(alpha, b, flops_norm, sens_norm)

            if b == B_LIST[0] and alpha == ALPHA_LIST[0]:
                print(f"\n[示例] b={b}, alpha={alpha} 各层剪枝率:")
                for layer, r in rates.items():
                    print(f"  {layer:<40} {r:.4f}")
                print()

            pruned = physical_prune(model.cpu(), rates)
            pruned.to(device)

            acc   = evaluate(pruned, val_loader, device)
            flops = calc_model_flops(pruned.cpu())

            delta          = baseline_acc - acc
            flops_ratio    = flops / TOTAL_FLOPS_ORIGINAL
            flops_reduc    = 1.0 - flops_ratio
            score          = (acc / baseline_acc) / flops_ratio

            results.append(dict(b=b, alpha=alpha, acc=acc, delta=delta,
                                flops=flops, flops_reduction=flops_reduc,
                                score=score, rates=rates))

            print(f"{b:>4} {alpha:>6.2f} | {acc:>7.4f} {delta:>+7.4f} "
                  f"{flops/1e9:>9.3f} {flops_reduc:>8.2%} {score:>8.4f}")

            del pruned
            torch.cuda.empty_cache()
            model.to(device)

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*62}")
    print("综合评分排序（score = acc保留率 / FLOPs保留率，越高越好）")
    print(f"{'='*62}")
    print(f"{'Rank':>5} {'b':>4} {'alpha':>6} {'Acc':>7} {'压缩率':>8} {'score':>8}")
    print("-" * 45)
    for rank, r in enumerate(results_sorted, 1):
        print(f"{rank:>5} {r['b']:>4} {r['alpha']:>6.2f} "
              f"{r['acc']:>7.4f} {r['flops_reduction']:>8.2%} {r['score']:>8.4f}")

    out_path = os.path.join(OUTPUT_DIR, "pruning_screen_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Baseline acc: {baseline_acc:.4f}, FLOPs: {TOTAL_FLOPS_ORIGINAL/1e9:.3f}G\n\n")
        f.write("各组剪枝率明细\n" + "=" * 100 + "\n")
        for r in results:
            f.write(f"b={r['b']}, alpha={r['alpha']:.2f} | acc={r['acc']:.4f} "
                    f"delta={r['delta']:+.4f} FLOPs={r['flops']/1e9:.3f}G "
                    f"压缩={r['flops_reduction']:.2%} score={r['score']:.4f}\n")
            for layer, rate in r["rates"].items():
                f.write(f"    {layer:<40} r={rate:.4f}\n")
            f.write("\n")
        f.write("\n综合评分排序\n")
        f.write(f"{'Rank':>5} {'b':>4} {'alpha':>6} {'Acc':>7} {'压缩率':>8} {'score':>8}\n")
        f.write("-" * 50 + "\n")
        for rank, r in enumerate(results_sorted, 1):
            f.write(f"{rank:>5} {r['b']:>4} {r['alpha']:>6.2f} "
                    f"{r['acc']:>7.4f} {r['flops_reduction']:>8.2%} {r['score']:>8.4f}\n")

    print(f"\n详细结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
