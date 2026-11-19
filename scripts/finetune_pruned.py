"""
finetune_pruned.py

对粗筛选出的4组剪枝配置分别执行物理剪枝 + 微调，保存最佳模型。

选定的4组：
  (b=5,  alpha=0.30)  压缩29.54%  综合score最高
  (b=10, alpha=0.30)  压缩24.97%  未微调精度最高
  (b=5,  alpha=0.40)  压缩38.71%  中等激进
  (b=10, alpha=0.40)  压缩32.82%  与上组对比b的影响

微调超参数设计：
  - 参考论文恒定lr=0.001，改进为CosineAnnealingLR（更平滑的衰减）
  - Epochs = 25（原始训练90的约四分之一，比论文略多以充分恢复）
  - Early stopping patience = 8（防止过拟合）
  - Optimizer: SGD + momentum=0.9 + weight_decay=1e-4（与原始训练一致）
  - 初始lr = 0.001（论文值），CosineAnnealingLR衰减至1e-5
"""

import sys
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
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
TRAIN_SPLIT_FILE = os.environ.get("TRAIN_SPLIT_FILE", str(ROOT / "splits" / "aerial_split_train.txt"))
VAL_SPLIT_FILE = os.environ.get("VAL_SPLIT_FILE", str(ROOT / "splits" / "aerial_split_val.txt"))
DATA_ROOT = os.environ.get("AERIAL_DATA_ROOT", str(ROOT / "data" / "Aerial_Landscapes"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(ROOT / "checkpoints" / "pruned_models"))
NUM_CLASSES     = 15
BATCH_SIZE      = 256

# ================================================================
# 微调超参数
# ================================================================
FINETUNE_EPOCHS = 25
LR_INIT         = 0.001
LR_MIN          = 1e-5
WEIGHT_DECAY    = 1e-4
MOMENTUM        = 0.9
PATIENCE        = 8      # early stopping

# ================================================================
# 待微调的4组配置
# ================================================================
CANDIDATES = [
    {"b": 5,  "alpha": 0.30},
    {"b": 10, "alpha": 0.30},
    {"b": 5,  "alpha": 0.40},
    {"b": 10, "alpha": 0.40},
]

# ================================================================
# 敏感度 & FLOPs（与Pruning_screen.py保持一致）
# ================================================================
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
BASELINE_ACC         = 0.8758


# ================================================================
# 归一化 & 剪枝率计算
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
# 剪枝工具函数
# ================================================================
def get_keep_idx(weight, prune_ratio):
    n = weight.shape[0]
    n_keep = max(1, n - max(1, int(n * prune_ratio)))
    l1 = weight.abs().sum(dim=[1, 2, 3])
    _, sorted_idx = torch.sort(l1, descending=True)
    keep, _ = torch.sort(sorted_idx[:n_keep])
    return keep

def conv_keep_out(conv, keep):
    nc = nn.Conv2d(conv.in_channels, len(keep),
                   kernel_size=conv.kernel_size, stride=conv.stride,
                   padding=conv.padding, bias=(conv.bias is not None))
    nc.weight.data = conv.weight.data[keep].clone()
    if conv.bias is not None:
        nc.bias.data = conv.bias.data[keep].clone()
    return nc

def conv_keep_in(conv, keep):
    nc = nn.Conv2d(len(keep), conv.out_channels,
                   kernel_size=conv.kernel_size, stride=conv.stride,
                   padding=conv.padding, bias=(conv.bias is not None))
    nc.weight.data = conv.weight.data[:, keep, :, :].clone()
    if conv.bias is not None:
        nc.bias.data = conv.bias.data.clone()
    return nc

def bn_keep(bn, keep):
    nb = nn.BatchNorm2d(len(keep))
    nb.weight.data  = bn.weight.data[keep].clone()
    nb.bias.data    = bn.bias.data[keep].clone()
    nb.running_mean = bn.running_mean[keep].clone()
    nb.running_var  = bn.running_var[keep].clone()
    nb.eps          = bn.eps
    nb.momentum     = bn.momentum
    return nb


# ================================================================
# 物理剪枝（与Pruning_screen.py完全一致）
# ================================================================
def physical_prune(model, rates):
    m = copy.deepcopy(model)

    s1b0 = m[1][0]; s1b1 = m[1][1]
    s2b0 = m[2][0]; s2b1 = m[2][1]
    s3b0 = m[3][0]; s3b1 = m[3][1]
    s4b0 = m[4][0]; s4b1 = m[4][1]

    # Stage1
    k = get_keep_idx(s1b0.conv1.weight.data, rates["stage1_block1_conv1"])
    s1b0.conv1 = conv_keep_out(s1b0.conv1, k); s1b0.bn1 = bn_keep(s1b0.bn1, k)
    s1b0.conv2 = conv_keep_in(s1b0.conv2, k)

    k = get_keep_idx(s1b1.conv1.weight.data, rates["stage1_block2_conv1"])
    s1b1.conv1 = conv_keep_out(s1b1.conv1, k); s1b1.bn1 = bn_keep(s1b1.bn1, k)
    s1b1.conv2 = conv_keep_in(s1b1.conv2, k)

    # Stage2 ds_block conv1
    k = get_keep_idx(s2b0.conv1.weight.data, rates["stage2_block1_conv1"])
    s2b0.conv1 = conv_keep_out(s2b0.conv1, k); s2b0.bn1 = bn_keep(s2b0.bn1, k)
    s2b0.conv2 = conv_keep_in(s2b0.conv2, k)

    # Stage2 ds_block conv2+conv3 联合
    k2 = get_keep_idx(s2b0.conv3.weight.data, rates["stage2_block1_conv2_shortcut"])
    s2b0.conv2 = conv_keep_out(s2b0.conv2, k2); s2b0.bn2 = bn_keep(s2b0.bn2, k2)
    s2b0.conv3 = conv_keep_out(s2b0.conv3, k2)

    # Stage2 id_block step3: 对齐block输入
    s2b1.conv1 = conv_keep_in(s2b1.conv1, k2)
    s2b1.conv2 = conv_keep_out(s2b1.conv2, k2); s2b1.bn2 = bn_keep(s2b1.bn2, k2)

    # Stage2 id_block step4: conv1自身剪枝
    k = get_keep_idx(s2b1.conv1.weight.data, rates["stage2_block2_conv1"])
    s2b1.conv1 = conv_keep_out(s2b1.conv1, k); s2b1.bn1 = bn_keep(s2b1.bn1, k)
    s2b1.conv2 = conv_keep_in(s2b1.conv2, k)

    # 跨stage: s2→s3
    s3b0.conv1 = conv_keep_in(s3b0.conv1, k2)
    s3b0.conv3 = conv_keep_in(s3b0.conv3, k2)

    # Stage3 ds_block conv1
    k = get_keep_idx(s3b0.conv1.weight.data, rates["stage3_block1_conv1"])
    s3b0.conv1 = conv_keep_out(s3b0.conv1, k); s3b0.bn1 = bn_keep(s3b0.bn1, k)
    s3b0.conv2 = conv_keep_in(s3b0.conv2, k)

    # Stage3 ds_block 联合
    k3 = get_keep_idx(s3b0.conv3.weight.data, rates["stage3_block1_conv2_shortcut"])
    s3b0.conv2 = conv_keep_out(s3b0.conv2, k3); s3b0.bn2 = bn_keep(s3b0.bn2, k3)
    s3b0.conv3 = conv_keep_out(s3b0.conv3, k3)

    # Stage3 id_block step3
    s3b1.conv1 = conv_keep_in(s3b1.conv1, k3)
    s3b1.conv2 = conv_keep_out(s3b1.conv2, k3); s3b1.bn2 = bn_keep(s3b1.bn2, k3)

    # Stage3 id_block step4
    k = get_keep_idx(s3b1.conv1.weight.data, rates["stage3_block2_conv1"])
    s3b1.conv1 = conv_keep_out(s3b1.conv1, k); s3b1.bn1 = bn_keep(s3b1.bn1, k)
    s3b1.conv2 = conv_keep_in(s3b1.conv2, k)

    # 跨stage: s3→s4
    s4b0.conv1 = conv_keep_in(s4b0.conv1, k3)
    s4b0.conv3 = conv_keep_in(s4b0.conv3, k3)

    # Stage4 ds_block conv1
    k = get_keep_idx(s4b0.conv1.weight.data, rates["stage4_block1_conv1"])
    s4b0.conv1 = conv_keep_out(s4b0.conv1, k); s4b0.bn1 = bn_keep(s4b0.bn1, k)
    s4b0.conv2 = conv_keep_in(s4b0.conv2, k)

    # Stage4 ds_block 联合
    k4 = get_keep_idx(s4b0.conv3.weight.data, rates["stage4_block1_conv2_shortcut"])
    s4b0.conv2 = conv_keep_out(s4b0.conv2, k4); s4b0.bn2 = bn_keep(s4b0.bn2, k4)
    s4b0.conv3 = conv_keep_out(s4b0.conv3, k4)

    # Stage4 id_block step3
    s4b1.conv1 = conv_keep_in(s4b1.conv1, k4)
    s4b1.conv2 = conv_keep_out(s4b1.conv2, k4); s4b1.bn2 = bn_keep(s4b1.bn2, k4)

    # Stage4 id_block step4
    k = get_keep_idx(s4b1.conv1.weight.data, rates["stage4_block2_conv1"])
    s4b1.conv1 = conv_keep_out(s4b1.conv1, k); s4b1.bn1 = bn_keep(s4b1.bn1, k)
    s4b1.conv2 = conv_keep_in(s4b1.conv2, k)

    # Linear
    old_fc = m[7]
    new_fc = nn.Linear(len(k4), old_fc.out_features)
    new_fc.weight.data = old_fc.weight.data[:, k4].clone()
    new_fc.bias.data   = old_fc.bias.data.clone()
    m[7] = new_fc

    return m


# ================================================================
# FLOPs计算
# ================================================================
def calc_model_flops(model):
    total = [0]; count = [0]; handles = []
    def hook(mod, inp, out):
        count[0] += 1
        if count[0] == 1: return
        _, ic, _, _ = inp[0].shape
        kh, kw = mod.kernel_size
        oh, ow = out.shape[2], out.shape[3]
        total[0] += kh * kw * ic * mod.out_channels * oh * ow
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for h in handles: h.remove()
    return total[0]


# ================================================================
# 数据加载
# ================================================================
def load_dataloaders(batch_size):
    train_samples = load_split_from_file(TRAIN_SPLIT_FILE)
    val_samples   = load_split_from_file(VAL_SPLIT_FILE)
    train_transform, val_transform = get_transforms()
    train_ds = AerialLandscapeDataset(DATA_ROOT, train_samples, transform=train_transform)
    val_ds   = AerialLandscapeDataset(DATA_ROOT, val_samples,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ================================================================
# 验证
# ================================================================
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
# 微调单个模型
# ================================================================
def finetune(model, train_loader, val_loader, tag, device, log_file):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_INIT,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=LR_MIN)

    best_acc   = 0.0
    best_wts   = copy.deepcopy(model.state_dict())
    no_improve = 0
    save_path  = os.path.join(OUTPUT_DIR, f"pruned_{tag}.pth")

    print(f"\n{'='*60}")
    print(f"微调: {tag}")
    print(f"{'='*60}")
    log_file.write(f"\n{'='*60}\n微调: {tag}\n{'='*60}\n")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        t0 = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs.detach(), 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc  = correct / total
        train_loss = running_loss / len(train_loader)
        val_acc    = evaluate(model, val_loader, device)
        lr_now     = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        scheduler.step()

        line = (f"Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
                f"loss={train_loss:.4f} train={train_acc:.4f} "
                f"val={val_acc:.4f} lr={lr_now:.2e} t={elapsed:.1f}s")
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, save_path)
            print(f"  -> 最佳模型已保存 val_acc={best_acc:.4f}")
            log_file.write(f"  -> 最佳模型已保存 val_acc={best_acc:.4f}\n")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}（{PATIENCE} epochs无提升）")
                log_file.write(f"Early stopping at epoch {epoch}\n")
                break

    model.load_state_dict(best_wts)
    return model, best_acc


# ================================================================
# 主流程
# ================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载原始模型
    base_model = get_resnet18(num_classes=NUM_CLASSES)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                          weights_only=False))
    base_model.eval()

    # 数据
    train_loader, val_loader = load_dataloaders(BATCH_SIZE)

    # 归一化
    flops_norm = divmax_normalize(FLOPS_RAW)
    sens_norm  = minmax_normalize(SENSITIVITY_RAW)

    # 汇总日志
    log_path = os.path.join(OUTPUT_DIR, "finetune_results.txt")
    summary  = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Baseline acc: {BASELINE_ACC:.4f}, "
                       f"FLOPs: {TOTAL_FLOPS_ORIGINAL/1e9:.3f}G\n\n")

        for cfg in CANDIDATES:
            b, alpha = cfg["b"], cfg["alpha"]
            tag = f"b{b}_alpha{alpha:.2f}"

            rates = compute_pruning_rates(alpha, b, flops_norm, sens_norm)
            pruned = physical_prune(base_model, rates)

            flops        = calc_model_flops(pruned.cpu())
            flops_reduc  = 1.0 - flops / TOTAL_FLOPS_ORIGINAL

            # 微调前精度（供对比）
            pruned.to(device)
            acc_before = evaluate(pruned, val_loader, device)
            pruned.cpu()

            # 微调
            finetuned, best_acc = finetune(
                pruned, train_loader, val_loader, tag, device, log_file)

            delta = BASELINE_ACC - best_acc
            score = (best_acc / BASELINE_ACC) / (flops / TOTAL_FLOPS_ORIGINAL)

            summary.append(dict(
                tag=tag, b=b, alpha=alpha,
                flops=flops, flops_reduction=flops_reduc,
                acc_before=acc_before,
                acc_after=best_acc,
                delta=delta, score=score
            ))

            result_line = (f"{tag} | 微调前={acc_before:.4f} 微调后={best_acc:.4f} "
                           f"ΔAcc={delta:+.4f} 压缩={flops_reduc:.2%} score={score:.4f}")
            print(f"\n{result_line}")
            log_file.write(f"\n{result_line}\n")

            # 释放显存
            del finetuned
            torch.cuda.empty_cache()

        # 汇总表
        print(f"\n\n{'='*70}")
        print("微调结果汇总（按微调后score排序）")
        print(f"{'='*70}")
        header = f"{'配置':<20} {'微调前':>8} {'微调后':>8} {'ΔAcc':>8} {'压缩率':>8} {'score':>8}"
        print(header); print("-" * len(header))
        log_file.write(f"\n\n{'='*70}\n微调结果汇总\n{header}\n{'-'*len(header)}\n")

        for r in sorted(summary, key=lambda x: x["score"], reverse=True):
            line = (f"{r['tag']:<20} {r['acc_before']:>8.4f} {r['acc_after']:>8.4f} "
                    f"{r['delta']:>+8.4f} {r['flops_reduction']:>8.2%} {r['score']:>8.4f}")
            print(line)
            log_file.write(line + "\n")

    print(f"\n日志已保存至: {log_path}")
    print(f"模型已保存至: {OUTPUT_DIR}/pruned_*.pth")


if __name__ == "__main__":
    main()