import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import copy
import numpy as np

import os as _os
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(_os.environ.get("PRUNING_ROOT", _Path(__file__).resolve().parents[1])).resolve()
_sys.path.insert(0, str(ROOT / "src"))

from models.resnet import get_resnet18, Residual
from data.dataset import AerialLandscapeDataset, get_transforms
from utils.split_data import load_split_from_file

# ==================== 配置 ====================
DATA_ROOT = os.environ.get("AERIAL_DATA_ROOT", str(ROOT / "data" / "Aerial_Landscapes"))
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints" / "best_resnet18_aerial.pth"))
VAL_SPLIT_FILE = os.environ.get("VAL_SPLIT_FILE", str(ROOT / "splits" / "aerial_split_val.txt"))
BATCH_SIZE = 256
PRUNING_RATES = [0.05, 0.10, 0.15, 0.20]
NUM_REPEATS = 4  # 每个剪枝率测量4次


# ==================== 工具函数 ====================

def get_conv_layers(model):
    """
    获取模型中所有可剪枝的卷积层（排除第一个7×7卷积和最后的全连接层）
    """
    layer_info = []

    # 遍历模型的所有模块来找残差块（索引1-4是b2-b5）
    for stage_idx in range(1, 5):  # 1=b2, 2=b3, 3=b4, 4=b5
        stage = model[stage_idx]
        for block_idx, block in enumerate(stage):
            if isinstance(block, Residual):
                # 判断是否是下采样块（有 conv3）
                is_downsample = block.conv3 is not None

                # 构建层名称
                block_name = f"{stage_idx}.{block_idx}"

                # 第一个卷积层（总是可剪枝）
                layer_info.append({
                    'name': f"{block_name}.conv1",
                    'bn_name': f"{block_name}.bn1",
                    'module': block.conv1,
                    'bn': block.bn1,
                    'is_downsample_block': is_downsample,
                    'conv_position': 'first',
                    'block': block,
                    'stage_idx': stage_idx,
                    'block_idx': block_idx
                })

                # 第二个卷积层（根据规则决定是否剪枝）
                if is_downsample:
                    # 下采样块：第二层需要根据 shortcut 剪枝
                    layer_info.append({
                        'name': f"{block_name}.conv2",
                        'bn_name': f"{block_name}.bn2",
                        'module': block.conv2,
                        'bn': block.bn2,
                        'is_downsample_block': is_downsample,
                        'conv_position': 'second_downsample',
                        'block': block,
                        'shortcut': block.conv3,  # shortcut 层（没有 BN）
                        'stage_idx': stage_idx,
                        'block_idx': block_idx
                    })
                else:
                    # 恒等映射块：第二层不剪枝
                    layer_info.append({
                        'name': f"{block_name}.conv2",
                        'bn_name': f"{block_name}.bn2",
                        'module': block.conv2,
                        'bn': block.bn2,
                        'is_downsample_block': is_downsample,
                        'conv_position': 'second_identity',
                        'block': block,
                        'prunable': False,
                        'stage_idx': stage_idx,
                        'block_idx': block_idx
                    })

    return layer_info


def compute_l1_norm(conv_layer):
    """计算卷积核的 L1-norm (按输出通道)"""
    weight = conv_layer.weight.data
    l1_norm = weight.abs().view(weight.size(0), -1).sum(dim=1)
    return l1_norm


def get_prune_indices(l1_norm, pruning_rate):
    """根据 L1-norm 获取要剪枝的通道索引"""
    n_channels = l1_norm.size(0)
    n_prune = int(n_channels * pruning_rate)

    if n_prune <= 0:
        return torch.tensor([], dtype=torch.long, device=l1_norm.device)

    _, indices = torch.topk(l1_norm, n_prune, largest=False)
    return indices.sort()[0]


def prune_conv_layer_and_bn(conv_layer, bn_layer, prune_indices):
    """
    对卷积层和其对应的BN层进行剪枝
    """
    if len(prune_indices) == 0:
        keep_indices = torch.arange(conv_layer.out_channels, device=conv_layer.weight.device)
    else:
        n_channels = conv_layer.out_channels
        keep_indices = torch.ones(n_channels, dtype=torch.bool, device=conv_layer.weight.device)
        keep_indices[prune_indices] = False

    # 剪枝卷积层
    weight = conv_layer.weight.data[keep_indices]
    conv_layer.weight = nn.Parameter(weight)
    conv_layer.out_channels = weight.size(0)

    if conv_layer.bias is not None:
        bias = conv_layer.bias.data[keep_indices]
        conv_layer.bias = nn.Parameter(bias)

    # 剪枝BN层（如果有）
    if bn_layer is not None:
        bn_layer.weight = nn.Parameter(bn_layer.weight.data[keep_indices])
        bn_layer.bias = nn.Parameter(bn_layer.bias.data[keep_indices])
        bn_layer.running_mean = bn_layer.running_mean.data[keep_indices]
        bn_layer.running_var = bn_layer.running_var.data[keep_indices]
        bn_layer.num_features = keep_indices.sum().item()

    return keep_indices


def prune_conv_layer_input(conv_layer, prune_indices):
    """
    对卷积层的输入通道进行剪枝
    """
    if len(prune_indices) == 0:
        return

    n_channels = conv_layer.in_channels
    keep_indices = torch.ones(n_channels, dtype=torch.bool, device=conv_layer.weight.device)
    keep_indices[prune_indices] = False

    weight = conv_layer.weight.data[:, keep_indices, :, :]
    conv_layer.weight = nn.Parameter(weight)
    conv_layer.in_channels = weight.size(1)


def prune_shortcut_conv(conv_layer, prune_indices):
    """
    对 shortcut 的 1x1 卷积层进行剪枝（没有 BN）
    """
    if len(prune_indices) == 0:
        keep_indices = torch.arange(conv_layer.out_channels, device=conv_layer.weight.device)
    else:
        n_channels = conv_layer.out_channels
        keep_indices = torch.ones(n_channels, dtype=torch.bool, device=conv_layer.weight.device)
        keep_indices[prune_indices] = False

    weight = conv_layer.weight.data[keep_indices]
    conv_layer.weight = nn.Parameter(weight)
    conv_layer.out_channels = weight.size(0)

    if conv_layer.bias is not None:
        bias = conv_layer.bias.data[keep_indices]
        conv_layer.bias = nn.Parameter(bias)

    return keep_indices


def get_module_by_name(model, name):
    """根据名称获取模型中的模块"""
    modules = name.split('.')
    module = model
    for m in modules:
        if m.isdigit():
            module = module[int(m)]
        else:
            module = getattr(module, m)
    return module


def update_stage_block_inputs(model, stage_idx, start_block_idx, prune_indices):
    """
    更新同一stage内后续block的输入通道
    """
    stage = model[stage_idx]
    for block_idx in range(start_block_idx + 1, len(stage)):
        # 更新后续block的conv1输入通道
        next_conv1_name = f"{stage_idx}.{block_idx}.conv1"
        next_conv1 = get_module_by_name(model, next_conv1_name)
        prune_conv_layer_input(next_conv1, prune_indices)

        # 如果后续block也是下采样块，还需要更新其conv2的输入通道
        block = stage[block_idx]
        if isinstance(block, Residual) and block.conv3 is not None:  # 下采样块
            next_conv2_name = f"{stage_idx}.{block_idx}.conv2"
            next_conv2 = get_module_by_name(model, next_conv2_name)
            prune_conv_layer_input(next_conv2, prune_indices)


def create_pruned_model(original_model, layer_to_prune, pruning_rate):
    """
    创建剪枝后的模型副本
    """
    # 深拷贝模型
    pruned_model = copy.deepcopy(original_model)

    # 获取目标层
    target_layer = get_module_by_name(pruned_model, layer_to_prune['name'])
    target_bn = get_module_by_name(pruned_model, layer_to_prune['bn_name'])

    # 计算 L1-norm 并获取剪枝索引
    l1_norm = compute_l1_norm(target_layer)
    prune_indices = get_prune_indices(l1_norm, pruning_rate)

    # 根据层的位置执行不同的剪枝策略
    if layer_to_prune['conv_position'] == 'first':
        # 第一个卷积层：正常剪枝
        keep_mask = prune_conv_layer_and_bn(target_layer, target_bn, prune_indices)

        # 如果是下采样块，需要同步剪枝 shortcut 层（conv3，没有 BN）
        if layer_to_prune['is_downsample_block']:
            shortcut_name = layer_to_prune['name'].replace('conv1', 'conv3')
            shortcut_layer = get_module_by_name(pruned_model, shortcut_name)
            prune_shortcut_conv(shortcut_layer, prune_indices)

        # 找到同一 block 的 conv2 并剪枝其输入通道
        conv2_name = layer_to_prune['name'].replace('conv1', 'conv2')
        conv2_layer = get_module_by_name(pruned_model, conv2_name)
        prune_conv_layer_input(conv2_layer, prune_indices)

        # ❗ 修复1：更新同一stage内后续block的输入通道
        update_stage_block_inputs(pruned_model,
                                  layer_to_prune['stage_idx'],
                                  layer_to_prune['block_idx'],
                                  prune_indices)

    elif layer_to_prune['conv_position'] == 'second_downsample':
        # 下采样块的第二层：需要先处理conv1，再处理conv2
        conv1_name = layer_to_prune['name'].replace('conv2', 'conv1')
        conv1_layer = get_module_by_name(pruned_model, conv1_name)
        conv1_bn = get_module_by_name(pruned_model, conv1_name.replace('conv', 'bn'))

        # 先计算conv1的L1-norm和剪枝索引
        conv1_l1 = compute_l1_norm(conv1_layer)
        conv1_prune_indices = get_prune_indices(conv1_l1, pruning_rate)

        # 剪枝conv1和其BN
        conv1_keep_mask = prune_conv_layer_and_bn(conv1_layer, conv1_bn, conv1_prune_indices)

        # 剪枝shortcut（根据conv1的剪枝结果）
        shortcut_name = layer_to_prune['name'].replace('conv2', 'conv3')
        shortcut_layer = get_module_by_name(pruned_model, shortcut_name)
        shortcut_keep_mask = prune_shortcut_conv(shortcut_layer, conv1_prune_indices)

        # 现在剪枝conv2，输出通道与shortcut保持一致，输入通道与conv1输出保持一致
        # 由于conv1输出通道已经改变，需要先剪枝conv2的输入通道
        prune_conv_layer_input(target_layer, conv1_prune_indices)

        # 然后剪枝conv2的输出通道（与shortcut一致）
        prune_conv_layer_and_bn(target_layer, target_bn, conv1_prune_indices)

        # ❗ 修复2：更新同一stage内后续block的输入通道
        update_stage_block_inputs(pruned_model,
                                  layer_to_prune['stage_idx'],
                                  layer_to_prune['block_idx'],
                                  conv1_prune_indices)

    elif layer_to_prune['conv_position'] == 'second_identity':
        # 恒等映射块的第二层：不剪枝
        pass

    return pruned_model


def evaluate_model(model, val_loader, device):
    """评估模型在验证集上的 top-1 准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def analyze_layer_sensitivity(original_model, layer_info, val_loader, device):
    """分析单个层的敏感度"""
    if not layer_info.get('prunable', True):
        return 0.0, {'skipped': True, 'reason': '恒等映射块第二层不剪枝'}

    # 获取原始准确率
    original_acc = evaluate_model(original_model, val_loader, device)

    delta_accs = []

    for pruning_rate in PRUNING_RATES:
        acc_diffs = []

        for repeat in range(NUM_REPEATS):
            try:
                pruned_model = create_pruned_model(original_model, layer_info, pruning_rate)
                pruned_acc = evaluate_model(pruned_model, val_loader, device)
                delta_acc = original_acc - pruned_acc
                acc_diffs.append(delta_acc)
            except Exception as e:
                print(f"    Error at rate {pruning_rate}: {str(e)}")
                acc_diffs.append(1.0)

        avg_delta = np.mean(acc_diffs)
        delta_accs.append((pruning_rate, avg_delta))
        print(f"    剪枝率 {pruning_rate * 100:.0f}%: 平均精度下降 {avg_delta * 100:.2f}%")

    # 计算平均下降斜率
    slopes = [delta / rate for rate, delta in delta_accs if rate > 0]
    sensitivity = np.mean(slopes) if len(slopes) > 0 else 0.0

    return sensitivity, {
        'original_acc': original_acc,
        'delta_accs': delta_accs,
        'slopes': slopes,
        'sensitivity': sensitivity
    }


# ==================== 主函数 ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载验证数据集
    print("Loading validation dataset...")
    val_samples = load_split_from_file(VAL_SPLIT_FILE)
    _, val_transform = get_transforms()
    val_dataset = AerialLandscapeDataset(DATA_ROOT, val_samples, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)

    # 加载原始模型
    print(f"Loading model from {MODEL_PATH}...")
    model = get_resnet18(num_classes=15)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 获取原始准确率
    print("Evaluating original model...")
    original_acc = evaluate_model(model, val_loader, device)
    print(f"Original Validation Accuracy: {original_acc * 100:.2f}%")

    # 获取所有可剪枝层
    print("Identifying prunable layers...")
    layer_info_list = get_conv_layers(model)
    print(f"Found {len(layer_info_list)} conv layers to analyze")

    for i, info in enumerate(layer_info_list):
        print(
            f"  {i + 1}. {info['name']} (downsample: {info['is_downsample_block']}, prunable: {info.get('prunable', True)})")

    # 逐层分析敏感度
    print("\n" + "=" * 60)
    print("Starting layer sensitivity analysis...")
    print("=" * 60 + "\n")

    results = []

    for idx, layer_info in enumerate(layer_info_list):
        print(f"\n[{idx + 1}/{len(layer_info_list)}] Analyzing {layer_info['name']}...")

        if not layer_info.get('prunable', True):
            print(f"  -> Skipped (恒等映射块第二层)")
            results.append({
                'layer_name': layer_info['name'],
                'sensitivity': 0.0,
                'skipped': True,
                'reason': '恒等映射块第二层不剪枝'
            })
            continue

        # 重新加载原始模型
        model = get_resnet18(num_classes=15)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        sensitivity, details = analyze_layer_sensitivity(model, layer_info, val_loader, device)

        results.append({
            'layer_name': layer_info['name'],
            'sensitivity': sensitivity,
            'skipped': False,
            'details': details
        })

        print(f"  -> Sensitivity: {sensitivity:.4f}")

    # 保存结果
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)

    # 打印敏感度排名
    print("\n层敏感度排名（从高到低）:")
    print("-" * 60)
    prunable_results = [r for r in results if not r.get('skipped', False)]
    sorted_results = sorted(prunable_results, key=lambda x: x['sensitivity'], reverse=True)

    for idx, r in enumerate(sorted_results):
        print(f"{idx + 1:2d}. {r['layer_name']:30s}  Sensitivity: {r['sensitivity']:.4f}")

    # 保存 JSON
    import json
    with open(os.path.join(ROOT, 'results', 'layer_sensitivity_results.json'), 'w', encoding='utf-8') as f:
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            if 'details' in r_copy and r_copy['details']:
                details = r_copy['details'].copy()
                if 'delta_accs' in details:
                    details['delta_accs'] = [(float(r), float(d)) for r, d in details['delta_accs']]
                if 'slopes' in details:
                    details['slopes'] = [float(s) for s in details['slopes']]
                r_copy['details'] = details
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to 'layer_sensitivity_results.json'")

    # 保存 CSV
    import csv
    with open(os.path.join(ROOT, 'results', 'layer_sensitivity_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Layer Name', 'Sensitivity', 'Skipped', 'Reason'])
        for idx, r in enumerate(sorted_results):
            writer.writerow([idx + 1, r['layer_name'], f"{r['sensitivity']:.4f}", 'No', '-'])
        for r in results:
            if r.get('skipped', False):
                writer.writerow(['-', r['layer_name'], '0.0000', 'Yes', r.get('reason', '')])
    print(f"Results saved to 'layer_sensitivity_results.csv'")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()