import os
import random


def split_and_save_dataset(root_dir, train_ratio=0.8, output_file_prefix="dataset_split"):
    """
    Splits dataset into train and validation sets and saves the splits to files.
    """
    classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    all_samples = []
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.jpg')]
        random.shuffle(imgs)
        n_train = int(len(imgs) * train_ratio)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]

        all_samples.extend([(img_path, class_to_idx[cls_name]) for img_path in train_imgs])
        all_samples.extend([(img_path, class_to_idx[cls_name]) for img_path in val_imgs])

    # 4:1 划分策略
    train_samples = [s for i, s in enumerate(all_samples) if i % 5 != 4]
    val_samples = [s for i, s in enumerate(all_samples) if i % 5 == 4]

    # 保存划分结果
    with open(f"{output_file_prefix}_train.txt", 'w') as f:
        for path, idx in train_samples:
            f.write(f"{path} {idx}\n")

    with open(f"{output_file_prefix}_val.txt", 'w') as f:
        for path, idx in val_samples:
            f.write(f"{path} {idx}\n")

    print(f"Dataset split saved to {output_file_prefix}_train.txt and {output_file_prefix}_val.txt")
    return train_samples, val_samples


def load_split_from_file(file_path):
    """从文件加载划分结果"""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_path = " ".join(parts[:-1])
            class_idx = int(parts[-1])
            samples.append((img_path, class_idx))
    return samples