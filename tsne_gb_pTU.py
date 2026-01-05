"""
Generate Independent t-SNE with Strict Mutual Exclusion.

Logic:
    - Independent t-SNE for Image and Text.
    - Discrete Color Mapping.
    - [Strict Mode]: If a sample has >= 2 labels from the selected classes, it is DISCARDED.
      Only samples with EXACTLY ONE label from the selected set are kept.
      This creates clean, separated islands by removing ambiguous samples.
    - [Priority Mode]: If strict is False, samples with >= 2 labels are kept,
      and labeled according to the order of classes provided in the command line.

Usage:
    # 1. 开启 strict 模式：丢弃重叠样本，画出分离的簇
    python tsne_raw_flickr.py --dset flickr25k --classes 0 12 16 19 23 --perplexity 30 --strict

    # 2. 关闭 strict 模式：保留重叠样本，但在着色时优先显示前面的类 (例如优先显示日落20，而非天空0)
    python tsne_raw_flickr.py --dset flickr25k --classes 20 0 12 16 23 --perplexity 30
"""

import argparse
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sympy import false

from utils.dset import PreDataset
from utils.viz_utils import labels_to_classes

DEFAULT_SPLITS = ('test', 'retrieval')
TSNE_SEED = 1234

MIRFLICKR_LABELS = [
    "0:sky", "1:clouds", "2:water", "3:sea", "4:river", "5:lake",
    "6:people", "7:portrait", "8:male", "9:female", "10:baby", "11:night",
    "12:plant_life", "13:tree", "14:flower", "15:animals", "16:dog", "17:bird",
    "18:food", "19:structures", "20:sunset", "21:indoor", "22:transport", "23:car"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Independent Raw dataset t-SNE visualizer')
    parser.add_argument('--data_path', type=str,
                        default='f:/project/pythonCode/Cross-modal retrieval/Granular-ball/GBR/data/Dataset',
                        help='root directory containing dataset .mat files')
    parser.add_argument('--dset', type=str, default='coco',
                        choices=['flickr25k', 'coco', 'nuswide'],
                        help='dataset name')
    parser.add_argument('--output_dir', type=str, default='raw_tsne_figs_strict_single_label_coco_fin',
                        help='directory to save t-SNE figures')
    parser.add_argument('--color_by_label', type=int, choices=[0, 1], default=1,
                        help='1: color points by label (default); 0: use fixed colors')
    parser.add_argument('--single_label_only', type=int, choices=[0, 1], default=1,
                        help='1: filter to samples that have exactly one label; 0: use all samples')
    parser.add_argument('--show_axis', type=int, choices=[0, 1], default=1,
                        help='1: keep axis visible on plots; 0: hide axis (clean view)')
    parser.add_argument('--show_cluster_ids', type=int, choices=[0, 1], default=1,
                        help='1: annotate each cluster with its label ID; 0: no annotation')
    # parser.add_argument('--cluster_id_labels', type=int, nargs='+', default=[1,25,32,38,39,43,69,71,72,76,78,79],
    #                     help='Label IDs whose annotations should be placed at their largest area sub-cluster instead of the centroid')

    parser.add_argument('--cluster_id_labels', type=int, nargs='+', default=None,
                        help='Label IDs whose annotations should be placed at their largest area sub-cluster instead of the centroid')

    # 核心参数：手动指定类别 (顺序决定非Strict模式下的优先级)
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Specific class IDs to visualize. ORDER MATTERS for priority labeling!')

    parser.add_argument('--top_k', type=int, default=40, help='Top K classes.')
    parser.add_argument('--bottom_k', type=int, default=None, help='Bottom K classes.')

    parser.add_argument('--perplexity', type=int, default=15,
                        help='t-SNE perplexity.')

    # [开关] 开启后，如果一个样本命中 >=2 个选中label，直接丢弃
    parser.add_argument('--strict', action='store_true',default=False,
                        help='If True, DROP samples that belong to >= 2 selected classes. Keep only pure samples.')

    return parser.parse_args()


def load_selected_splits(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, texts, labels = [], [], []
    for split in DEFAULT_SPLITS:
        dataset = PreDataset(
            data_path=args.data_path,
            data_split=split,
            dataname=args.dset,
            flag='target',
        )
        images.append(dataset.images)
        texts.append(dataset.texts)
        labels.append(dataset.labels)

    img_all = np.concatenate(images, axis=0).astype(np.float32)
    txt_all = np.concatenate(texts, axis=0).astype(np.float32)
    lab_all = np.concatenate(labels, axis=0).astype(np.float32)
    return img_all, txt_all, lab_all


def filter_single_label_samples(img_feats: np.ndarray,
                                 txt_feats: np.ndarray,
                                 labels_onehot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if labels_onehot.ndim != 2:
        return img_feats, txt_feats, labels_onehot
    label_sums = labels_onehot.sum(axis=1)
    mask = label_sums == 1
    if not np.any(mask):
        raise ValueError("No single-label samples found; cannot filter.")
    return img_feats[mask], txt_feats[mask], labels_onehot[mask]


def build_distinct_cmap(num_classes: int) -> ListedColormap:
    if num_classes <= 10:
        return plt.get_cmap('tab10', num_classes)
    if num_classes <= 20:
        return plt.get_cmap('tab20', num_classes)

    # Combine multiple qualitative palettes for mid-sized class counts
    base_maps = ['tab20', 'tab20b', 'tab20c', 'Set3']
    colors = []

    def append_filtered_colors(cmap_name: str) -> None:
        cmap = plt.get_cmap(cmap_name)
        new_colors = cmap(np.linspace(0, 1, cmap.N))
        for color in new_colors:
            if np.min(color[:3]) >= 0.95:  # skip near-white colors
                continue
            colors.append(color)
            if len(colors) >= num_classes:
                return

    for cmap_name in base_maps:
        append_filtered_colors(cmap_name)
        if len(colors) >= num_classes:
            break

    if len(colors) < num_classes:
        # Fallback: sample evenly from a high-diversity map
        extra_cmap = plt.get_cmap('gist_ncar')
        for color in extra_cmap(np.linspace(0, 1, num_classes - len(colors))):
            if np.min(color[:3]) >= 0.95:
                continue
            colors.append(color)
            if len(colors) >= num_classes:
                break

    return ListedColormap(colors[:num_classes])


def largest_cluster_center(points: np.ndarray,
                           eps: float = 5.0,
                           min_samples: int = 5) -> np.ndarray:
    if points.shape[0] == 0:
        return None
    if points.shape[0] <= min_samples:
        return points.mean(axis=0)

    # 自适应 eps：根据坐标标准差调节尺度，避免固定值不适配
    adaptive_eps = max(np.std(points, axis=0).mean() * 0.6, 1e-3)
    clustering = DBSCAN(eps=max(eps, adaptive_eps), min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    if unique_clusters.size == 0:
        return points.mean(axis=0)

    best_center = points.mean(axis=0)
    best_area = -np.inf
    for cluster_id in unique_clusters:
        cluster_points = points[labels == cluster_id]
        if cluster_points.size == 0:
            continue
        span_x = cluster_points[:, 0].max() - cluster_points[:, 0].min()
        span_y = cluster_points[:, 1].max() - cluster_points[:, 1].min()
        area = span_x * span_y
        if area > best_area:
            best_area = area
            best_center = cluster_points.mean(axis=0)

    return best_center


def filter_exclusive_and_priority(img_feats: np.ndarray,
                                  txt_feats: np.ndarray,
                                  labels_onehot: np.ndarray,
                                  target_ids: List[int],
                                  strict: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实现了严格互斥逻辑 + 优先级归类逻辑。
    """
    print(f'  [Filter] Processing classes: {target_ids}')
    print(f'  [Filter] Strategy: {"Strict Mutual Exclusion (Drop Overlaps)" if strict else "Priority Assignment (Keep Overlaps)"}')

    target_ids_array = np.array(target_ids)

    # 1. 制作掩码，只保留选中的列，其他列视为0
    col_mask = np.zeros(labels_onehot.shape[1], dtype=bool)
    col_mask[target_ids_array] = True

    labels_masked = labels_onehot.copy()
    labels_masked[:, ~col_mask] = 0

    # 2. 计算每个样本命中了几次
    hit_counts = labels_masked.sum(axis=1)

    # 3. 筛选行 (Row Selection)
    if strict:
        # 【严格模式】：丢弃重叠！
        # 只保留 hit_counts 正好等于 1 的样本
        valid_rows = hit_counts == 1
        dropped_count = np.sum(hit_counts > 1)
        print(f'  [Filter] Strict Mode ON. Dropped {dropped_count} samples because they had >= 2 labels.')
    else:
        # 【普通模式】：保留重叠
        # 只要 hit_counts > 0 即可
        valid_rows = hit_counts > 0

    if np.sum(valid_rows) == 0:
        raise ValueError("No samples left after filtering! Check your class IDs.")

    # 4. 提取数据
    img_filtered = img_feats[valid_rows]
    txt_filtered = txt_feats[valid_rows]
    labels_subset = labels_masked[valid_rows]

    # 5. 生成单标签 (Label Assignment)
    # 我们使用“优先级”逻辑：按照 target_ids 在列表中的顺序依次认领
    # 例如 target_ids = [20, 0]，如果样本同时是 20 和 0，它会被先遍历到的 20 认领

    final_labels = np.full(len(img_filtered), -1, dtype=int)

    for lbl in target_ids:
        # 找出当前属于该类别的样本
        is_class = labels_subset[:, lbl] == 1
        # 只有那些【尚未被前面高优先级类别认领】的样本，才会被当前类别认领
        mask_assign = is_class & (final_labels == -1)
        final_labels[mask_assign] = lbl

    # 兜底：如果还有 -1 (理论上不可能)，用 argmax 填充
    if np.any(final_labels == -1):
        print("  [Warning] Some samples were not assigned by priority logic. Fallback to argmax.")
        unset_mask = final_labels == -1
        final_labels[unset_mask] = np.argmax(labels_subset[unset_mask], axis=1)

    print(f'  [Filter] Retained {len(final_labels)} samples.')
    print(f'  [Debug] Classes present in final data: {np.unique(final_labels)}')

    return img_filtered, txt_filtered, final_labels


def filter_top_k_classes(img_feats, txt_feats, classes, top_k):
    unique_ids = np.unique(classes)
    if top_k is None or top_k >= len(unique_ids):
        return img_feats, txt_feats, classes
    unique, counts = np.unique(classes, return_counts=True)
    top_k_indices = np.argsort(counts)[::-1][:top_k]
    target_ids = unique[top_k_indices]
    mask = np.isin(classes, target_ids)
    return img_feats[mask], txt_feats[mask], classes[mask]


def filter_bottom_k_classes(img_feats, txt_feats, classes, bottom_k):
    unique_ids = np.unique(classes)
    if bottom_k is None or bottom_k >= len(unique_ids):
        return img_feats, txt_feats, classes
    unique, counts = np.unique(classes, return_counts=True)
    bottom_k_indices = np.argsort(counts)[:bottom_k]
    target_ids = unique[bottom_k_indices]
    mask = np.isin(classes, target_ids)
    return img_feats[mask], txt_feats[mask], classes[mask]


def run_tsne(data: np.ndarray, perplexity: int, desc: str = "t-SNE") -> np.ndarray:
    safe_perplexity = min(perplexity, max(data.shape[0] // 3, 5))
    print(f"  -> Starting {desc} (n={data.shape[0]}, perp={safe_perplexity})...")

    # === 关键修改 ===
    # 1. early_exaggeration=30: 强力推开不同的簇，制造 Gap
    # 2. init='random': 打断全局连续性，更容易形成独立岛屿
    tsne = TSNE(n_components=2,
                random_state=TSNE_SEED,
                perplexity=safe_perplexity,
                early_exaggeration=50,  # <--- 默认为12，改为30能制造大间隙！
                init='random',  # <--- 默认为pca，改为random能避免粘连
                learning_rate='auto',
                n_jobs=-1)
    return tsne.fit_transform(data)


def save_label_shape_figure(emb: np.ndarray, mask: np.ndarray, out_path: str,
                            label_id: int, color) -> None:
    if not np.any(mask):
        return

    base_out, _ = os.path.splitext(out_path)
    shape_path = f"{base_out}_label{label_id}_shape.png"

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.scatter(emb[mask, 0], emb[mask, 1], color=[color], s=12, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(shape_path, dpi=300)
    plt.close()


def plot_embedding(emb: np.ndarray, labels: np.ndarray, title: str, out_path: str,
                   color_by_label: bool, fallback_color: str, show_axis: bool,
                   annotate_clusters: bool,
                   labels_use_largest_cluster: Optional[set]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 8))

    unique_labels = np.sort(np.unique(labels))
    highlight_label = 14
    mask_highlight = (labels == highlight_label)
    highlight_color = (1.0, 0.4, 0.7, 1.0)

    if color_by_label:
        num_classes = len(unique_labels)

        # 离散映射：确保颜色区分度最大化
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        mapped_labels = np.array([label_to_idx[l] for l in labels])

        cmap = build_distinct_cmap(num_classes)

        if highlight_label in label_to_idx:
            idx = label_to_idx[highlight_label]
            denom = max(num_classes - 1, 1)
            highlight_color = cmap(idx / denom)
        else:
            highlight_color = (1.0, 0.4, 0.7, 1.0)

        mask_background = ~mask_highlight
        if np.any(mask_background):
            plt.scatter(
                emb[mask_background, 0], emb[mask_background, 1],
                c='#d3d3d3',
                s=10,
                alpha=0.3,
            )
        if np.any(mask_highlight):
            plt.scatter(
                emb[mask_highlight, 0], emb[mask_highlight, 1],
                color=[highlight_color],
                s=10,
                alpha=0.9,
            )

    else:
        mask_background = ~mask_highlight
        if np.any(mask_background):
            plt.scatter(
                emb[mask_background, 0], emb[mask_background, 1],
                c='#d3d3d3',
                s=10,
                alpha=0.3,
            )
        if np.any(mask_highlight):
            plt.scatter(
                emb[mask_highlight, 0], emb[mask_highlight, 1],
                color=[highlight_color],
                s=10,
                alpha=0.9,
            )

    if not show_axis:
        plt.axis('off')

    ax = plt.gca()
    base_size = plt.rcParams.get('font.size', 10)
    tick_fontsize = base_size * 2
    ax.tick_params(labelsize=tick_fontsize, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    save_label_shape_figure(emb, mask_highlight, out_path, highlight_label, highlight_color)

    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    args.color_by_label = bool(args.color_by_label)
    args.single_label_only = bool(args.single_label_only)
    args.show_axis = bool(args.show_axis)
    args.show_cluster_ids = bool(args.show_cluster_ids)
    args.cluster_id_labels = set(args.cluster_id_labels or [])

    print("-" * 60)
    print("MIRFlickr Class ID Mapping:")
    for i in range(0, len(MIRFLICKR_LABELS), 4):
        print(" | ".join(MIRFLICKR_LABELS[i:i+4]))
    print("-" * 60)

    print(f'Loading raw features for {args.dset} from {args.data_path} ...')
    img_feats, txt_feats, labels = load_selected_splits(args)

    if args.single_label_only:
        img_feats, txt_feats, labels = filter_single_label_samples(img_feats, txt_feats, labels)
    else:
        print('Skipping single-label-only filter. Using all samples, including multi-label ones.')

    n_orig = img_feats.shape[0]
    print(f'Loaded - Images: {n_orig}, Texts: {txt_feats.shape[0]}')

    run_mode = "all"

    # ==========================
    # 核心筛选逻辑
    # ==========================
    # 1. 优先级最高：指定具体类别
    if args.classes is not None:
        target_ids = args.classes
        # 调用核心函数：支持 strict 互斥 和 优先级归类
        img_feats, txt_feats, classes = filter_exclusive_and_priority(
            img_feats, txt_feats, labels, target_ids, strict=args.strict
        )
        mode_suffix = "strict" if args.strict else "priority"
        run_mode = f"classes_{mode_suffix}"

    # 2. Top/Bottom K (使用默认单标签逻辑)
    else:
        try:
            classes = labels_to_classes(labels)
        except Exception:
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                classes = np.argmax(labels, axis=1)
            else:
                classes = labels

        if args.bottom_k is not None:
            img_feats, txt_feats, classes = filter_bottom_k_classes(img_feats, txt_feats, classes, args.bottom_k)
            run_mode = f"bottom{args.bottom_k}"

        elif args.top_k is not None:
            img_feats, txt_feats, classes = filter_top_k_classes(img_feats, txt_feats, classes, args.top_k)
            run_mode = f"top{args.top_k}"

        else:
             print("No filter selected. Using top 10 as default.")
             img_feats, txt_feats, classes = filter_top_k_classes(img_feats, txt_feats, classes, 10)
             run_mode = "top10_default"

    print(f'Final visualization dataset size: {img_feats.shape[0]} samples')

    # =========================================================
    # INDEPENDENT t-SNE (独立降维)
    # =========================================================
    print('\n=== Running Independent t-SNE ===')
    print('Processing Image Features...')
    img_emb = run_tsne(img_feats, args.perplexity, desc="Image t-SNE")

    print('Processing Text Features...')
    txt_emb = run_tsne(txt_feats, args.perplexity, desc="Text t-SNE")

    base_name = f'{args.dset}_raw_{run_mode}_perp{args.perplexity}_independent'

    img_path = os.path.join(args.output_dir, f'{base_name}_image.png')
    txt_path = os.path.join(args.output_dir, f'{base_name}_text.png')

    print("Generating plots...")
    target_label_set = args.cluster_id_labels if args.show_cluster_ids else set()

    plot_embedding(img_emb, classes, f'Raw Image Features\n(Mode: {run_mode}, Perp: {args.perplexity})', img_path,
                   args.color_by_label, 'tab:blue', args.show_axis, args.show_cluster_ids, target_label_set)
    plot_embedding(txt_emb, classes, f'Raw Text Features\n(Mode: {run_mode}, Perp: {args.perplexity})', txt_path,
                   args.color_by_label, 'tab:orange', args.show_axis, args.show_cluster_ids, target_label_set)

    print(f'Figures saved to:\n  {img_path}')


if __name__ == '__main__':
    main()