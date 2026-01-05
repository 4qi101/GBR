import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
except ImportError:  # pragma: no cover - optional dependency
    plt = None
    TSNE = None

from src.mgbcc_style_balls import GBList
from utils.experiment_utils import build_experiment_tag


def labels_to_classes(labels_np: np.ndarray) -> np.ndarray:
    if labels_np.ndim == 1:
        return labels_np.astype(int)
    if labels_np.ndim == 2 and labels_np.shape[1] == 1:
        return labels_np[:, 0].astype(int)
    return labels_np.argmax(axis=1).astype(int)


def plot_modality_alignment(img_feats: np.ndarray,
                            txt_feats: np.ndarray,
                            out_path: str,
                            seed: int = 1234) -> None:
    if plt is None or TSNE is None:
        print('[Viz] matplotlib or scikit-learn not available, skip modality alignment plot.')
        return
    if img_feats.shape != txt_feats.shape:
        return

    all_feats = np.concatenate([img_feats, txt_feats], axis=0)
    perplexity = min(30, max(5, len(all_feats) // 50))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init='pca')
    emb = tsne.fit_transform(all_feats)
    n = img_feats.shape[0]
    img_2d = emb[:n]
    txt_2d = emb[n:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(img_2d[:, 0], img_2d[:, 1], c='tab:blue', marker='o', s=10, alpha=0.5, label='Image')
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c='tab:orange', marker='^', s=10, alpha=0.5, label='Text')
    plt.legend()
    plt.title('Modality Alignment (Image vs Text)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_semantic_clusters_with_granular_balls(img_feats: np.ndarray,
                                               labels_np: np.ndarray,
                                               p: int,
                                               device: torch.device,
                                               out_path: str,
                                               seed: int = 1234) -> None:
    if plt is None or TSNE is None:
        print('[Viz] matplotlib or scikit-learn not available, skip granular ball plot.')
        return

    feats_tensor = torch.as_tensor(img_feats, dtype=torch.float32, device=device)
    gblist = GBList(feats_tensor, p=p)
    centers = gblist.get_centers().detach().cpu().numpy()
    ball_ids = gblist.y_parts.detach().cpu().numpy()

    if centers.shape[0] > 0:
        tsne_input = np.concatenate([img_feats, centers], axis=0)
    else:
        tsne_input = img_feats

    perplexity = min(30, max(5, len(tsne_input) // 50))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init='pca')
    emb = tsne.fit_transform(tsne_input)
    pts_2d = emb[: img_feats.shape[0]]
    ctr_2d = emb[img_feats.shape[0] : img_feats.shape[0] + centers.shape[0]] if centers.shape[0] > 0 else None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=ball_ids, cmap='tab20', s=8, alpha=0.6)
    plt.colorbar(scatter, label='Granular Ball ID')
    if ctr_2d is not None and len(ctr_2d) > 0:
        plt.scatter(ctr_2d[:, 0], ctr_2d[:, 1], marker='*', s=150,
                    edgecolors='k', facecolors='none', linewidths=1.5,
                    label='Granular Ball Centers')
        plt.legend()
    plt.title('Semantic Clusters with Granular Ball Centers (Colored by Ball)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def visualize_embeddings_after_training(img_net: torch.nn.Module,
                                        txt_net: torch.nn.Module,
                                        loader: Any,
                                        device: torch.device,
                                        args,
                                        epoch: Optional[int] = None) -> None:
    if plt is None or TSNE is None:
        print('[Viz] matplotlib or scikit-learn not available, skip visualization.')
        return

    max_points = getattr(args, 'viz_max_points', 2000)
    img_list = []
    txt_list = []
    label_list = []
    total = 0

    img_net.eval()
    txt_net.eval()
    with torch.no_grad():
        for images, texts, labels, indices in loader:
            images = images.to(device)
            texts = texts.to(device)
            _, _, _, img_code = img_net(images)
            _, _, _, txt_code = txt_net(texts)
            v_img = F.normalize(img_code, dim=1, eps=1e-8)
            v_txt = F.normalize(txt_code, dim=1, eps=1e-8)
            v_img = torch.nan_to_num(v_img, nan=0.0, posinf=1.0, neginf=-1.0)
            v_txt = torch.nan_to_num(v_txt, nan=0.0, posinf=1.0, neginf=-1.0)
            img_list.append(v_img.cpu())
            txt_list.append(v_txt.cpu())
            label_list.append(labels.cpu())
            total += images.size(0)
            if total >= max_points:
                break

    if not img_list:
        print('[Viz] No samples collected for visualization, skip.')
        return

    img_feats = torch.cat(img_list, dim=0)[:max_points].numpy()
    txt_feats = torch.cat(txt_list, dim=0)[:max_points].numpy()
    labels_np = torch.cat(label_list, dim=0)[:max_points].numpy()

    base_tag = build_experiment_tag(args)
    if epoch is not None:
        tag = f"{base_tag}_ep{epoch}"
    else:
        tag = base_tag
    out_dir = os.path.join(args.save_dir, 'figs')
    os.makedirs(out_dir, exist_ok=True)

    align_path = os.path.join(out_dir, f'{tag}_modality_alignment.png')
    plot_modality_alignment(img_feats, txt_feats, align_path, seed=args.seed)

    gb_path = os.path.join(out_dir, f'{tag}_granular_balls.png')
    plot_semantic_clusters_with_granular_balls(
        img_feats,
        labels_np,
        p=args.mgbcc_p,
        device=device,
        out_path=gb_path,
        seed=args.seed,
    )
