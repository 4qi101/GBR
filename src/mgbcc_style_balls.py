# utils/mgbcc_style_balls.py
# -*- coding: utf-8 -*-
"""
完全按照MGBCC的粒球生成和匹配方法
[Optimized] Robust GPU KMeans included.
"""

import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from utils.debug_tools import log_mask_stats, log_similarity_stats


def kmeans_gpu(data, n_clusters, max_iter=50, tol=1e-4, seed=None):
    """
    GPU版KMeans聚类 - 鲁棒版本

    Args:
        data: [N, D] tensor on GPU
        n_clusters: 聚类数量
        max_iter: 最大迭代次数

    Returns:
        labels: [N] 聚类标签
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = data.device
    n_samples, n_features = data.shape

    if n_samples <= n_clusters:
        return torch.arange(n_samples, device=device)

    # 1. 初始化中心 (Random For Speed, KMeans++ is too slow for per-batch loop)
    # 在 Batch 级训练中，随机初始化通常足够且极快
    indices = torch.randperm(n_samples, device=device)[:n_clusters]
    centers = data[indices].clone()

    labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    for i in range(max_iter):
        # E-step: 使用欧氏距离分配簇
        dists = torch.cdist(data, centers)
        new_labels = torch.argmin(dists, dim=1)

        # Check convergence
        if i > 0 and torch.equal(labels, new_labels):
            break
        labels = new_labels

        # M-step: 聚合每个簇的样本向量
        counts = torch.bincount(labels, minlength=n_clusters).float()
        mask_non_empty = counts > 0

        centers_sum = torch.zeros_like(centers)
        centers_sum.index_add_(0, labels, data)

        new_centers = centers_sum / counts.unsqueeze(1).clamp(min=1e-6)

        # 处理空簇：重新随机选择样本作为中心
        if (~mask_non_empty).any():
            empty_indices = (~mask_non_empty).nonzero().squeeze(1)
            rand_indices = torch.randint(0, n_samples, (empty_indices.size(0),), device=device)
            new_centers[empty_indices] = data[rand_indices]

        centers = torch.where(mask_non_empty.unsqueeze(1), new_centers, new_centers)

    return labels


class GranularBall:
    """完全按照MGBCC的粒球类"""

    def __init__(self, data, indices):
        self.data = data
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=self.data.device).reshape(-1)
        self.num_smp, self.dim = data.shape
        self.center = self.data.mean(0)
        # 计算各点到中心点距离
        arr = torch.norm(self.data - self.center, p=2, dim=1)
        # 各点到中心点距离的平均值
        self.r = arr.mean()

    def split_balls(self, p):
        # 将该粒球划分为k个小的粒球
        k = max(self.num_smp // p, 1)

        # GPU上检查去重 (简易版，只检查前几个维度或直接忽略以加速)
        # 完整 unique 在高维数据上可能较慢，这里假设 batch 内可能有少量重复
        # 如果追求极致速度，可以注释掉下面两行
        if self.num_smp > 1000:  # 仅在数据量大时考虑去重上限
            unique_count = self.num_smp  # 假设不重复
        else:
            unique_count = torch.unique(self.data, dim=0).shape[0]

        k = min(k, unique_count)

        if p == 1 or k >= self.num_smp:
            y_part = torch.arange(self.num_smp, device=self.data.device)
        else:
            # 使用鲁棒的 GPU KMeans
            y_part = kmeans_gpu(self.data, n_clusters=k)

        sub_balls = []
        # 构建子球对象
        # 获取实际存在的 label (kmeans 可能会合并)
        present_labels, y_part_new = torch.unique(y_part, return_inverse=True)

        for i in range(present_labels.numel()):
            mask = (y_part_new == i)
            # 此时 mask 必定不为空，因为 i 来自 unique(y_part)
            sub_data = self.data[mask]
            sub_indices = self.indices[mask]
            new_ball = GranularBall(sub_data, sub_indices)
            sub_balls.append(new_ball)

        # 返回的是重新映射后的连续标签，范围为 [0, num_balls-1]
        return sub_balls, y_part_new


class GBList:
    """完全按照MGBCC的粒球列表类"""

    def __init__(self, data, p=8):
        self.data = data
        self.indices = torch.arange(data.shape[0], device=data.device)
        self.y_parts = None
        self.granular_balls = [GranularBall(data, self.indices)]
        self.split_granular_balls(p)

    def __len__(self):
        return len(self.granular_balls)

    def __getitem__(self, i):
        return self.granular_balls[i]

    def split_granular_balls(self, p):
        gb_list, y_parts = self[0].split_balls(p)
        self.granular_balls = gb_list
        self.y_parts = y_parts

    def get_centers(self):
        if not self.granular_balls:
            return torch.empty((0, self.data.shape[1]), device=self.data.device)
        return torch.vstack(list(map(lambda x: x.center, self.granular_balls)))

    def get_rs(self):
        if not self.granular_balls:
            return torch.tensor([], device=self.data.device)
        return torch.vstack(list(map(lambda x: x.r, self.granular_balls))).squeeze()

    @torch.no_grad
    def affinity(self, spread=3, sim_threshold=0.5):
        centers = self.get_centers()
        if centers.size(0) == 0:
            return torch.zeros((0, 0), device=self.data.device)

        # 1. 几何距离
        dist = torch.cdist(centers, centers)
        rs = self.get_rs()
        if rs.ndim == 0: rs = rs.unsqueeze(0)
        extra = rs.unsqueeze(0) + rs.unsqueeze(-1)
        geo_mask = (dist <= extra)

        # 2. 相似度
        centers_norm = F.normalize(centers, dim=1, eps=1e-8)
        similarity = centers_norm @ centers_norm.T
        sim_mask = (similarity > sim_threshold)

        # 3. 结合
        indicate = geo_mask & sim_mask
        return indicate.float()


class MVGBList:
    """完全按照MGBCC的多视图粒球列表类"""

    def __init__(self, mv_data, p=8):
        self.num_view = len(mv_data)
        self.gblists = []
        for i in range(self.num_view):
            gblist = GBList(mv_data[i], p=p)
            self.gblists.append(gblist)

    def __len__(self):
        return self.num_view

    def __getitem__(self, i):
        return self.gblists[i]


def relation_of_views_gblists_tensor(view0: GBList, view1: GBList, t=0.1):
    y_parts0 = view0.y_parts
    y_parts1 = view1.y_parts

    num_gb0 = len(view0)
    num_gb1 = len(view1)

    one_hot0 = one_hot(y_parts0, num_classes=num_gb0).float()
    one_hot1 = one_hot(y_parts1, num_classes=num_gb1).float()

    mask = one_hot0.T @ one_hot1

    num_gb_set0 = one_hot0.sum(dim=0).view((-1, 1))
    num_gb_set1 = one_hot1.sum(dim=0).view((1, -1))

    num_gb_min = torch.min(num_gb_set0, num_gb_set1)
    num_gb_min[num_gb_min == 0] = 1.0  # Avoid div by zero

    # 硬匹配：重叠比例超过阈值 t 记为 1，否则为 0
    mask = (mask / num_gb_min) > t
    return mask.float()


def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4):
    merged_tensor = torch.zeros((n + m, n + m), device=tensor1.device)
    merged_tensor[:n, :n] = tensor1
    merged_tensor[:n, n:n + m] = tensor2
    merged_tensor[n:n + m, :n] = tensor3
    merged_tensor[n:n + m, n:n + m] = tensor4
    return merged_tensor


class MultiviewGCLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        match_threshold: float = 0.1,
        sim_threshold: float = 0.5,
        use_affinity: bool = True,
    ):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature
        self.match_threshold = match_threshold
        self.sim_threshold = sim_threshold
        self.use_affinity = use_affinity

    def forward(self, views: MVGBList):
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        num_views = len(views)

        for i in range(num_views):
            mask_i_intra = views[i].affinity(sim_threshold=self.sim_threshold) if self.use_affinity else torch.eye(
                len(views[i]), device=device)

            for j in range(i + 1, num_views):
                mask_j_intra = views[j].affinity(sim_threshold=self.sim_threshold) if self.use_affinity else torch.eye(
                    len(views[j]), device=device)

                mask_inter = relation_of_views_gblists_tensor(
                    views[i], views[j], t=self.match_threshold
                )

                ni, nj = len(views[i]), len(views[j])
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra)

                # Zero out diagonal
                pos_mask.fill_diagonal_(0)
                pos_mask_bool = pos_mask > 0
                if not pos_mask_bool.any():
                    continue

                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()
                x = torch.cat((centers_i, centers_j), dim=0)

                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)

                logits = sim_x / self.t

                log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

                # 硬匹配：正样本权重为 0/1
                pos_mask_float = pos_mask_bool.float()
                pos_counts = pos_mask_float.sum(dim=1)

                valid_rows = pos_counts > 0
                if valid_rows.any():
                    row_loss = -(log_prob * pos_mask_float).sum(dim=1) / pos_counts.clamp_min(1.0)
                    loss += row_loss[valid_rows].mean()

        return loss / (num_views * (num_views - 1) / 2)