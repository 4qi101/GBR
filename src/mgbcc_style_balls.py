# utils/mgbcc_style_balls.py
# -*- coding: utf-8 -*-
"""
完全按照MGBCC的粒球生成和匹配方法
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.nn.functional import one_hot

from utils.debug_tools import log_mask_stats, log_similarity_stats


class GranularBall:
    """完全按照MGBCC的粒球类"""

    def __init__(self, data, indices):
        """
        :param data: 样本特征
        :param indices: 样本索引
        """
        self.data = data
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=self.data.device).reshape(-1)
        self.num_smp, self.dim = data.shape
        self.center = self.data.mean(0)
        arr = torch.norm(self.data - self.center, p=2, dim=1)
        # 各点到中心点距离的平均值
        self.r = arr.mean()

    def split_balls(self, p):
        # 将该粒球划分为k个小的粒球
        k = max(self.num_smp // p, 1)
        data = self.data.detach().cpu().numpy()
        # 当data中有大量重复时
        k = min(k, np.unique(data, axis=0).shape[0])
        if p == 1:
            y_part = np.arange(data.shape[0])
        else:
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            y_part = kmeans.fit_predict(data)
        # 根据聚类结果对样本进行划分
        y_part = torch.from_numpy(y_part).to(self.data.device, dtype=torch.long)
        sub_balls = []
        for i in range(k):
            mask = (y_part == i)
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            sub_data = self.data[idx]
            sub_indices = self.indices[idx]
            new_ball = GranularBall(sub_data, sub_indices)
            sub_balls.append(new_ball)
        return sub_balls, y_part


class GBList:
    """完全按照MGBCC的粒球列表类"""

    def __init__(self, data, p=8):
        self.data = data
        self.indices = torch.arange(data.shape[0], device=data.device)
        self.y_parts = None
        self.granular_balls = [GranularBall(data, self.indices)]  # gbs is initialized with all data
        self.split_granular_balls(p)

    def __len__(self):
        return len(self.granular_balls)

    def __getitem__(self, i):
        return self.granular_balls[i]

    def split_granular_balls(self, p):
        """
        Split the balls, initialize the balls list.
        :param p: If the number of samples of a ball is less than this value, stop splitting.
        """
        gb_list, y_parts = self[0].split_balls(p)
        self.granular_balls = gb_list
        self.y_parts = y_parts

    def get_centers(self):
        """
        :return: the center of each ball.
        """
        return torch.vstack(list(map(lambda x: x.center, self.granular_balls)))

    def get_rs(self):
        """
        :return: 返回半径r
        """
        return torch.vstack(list(map(lambda x: x.r, self.granular_balls))).squeeze()

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        list_indices = [ball.indices for ball in self.granular_balls]
        return torch.cat(list_data, dim=0), torch.cat(list_indices, dim=0)

    def del_ball(self, min_smp=0):
        T_ball = []
        for ball in self.granular_balls:
            if ball.num_smp >= min_smp:
                T_ball.append(ball)
        self.granular_balls = T_ball
        self.data, self.indices = self.get_data()

    @torch.no_grad
    def affinity(self, spread=3, sim_threshold=0.5):
        """粒球几何覆盖关系 + 相似度约束
        
        Args:
            spread: 传递邻域扩展次数（已禁用，避免过度泛化）
            sim_threshold: 中心相似度阈值，只有相似度 > 此值的球才能匹配
        
        Returns:
            indicate: 匹配矩阵，1表示两个粒球匹配
        """
        # 获取所有中心点
        centers = self.get_centers()
        
        # 1. 几何距离判断：dist(i,j) <= r_i + r_j
        dist = torch.cdist(centers, centers)
        rs = self.get_rs()
        extra = rs.unsqueeze(0) + rs.unsqueeze(-1)
        geo_mask = (dist <= extra)
        
        # 2. 相似度判断：归一化后计算余弦相似度
        centers_norm = torch.nn.functional.normalize(centers, dim=1, eps=1e-8)
        similarity = centers_norm @ centers_norm.T
        log_similarity_stats(similarity, prefix=f"Affinity(sim_thr={sim_threshold:.3f})")
        sim_mask = (similarity > sim_threshold)
        
        # 3. 两者都满足才匹配
        indicate = geo_mask & sim_mask
        indicate = indicate.type(torch.float32)
        
        # spread 传递扩展已禁用（会导致过度匹配）
        # indicate = transitive_neighbor_relations(indicate, spread)
        
        return indicate


class MVGBList:
    """完全按照MGBCC的多视图粒球列表类"""

    def __init__(self, mv_data, p=8):
        """
        :param mv_data: 多视图数据
        :param p:
        """
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
    """
    完全按照MGBCC的relation_of_views_gblists_tensor函数
    """
    y_parts0 = view0.y_parts
    y_parts1 = view1.y_parts
    num_gb = len(view0)
    # n * k
    one_hot0 = one_hot(y_parts0, num_classes=num_gb).float()
    # n * k
    one_hot1 = one_hot(y_parts1, num_classes=num_gb).float()
    mask = one_hot0.T @ one_hot1
    num_gb_set0 = one_hot0.sum(dim=0).view((-1, 1))
    num_gb_set1 = one_hot1.sum(dim=0).view((1, -1))
    num_gb_min = torch.min(num_gb_set0, num_gb_set1)
    mask = (mask / num_gb_min) > t
    return mask.float()


def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4):
    """完全按照MGBCC的merge_tensors函数"""
    # 创建一个大小为 (n+m) * (n+m) 的零张量
    merged_tensor = torch.zeros((n + m, n + m))

    # 填充第一个 tensor 张量
    merged_tensor[:n, :n] = tensor1

    # 填充第二个 tensor 张量
    merged_tensor[:n, n:n + m] = tensor2

    # 填充第三个 tensor 张量
    merged_tensor[n:n + m, :n] = tensor3

    # 填充第四个 tensor 张量
    merged_tensor[n:n + m, n:n + m] = tensor4

    return merged_tensor


class MultiviewGCLoss(torch.nn.Module):
    """完全按照MGBCC的MultiviewGCLoss类"""

    def __init__(self, temperature=1., match_threshold=0.1, sim_threshold=0.5, use_affinity=True):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature
        self.match_threshold = match_threshold
        self.sim_threshold = sim_threshold
        self.use_affinity = use_affinity

    def forward(self, views: MVGBList):
        # 统一设备
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        # 两两视图之间进行对比
        num_views = len(views)
        for i in range(num_views):
            if self.use_affinity:
                mask_i_intra = views[i].affinity(sim_threshold=self.sim_threshold)
            else:
                mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                # 计算掩码
                if self.use_affinity:
                    mask_j_intra = views[j].affinity(sim_threshold=self.sim_threshold)
                else:
                    mask_j_intra = torch.eye(len(views[j]), device=device)
                mask_inter = relation_of_views_gblists_tensor(
                    views[i], views[j], t=self.match_threshold
                )
                log_mask_stats(
                    mask_i_intra,
                    mask_inter,
                    prefix=f"view{i}-{j}"
                )
                # 两个视图的粒球数量
                ni, nj = len(views[i]), len(views[j])
                # 合并视图内和视图间的掩码矩阵
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra).to(device)
                num_ins = ni + nj
                idx = torch.arange(0, num_ins, device=device)
                pos_mask[idx, idx] = 0
                pos_mask_bool = pos_mask > 0
                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()
                x = torch.cat((centers_i, centers_j), dim=0)
                # 计算相似度，这里就是矩阵相乘
                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
                logits = sim_x / self.t
                log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                pos_mask_float = pos_mask_bool.float()
                pos_counts = pos_mask_float.sum(dim=1)
                valid_rows = pos_counts > 0
                if not valid_rows.any():
                    continue
                row_loss = -(log_prob * pos_mask_float).sum(dim=1) / pos_counts.clamp_min(1.0)
                loss += row_loss[valid_rows].mean()
        return loss / (num_views * (num_views - 1) / 2)

