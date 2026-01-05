"""
粒球损失函数工具模块
用于在GBR backbone上使用粒球对比学习
与UCCH项目保持一致的使用方式
"""

import torch
import torch.nn.functional as F
from src.mgbcc_style_balls import MVGBList, MultiviewGCLoss, relation_of_views_gblists_tensor


def compute_granular_ball_loss(
    img_features,
    txt_features,
    p=8,
    temperature=1.0,
    match_threshold=0.1,
    criterion=None,
    return_aux: bool = False,
):
    """
    计算粒球对比损失（与UCCH项目一致）
    
    Args:
        img_features: 图像特征 [batch_size, feature_dim]
        txt_features: 文本特征 [batch_size, feature_dim]
        p: MGBCC粒度参数，控制粒球大小
        temperature: 对比学习温度参数
        match_threshold: 粒球交集阈值
        criterion: 复用的 MultiviewGCLoss 实例，可选
    
    Returns:
        loss: 粒球对比损失值
    """
    # 归一化特征
    v_img = F.normalize(img_features, dim=1, eps=1e-8)
    v_txt = F.normalize(txt_features, dim=1, eps=1e-8)
    
    # 处理NaN和Inf
    v_img = torch.nan_to_num(v_img, nan=0.0, posinf=1.0, neginf=-1.0)
    v_txt = torch.nan_to_num(v_txt, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 创建多视图粒球列表（与UCCH一致）
    mv_data = [v_img, v_txt]
    mv_gb_list = MVGBList(mv_data, p=p)
    
    # 创建粒球对比损失函数（与UCCH一致）
    if criterion is None:
        criterion = MultiviewGCLoss(
            temperature=temperature,
            match_threshold=match_threshold,
        )

    # 计算损失
    loss_gb = criterion(mv_gb_list)

    if not return_aux:
        return loss_gb

    gb_img = mv_gb_list[0]
    gb_txt = mv_gb_list[1]

    centers_img = gb_img.get_centers()
    centers_txt = gb_txt.get_centers()

    assign_img = gb_img.y_parts
    assign_txt = gb_txt.y_parts

    relation_matrix = relation_of_views_gblists_tensor(
        gb_img,
        gb_txt,
        t=match_threshold,
    )

    return (
        loss_gb,
        centers_img,
        centers_txt,
        assign_img,
        assign_txt,
        relation_matrix,
    )
