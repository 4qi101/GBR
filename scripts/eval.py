import torch
from utils.metric import calc_map_k, calc_and_save_pr


def eval_retrieval_target(imgNet, txtNet, test_dl, retrieval_dl, db_name=None, device=None, topk=None):
    """
    评估跨模态检索性能
    
    Args:
        imgNet: 图像编码网络
        txtNet: 文本编码网络
        test_dl: 测试集DataLoader
        retrieval_dl: 检索集DataLoader
        db_name: 数据集名称（可选）
    
    Returns:
        mapi2t: 图像到文本的MAP值
        mapt2i: 文本到图像的MAP值
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    
    imgNet.eval()
    txtNet.eval()
    
    with torch.no_grad():
        # 处理测试集
        for i, (im_t, txt_t, label_t, id_target) in enumerate(test_dl):
            im_t = im_t.to(device)
            txt_t = txt_t.to(device)
            label_t = label_t.to(device)
            
            _, _, _, Im_code = imgNet(im_t)
            _, _, _, Txt_code = txtNet(txt_t)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            test_dl_dict['img_code'].append(Im_code)
            test_dl_dict['txt_code'].append(Txt_code)
            test_dl_dict['label'].append(label_t)

        # 处理检索集
        for i, (im_t, txt_t, label_t, id_target) in enumerate(retrieval_dl):
            im_t_db = im_t.to(device)
            txt_t_db = txt_t.to(device)
            label_t_db = label_t.to(device)
            
            _, _, _, Im_code = imgNet(im_t_db)
            _, _, _, Txt_code = txtNet(txt_t_db)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            retrieval_dl_dict['img_code'].append(Im_code)
            retrieval_dl_dict['txt_code'].append(Txt_code)
            retrieval_dl_dict['label'].append(label_t_db)

    query_img = torch.cat(test_dl_dict['img_code'], dim=0).to(device)
    query_txt = torch.cat(test_dl_dict['txt_code'], dim=0).to(device)
    query_label = torch.cat(test_dl_dict['label'], dim=0).to(device)

    retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0).to(device)
    retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0).to(device)
    retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0).to(device)

    # 计算i2t的MAP
    mapi2t = calc_map_k(query_img, retrieval_txt, query_label, retrieval_label, k=topk)
    # 计算t2i的MAP
    mapt2i = calc_map_k(query_txt, retrieval_img, query_label, retrieval_label, k=topk)

    imgNet.train()
    txtNet.train()
    
    return mapi2t.item(), mapt2i.item()



def eval_pr_curves(imgNet, txtNet, test_dl, retrieval_dl,
                   db_name=None, device=None,
                   method_name='GBR', bit=None,
                   save_dir='pr_curves_data'):
    """Compute and save PR curves (I2T & T2I) using current networks.

    This function mirrors eval_retrieval_target's code extraction, but instead
    of computing MAP, it calls calc_and_save_pr to obtain and persist PR data.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}

    imgNet.eval()
    txtNet.eval()

    with torch.no_grad():
        for _, (im_t, txt_t, label_t, _) in enumerate(test_dl):
            im_t = im_t.to(device)
            txt_t = txt_t.to(device)
            label_t = label_t.to(device)

            _, _, _, Im_code = imgNet(im_t)
            _, _, _, Txt_code = txtNet(txt_t)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            test_dl_dict['img_code'].append(Im_code)
            test_dl_dict['txt_code'].append(Txt_code)
            test_dl_dict['label'].append(label_t)

        for _, (im_t, txt_t, label_t, _) in enumerate(retrieval_dl):
            im_t_db = im_t.to(device)
            txt_t_db = txt_t.to(device)
            label_t_db = label_t.to(device)

            _, _, _, Im_code = imgNet(im_t_db)
            _, _, _, Txt_code = txtNet(txt_t_db)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            retrieval_dl_dict['img_code'].append(Im_code)
            retrieval_dl_dict['txt_code'].append(Txt_code)
            retrieval_dl_dict['label'].append(label_t_db)

    query_img = torch.cat(test_dl_dict['img_code'], dim=0)
    query_txt = torch.cat(test_dl_dict['txt_code'], dim=0)
    query_label = torch.cat(test_dl_dict['label'], dim=0)

    retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0)
    retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0)
    retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0)

    if bit is None:
        bit = query_img.shape[1]

    calc_and_save_pr(
        q_b=query_img,
        r_b=retrieval_txt,
        q_l=query_label,
        r_l=retrieval_label,
        method_name=method_name,
        task_name='I2T',
        bit=bit,
        save_dir=save_dir,
    )

    calc_and_save_pr(
        q_b=query_txt,
        r_b=retrieval_img,
        q_l=query_label,
        r_l=retrieval_label,
        method_name=method_name,
        task_name='T2I',
        bit=bit,
        save_dir=save_dir,
    )

    imgNet.train()
    txtNet.train()

