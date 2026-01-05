import os
import numpy as np
import scipy
import torch
import torch.nn.functional as F


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.t()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose(0,1)) > 0).astype(np.float32)
        tsum =np.int32(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

def calc_cosine_dist(query, retrieval):
    # q = B2.shape[1]
    # if len(B1.shape) < 2:
    #     B1 = B1.unsqueeze(0)
    # distH = 0.5 * (q - B1.mm(B2.t()))

    query = query.unsqueeze(0)
    dis = scipy.spatial.distance.cdist(query, retrieval, 'cosine')  # 算的是余弦距离不是相似度
    dis = torch.as_tensor(dis)
    return dis

# TODO (n,d)×(m,d) -> (n,m)
def calc_cosine_dist2(query, retrieval):
    # q = B2.shape[1]
    # if len(B1.shape) < 2:
    #     B1 = B1.unsqueeze(0)
    # distH = 0.5 * (q - B1.mm(B2.t()))

    # query = query.unsqueeze(0)
    dis = scipy.spatial.distance.cdist(query, retrieval, 'cosine') # 算的是余弦距离不是相似度
    dis = torch.as_tensor(dis)
    return dis

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}

    num_query = query_L.shape[0]
    hamm = calc_hamming_dist(qB, rB)  # 提前算，不用每步都算

    # print('hamm ', hamm.shape)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        # hamm = calc_cosine_dist(qB[iter, :], rB)
        _, ind = torch.sort(hamm[iter])
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_and_save_pr(q_b, r_b, q_l, r_l, method_name, task_name, bit, save_dir='pr_curves_data'):
    """Calculate average precision-recall curve and save to .npy.

    q_b, r_b: binary codes {-1, 1} or {0, 1} for query and retrieval sets.
    q_l, r_l: one-hot labels for query and retrieval sets.
    """

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    q_b = _to_numpy(q_b)
    r_b = _to_numpy(r_b)
    q_l = _to_numpy(q_l)
    r_l = _to_numpy(r_l)

    if q_b.ndim == 1:
        q_b = q_b.reshape(1, -1)
    if r_b.ndim == 1:
        r_b = r_b.reshape(1, -1)
    if q_l.ndim == 1:
        q_l = q_l.reshape(1, -1)
    if r_l.ndim == 1:
        r_l = r_l.reshape(1, -1)

    q_b = np.where(q_b > 0, 1, -1).astype(np.int8)
    r_b = np.where(r_b > 0, 1, -1).astype(np.int8)

    num_query = q_b.shape[0]
    num_retrieval = r_b.shape[0]
    code_length = q_b.shape[1]

    inner = np.matmul(q_b, r_b.T).astype(np.int32)
    dist = 0.5 * (code_length - inner).astype(np.float32)

    gnd = np.matmul(q_l, r_l.T) > 0
    gnd = gnd.astype(np.float32)

    order = np.argsort(dist, axis=1)
    gnd_sorted = np.take_along_axis(gnd, order, axis=1)

    num_rel = np.sum(gnd_sorted, axis=1)
    valid = num_rel > 0

    if not np.any(valid):
        precision = np.zeros(num_retrieval, dtype=np.float32)
        recall = np.zeros(num_retrieval, dtype=np.float32)
    else:
        gnd_valid = gnd_sorted[valid]
        num_rel_valid = num_rel[valid][:, None]

        cumsum = np.cumsum(gnd_valid, axis=1)
        ranks = np.arange(1, num_retrieval + 1, dtype=np.float32).reshape(1, -1)

        precision_q = cumsum / ranks
        recall_q = cumsum / num_rel_valid

        precision = precision_q.mean(axis=0).astype(np.float32)
        recall = recall_q.mean(axis=0).astype(np.float32)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    filename = "{}_{}_{}.npy".format(method_name, task_name, bit)
    out_path = os.path.join(save_dir, filename)
    np.save(out_path, {"P": precision, "R": recall})

    return precision, recall

