from utils.dset import PreDataset
from torch.utils.data import DataLoader


def get_dataloaders(args):
    """动态创建数据加载器，避免import时副作用
    
    Args:
        args: 配置参数对象
        
    Returns:
        train_dl, test_dl, retrieval_dl: 训练/测试/检索数据加载器
    """
    dataset_root = args.data_path
    dataname = args.dset
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # 创建数据集
    train_ds = PreDataset(data_path=dataset_root, data_split='train', dataname=dataname, flag='target')
    test_ds = PreDataset(data_path=dataset_root, data_split='test', dataname=dataname, flag='target')
    retrieval_ds = PreDataset(data_path=dataset_root, data_split='retrieval', dataname=dataname, flag='target')
    
    # 创建数据加载器
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, drop_last=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, drop_last=False)
    retrieval_dl = DataLoader(dataset=retrieval_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    
    return train_dl, test_dl, retrieval_dl





if __name__ == '__main__':

    pass
