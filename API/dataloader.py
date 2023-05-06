# from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
# from .dataloader_mhws import load_data as load_mhws
from .dataloader_sst import load_data as load_sst

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mhws':
        data_root = data_root+"mhws_1982_2020/2020/"
        return load_mhws(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'sst':
        data_root = data_root+"sst/"
        return load_sst(batch_size, val_batch_size, data_root, num_workers)