from nilearn import datasets
from torch.utils.data import DataLoader, TensorDataset
import warnings
import torch


#  下载数据集
def get_dataset():
    warnings.filterwarnings("ignore")
    abide_dataset = datasets.fetch_abide_pcp('ABIDE-871/',
                                             derivatives=['rois_aal'],
                                             pipeline='cpac',
                                             band_pass_filtering=True,
                                             global_signal_regression=True,
                                             quality_checked=True,
                                             legacy_format=True)
    y_label = abide_dataset["phenotypic"]["DX_GROUP"] - 1
    rois = abide_dataset["rois_aal"]
    return rois, y_label


def split_dataset(seq_length):
    train_x = []
    train_y = []
    rois, y_label = get_dataset()
    for i, roi in enumerate(rois):
        _l, _n = roi.shape
        for j in range(0, _l // seq_length):
            train_x.append(roi[j: seq_length + j, :])
            train_y.append(y_label[i])
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y).view(-1)
    train_y = torch.nn.functional.one_hot(train_y.to(torch.int64))
    all_datasets = TensorDataset(train_x, train_y)
    lens = len(all_datasets)
    print(f"total lens of the dataset is {lens}")
    train = int(lens * 0.7)
    val = int(lens * 0.2)
    test = lens - train - val
    train_set, val_set, test_set = torch.utils.data.random_split(all_datasets, [train, val, test])
    return train_set, val_set, test_set


def load_data(seq_length: int, batch_size: int):
    train_set, val_set, test_set = split_dataset(seq_length)
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    return train_dataloader, test_dataloader, val_dataloader
