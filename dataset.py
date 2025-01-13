from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms
import random
import numpy as np


class RNA_Mg_Dataset(Dataset):
    def __init__(self, dataset_label_list, transform):
        self.dataset_label_list = dataset_label_list
        self.transform = transform

    def __getitem__(self, item):
        (data_path, label_path) = self.dataset_label_list[item]
        data = np.load(data_path, allow_pickle=True)
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            label = [float(x) for x in line.split(',')]
            label = np.array(label)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset_label_list)


def create_datasets(dataset_dir, lables_dir):
    list_data_path = glob.glob(os.path.join(dataset_dir, "*.npy"))
    combined_list = []
    for data_path in list_data_path:
        label_path = os.path.join(lables_dir, os.path.basename(data_path).replace('.npy', '.txt'))
        if not os.path.exists(label_path):
            print(f'找不到{label_path}')
            continue
        # 有的数据里面有nan值，需要去掉
        data = np.load(data_path, allow_pickle=True)
        if any(np.isnan(data.reshape(-1))):
            print(f'{data_path}数据里面有nan值')
            continue

        combined_list.append([data_path, label_path])
    print(f"npy数据个数为{len(list_data_path)}个，找到对应的txt label为{len(combined_list)}个")
    print(f"匹配得到的data-label数据对{len(combined_list)}个")
    random.shuffle(combined_list)
    p1 = int(len(combined_list) * 0.7)
    p2 = int(len(combined_list) * 0.9)
    train_list, val_list, test_list = combined_list[:p1], combined_list[p1:p2], combined_list[p2:]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = RNA_Mg_Dataset(train_list, transform)
    val_dataset = RNA_Mg_Dataset(val_list, transform)
    test_dataset = RNA_Mg_Dataset(test_list, transform)
    print(test_list)
    print(f'构建数据集，训练集大小：{len(train_dataset)}，验证集大小：{len(val_dataset)}，测试集大小：{len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    dataset_dir = 'force'
    lables_dir = 'output_labels'
    #train_dataset = create_datasets(dataset_dir, lables_dir)
    train_dataset, val_dataset, test_dataset = create_datasets(dataset_dir, lables_dir)
    # for data, label in train_dataset:
    #     print(np.max(data), np.min(data), np.shape(data))