import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from RMSIF_NET import RMSIF_NET, BasicBlock
import data
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(model, valid_loader):  # 验证模块
    valid_loss = 0.0
    model.eval()  # 下面的不做反向传播，即为验证部分
    targets, outputs = [], []
    for step, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device).float()
        with torch.no_grad():
            output = model(data)
            loss = F.mse_loss(output, target)  # 计算loss
            valid_loss += loss.item() * data.size(0)  # 计算验证损失
            outputs.extend(output.tolist())
            targets.extend(target.tolist())
    valid_loss = valid_loss / len(valid_loader.dataset)
    r_s = spearmanr(outputs, targets)[0]
    return valid_loss, r_s


def train(model, train_loader, optimizer):
    train_loss = 0.0
    # 训练模块
    model.train()
    for step, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        output = model(data) # 得到输出
        loss = F.mse_loss(output, target)
        optimizer.zero_grad()  # 将梯度清零
        loss.backward()  # 求方向传播得梯度
        optimizer.step()  # 反向传播
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)  # 计算平均损失
    return train_loss


def main():
    dataset_dir = ''
    lables_dir = ''
    dataset_dir_val = ''
    lables_dir_val = ''
    dataset_dir_test = ('')
    lables_dir_test = ''
    # 构建数据集
    train_dataset, val_dataset, test_dataset = data.create_datasets(
        dataset_dir, lables_dir, dataset_dir_val, lables_dir_val, dataset_dir_test, lables_dir_test)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)

    # 初始化神经网络
    model = RMSIF_NET(BasicBlock=BasicBlock, num_classes=1).to(device)

    save_train_loss = []
    save_Valid_loss = []
    total_epochs = 30  # 设置学习轮次为25
    best_model_path = 'models.pt'  # 最佳模型的保存路径
    for epoch in range(total_epochs):  # 开始循环训练
        # 学习率
        learning_rate = 0.01  # 0.03
        if epoch > 30:
            learning_rate = 0.001
        if epoch > 90:
            learning_rate = 0.001
        if epoch > 200:
            learning_rate = 0.0001

        # 在训练集上训练一轮
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降
        train_loss = train(model, train_loader, optimizer)
        save_train_loss.append(train_loss)

        # 计算验证集上的性能
        valid_loss, spcc = validation(model, valid_loader)
        save_Valid_loss.append(valid_loss)

        print(f'Epoch={epoch + 1}, TrainLoss={train_loss:.4f}, Validation: Loss={valid_loss:.4f}')

    # 在所有训练结束后保存模型
    torch.save(model.state_dict(), best_model_path)
    print(f'Model saved after {total_epochs} epochs.')

    # 计算测试集上的性能
    model.load_state_dict(torch.load(best_model_path))
    test_loss, spcc = validation(model, test_loader)
    print(f'Test Set: Loss={test_loss:.4f}, spcc={spcc:.4f}')


if __name__ == '__main__':
    main()
