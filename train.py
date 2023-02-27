import os
import torch
import datetime
import numpy as np
from visualdl import LogWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils.Model import mini_XCEPTION
from utils.dataset import FER2013

num_epochs = 200
log_step = 100      # 打印info的间隔步数
num_workers = 10    # 线程数

# output文件夹，会根据当前时间命名文件夹。
base_path = 'output/{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
writter = LogWriter(logdir=base_path)

batch_size = 32
input_size = (48, 48)
num_classes = 7
patience = 50

if not os.path.exists(base_path):
    os.makedirs(base_path)

# 定义模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mini_XCEPTION(num_classes=7)
model.to(device)

# 数据加载
train_dataset = FER2013("train", input_size=input_size)
test_dataset = FER2013("test", input_size=input_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 优化器
optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       factor=0.1,
                                                       patience=int(patience / 4),
                                                       verbose=True)


def train_f():
    # 训练
    best_acc = 0
    step = 0
    for Epoch in range(0, num_epochs):
        total_train_loss, total_test_loss = 0, 0
        total_train_acc, total_test_acc = 0, 0
        count = 0
        end_index = len(train_loader) - 1
        model.train()
        for index, (labels, imgs) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels_pd = model(imgs)
            # 记录acc和loss
            acc = accuracy_score(np.argmax(labels_pd.cpu().detach().numpy(), axis=-1), labels)
            total_train_acc += acc
            loss = loss_fn(labels_pd, labels.to(device))
            total_train_loss += loss.item()
            count += 1
            # 更新梯度
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_mean_acc = total_train_acc / count
            epoch_mean_loss = total_train_loss / count

            step += 1
            writter.add_scalar(tag="train_acc", step=step, value=epoch_mean_acc)
            writter.add_scalar(tag="train_loss", step=step, value=epoch_mean_loss)

            if index % log_step == 0 or index == end_index:
                print("e:{}\titer:{}/{}\tloss:{:.4f}\tacc:{:.4f}".format(Epoch, index, end_index,
                                                                         epoch_mean_loss,
                                                                         epoch_mean_acc))
        count = 0
        model.eval()
        for index, (labels, imgs) in enumerate(test_loader):
            labels_pd = model(imgs.to(device))
            acc = accuracy_score(np.argmax(labels_pd.cpu().detach().numpy(), axis=-1), labels)
            loss = loss_fn(labels_pd, labels.to(device))
            total_test_loss += loss.item()
            total_test_acc += acc
            count += 1

        mean_test_loss = total_test_loss / count
        mean_test_acc = total_test_acc / count
        
        scheduler.step(mean_test_acc)
        print("evla\tloss:{:.4f}\tacc:{:.4f}".format(mean_test_loss, mean_test_acc))

        writter.add_scalar(tag="test_acc", step=Epoch, value=mean_test_acc)
        writter.add_scalar(tag="test_loss", step=Epoch, value=mean_test_loss)

        if (total_test_acc / count) > best_acc:
            torch.save(model.state_dict(), "{}/E{}_acc_{:.4f}.pth".format(base_path, Epoch, total_test_acc / count))
            best_acc = total_test_acc / count
            print("saved best model")


if __name__ == "__main__":
    train_f()
