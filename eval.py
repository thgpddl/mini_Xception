import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#
from utils.Model import mini_XCEPTION
from utils.dataset import FER2013


class DrawConfusionMatrix:
    def __init__(self, labels_name):
        """

        :param num_classes: 分类数目
        """
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def draw(self):
        per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
        for i in range(self.num_classes):
            self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比

        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(7), self.labels_name)  # y轴标签
        plt.xticks(range(7), self.labels_name, rotation=45)  # x轴标签

        for x in range(7):
            for y in range(7):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()


def eval():
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                                                           'neutral'])
    total_test_loss = 0
    total_test_acc = 0
    count = 0
    model.eval()
    for index, (labels, imgs) in enumerate(test_loader):
        labels_pd = model(imgs.to(device))
        predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)
        labels_np = labels.numpy()
        drawconfusionmatrix.update(predict_np, labels_np)
        acc = sum(predict_np == labels_np)
        loss = loss_fn(labels_pd, labels.to(device))
        total_test_loss += loss.item()
        total_test_acc += acc
        count += 1

    mean_test_loss = total_test_loss / count
    mean_test_acc = total_test_acc / count
    print("evla\tloss:{:.4f}\tacc:{:.4f}".format(mean_test_loss, mean_test_acc))
    drawconfusionmatrix.draw()


if __name__ == "__main__":
    num_workers = 0  # 线程数

    # output文件夹，会根据当前时间命名文件夹。

    batch_size = 32
    input_size = (48, 48)
    num_classes = 7

    # 定义模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mini_XCEPTION(num_classes=7)
    model.load_state_dict(torch.load("output/E370_acc_0.6504.pth", map_location=device))
    model.to(device)

    # 数据加载
    test_dataset = FER2013("test", input_size=input_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 优化器
    loss_fn = torch.nn.CrossEntropyLoss()

    # 开始评估
    eval()
