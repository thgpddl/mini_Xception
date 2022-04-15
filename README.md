# mini_Xception
用于表情识别的轻量级卷积神经网络

来自论文[《Real-time Convolutional Neural Networks for Emotion and Gender Classification》](https://arxiv.org/pdf/1710.07557v1.pdf)

当然有官方的项目:[oarriaga/face_classification](https://github.com/oarriaga/face_classification)

论文解析及复现总结移步[ CSDN]()

需要注意的是：
> 官方使用keras在fer2013数据集上的表情识别精度达到了66%，本项目使用pytorch实现的精度达到了65%

# 1、安装轮子
使用命令：
> pip install -r requirements.txt

如果太慢，可以加个清华源：
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 2、训练
运行train.py脚本：
- num_epochs = 200
- log_step = 100      # 打印info的间隔步数
- num_workers = 0    # 线程数

# 3、测试
测试单幅图像，运行frame.py脚本
摄像头实时预测，运行video.py脚本

# 4、DeBug
1. 出现“BrokenPipeError: [Errno 32] Broken pipe”，把线程数num_workers=0即可。
