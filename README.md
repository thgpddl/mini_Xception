# mini_Xception
用于表情识别的轻量级卷积神经网络

来自论文[《Real-time Convolutional Neural Networks for Emotion and Gender Classification》](https://arxiv.org/pdf/1710.07557v1.pdf)

当然有官方的项目：[oarriaga/face_classification](https://github.com/oarriaga/face_classification)

关于论文详解，可以移步博客：[Real-time Convolutional Neural Networks for Emotion and Gender Classification--O Arriaga](https://blog.csdn.net/qq_40243750/article/details/124208527)，需要重点关注的是：
> 论文中基于kera实现的，在fer2013数据集上达到了66%的精度。本文是基于Pytorch实现的，最高只能达到65%的精度。对于这1%~2%精度差异未找到原因，只能归结于框架的不同（不同框架之间的效果会有差异）。

我复现论文的总结，可以移步：[Pytorch实现表情识别卷积神经网络网络：mini_Xception
](https://blog.csdn.net/qq_40243750/article/details/124226066?spm=1001.2014.3001.5501)

# 1、安装轮子
使用命令：
> pip install -r requirements.txt

如果太慢，可以加个清华源：
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 2、下载数据集
下将链接中的train.csv和test.csv文件下载下来：[数据集](https://www.aliyundrive.com/s/fQz68x23mtk)

然后在mini_Xception根目录创建dataset文件夹，将train.csv和test.csv文件放在dataset文件夹中即可


# 3、训练
运行train.py脚本：
- num_epochs = 200
- log_step = 100      # 打印info的间隔步数
- num_workers = 16    # 线程数

# 4、eval
运行eval.py脚本，会计算出测试集的精度和loss，并且显示出混淆矩阵，并保存为图片。
![ConfusionMatrix](https://user-images.githubusercontent.com/48787805/163796143-8d134aa7-9e51-433b-9da8-61c651f4bb5d.png)



# 5、测试
测试单幅图像，运行frame.py脚本
摄像头实时预测，运行video.py脚本

# 6、DeBug
1. 出现“BrokenPipeError: [Errno 32] Broken pipe”，把线程数num_workers=0即可。
