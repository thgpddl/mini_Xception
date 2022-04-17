import torch
from torchsummary import summary
from utils.Model import mini_XCEPTION

model=mini_XCEPTION().cuda()
print(model)
summary(model,input_size=(1,48,48))