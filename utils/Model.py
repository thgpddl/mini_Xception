from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1):
        super(SeparableConv2d, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# residual depth-wise separable convolutions
class RDWSC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(RDWSC, self).__init__()

        self.left = nn.Sequential(SeparableConv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(),
                                   SeparableConv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))

        self.right = nn.Sequential(nn.Conv2d(input_channels, output_channels, (1, 1), stride=(2, 2)),
                                  nn.BatchNorm2d(output_channels))

    def forward(self, x):
        right = self.right(x)
        left = self.left(x)
        output = right + left
        return output


class mini_XCEPTION(nn.Module):
    def __init__(self, num_classes=7):
        super(mini_XCEPTION, self).__init__()

        self.base = nn.Sequential(nn.Conv2d(1, 8, (3, 3), (1, 1)),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 8, (3, 3), stride=(1, 1)),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU())
        self.module1 = RDWSC(input_channels=8, output_channels=16)
        self.module2 = RDWSC(input_channels=16, output_channels=32)
        self.module3 = RDWSC(input_channels=32, output_channels=64)
        self.module4 = RDWSC(input_channels=64, output_channels=128)

        # output
        self.conv=nn.Conv2d(128, num_classes, kernel_size=(3, 3),padding=1)

    def forward(self, x):
        x = self.base(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x=self.conv(x)
        x=x.mean(axis=[-1,-2])  # avgpool
        return x

