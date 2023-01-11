from torch import nn
import torch
import  torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 1x32x32
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 6x28x28
        self.s2 = nn.AdaptiveAvgPool2d((14, 14))
        # 6x14x14
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 16x10x10
        self.s4 = nn.AdaptiveAvgPool2d((5, 5))
        # 16x5x5

        # upsampling
        self.u1 = nn.UpsamplingBilinear2d(scale_factor=8)
        # 16x40x40
        self.u2 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=5)
        # 10x36x36
        self.u3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5)
        # 10x32x32
        

    def forward(self, x):
        out = F.relu(self.c1(x))
        out = F.relu(self.s2(out))
        out = F.relu(self.c3(out))
        out = F.relu(self.s4(out))

        out = F.relu(self.u1(out))
        out = F.relu(self.u2(out))
        out = F.relu(self.u3(out))

        return out

if __name__ == '__main__':
    input = torch.randn((1, 1, 32, 32))
    model = LeNet()
    out = model(input)
    print(out.shape)