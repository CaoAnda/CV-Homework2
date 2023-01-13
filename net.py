from torch import nn
import torch
import  torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 3x28x28
        self.conv1 = nn.Sequential(     
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2, padding=0)
        )

        # 6x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, stride=1, padding=0), #input_size=(6*14*14)，output_size=16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    ##input_size=(16*10*10)，output_size=(16*5*5)
        )
        # 16x5x5

        # upsampling
        self.upsamp = nn.Sequential(
            nn.Upsample(scale_factor=6),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3),
            nn.ReLU(),
        )
        # deconv
        self.deconv = nn.Sequential(
            # TODO:finish deconv
            nn.ConvTranspose2d(16, 32, kernel_size=),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3),
            nn.ReLU(),
        )
        # 2x28x28
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.upsamp(out)

        return out

if __name__ == '__main__':
    input = torch.randn((1, 3, 28, 28))
    # print(input.numel())
    model = LeNet()

    out = model(input)
    print(out.shape)