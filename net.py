from torch import nn
import torch

class LeNet(nn.Module):
    def __init__(self, num_classes, decoding_method) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.decoding_method = decoding_method
        
        # 3x28x28
        self.conv1 = nn.Sequential(     
            nn.Conv2d(3, 6, kernel_size=5, padding=2),  # 3x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 6x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5),    # 64x10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 64x5x5

        # upsampling
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 64x10x10
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # 64x8x8
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 64x16x16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # 64x14x14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 64x28x28
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
            nn.ReLU(),
        )

        # deconv
        self.deconv = nn.Sequential(
            # TODO:finish deconv
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2), # 32x11x11
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2), # 32x25x25
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5), # 32x29x29
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=2),
            nn.ReLU(),
        )
        # out_channelsx28x28
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.decoding_method == 'upsample':
            out = self.upsample(out)
        elif self.decoding_method == 'deconv':
            out = self.deconv(out)
        return out

if __name__ == '__main__':
    input = torch.randn((1, 3, 28, 28))
    # print(input.numel())
    model = LeNet(num_classes=2)

    out = model(input)
    print(out.shape)