from torch import nn


class FCN(nn.Module):
    def __init__(self, in_nc, nc, num_classes):
        super().__init__()

        self.down_sampling = nn.Sequential(
            nn.Conv2d(in_nc, nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(nc, nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(nc*2, nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.out = nn.Conv2d(nc*4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.down_sampling(x)
        x = self.up_sampling(x)
        x = self.out(x)

        return x
