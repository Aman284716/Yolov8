import torch
import torch.nn as nn

# Define the Conv module


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Define the C2f module


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(C2f, self).__init__()
        self.conv1 = Conv(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = Conv(out_channels // 2, out_channels //
                          2, kernel_size=3, padding=1)
        self.conv3 = Conv(out_channels // 2, out_channels, kernel_size=1)
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            return self.conv3(self.conv2(self.conv1(x))) + x
        else:
            return self.conv3(self.conv2(self.conv1(x)))

# Define the SPPF module


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[5]):
        super(SPPF, self).__init__()
        self.pool = nn.ModuleList(
            [nn.MaxPool2d(size, stride=1, padding=size // 2) for size in pool_sizes])
        self.conv = Conv(in_channels * (len(pool_sizes) + 1),
                         out_channels, kernel_size=1)

    def forward(self, x):
        pooled = [x] + [p(x) for p in self.pool]
        return self.conv(torch.cat(pooled, dim=1))

# Define the Concat module


class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

# Define the Detect module


class Detect(nn.Module):
    def __init__(self, nc, anchors=9):
        super(Detect, self).__init__()
        self.nc = nc
        self.anchors = anchors
        self.detect = nn.ModuleList([
            # Adjust channels as needed
            Conv(256, anchors * (nc + 5), kernel_size=1)
            for _ in range(3)  # For each scale
        ])

    def forward(self, x):
        return [d(x) for d in self.detect]

# Define the YOLOv8 model


class YOLOv8(nn.Module):
    def __init__(self, nc=80):
        super(YOLOv8, self).__init__()
        # Backbone
        self.backbone = nn.Sequential(
            Conv(3, 64, kernel_size=3, stride=2, padding=1),
            Conv(64, 128, kernel_size=3, stride=2, padding=1),
            C2f(128, 128),
            Conv(128, 256, kernel_size=3, stride=2, padding=1),
            C2f(256, 256),
            Conv(256, 512, kernel_size=3, stride=2, padding=1),
            C2f(512, 512),
            Conv(512, 1024, kernel_size=3, stride=2, padding=1),
            C2f(1024, 1024),
            SPPF(1024, 1024, [5])
        )

        # Head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(1),
            C2f(512, 256),

            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(1),
            C2f(256, 128),

            Conv(128, 256, kernel_size=3, stride=2),
            Concat(1),
            C2f(256, 512),

            Conv(512, 512, kernel_size=3, stride=2),
            Concat(1),
            C2f(512, 1024),

            Detect(nc)
        )

    def forward(self, x):
        # Backbone
        x1 = self.backbone[0:2](x)
        x2 = self.backbone[2:5](x1)
        x3 = self.backbone[5:8](x2)
        x4 = self.backbone[8:](x3)

        # Head
        x = self.head[0:2](x4)
        x = self.head[2:5](x)
        x = self.head[5:8](x)
        x = self.head[8:11](x)
        out = self.head[11:](x)
        return out
