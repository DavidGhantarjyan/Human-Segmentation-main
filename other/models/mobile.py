import torch.nn as nn
import torch.nn.functional as F


class CBG(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, groups=1, bias=True):
        super(CBG, self).__init__()
        self.stride = stride
        self.padding = (kernel_size - 1) // 2  # Reflective padding to preserve dimensions after convolution
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=self.padding, padding_mode='reflect' if stride == 1 else 'zeros',
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_nc)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        # if self.stride == 1:
        #     x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')  # Reflect padding
        # else:
        #     x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        out = self.conv(x)  # Convolution operation
        out = self.bn(out)  # Normalize activations
        out = self.relu6(out)  # Apply LeakyReLU activation
        return out


class MobileNetV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(MobileNetV1Block, self).__init__()
        self.depthwise = CBG(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels,
                             bias=bias)  # Depthwise
        self.pointwise = CBG(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)  # Pointwise

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        return x


class MobileNetV2Block(nn.Module):
    """
    Inverted residual block with expansion.
    """

    def __init__(self, in_channels, out_channels, stride, expansion_factor=2, bias=True):
        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        mid_channels = int(in_channels * expansion_factor)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand = CBG(in_channels, mid_channels, kernel_size=1, stride=1, bias=bias)
        self.depthwise = CBG(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, bias=bias)
        self.pointwise = CBG(mid_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.use_residual:
            x += residual
        return x


class MobileNetv(nn.Module):
    def __init__(self, n_channels, block_type, **kwargs):
        """
        Parameters:
        - n_channels: Number of input channels
        - block_type: The block class to use (e.g., MobileNetV1Block, MobileNetV2Block)
        - kwargs: Additional arguments for the block type
        """
        super(MobileNetv, self).__init__()

        def conv_dw_block(in_channels, out_channels, stride):
            return nn.Sequential(
                block_type(in_channels, out_channels, stride, **kwargs),
                block_type(out_channels, out_channels, 1, **kwargs)
            )

        self.layer1 = nn.Sequential(
            CBG(n_channels, 32, kernel_size=3, stride=1, bias=True),  # Initial conv layer
            block_type(32, 64, 1, **kwargs),  # Block type layer
            conv_dw_block(64, 128, 2)  # Downsampling layer
        )
        self.layer2 = nn.Sequential(
            conv_dw_block(128, 256, 2)
        )
        self.layer3 = nn.Sequential(
            conv_dw_block(256, 512, 2),
            *[block_type(512, 512, 1, **kwargs) for _ in range(5)]
        )
        self.layer4 = nn.Sequential(
            conv_dw_block(512, 1024, 2)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
