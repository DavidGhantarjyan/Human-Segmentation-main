import torch.nn as nn
import torch.nn.functional as F

class CBG(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, groups=1, bias=True, use_activation=True):
        super(CBG, self).__init__()
        self.stride = stride
        self.padding = (kernel_size - 1) // 2  # Reflective padding to preserve dimensions after convolution
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=self.padding, padding_mode='reflect' if stride == 1 else 'zeros',
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_nc)
        self.use_activation = use_activation
        if self.use_activation:
            self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_activation:
            x = self.relu6(x)
        return x

class MobileNetV1Block(nn.Module):
    # original mobilenet block use relu no relu 6
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(MobileNetV1Block, self).__init__()
        self.depthwise = CBG(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels,
                             bias=bias, use_activation=True)  # Depthwise
        self.pointwise = CBG(in_channels, out_channels, kernel_size=1, stride=1, bias=bias, use_activation=True)  # Pointwise

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
        self.expand = CBG(in_channels, mid_channels, kernel_size=1, stride=1, bias=bias, use_activation=True)
        self.depthwise = CBG(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, bias=bias, use_activation=True)
        self.pointwise = CBG(mid_channels, out_channels, kernel_size=1, stride=1, bias=bias, use_activation=False)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.use_residual:
            x += residual
        return x


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=2, use_se=True, bias=True):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.use_se = use_se
        mid_channels = int(in_channels * expansion_factor)
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = CBG(in_channels, mid_channels, kernel_size=1, stride=1,
                          bias=bias, use_activation=False) if expansion_factor > 1 else None

        self.depthwise = CBG(mid_channels, mid_channels, kernel_size=3, stride=stride,
                             groups=mid_channels, bias=bias, use_activation=False)

        if self.use_se:
            se_channels = max(1, mid_channels // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, mid_channels, kernel_size=1, bias=True),
                nn.Hardsigmoid(inplace=True)
            )

        self.pointwise = CBG(mid_channels, out_channels, kernel_size=1, stride=1, bias=bias, use_activation=False)

    def h_swish(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

    def forward(self, x):
        residual = x

        if self.expand is not None:
            x = self.expand(x)
            x = self.h_swish(x)
        else:
            x = x

        x = self.depthwise(x)
        x = self.h_swish(x)

        # Squeeze-and-Excitation
        if self.use_se:
            se_weights = self.se(x)
            x = x * se_weights

        x = self.pointwise(x)
        # After pointwise in MobileNetV3 there is usually no activation (linear output),
        # so we don't apply h_swish here, we leave it as it is after CBG

        if self.use_residual:
            x += residual

        return x

class MobileNetv(nn.Module):
    def __init__(self, n_channels, block_type, alpha, **kwargs):
        """
        Parameters:
        - n_channels: Number of input channels
        - block_type: The block class to use (e.g., MobileNetV1Block, MobileNetV2Block)
        - kwargs: Additional arguments for the block type
        """
        super(MobileNetv, self).__init__()
        self.alpha = alpha
        def conv_dw_block(in_channels, out_channels, stride):
            # change conv_dw channels count
            in_channels = int(in_channels * self.alpha)
            out_channels = int(out_channels * self.alpha)
            return nn.Sequential(
                block_type(in_channels, out_channels, stride, **kwargs),
                block_type(out_channels, out_channels, 1, **kwargs)
            )
        self.layer1 = nn.Sequential(
            CBG(n_channels, int(32 * self.alpha), kernel_size=3, stride=1, bias=True),  # Initial conv layer
            block_type(int(32 * self.alpha), int(64 * self.alpha), 1, **kwargs),  # Block type layer
            conv_dw_block(64, 128, 2)  # Downsampling layer
        )
        self.layer2 = nn.Sequential(
            conv_dw_block(128, 256, 2)
        )

        self.layer3 = nn.Sequential(
            conv_dw_block(256, 512, 2),
            *[block_type(int(512 * self.alpha), int(512 * self.alpha), 1, **kwargs) for _ in range(5)]
        )
        self.layer4 = nn.Sequential(
            conv_dw_block(512, 1024, 2)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * self.alpha), 1000)

        # self.dropout3 = nn.Dropout2d(p=0.3) if use_dropout  else nn.Identity()
        # self.dropout4 = nn.Dropout2d(p=0.3) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.dropout3(x)
        x = self.layer4(x)
        # x = self.dropout4(x)
        x = self.avg(x)
        x = x.view(-1,  int(1024 * self.alpha))
        x = self.fc(x)
        return x
