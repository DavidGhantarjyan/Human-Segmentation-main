import torch.nn as nn
import torch.nn.functional as F

class CBG(nn.Module):
    """
        Convolution-BatchNorm-Activation block for MobileNet architectures.

        Combines a convolution layer, batch normalization, and optional ReLU6 activation
        with reflective or zero padding to preserve spatial dimensions.
        """
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, groups=1, bias=True, use_activation=True):
        """
            Args:
                in_nc (int): Number of input channels.
                out_nc (int): Number of output channels.
                kernel_size (int): Convolution kernel size, default 3.
                stride (int): Convolution stride, default 1.
                groups (int): Number of groups for grouped convolution, default 1.
                bias (bool): Whether to include bias in convolution, default True.
                use_activation (bool): Whether to apply ReLU6 activation, default True.
                """
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
        """
            Args:
                x (torch.Tensor): Input tensor, shape (N, in_nc, H, W).

            Returns:
                torch.Tensor: Output tensor, shape (N, out_nc, H', W').
                """
        x = self.conv(x)
        x = self.bn(x)
        if self.use_activation:
            x = self.relu6(x)
        return x

class MobileNetV1Block(nn.Module):
    """
        MobileNetV1-style block with depthwise separable convolution.

        Consists of a depthwise 3x3 convolution followed by a pointwise 1x1 convolution,
        both with ReLU6 activation (note: original MobileNetV1 uses ReLU).
        """
    # original mobilenet block use relu no relu 6
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        """
            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                stride (int): Stride for depthwise convolution, default 1.
                bias (bool): Whether to include bias, default True.
                """
        super(MobileNetV1Block, self).__init__()
        self.depthwise = CBG(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels,
                             bias=bias, use_activation=True)  # Depthwise
        self.pointwise = CBG(in_channels, out_channels, kernel_size=1, stride=1, bias=bias, use_activation=True)  # Pointwise

    def forward(self, x):
        """
            Args:
                x (torch.Tensor): Input tensor, shape (N, in_channels, H, W).

            Returns:
                torch.Tensor: Output tensor, shape (N, out_channels, H', W').
                """
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        return x


class MobileNetV2Block(nn.Module):
    """
        MobileNetV2-style inverted residual block with expansion.

        Consists of expansion (1x1), depthwise (3x3), and projection (1x1) convolutions
        with a residual connection if stride=1 and channel counts match.
        """
    def __init__(self, in_channels, out_channels, stride, expansion_factor=2, bias=True):
        """
            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                stride (int): Stride for depthwise convolution.
                expansion_factor (float): Channel expansion factor, default 2.
                bias (bool): Whether to include bias, default True.
                """
        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        mid_channels = int(in_channels * expansion_factor) # Ensure at least 1 channel
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand = CBG(in_channels, mid_channels, kernel_size=1, stride=1, bias=bias, use_activation=True)
        self.depthwise = CBG(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, bias=bias, use_activation=True)
        self.pointwise = CBG(mid_channels, out_channels, kernel_size=1, stride=1, bias=bias, use_activation=False)

    def forward(self, x):
        """
            Args:
                x (torch.Tensor): Input tensor, shape (N, in_channels, H, W).

            Returns:
                torch.Tensor: Output tensor, shape (N, out_channels, H', W').
                """
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.use_residual:
            x += residual
        return x


class MobileNetV3Block(nn.Module):
    """
        MobileNetV3-style block with expansion, depthwise, and squeeze-and-excitation (SE).

        Uses h-swish activation, optional SE module, and residual connection if applicable.
        """
    def __init__(self, in_channels, out_channels, stride, expansion_factor=2, use_se=True, bias=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for depthwise convolution.
            expansion_factor (float): Channel expansion factor, default 2.
            use_se (bool): Whether to include SE module, default True.
            bias (bool): Whether to include bias, default True.
                """
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.use_se = use_se
        mid_channels = int(in_channels * expansion_factor) # Ensure at least 1 channel
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
        """
        Hard-swish activation: x * ReLU6(x + 3) / 6.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with h-swish applied.
                """
        return x * F.relu6(x + 3, inplace=True) / 6

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor, shape (N, out_channels, H', W').
        """
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
    """
    Customizable MobileNet-like backbone using specified block type (V1, V2, or V3).

    Scales channel counts with width multiplier (alpha) and outputs classification logits.
    Supports optional dropout for regularization.
        """
    def __init__(self, n_channels, block_type, alpha, **kwargs):
        """
        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB).
            block_type (nn.Module): Block class (e.g., MobileNetV1Block, MobileNetV2Block, MobileNetV3Block).
            alpha (float): Width multiplier to scale channel counts.
            use_dropout (bool): Whether to apply dropout in layer3 and layer4, default False.
            **kwargs: Additional arguments for block_type (e.g., expansion_factor, use_se).
                """
        super(MobileNetv, self).__init__()
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        self.alpha = alpha
        def conv_dw_block(in_channels, out_channels, stride):
            """
            Helper to create a block with scaled channels and two block_type layers.

            Args:
                in_channels (int): Input channels.
                out_channels (int): Output channels.
                stride (int): Stride for the first block.

            Returns:
                nn.Sequential: Block sequence.
                        """
            in_channels = int(in_channels * self.alpha) # Ensure at least 1 channel
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
        """
        Args:
            x (torch.Tensor): Input tensor, shape (N, n_channels, H, W).

        Returns:
            torch.Tensor: Output logits, shape (N, 1000).
                """
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
