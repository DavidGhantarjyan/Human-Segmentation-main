import torch
import torch.nn as nn
from other.models.mobile import CBG, MobileNetv, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block
from other.models.guided_filter import ConvGuidedFilter




class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            CBG(in_nc, out_nc, kernel_size=kernel_size, stride=stride),
            CBG(out_nc, out_nc, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, inputs):
        return self.conv_block(inputs)


class Downsample_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1):
        super(Downsample_Block, self).__init__()
        self.conv_block = ConvBlock(in_nc, out_nc, kernel_size=kernel_size, stride=stride)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling for downsampling

    def forward(self, inputs):
        x = self.conv_block(inputs)
        p = self.maxpool(x)  # Apply max pooling for spatial downsampling
        return x, p

class Upsample_Block(nn.Module):
    def __init__(self, in_nc, out_nc, block_type=None, kernel_size=3, stride=1, skip_connect=True, upsample_type='transpose', **kwargs):
        super(Upsample_Block, self).__init__()
        self.skip_connect = skip_connect  # Set skip_connect flag
        self.upsample_type = upsample_type
        if self.upsample_type == 'transpose':
            # Transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif self.upsample_type == 'interpolate':
            # Interpolation followed by convolution for upsampling
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=(kernel_size - 1) // 2))

        if self.skip_connect:
            if block_type:
                self.conv_block = block_type(2 * out_nc, out_nc,
                                            stride=stride)  # For skip connection
                self.final_block = block_type(out_nc, out_nc,stride=stride)
            else:
                self.conv_block = ConvBlock(2 * out_nc, out_nc,
                                                stride=stride, kernel_size=kernel_size)
                self.final_block = ConvBlock(out_nc, out_nc,stride=stride,kernel_size=kernel_size)

        else:
            if block_type:
                self.conv_block = block_type(out_nc, out_nc,
                                             stride=stride)  # For skip connection
                self.final_block = block_type(out_nc, out_nc,stride=stride)
            else:
                self.conv_block = ConvBlock(out_nc, out_nc,
                                            stride=stride, kernel_size=kernel_size)
                self.final_block = ConvBlock(out_nc, out_nc,
                                                stride=stride, kernel_size=kernel_size)


    @staticmethod
    def center_crop(output, skip_height, skip_width):
        _, _, h, w = output.size()
        start_x = (w - skip_width) // 2
        start_y = (h - skip_height) // 2
        return output[:, :, start_y:start_y + skip_height, start_x:start_x + skip_width]

    def forward(self, inputs, skip):
        x = self.up(inputs)  # Upsample the input tensor
        _, _, h_skip, w_skip = skip.size()
        x = self.center_crop(x, h_skip, w_skip)
        if self.skip_connect:
            x = torch.cat([x, skip], dim=1)  # Concatenate the skip connection
        x = self.conv_block(x)
        x = self.final_block(x)
        return x


class mobilenet(nn.Module):
    def __init__(self, n_channels, block_type, alpha):
        super(mobilenet, self).__init__()
        self.model = MobileNetv(n_channels, block_type, alpha)
        # self.dropout3 = nn.Dropout2d(p=0.3) if use_dropout else nn.Identity()
        # self.dropout4 = nn.Dropout2d(p=0.3) if use_dropout else nn.Identity()

    def forward(self, x):
        out1 = self.model.layer1(x)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        # out3 = self.dropout3(out3)
        out4 = self.model.layer4(out3)
        # out4 = self.dropout4(out4)

        return out1, out2, out3, out4


class UNet_MobileNet(nn.Module):
    def __init__(self, n_channels, num_classes, block_type, upsample_type, alpha=1.0, use_dropout=False):
        super(UNet_MobileNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        # Backbone MobileNetv1/2
        self.backbone = mobilenet(n_channels, block_type, alpha)

        self.up1 = Upsample_Block(int(alpha * 1024), int(alpha * 512), block_type=block_type,
                                  upsample_type=upsample_type)
        self.up2 = Upsample_Block(int(alpha * 512), int(alpha * 256), block_type=block_type,
                                  upsample_type=upsample_type)
        self.up3 = Upsample_Block(int(alpha * 256), int(alpha * 128), block_type=block_type,
                                  upsample_type=upsample_type)
        self.up4 = Upsample_Block(int(alpha * 128), int(alpha * 64), block_type=block_type, skip_connect=False,
                                  upsample_type=upsample_type)

        self.dropout1 = nn.Dropout2d(p=0.5) if use_dropout else nn.Identity()
        self.dropout2 = nn.Dropout2d(p=0.4) if use_dropout else nn.Identity()
        self.dropout3 = nn.Dropout2d(p=0.3) if use_dropout else nn.Identity()
        self.dropout4 = nn.Dropout2d(p=0.2) if use_dropout else nn.Identity()

        self.out = nn.Conv2d(int(alpha * 64), num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features using MobileNetv1
        s1, s2, s3, s4 = self.backbone(x)
        # Upsampling with skip connections
        u1 = self.dropout1(self.up1(s4, s3))
        u2 = self.dropout2(self.up2(u1, s2))
        u3 = self.dropout3(self.up3(u2, s1))
        u4 = self.dropout4(self.up4(u3, x))

        # u1 = self.up1(s4, s3)
        # u2 = self.up2(u1, s2)
        # u3 = self.up3(u2, s1)
        # u4 = self.up4(u3,x)
        outputs = self.out(u4)
        return outputs

class UNet_MobileNetWithGuidedFilter(nn.Module):
    def __init__(self, n_channels, num_classes, block_type, upsample_type, alpha=1.0, use_dropout=False):
        super(UNet_MobileNetWithGuidedFilter, self).__init__()
        self.unet = UNet_MobileNet(n_channels=n_channels, num_classes=num_classes,
                                   block_type=block_type, upsample_type=upsample_type,
                                   alpha=alpha, use_dropout=use_dropout)
        self.guided_filter = ConvGuidedFilter(radius=1)

    def forward(self, x_lr, x_hr):
        y_lr = self.unet(x_lr)
        q = self.guided_filter(x_lr=x_lr, y_lr=y_lr, x_hr=x_hr)
        return y_lr, q