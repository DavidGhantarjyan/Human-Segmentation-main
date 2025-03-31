import torch
import torch.nn as nn
from other.models.mobile import CBG, MobileNetv, MobileNetV1Block, MobileNetV2Block



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

# s4: torch.Size([3, 1024, 40, 23]) -> torch.Size([3, 512, 80, 45]) ->
# torch.Size([3, 512, 80, 45]) + torch.Size([3, 512, 80, 45]) = torch.Size([3, 1024, 80, 45]) -> torch.Size([3, 512, 80, 45])
# -> torch.Size([3, 512, 80, 45])
class Upsample_Block(nn.Module):
    def __init__(self, in_nc, out_nc, block_type=None, kernel_size=3, stride=1, skip_connect=True, upsample_type='transpose', **kwargs):
        super(Upsample_Block, self).__init__()
        self.skip_connect = skip_connect  # Set skip_connect flag
        self.upsample_type = upsample_type
        #  torch.Size([3, 1024, 40, 23]) -> torch.Size([3, 512, 81, 47])
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
        # s4: torch.Size([3, 1024, 40, 23]) -> torch.Size([3, 512, 80, 45]) ->
        # torch.Size([3, 512, 80, 45]) + torch.Size([3, 512, 80, 45]) = torch.Size([3, 1024, 80, 45]) -> torch.Size([3, 512, 80, 45])
        # -> torch.Size([3, 512, 80, 45])

        #  torch.Size([3, 1024, 40, 23]) -> torch.Size([3, 512, 81, 47])
        x = self.up(inputs)  # Upsample the input tensor
        _, _, h_skip, w_skip = skip.size()
        # torch.Size([3, 512, 81, 47]) -> torch.Size([3, 512, 80, 45])
        x = self.center_crop(x, h_skip, w_skip)
        if self.skip_connect:
            #  torch.Size([3, 512, 80, 45]) + torch.Size([3, 512, 80, 45]) = torch.Size([3, 1024, 80, 45])
            x = torch.cat([x, skip], dim=1)  # Concatenate the skip connection
        # torch.Size([3, 1024, 80, 45]) -> torch.Size([3, 512, 80, 45]) or torch.Size([3, 512, 80, 45]) -> torch.Size([3, 512, 80, 45])
        x = self.conv_block(x)
        x = self.final_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(UNet, self).__init__()

        # Downsampling (encoder) blocks
        self.down1 = Downsample_Block(n_channels, 64)
        self.down2 = Downsample_Block(64, 128)
        self.down3 = Downsample_Block(128, 256)
        self.down4 = Downsample_Block(256, 512)

        # Bottleneck block
        self.b = ConvBlock(512, 1024)

        # Upsampling (decoder) blocks
        self.up1 = Upsample_Block(1024, 512)  # Here skip_connect=False
        self.up2 = Upsample_Block(512, 256)
        self.up3 = Upsample_Block(256, 128)
        self.up4 = Upsample_Block(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs):
        s1, p1 = self.down1(inputs)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)
        # print('s1:',s1.shape)
        # print('s2:',s2.shape)
        # print('s3:',s3.shape)
        # print('s4:',s4.shape)

        b = self.b(p4)
        # print('b:',b.shape)
        # For the first upsampling, skip_connect=False, no skip connections
        u1 = self.up1(b, s4)
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)
        # print('u1:',u1.shape)
        # print('u2:',u2.shape)
        # print('u3:',u3.shape)
        # print('u4:',u4.shape)
        outputs = self.out(u4)
        return outputs


class mobilenet(nn.Module):
    def __init__(self, n_channels, block_type):
        super(mobilenet, self).__init__()
        self.model = MobileNetv(n_channels, block_type)

    def forward(self, x):
        out1 = self.model.layer1(x)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)

        return out1, out2, out3, out4


class UNet_MobileNet(nn.Module):
    def __init__(self, n_channels, num_classes, block_type,upsample_type):
        super(UNet_MobileNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        # Backbone MobileNetv1/2
        self.backbone = mobilenet(n_channels, block_type)

        self.up1 = Upsample_Block(1024, 512, block_type=block_type,upsample_type=upsample_type)
        self.up2 = Upsample_Block(512, 256, block_type=block_type,upsample_type=upsample_type)
        self.up3 = Upsample_Block(256, 128, block_type=block_type,upsample_type=upsample_type)
        self.up4 = Upsample_Block(128, 64, block_type=block_type,skip_connect=False,upsample_type=upsample_type)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features using MobileNetv1
        s1, s2, s3, s4 = self.backbone(x)
        # print('s1:',s1.shape)
        # print('s2:',s2.shape)
        # print('s3:',s3.shape)
        # print('s4:',s4.shape)
        # Upsampling with skip connections
        u1 = self.up1(s4, s3)
        #
        # print('u1:',u1.shape)
        u2 = self.up2(u1, s2)
        # print('u2:',u2.shape)
        # torch.Size([3, 128, 320, 180])
        u3 = self.up3(u2, s1)
        # print('u3:',u3.shape)
        # print(x.shape)
        # u3: torch.Size([1, 128, 320, 180]) -> u4: torch.Size([1, 64, 640, 360])
        u4 = self.up4(u3,x)
        # print('u4:',u4.shape)
        # Final output layer
        # u4: torch.Size([1, 64, 640, 360]) -> torch.Size([1, 1, 640, 360])
        outputs = self.out(u4)
        return outputs
