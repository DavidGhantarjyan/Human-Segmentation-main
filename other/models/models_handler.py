from other.models.unet_models import UNet_MobileNet, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block
import torch
from fvcore.nn import FlopCountAnalysis
# from other.models.unet_models import UNet, UNet_MobileNet, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block

mobilenet_block_names = {
    "MobileNetV1Block": MobileNetV1Block,
    "MobileNetV2Block": MobileNetV2Block,
    "MobileNetV3Block": MobileNetV3Block
}


# unet = lambda: UNet(n_channels=3, num_classes=1)
unet_mobilenetv1 = lambda: UNet_MobileNet(n_channels=3, num_classes=1,
                                          block_type=mobilenet_block_names['MobileNetV1Block'],
                                          upsample_type="transpose", use_dropout=False)
unet_mobilenetv2 = lambda: UNet_MobileNet(n_channels=3, num_classes=1,
                                          block_type=mobilenet_block_names['MobileNetV2Block'],
                                          upsample_type="transpose", alpha=0.45, use_dropout=False)
unet_mobilenetv3 = lambda: UNet_MobileNet(n_channels=3, num_classes=1,block_type=mobilenet_block_names['MobileNetV3Block'],upsample_type="transpose", alpha=0.3
                                          ,use_dropout=False)


MODELS = {
    # # 40797761
    # "UNet": unet,
    # 11860585
    "UNet_MobileNet": unet_mobilenetv1,
    # 26013993
    "UNet_MobileNetv2": unet_mobilenetv2,
    # 35596825
    "UNet_MobileNetv3": unet_mobilenetv3,
}
NAMES = [*MODELS.keys()]
MODELS_COUNT = len(NAMES)
if MODELS_COUNT == 0:
    raise Exception("You must specify at least one model")


# ***********************************************************************************
# count_parameters,estimate_vram_usage

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram_usage(model, include_gradients=True):
    # Calculate total number of parameters
    total_params = count_parameters(model)
    param_memory = total_params * 4  # 4 bytes per float32 parameter

    # If gradients are stored (common in training), double the memory usage
    if include_gradients:
        param_memory *= 2

    # Convert to (GB)
    return param_memory / (1024 ** 3)

def get_flops(model, input_shape=(1, 3, 320, 180), device="gpu"):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    return flops.total() / 1e9  # GFLOPs