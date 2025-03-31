from other.models.unet_models import UNet, UNet_MobileNet, MobileNetV1Block, MobileNetV2Block

mobilenet_block_names = {
    "MobileNetV1Block": MobileNetV1Block,
    "MobileNetV2Block": MobileNetV2Block,
    "Other": None
}

unet = lambda: UNet(n_channels=3, num_classes=1)
unet_mobilenetv1 = lambda: UNet_MobileNet(n_channels=3, num_classes=1,
                                          block_type=mobilenet_block_names['MobileNetV1Block'],
                                          upsample_type="transpose")
unet_mobilenetv2 = lambda: UNet_MobileNet(n_channels=3, num_classes=1,
                                          block_type=mobilenet_block_names['MobileNetV2Block'],
                                          upsample_type="transpose")

MODELS = {
    "UNet": unet,
    "UNet_MobileNet": unet_mobilenetv1,
    "UNet_MobileNetv2": unet_mobilenetv2,
    "Other": None
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
