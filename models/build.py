from .resnet import ResNet


def build_model(config):
    "Model builder."

    model_type = config.MODEL.TYPE

    if model_type == 'resnet':
        model = ResNet(config.RESNET.NUM_BLOCKS)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 