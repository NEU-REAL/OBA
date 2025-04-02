from .models import *


def get_network(network, depth, dataset):
    if network == 'vgg':
        return VGG(depth=depth, dataset=dataset)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset)
    elif network == 'presnet':
        return presnet(depth=depth, dataset=dataset)
    else:
        raise NotImplementedError


def get_bottleneck_builder(network):
    if 'vgg' in network:
        return BottleneckVGG
    elif 'presnet' in network:
        return BottleneckPResNet
    elif 'resnet' in network:
        return BottleneckResNet
    else:
        raise NotImplementedError


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)
