import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.resnet import BasicBlock, ResNet


class SmallCNN(nn.Module):
    """Small CNN branch placeholder for 64x64 patch inputs."""

    def __init__(self, output_features, input_channels=3):
        super(SmallCNN, self).__init__()
        validate_input_channels(input_channels)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_features)
        )

    def forward(self, x):
        return self.net(x)


def validate_input_channels(input_channels):
    """Validate the number of image channels expected by a branch."""
    if int(input_channels) < 1:
        raise ValueError(f'input_channels must be at least 1. Got: {input_channels}')


def adapt_conv_weight(old_weight, input_channels):
    """Adapt pretrained first-layer weights to the requested input channel count."""
    validate_input_channels(input_channels)

    old_channels = old_weight.shape[1]

    if input_channels == old_channels:
        return old_weight.clone()

    if input_channels == 1:
        return old_weight.mean(dim=1, keepdim=True)

    new_weight = old_weight.new_empty(old_weight.shape[0], input_channels, old_weight.shape[2], old_weight.shape[3])

    channels_to_copy = min(old_channels, input_channels)
    new_weight[:, :channels_to_copy] = old_weight[:, :channels_to_copy]

    if input_channels > old_channels:
        channel_mean = old_weight.mean(dim=1, keepdim=True)
        new_weight[:, old_channels:] = channel_mean.repeat(1, input_channels - old_channels, 1, 1)

    return new_weight


def replace_first_conv(network, input_channels, kernel_size=None, stride=None, padding=None, pretrained_stem=False, old_weight=None):
    """Replace a ResNet first convolution while preserving or initialising its weights."""
    validate_input_channels(input_channels)

    old_conv = network.conv1
    kernel_size = kernel_size if kernel_size is not None else old_conv.kernel_size
    stride = stride if stride is not None else old_conv.stride
    padding = padding if padding is not None else old_conv.padding
    old_weight = old_weight if old_weight is not None else old_conv.weight.detach().clone()

    network.conv1 = nn.Conv2d(input_channels, old_conv.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    if pretrained_stem:
        with torch.no_grad():
            network.conv1.weight.copy_(adapt_conv_weight(old_weight, input_channels))
    else:
        nn.init.kaiming_normal_(network.conv1.weight, mode='fan_out', nonlinearity='relu')

    return network


def update_resnet_for_small_inputs(network, input_channels=3, pretrained_stem=False):
    """Replace the standard ImageNet stem with a small-input stem."""
    old_weight = network.conv1.weight.detach().clone()
    small_stem_weight = old_weight[:, :, 2:5, 2:5]
    network = replace_first_conv(network=network, input_channels=input_channels, kernel_size=3, stride=1, padding=1,
                                 pretrained_stem=pretrained_stem, old_weight=small_stem_weight)
    network.maxpool = nn.Identity()

    return network


def get_resnet_stage_modules(network):
    """Return ResNet stages in input-to-output order."""
    return [
        [network.conv1, network.bn1],
        [network.layer1],
        [network.layer2],
        [network.layer3],
        [network.layer4]
    ]


def set_frozen_resnet_stages_eval(network):
    """Keep frozen ResNet stages in eval mode so BatchNorm statistics do not update."""
    frozen_stages = getattr(network, '_frozen_stages', 0)

    if frozen_stages <= 0:
        return

    for stage in get_resnet_stage_modules(network)[:frozen_stages]:
        for module in stage:
            module.eval()


def freeze_resnet_stages(network, frozen_stages):
    """Freeze ResNet stages from the image input side towards the deeper blocks."""
    if frozen_stages < 0 or frozen_stages > 5:
        raise ValueError('frozen_stages must be between 0 and 5.')

    network._frozen_stages = int(frozen_stages)

    for stage in get_resnet_stage_modules(network)[:frozen_stages]:
        for module in stage:
            for parameter in module.parameters():
                parameter.requires_grad = False

    set_frozen_resnet_stages_eval(network)


def build_custom_resnet(layer_config, output_features, small_input_stem, input_channels):
    """Build an untrained BasicBlock ResNet branch."""
    network = ResNet(block=BasicBlock, layers=layer_config, num_classes=output_features)

    if small_input_stem:
        network = update_resnet_for_small_inputs(network, input_channels=input_channels, pretrained_stem=False)
    else:
        network = replace_first_conv(network, input_channels=input_channels, pretrained_stem=False)

    return network


def build_branch(network_name, output_features=128, frozen_stages=0, small_input_stem=True, input_channels=3):
    """Build one quadruplet branch and return a feature vector."""
    network_name = network_name.lower()
    validate_input_channels(input_channels)

    if network_name == 'resnet18_pretrained':
        network = resnet18(weights=ResNet18_Weights.DEFAULT)
        if small_input_stem:
            network = update_resnet_for_small_inputs(network, input_channels=input_channels, pretrained_stem=True)
        else:
            network = replace_first_conv(network, input_channels=input_channels, pretrained_stem=True)
        network.fc = nn.Linear(network.fc.in_features, output_features)
        freeze_resnet_stages(network, frozen_stages=frozen_stages)
        return network

    if network_name == 'resnet18_untrained':
        if frozen_stages != 0:
            raise ValueError('Do not freeze stages in an untrained ResNet unless you intentionally want fixed random filters.')
        network = resnet18(weights=None)
        if small_input_stem:
            network = update_resnet_for_small_inputs(network, input_channels=input_channels, pretrained_stem=False)
        else:
            network = replace_first_conv(network, input_channels=input_channels, pretrained_stem=False)
        network.fc = nn.Linear(network.fc.in_features, output_features)
        return network

    if network_name == 'resnet34_pretrained':
        network = resnet34(weights=ResNet34_Weights.DEFAULT)
        if small_input_stem:
            network = update_resnet_for_small_inputs(network, input_channels=input_channels, pretrained_stem=True)
        else:
            network = replace_first_conv(network, input_channels=input_channels, pretrained_stem=True)
        network.fc = nn.Linear(network.fc.in_features, output_features)
        freeze_resnet_stages(network, frozen_stages=frozen_stages)
        return network

    if network_name == 'resnet34_untrained':
        if frozen_stages != 0:
            raise ValueError('Do not freeze stages in an untrained ResNet unless you intentionally want fixed random filters.')
        network = resnet34(weights=None)
        if small_input_stem:
            network = update_resnet_for_small_inputs(network, input_channels=input_channels, pretrained_stem=False)
        else:
            network = replace_first_conv(network, input_channels=input_channels, pretrained_stem=False)
        network.fc = nn.Linear(network.fc.in_features, output_features)
        return network

    if network_name == 'resnet10_untrained':
        if frozen_stages != 0:
            raise ValueError('Do not freeze stages in an untrained ResNet unless you intentionally want fixed random filters.')
        return build_custom_resnet(layer_config=[1, 1, 1, 1], output_features=output_features, small_input_stem=small_input_stem, input_channels=input_channels)

    if network_name == 'resnet14_untrained':
        if frozen_stages != 0:
            raise ValueError('Do not freeze stages in an untrained ResNet unless you intentionally want fixed random filters.')
        return build_custom_resnet(layer_config=[1, 1, 2, 2], output_features=output_features, small_input_stem=small_input_stem, input_channels=input_channels)

    if network_name == 'small_cnn':
        if frozen_stages != 0:
            raise ValueError('frozen_stages is only valid for pretrained ResNet branches.')
        return SmallCNN(output_features=output_features, input_channels=input_channels)

    raise ValueError(f'Unknown network_name: {network_name}')


class Quadruplet(nn.Module):

    def __init__(self, num_of_pts, tasks_classes, network_name='resnet18_pretrained', branch_features=128, frozen_stages=0, small_input_stem=True, input_channels=3):
        super(Quadruplet, self).__init__()
        validate_input_channels(input_channels)

        if num_of_pts < 1 or num_of_pts > 30:
            raise ValueError('num_of_pts must be between 1 and 30.')

        self.input_channels = int(input_channels)
        self.net1 = build_branch(network_name=network_name, output_features=branch_features, frozen_stages=frozen_stages, small_input_stem=small_input_stem, input_channels=self.input_channels)
        self.net2 = build_branch(network_name=network_name, output_features=branch_features, frozen_stages=frozen_stages, small_input_stem=small_input_stem, input_channels=self.input_channels)
        self.net3 = build_branch(network_name=network_name, output_features=branch_features, frozen_stages=frozen_stages, small_input_stem=small_input_stem, input_channels=self.input_channels)
        self.net4 = build_branch(network_name=network_name, output_features=branch_features, frozen_stages=frozen_stages, small_input_stem=small_input_stem, input_channels=self.input_channels)

        self.num_of_pts = num_of_pts
        self.num_of_tasks = len(tasks_classes)
        self.num_of_classes = [len(task_classes) for _ in range(self.num_of_pts) for task_classes in tasks_classes]

        combined_features = branch_features * 4
        self.output_heads = nn.ModuleList([nn.Linear(combined_features, class_count) for class_count in self.num_of_classes])

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f'Expected input shape [batch, 4, channels, height, width], got {tuple(x.shape)}.')

        if x.shape[1] != 4:
            raise ValueError(f'Quadruplet expects exactly 4 sub-patches per sample, got {x.shape[1]}.')

        if x.shape[2] != self.input_channels:
            raise ValueError(f'Quadruplet was built for {self.input_channels} input channels, got {x.shape[2]}.')

        net1_out = self.net1(x[:, 0])
        net2_out = self.net2(x[:, 1])
        net3_out = self.net3(x[:, 2])
        net4_out = self.net4(x[:, 3])

        net_out = torch.cat((net1_out, net2_out, net3_out, net4_out), 1)
        outputs = tuple(output_head(net_out) for output_head in self.output_heads)

        return outputs

    def train(self, mode=True):
        """Set training mode while keeping frozen ResNet stages fixed."""
        super().train(mode)

        if mode:
            self.set_frozen_stages_eval()

        return self

    def set_frozen_stages_eval(self):
        """Set frozen stages in each branch to eval mode."""
        for branch in (self.net1, self.net2, self.net3, self.net4):
            set_frozen_resnet_stages_eval(branch)
