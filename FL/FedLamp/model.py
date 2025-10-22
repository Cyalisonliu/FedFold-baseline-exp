import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        scaler = Scaler(rate)

        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  scaler,
                  norm,
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            norm = nn.BatchNorm2d(hidden_size[i + 1], momentum=None, track_running_stats=track)
            scaler = Scaler(rate)
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           scaler,
                           norm,
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out


def CNN(n_class, model_rate=1, track=False):
    classes_size = n_class
    # model_rate=0.0625
    conv_hidden_size = [64, 128, 256, 512]
    data_shape = [3, 32, 32]
    hidden_size = [int(np.ceil(model_rate * x)) for x in conv_hidden_size]
    scaler_rate = int(1 / model_rate)
    model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track)
    model.apply(init_param)
    return model


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, rate, track):
        super(Block, self).__init__()
        n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        out = F.relu(self.scaler(out))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        # out = self.conv2(F.relu(self.scaler(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, rate, track):
        super(Bottleneck, self).__init__()
        n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        n3 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = n3
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = F.relu(self.n2(self.scaler(out)))
        out = self.conv2(out)
        out = F.relu(self.n3(self.scaler(out)))
        out = self.conv3(out)
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, rate, track):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, rate=rate, track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, rate=rate, track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, rate=rate, track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, rate=rate, track=track)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion, momentum=None, track_running_stats=track)
        self.scaler = Scaler(rate)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rate, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x = input
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(n_class, model_rate=1, track=False):
    conv_hidden_size = [64, 128, 256, 512]
    data_shape = [3, 32, 32]
    classes_size = n_class
    hidden_size = [int(np.ceil(model_rate * x)) for x in conv_hidden_size]
    scaler_rate = model_rate
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, scaler_rate, track)
    model.apply(init_param)
    return model


def resnet152(model_rate=1, track=False):
    conv_hidden_size = [64, 128, 256, 512]
    data_shape = [3, 32, 32]
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in conv_hidden_size]
    scaler_rate = model_rate
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], classes_size, scaler_rate, track)
    model.apply(init_param)
    return model


# class BasicBlock(nn.Module):
#     expansion = 1  # For ResNet18/34, expansion = 1 (no change in channels in shortcut)
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # This handles the projection shortcut for downsampling and channel mismatch
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)  # Add the shortcut to the output
#         out = F.relu(out)
#         return out
#
#
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):  # Set num_classes to 10 for CIFAR10 or 100 for CIFAR100
#         super(ResNet18, self).__init__()
#         self.in_channels = 64  # Initial number of input channels
#
#         # Modify the first conv layer to match smaller CIFAR images (32x32)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Smaller kernel size
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.Identity()  # Remove maxpooling for smaller images
#
#         # Create 4 layers with BasicBlock, matching ResNet18 architecture
#         self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # No downsampling in the first layer
#         self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # Downsampling in the second layer
#         self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # Downsampling in the third layer
#         self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # Downsampling in the fourth layer
#
#         # Adaptive average pooling and fully connected layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (1, 1)
#         self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)  # Final fully connected layer
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         """Helper function to create a layer of residual blocks"""
#         layers = []
#         # First block might need to downsample the input (use stride)
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels  # Update the in_channels for next blocks
#
#         # Remaining blocks use stride=1
#         for _ in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # Initial convolution + max pooling
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#
#         # Pass through 4 layers of residual blocks
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         # Pooling and fully connected layer
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x


# class BottleneckBlock(nn.Module):
#     expansion = 4  # Bottleneck layers expand channels by 4x
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BottleneckBlock, self).__init__()
#         # First convolution: 1x1 conv reduces dimensionality
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#         # Second convolution: 3x3 conv maintains dimensionality
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # Third convolution: 1x1 conv restores dimensionality
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             # This handles the shortcut connection if dimensions change (when stride != 1 or in_channels != out_channels * 4)
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )
#
#     def forward(self, x):
#         # Standard forward pass through the Bottleneck block
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet152(nn.Module):
#     def __init__(self, num_classes=10):  # Set num_classes to 1000 for ImageNet or modify for CIFAR10/100
#         super(ResNet152, self).__init__()
#         self.in_channels = 64
#
#         # First conv layer: 7x7 conv followed by maxpool (for larger image datasets like ImageNet)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # Stages with residual blocks
#         self.layer1 = self._make_layer(BottleneckBlock, 64, 3, stride=1)   # 3 blocks
#         self.layer2 = self._make_layer(BottleneckBlock, 128, 8, stride=2)  # 8 blocks
#         self.layer3 = self._make_layer(BottleneckBlock, 256, 36, stride=2) # 36 blocks
#         self.layer4 = self._make_layer(BottleneckBlock, 512, 3, stride=2)  # 3 blocks
#
#         # Adaptive average pooling and final fully connected layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (1, 1)
#         self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)  # Fully connected layer
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         """Helper function to create a layer of residual blocks"""
#         layers = []
#         # First block might need to downsample the input (use stride)
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion  # Update the in_channels for the next blocks
#
#         # Remaining blocks use stride=1
#         for _ in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # Initial convolution + max pooling
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#
#         # Pass through 4 layers of residual blocks
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         # Pooling and fully connected layer
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x


class ResNet9(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = BasicBlock(64, 128, stride=2)
        self.layer2 = BasicBlock(128, 256, stride=2)
        self.layer3 = BasicBlock(256, 512, stride=2)

        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.model = models.vgg16(weights=None)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG9(nn.Module):
    def __init__(self, num_classes=47):  # Adjust num_classes based on your EMNIST split
        super(VGG9, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),  # Adjust this based on your input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = self.classifier(x)
        return x
