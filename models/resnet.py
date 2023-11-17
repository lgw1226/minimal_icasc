import torch
import torch.nn as nn
import collections


class BasicBlock(nn.Module):

    expansion = 1
    
    def __init__(self, name, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            collections.OrderedDict([
                (name + 'Conv2d', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                (name + 'BatchNorm2d', nn.BatchNorm2d(out_channels)),
                (name + 'ReLU', nn.ReLU(inplace=True)),
                (name + 'Conv2d2', nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)),
                (name + 'BatchNorm2d2', nn.BatchNorm2d(out_channels * self.expansion))
            ])
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                collections.OrderedDict([
                    (name + 'ShortCut' + 'Conv2d', nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)),
                    (name + 'ShortCut' + 'BatchNorm2d', nn.BatchNorm2d(out_channels * BasicBlock.expansion))
                ])
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))
    

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, parallel_last_layers=False):
        super().__init__()

        self.in_channels = 64
        self.num_classes = num_classes
        self.parallel_last_layers = parallel_last_layers
        
        self.layer1 = nn.Sequential(
            collections.OrderedDict([
                ('Layer1' + 'Conv2d', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
                ('Layer1' + 'BatchNorm2d', nn.BatchNorm2d(64)),
                ('Layer1' + 'ReLU', nn.ReLU(inplace=True))
            ])
        )
        self.layer2 = self._make_layer('Layer2', block, 64, num_blocks[0], 1)
        self.layer3 = self._make_layer('Layer3', block, 128, num_blocks[1], 2)
        self.layer4 = self._make_layer('Layer4', block, 256, num_blocks[2], 2)
        self.layer5 = self._make_layer('Layer5', block, 512, num_blocks[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.parallel_last_layers:
            self.last_layers = nn.ModuleDict()
            for idx in range(num_classes):
                self.last_layers.update({'Layer5' + 'Class' + str(idx + 1): self._make_layer('Layer5' + 'Class' + str(idx + 1), block, 1, num_blocks[3], 2, last=True)})
            self.fc = nn.Linear(num_classes * block.expansion, num_classes)
            
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.grads = None

    def _make_layer(self, name, block, out_channels, num_blocks, stride, last=False):

        strides = [stride] + [1] * (num_blocks - 1)
        blocks = collections.OrderedDict()
        if last:
            self.in_channels = 256
        for idx, stride in enumerate(strides):
            blocks.update({name + 'Block' + str(idx + 1): block(name + 'Block' + str(idx + 1), self.in_channels, out_channels, stride)})
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(blocks)
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.parallel_last_layers:
            x = self.layer5(x)
            x = self.global_average_pool(x)
        else:
            x = torch.cat([layer(x) for layer in self.last_layers.values()], dim=1)
            x = self.global_average_pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet18(num_classes, parallel_last_layers):

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, parallel_last_layers)


if __name__ == '__main__':

    from icecream import ic

    model = resnet18(3, True)
    batch_size = 4
    width = 3
    height = 3
    num_channels = 3
    input = torch.rand((batch_size, width, height, num_channels))
    output = model(input)
    ic(output)
