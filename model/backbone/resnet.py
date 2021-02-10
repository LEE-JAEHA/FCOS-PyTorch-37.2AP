import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_dilated(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1)
    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck2(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,small=False):
        super(Bottleneck2, self).__init__()
        self.small = small
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        ### added dilation part
        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv2_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=3, bias=False, dilation=3)
        self.bn2_3 = nn.BatchNorm2d(planes)
        ########################### dilation result concat part
        self.conv_attention = nn.Conv2d(planes * 3, planes, kernel_size=1, stride=1, bias=True)
        self.bn_attention = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        if self.small == False:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            # out = self.conv2(out)
            # out = self.bn2(out)

            dilation_1 = self.conv2(out)
            dilation_1 = self.bn2(dilation_1)
            dilation_1 = self.relu(dilation_1)

            dilation_2 = self.conv2_2(out)
            dilation_2 = self.bn2_2(dilation_2)
            dilation_2 = self.relu(dilation_2)

            dilation_3 = self.conv2_3(out)
            dilation_3 = self.bn2_3(dilation_3)
            dilation_3 = self.relu(dilation_3)

            attention_map = torch.cat((dilation_1, dilation_2, dilation_3), dim=1)

            out = self.conv_attention(attention_map)
            # out = self.bn_attention(out)
            # print(attention_map.shape)
            # input("TIME")
            # squeeze_=self.squeeze(attention_map)
            # excitation = self.excitation(squeeze_)
            # out = dilation_1 + dilation_2 + dilation_3
            ####
            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

class Bottleneck_Small(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Small, self).__init__()

        # 1 branch => 1x1 convolution
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        #2 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=6, bias=False,dilation=6)
        self.bn2 = nn.BatchNorm2d(planes)
        #3 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=12, bias=False, dilation=12)
        self.bn2_2 = nn.BatchNorm2d(planes)
        #4 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        self.conv2_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=18, bias=False, dilation=18)
        self.bn2_3 = nn.BatchNorm2d(planes)

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(planes)


        # dilation result concat part
        self.conv_attention = nn.Conv2d(planes * 4, planes, kernel_size=1, stride=1, bias=True)
        self.bn_attention = nn.BatchNorm2d(planes)


        #########################
        #
        # self.squeeze =  nn.AvgPool2d(kernel_size=86) # output channel 만큼의 1x1 / 86 부분은 img input이 300 300 일때
        # self.excitation =nn.Sequential(
        #     nn.Linear(planes,4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4,planes),
        #     nn.Sigmoid()
        # )
        # H * W * C

        ###########################

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # import pdb;
        # pdb.set_trace()
        # input("STRIDE : {0}".format(self.stride))
        x_h = x.size()[2]
        x_w = x.size()[3]
        # print(x_h,x_w)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        dilation_1 = self.conv2(out)
        dilation_1 = self.bn2(dilation_1)
        dilation_1 = self.relu(dilation_1)

        dilation_2 = self.conv2_2(out)
        dilation_2 = self.bn2_2(dilation_2)
        dilation_2 = self.relu(dilation_2)

        dilation_3 = self.conv2_3(out)
        dilation_3 = self.bn2_3(dilation_3)
        dilation_3 = self.relu(dilation_3)

        img = self.avg_pool(x)
        img = self.conv_1x1_2(img)
        img = self.bn_conv_1x1_2(img)
        img = self.relu(img)
        if self.stride == 2:
            img = F.upsample(img, size=(x_h // 2, x_w // 2), mode="bilinear")
        else:
            img = F.upsample(img, size=(x_h, x_w), mode="bilinear")

        # print(dilation_1.shape)
        # print(dilation_2.shape)
        # print(dilation_3.shape)
        # print(img.shape)
        # input("TIME")


        attention_map = torch.cat([dilation_1, dilation_2, dilation_3,img], dim=1)
        out = self.conv_attention(attention_map)
        out = self.bn_attention(out)
        out = self.relu(out)

        # out = self.bn_attention(out)
        # print(attention_map.shape)
        # input("TIME")
        # squeeze_=self.squeeze(attention_map)
        # excitation = self.excitation(squeeze_)
        # out = dilation_1 + dilation_2 + dilation_3
        ####
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, if_include_top=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2_small = self._make_layer_small(Bottleneck_Small, 128, layers[1], stride=2) # added have to change self._make_layer_small in self.planes to before value
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if if_include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.if_include_top = if_include_top

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer_small(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        tmp  = self.inplanes
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self.inplanes = tmp
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # input("before layer small")
        out3_small = self.layer2_small(x)  # added
        # input("After layer small")
        out3 = self.layer2(x)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)



        if self.if_include_top:
            x = self.avgpool(out5)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            # return (out3, out4, out5)
            # return (out3, out4, out5)
            return (out3, out4, out5,out3_small) #added

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('./resnet50.pth'), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
