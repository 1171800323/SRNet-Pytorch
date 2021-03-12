import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class Residual(nn.Module):
    def __init__(self, in_dim):
        super(Residual, self).__init__()
        temp_channels = in_dim // 4
        self.conv = nn.Sequential(nn.Conv2d(in_dim, temp_channels, kernel_size=1),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(temp_channels, temp_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(temp_channels, in_dim, kernel_size=1))
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        out = self.conv(x) + x
        return F.leaky_relu(self.bn(out))


class ResNet(nn.Module):
    def __init__(self, in_dim):
        super(ResNet, self).__init__()
        self.layer = nn.Sequential(Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim))

    def forward(self, x):
        return self.layer(x)


def conv_bn_relu(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
    return blk


# def dilated_conv(in_dim, padding=2, dilation=2):
#     blk = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=padding, dilation=dilation),
#                         nn.BatchNorm2d(in_dim),
#                         nn.LeakyReLU())
#     return blk


class EncoderNet(nn.Module):
    def __init__(self, in_dim):
        super(EncoderNet, self).__init__()
        layer1, layer2, layer3, layer4 = [], [], [], []

        layer1.append(conv_bn_relu(in_dim, 32))
        layer1.append(conv_bn_relu(32, 32))

        layer2.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        layer2.append(nn.LeakyReLU())
        layer2.append(conv_bn_relu(64, 64))
        layer2.append(conv_bn_relu(64, 64))

        layer3.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        layer3.append(nn.LeakyReLU())
        layer3.append(conv_bn_relu(128, 128))
        layer3.append(conv_bn_relu(128, 128))

        layer4.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        layer4.append(nn.LeakyReLU())
        layer4.append(conv_bn_relu(256, 256))
        layer4.append(conv_bn_relu(256, 256))

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

    def forward(self, x, get_feature_map=False):
        out = self.l1(x)
        out = self.l2(out)
        f1 = out
        out = self.l3(out)
        f2 = out
        out = self.l4(out)
        if get_feature_map:
            return out, [f2, f1]
        else:
            return out


class DecoderNet(nn.Module):
    def __init__(self, in_dim, feature_map_channels=None):
        super(DecoderNet, self).__init__()

        f1, f2, f3 = 0, 0, 0
        if feature_map_channels:
            f1, f2, f3 = feature_map_channels

        cat_channels = in_dim + f1
        self.conv1 = nn.Sequential(conv_bn_relu(cat_channels, 256),
                                   conv_bn_relu(256, 256))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU())

        cat_channels = 128 + f2
        self.conv2 = nn.Sequential(conv_bn_relu(cat_channels, 128),
                                   conv_bn_relu(128, 128))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU())

        cat_channels = 64 + f3
        self.conv3 = nn.Sequential(conv_bn_relu(cat_channels, 64),
                                   conv_bn_relu(64, 64))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU())
        self.conv4 = nn.Sequential(conv_bn_relu(32, 32),
                                   conv_bn_relu(32, 32))

    def forward(self, x, fuse=None, get_feature_map=False):
        if fuse and fuse[0] is not None:
            x = torch.cat([x, fuse[0]], dim=1)
        out = self.conv1(x)
        f1 = out

        out = self.deconv1(out)
        if fuse and fuse[1] is not None:
            out = torch.cat([out, fuse[1]], dim=1)
        out = self.conv2(out)
        f2 = out

        out = self.deconv2(out)
        if fuse and fuse[2] is not None:
            out = torch.cat([out, fuse[2]], dim=1)
        out = self.conv3(out)
        f3 = out

        out = self.deconv3(out)
        out = self.conv4(out)

        if get_feature_map:
            return out, [f1, f2, f3]
        else:
            return out


class TextConversionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(TextConversionNet, self).__init__()
        self.t_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(256))
        self.s_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(256))
        self.sk_decoder = DecoderNet(2 * 256)
        self.sk_out = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                    nn.Sigmoid())
        self.t_decoder = DecoderNet(2 * 256)
        self.t_conv = conv_bn_relu(64, 64)
        self.last = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x_t, x_s):
        out_t = self.t_encoder(x_t)
        out_s = self.s_encoder(x_s)

        out = torch.cat([out_t, out_s], dim=1)

        out_sk = self.sk_decoder(out)
        out_t = self.t_decoder(out)

        out_t = torch.cat([out_sk, out_t], dim=1)

        out_t = self.t_conv(out_t)

        return self.sk_out(out_sk), self.last(out_t)


class BackgroundInpaintingNet(nn.Module):
    def __init__(self, in_dim=3):
        super(BackgroundInpaintingNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(256)
        # self.dilation_net = nn.Sequential(dilated_conv(256),
        #                                   dilated_conv(256, padding=4, dilation=4),
        #                                   dilated_conv(256, padding=8, dilation=8))
        self.decoder = DecoderNet(256, [0, 128, 64])
        self.last = nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x):
        out, f_encoder = self.encoder(x, get_feature_map=True)
        out = self.resnet(out)
        # out = self.dilation_net(out)
        out, fuse = self.decoder(out, [None] + f_encoder, get_feature_map=True)
        return self.last(out), fuse


class FusionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(FusionNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(256)
        self.decoder = DecoderNet(256, [256, 128, 64])
        self.last = nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x, fuse):
        out = self.encoder(x)
        out = self.resnet(out)
        out = self.decoder(out, fuse)
        return self.last(out)


class Generator(nn.Module):
    def __init__(self, in_dim=3):
        super(Generator, self).__init__()
        self.text_conversion_net = TextConversionNet(in_dim)
        self.background_inpainting_net = BackgroundInpaintingNet(in_dim)
        self.fusion_net = FusionNet(in_dim)

    def forward(self, inputs):
        i_t, i_s = inputs
        o_sk, o_t = self.text_conversion_net(i_t, i_s)
        o_b, fuse = self.background_inpainting_net(i_s)
        o_f = self.fusion_net(o_t, fuse)
        return o_sk, o_t, o_b, o_f


class Discriminator(nn.Module):
    def __init__(self, in_dim=6):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=1),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return self.conv5(out)


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(torchvision.models.vgg19(pretrained=True).features)
        # 将模型设置为evaluation模式，只影响dropout和batchNorm
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)

            if i in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
