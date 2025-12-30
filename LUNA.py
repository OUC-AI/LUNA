import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class CFA(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, embedding_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels / 4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3,
                                   bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        out = self.relu(x_master + x_gpb)

        return out

class Adapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, skip=False, scale=1):
        super().__init__()
        self.skip = skip
        self.scale = scale
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.act = act_layer()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()  # Change to [B, H*W, C]

        out = self.fc2(self.act(self.fc1(x)))

        if self.skip:
            out = out + x

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, C, H, W)

        return self.scale * out

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class EMSA(nn.Module):
    def __init__(self, channels=2048):

        super(EMSA, self).__init__()
        self.aff = AFF(channels)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1,
                                                  bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.GELU()

    def forward(self, x):
        x1 = self.conv3x3(x)
        x1 = self.bn1(x1)
        x1_upsample = self.relu(self.bn_upsample_1(self.conv_upsample_1(x1)))

        x2 = self.conv5x5(x)
        x2 = self.bn2(x2)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2)))


        out = self.aff(x1_upsample,x2_upsample)
        return out

class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode="constant") -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:, 0 * self.g: 1 * self.g, :, :] = pad_x[
                                               :, 0 * self.g: 1 * self.g, cx - stride: cx - stride + h, cy: cy + w
                                               ]
        out[:, 1 * self.g: 2 * self.g, :, :] = pad_x[
                                               :, 1 * self.g: 2 * self.g, cx + stride: cx + stride + h, cy: cy + w
                                               ]
        out[:, 2 * self.g: 3 * self.g, :, :] = pad_x[
                                               :, 2 * self.g: 3 * self.g, cx: cx + h, cy - stride: cy - stride + w
                                               ]
        out[:, 3 * self.g: 4 * self.g, :, :] = pad_x[
                                               :, 3 * self.g: 4 * self.g, cx: cx + h, cy + stride: cy + stride + w
                                               ]

        out[:, 4 * self.g: 5 * self.g, :, :] = pad_x[
                                               :, 4 * self.g: 5 * self.g, cx + stride: cx + stride + h,
                                               cy + stride: cy + stride + w
                                               ]
        out[:, 5 * self.g: 6 * self.g, :, :] = pad_x[
                                               :, 5 * self.g: 6 * self.g, cx + stride: cx + stride + h,
                                               cy - stride: cy - stride + w
                                               ]
        out[:, 6 * self.g: 7 * self.g, :, :] = pad_x[
                                               :, 6 * self.g: 7 * self.g, cx - stride: cx - stride + h,
                                               cy + stride: cy + stride + w
                                               ]
        out[:, 7 * self.g: 8 * self.g, :, :] = pad_x[
                                               :, 7 * self.g: 8 * self.g, cx - stride: cx - stride + h,
                                               cy - stride: cy - stride + w
                                               ]

        return out

class ResidualBlockShift(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.shift = Shift8(groups=num_feat // 8, stride=1)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)

class SFE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SFE, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class X(nn.Module):
    def __init__(self, input_channels=64, height=224, width=224, res_scale=1, res_scale2=1):
        super(X, self).__init__()
        self.res_scale = res_scale
        self.res_scale2 = res_scale2

        self.layernorm1 = nn.LayerNorm([input_channels, height, width])
        self.layernorm2 = nn.LayerNorm([input_channels, height, width])
        self.relu = nn.GELU()

        self.color_attention = CFA(embedding_dim=input_channels)

        self.afpa = EMSA(channels=input_channels)

        self.adapter = Adapter(embedding_dim=input_channels, skip=True)

        self.sc_resblock = ResidualBlockShift(num_feat=input_channels)

    def forward(self, x):
        identity = x

        x = self.afpa(self.layernorm1(x))
        x = self.relu(x)


        attention_weights = self.color_attention(x)
        x = x * attention_weights

        x = x + identity * self.res_scale

        out = self.adapter(x)

        x = self.sc_resblock(self.layernorm2(x))
        x = self.relu(x)

        x = x + out * self.res_scale2
        return x

class IQANetwork(nn.Module):
    def __init__(self, input_dim=3, height=224, width=224, feature_dim=64, output_dim=1, num_blocks=1):
        super(IQANetwork, self).__init__()
        # Shallow Feature Extractor
        self.sfe = SFE(input_dim=input_dim, output_dim=feature_dim)

        self.body = make_layer(X, num_blocks, input_channels=feature_dim, height=height, width=width)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(feature_dim, output_dim)
        init_weights(self)
    def forward(self, x):

        x = self.sfe(x)
        identity = x

        x = self.body(x)
        x = x + identity

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

def main():

    input_dim = 3
    height, width = 256, 256
    feature_dim = 64
    output_dim = 1
    batch_size = 4
    num_blocks = 2

    model = IQANetwork(input_dim=input_dim, height=height, width=width,
                       feature_dim=feature_dim, output_dim=output_dim, num_blocks=num_blocks)
    print(model)

    input_tensor = torch.randn(batch_size, input_dim, height, width)

    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (
    batch_size, output_dim), f"Expected output shape: ({batch_size}, {output_dim}), but got {output.shape}"

if __name__ == "__main__":
    main()