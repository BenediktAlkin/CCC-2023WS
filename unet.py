from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, conv_ctor):
        super().__init__()
        self.conv1 = conv_ctor(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.conv2 = conv_ctor(dim_out, dim_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.residual = conv_ctor(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


class ResnetBlock1d(ResnetBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv1d)


class ResnetBlock2d(ResnetBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv2d)


class ResnetBlock3d(ResnetBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv3d)


class UpsampleConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, scale_factor=2, mode="nearest", conv_ctor=nn.Conv2d):
        super().__init__()
        assert kernel_size % 2 == 1
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = conv_ctor(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UpsampleConv1d(UpsampleConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv1d)


class UpsampleConv2d(UpsampleConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv2d)


class UpsampleConv3d(UpsampleConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_ctor=nn.Conv3d)


class Unet(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        # create ctors
        conv_ctor = nn.Conv2d
        block_ctor = ResnetBlock2d
        upsample_conv_ctor = UpsampleConv2d
        # stem
        self.stem = nn.Conv2d(5, dim, kernel_size=7, padding=3)

        # create properties of hourglass architecture
        in_dims = []
        out_dims = []
        strides = []
        for i in range(depth):
            # first block keeps dimension, later blocks double dimension
            if i == 0:
                in_dims.append(dim)
                out_dims.append(dim)
            else:
                in_dim = dim * 2 ** (i - 1)
                in_dims.append(in_dim)
                out_dims.append(in_dim * 2)
            # downsample (last block doesnt downsample)
            if i < depth - 1:
                strides.append(2)
            else:
                strides.append(1)

        # down path
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            dim_in = in_dims[i]
            dim_out = out_dims[i]
            stride = strides[i]
            if stride == 1:
                downsample_ctor = partial(conv_ctor, kernel_size=3, padding=1)
            else:
                assert stride == 2
                downsample_ctor = partial(conv_ctor, kernel_size=2, stride=2)
            block = nn.ModuleList([
                block_ctor(dim_in=dim_in, dim_out=dim_in),
                block_ctor(dim_in=dim_in, dim_out=dim_in),
                downsample_ctor(dim_in, dim_out),
            ])
            self.down_blocks.append(block)

        # middle block
        mid_dim = out_dims[-1]
        self.mid_block1 = block_ctor(dim_in=mid_dim, dim_out=mid_dim)
        self.mid_block2 = block_ctor(dim_in=mid_dim, dim_out=mid_dim)

        # up blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            dim_in = in_dims[i]
            dim_out = out_dims[i]
            stride = strides[depth - 1 - i]
            if stride == 1:
                upsample_ctor = partial(conv_ctor, kernel_size=3, padding=1)
            else:
                assert stride == 2
                upsample_ctor = upsample_conv_ctor
            block = nn.ModuleList([
                block_ctor(dim_in=dim_in + dim_out, dim_out=dim_out),
                block_ctor(dim_in=dim_in + dim_out, dim_out=dim_out),
                upsample_ctor(dim_out, dim_in),
            ])
            self.up_blocks.append(block)

        # final block
        self.final_res_block = block_ctor(dim_in=dim * 2, dim_out=dim)
        self.final_conv = conv_ctor(dim, 1, kernel_size=1)

    def forward(self, x):
        stack = []

        # stem
        x = self.stem(x)
        stack.append(x)

        # down blocks
        for block1, block2, downsample in self.down_blocks:
            x = block1(x)
            stack.append(x)
            x = block2(x)
            stack.append(x)
            x = downsample(x)

        # mid blocks
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        # up blocks
        for block1, block2, upsample in self.up_blocks:
            x = torch.cat((x, stack.pop()), dim=1)
            x = block1(x)
            x = torch.cat((x, stack.pop()), dim=1)
            x = block2(x)
            x = upsample(x)

        # final
        x = torch.cat((x, stack.pop()), dim=1)
        x = self.final_res_block(x)
        x = self.final_conv(x)
        return x
