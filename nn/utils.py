# -*- coding: utf-8 -*-

# @Time    : 19-5-27 上午11:16
# @Author  : zj


def conv_fc2output(inputs, batch_size, out_height, out_width):
    output = inputs.copy()
    # [N*H*W, C]
    local_connect_size, depth = output.shape[:2]
    # [N*H*W, C] -> [N, H, W, C]
    output = output.reshape(batch_size, out_height, out_width, depth)
    # [N, H, W, C] -> [N, C, H, W]
    return output.transpose((0, 3, 1, 2))


def conv_output2fc(inputs):
    output = inputs.copy()
    # [N, C, H, W]
    num, depth, height, width = output.shape[:4]

    # [N,C,H,W] —> [N,C,H*W]
    output = output.reshape(num, depth, -1)
    # [N,C,H*W] -> [N,H*W,C]
    output = output.transpose(0, 2, 1)
    # [N,H*W,C] -> [N*H*W,C]
    return output.reshape(-1, depth)


def pool_fc2output(inputs, batch_size, out_height, out_width):
    output = inputs.copy()
    # [N*C*H*W] -> [N, C, H, W]
    return output.reshape(batch_size, -1, out_height, out_width)


def pool_output2fc(inputs):
    return inputs.copy().reshape(-1)
