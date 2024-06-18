import os
import torch
import numpy as np
import math

from scipy.ndimage import convolve
from torchvision.transforms.functional import normalize
from .color_util import bgr2ycbcr
from .niqe import compute_feature, niqe
from .matlab_functions import imresize

def calculate_psnr(img, img2, crop_border=8, img_range=1.0, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.') 
    #检查 img 和 img2 的形状是否相同，如果不同，会引发一个断言错误，指出形状不匹配
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range
    #这些行将输入图像的像素值从 [0, img_range] 缩放到 [0, 255] 的范围内，以便进行后续计算
    if crop_border != 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]
    #如果 crop_border 不等于0，则从 img 和 img2 的四个边界（上、下、左、右）裁剪 crop_border 个像素
    mse = torch.mean((img - img2)**2)
    #计算 img 和 img2 之间所有像素的平均平方差（MSE）
    if mse == 0:
        return float('inf')
    #如果 mse 等于0，即图像完全相同，函数会返回正无穷大（inf），因为此时PSNR无限大
    return (10. * torch.log10(255. * 255. / mse)).item()
    #计算并返回 img 和 img2 之间的PSNR（以分贝为单位）

def calculate_psnr_batch(img, img2, crop_border=8, img_range=1.0, **kwargs):
    #计算两个批次图像（img 和 img2）之间的PSNR（峰值信噪比）
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    #断言检查
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range
    #像素缩放
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    #处理裁剪边界
    mse = torch.mean((img - img2)**2, dim=(1,2,3))  # 均方差
    valid_mask = (mse != 0.)  # 布尔掩码valid_mask，标识值不为零的MSE
    mse = mse[valid_mask]
    return (10. * torch.log10(255. * 255. / mse)).mean(), valid_mask.sum()  # batch-wise mean

def calculate_lpips_batch(img, img2, net_lpips, crop_border=8, img_range=1.0, **kwargs):
    #计算两个批次图像（img 和 img2）之间的LPIPS（perceptual similarity metric）#
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    #断言检查
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    #定义了RGB三个通道的均值和标准差，用于将图像归一化到 [-1, 1] 的范围内
    img = normalize(img, mean, std)
    img2 = normalize(img2, mean, std)
    #使用 normalize 函数将输入图像 img 和 img2 根据给定的均值和标准差进行归一化，将像素值缩放到 [-1, 1] 的范围内
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    #处理裁剪边界
    lpips = net_lpips(img, img2).squeeze(1,2,3) 
    # 使用 net_lpips 神经网络模型计算 img 和 img2 之间的LPIPS（感知相似度度量），并通过 squeeze(1,2,3) 函数将多余的维度压缩掉，得到批次上每对图像的LPIPS值
    valid_mask = (lpips != 0.)  # 布尔掩码
    lpips = lpips[valid_mask]
    return lpips.mean(), valid_mask.sum()   # 批次均值和有效数量


