import math

import torch
# import numpy as np
import torch.nn as nn


# 使用时，将其更名为bbox_iou放入到metrics.py里
# alpha=1时，认为与正常的IoU无异，alpha!=1时，进化为alpha-IoU，一般认为alpha=3时，IoU有较好的表现
# Focal——EIoU的思想可以用在其他的IoU变体上，但是Focal+SIoU存在问题，且目前尚不稳定
# gamma为Focal——EIoU的参数，默认0.5，可自行修改
# 此外，仍需要修改其他文件中的部分语句
# utils/loss.py中ComputeLoss Class中的__call__函数中修改一下：
# eps用来防治零溢事件

# GIoU：计算两框的组合与未交错区域之比
# DIoU：GIoU基础上，计算两框中间的欧氏距离
# CIoU：DIoU基础上，涉及长宽比
# EIoU：在CIOU的惩罚项基础上将纵横比的影响因子拆开分别计算目标框和锚框的长和宽
# SIoU：在之前的基础上，增加对角度（向量）的考虑
# WIoU：动态调整的IoU，需要一下的两个额外参数👇，和额外的class
# scale: scale为True时，WIoU会乘以一个系数
# monotonous: 3个输入分别代表WIoU的3个版本，None: origin v1, True: monotonic FM v2, False: non-monotonic FM v3

def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False,
             SIoU=False, WIoU=False, Focal=False, scale=False, monotonous=None, alpha=1, gamma=0.5, eps=1e-7):
    # box1 = box1.T
    # 获取边界框的坐标
    # if xywh:  # transform from xywh to xyxy
    #         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    #         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    #         b1_x1, b1_x2  = x1 - w1_, x1 + w1_
    #         b1_y1, b1_y2  = y1 - h1_, y1 + h1_
    #         b2_x1, b2_x2  = x2 - w2_, x2 + w2_
    #         b2_y1, b2_y2  = y2 - h2_, y2 + h2_
    #     else:  # x1, y1, x2, y2 = box1
    #         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    #         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    #         w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    #         w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    #
    #     # Intersection area
    #     inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
    #             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    else:  # 将xy(中心坐标)wh(候选框的宽高)转化为xyxy(四条边的所在位置)形态
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    # 联合区域
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
    #         (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    # 交叠区域
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        wise_scale = WIoU_Scale(1 - (inter / union), monotonous=monotonous)
    # 求出常规的IoU
    iou = inter / union
    # 根据所选定的IoU-Loss来进行计算
    if GIoU or DIoU or EIoU or CIoU or SIoU:
        # 计算出最小闭包的宽和高
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if SIoU:  # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            # angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            # return iou - 0.5 * (distance_cost + shape_cost)
            if Focal:  # SIoU
                return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                    inter / (union + eps), gamma)
            else:
                return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)
        if CIoU or DIoU or EIoU or WIoU:
            # c2为能覆盖锚框和目标框的最小矩形的对角线距离。
            c2 = cw ** 2 + ch ** 2 + eps
            # 计算锚框与目标框之间的欧氏距离。
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            # DIoU:两锚框的中心距离
            if DIoU:
                # return iou - rho2 / c2
                if Focal:  # DIoU
                    return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - rho2 / c2
            elif WIoU:
                if scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - iou) * torch.exp(
                        (rho2 / c2)), iou  # WIoU v3 https://arxiv.org/abs/2301.10051
                return iou, torch.exp((rho2 / c2))  # WIoU v1
            # CIoU:两锚框的长宽之比
            elif CIoU:
                # v用来表示长宽比的相似性
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                # return iou - (rho2 / c2 + v * alpha_ciou)
                if Focal:  # CIoU
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), \
                        torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))
            elif EIoU:
                # 在CIOU的惩罚项基础上将纵横比的影响因子拆开分别计算目标框和锚框的长和宽
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                # return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
                if Focal:  # EIoU
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:  # GIoU https://a.rxiv.org/pdf/1902.09630.pdf
            # c_area是他们的最小闭包（矩形）
            c_area = cw * ch + eps
            # return iou - (c_area - union) / c_area
            if Focal:  # GIoU
                return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps), gamma)
            else:
                return iou - torch.pow((c_area - union) / c_area + eps, alpha)
    elif Focal:  # 未采用任何优化的，纯粹的IoU
        return iou, torch.pow(inter / (union + eps), gamma)
    else:
        return iou


# 在utils/loss.py中，找到ComputeLoss类的__call__()函数，把Regression loss中计算iou的代码，换成下面这句
# iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False, EIoU=True)
# iou(prediction, target)


#
#
# def bbox_iou(box1,
#              box2,
#              xywh=True,
#              GIoU=False,
#              DIoU=False,
#              CIoU=False,
#              SIoU=False,
#              EIoU=False,
#              WIoU=False,
#              Focal=False,
#              alpha=1,
#              gamma=0.5,
#              scale=False,
#              monotonous=False,
#              eps=1e-7):
#     """
#     计算bboxes iou
#     Args:
#         box1: predict bboxes
#         box2: target bboxes
#         xywh: 将bboxes转换为xyxy的形式
#         GIoU: 为True时计算GIoU LOSS (yolov5自带)
#         DIoU: 为True时计算DIoU LOSS (yolov5自带)
#         CIoU: 为True时计算CIoU LOSS (yolov5自带，默认使用)
#         SIoU: 为True时计算SIoU LOSS (新增)
#         EIoU: 为True时计算EIoU LOSS (新增)
#         WIoU: 为True时计算WIoU LOSS (新增)
#         Focal: 为True时，可结合其他的XIoU生成对应的IoU变体，如CIoU=True，Focal=True时为Focal-CIoU
#         alpha: AlphaIoU中的alpha参数，默认为1，为1时则为普通的IoU，如果想采用AlphaIoU，论文alpha默认值为3，此时设置CIoU=True则为AlphaCIoU
#         gamma: Focal_XIoU中的gamma参数，默认为0.5
#         scale: scale为True时，WIoU会乘以一个系数
#         monotonous: 3个输入分别代表WIoU的3个版本，None: origin v1, True: monotonic FM v2, False: non-monotonic FM v3
#         eps: 防止除0
#     Returns:
#         iou
#     """
#     # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
#
#     # Get the coordinates of bounding boxes
#     if xywh:  # transform from xywh to xyxy
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#         w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
#         w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
#
#     # Intersection area
#     inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
#             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
#
#     # Union Area
#     union = w1 * h1 + w2 * h2 - inter + eps
#     if scale:
#         wise_scale = WIoU_Scale(1 - (inter / union), monotonous=monotonous)
#
#     # IoU
#     # iou = inter / union # ori iou
#     iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
#     if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
#         cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#         ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#         if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
#             rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
#                         b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
#                 with torch.no_grad():
#                     alpha_ciou = v / (v - iou + (1 + eps))
#                 if Focal:
#                     return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter / (union + eps),
#                                                                                                  gamma)  # Focal_CIoU
#                 return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
#             elif EIoU:
#                 rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
#                 rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
#                 cw2 = torch.pow(cw ** 2 + eps, alpha)
#                 ch2 = torch.pow(ch ** 2 + eps, alpha)
#                 if Focal:
#                     return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps), gamma)  # Focal_EIou
#                 return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
#             elif SIoU:
#                 # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
#                 s_cw, s_ch = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps, (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
#                 sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
#                 sin_alpha_1, sin_alpha_2 = torch.abs(s_cw) / sigma, torch.abs(s_ch) / sigma
#                 threshold = pow(2, 0.5) / 2
#                 sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
#                 angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
#                 rho_x, rho_y = (s_cw / cw) ** 2, (s_ch / ch) ** 2
#                 gamma = angle_cost - 2
#                 distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
#                 omiga_w, omiga_h = torch.abs(w1 - w2) / torch.max(w1, w2), torch.abs(h1 - h2) / torch.max(h1, h2)
#                 shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
#                 if Focal:
#                     return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
#                         inter / (union + eps), gamma)  # Focal_SIou
#                 return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
#             elif WIoU:
#                 if scale:
#                     return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - iou) * torch.exp((rho2 / c2)), iou  # WIoU v3 https://arxiv.org/abs/2301.10051
#                 return iou, torch.exp((rho2 / c2))  # WIoU v1
#             if Focal:
#                 return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)  # Focal_DIoU
#             return iou - rho2 / c2  # DIoU
#         c_area = cw * ch + eps  # convex area
#         if Focal:
#             return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
#         return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     if Focal:
#         return iou, torch.pow(inter / (union + eps), gamma)  # Focal_IoU
#     return iou  # IoU
#
#
class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean
    """
    iou_mean = 1.
    _momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True

    def __init__(self, iou, monotonous=False):
        self.iou = iou
        self.monotonous = monotonous
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1
