from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F



class PixelContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07

        self.max_samples = 1024
        self.max_views = 10


    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            # if ii <= 1: continue   # 不采样背景类
            this_q = Q[ii, :cache_size, :]   # 采样第i个类别的所有样本

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q  # x_的前cache_size个样本都是第i类的
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii   # y_的前cache_size行为ii
            sample_ptr += cache_size

        # 最后得到的X_为((classes-1)*cache_size, feature_dim), y_为((classes-1) * cache_size)
        return X_, y_

    def _contrastive(self, s_f, label, queue=None):
        # sample anchor feature
        anchor_feature = s_f.contiguous().view(-1, s_f.shape[-1])
        anchor_y = label.contiguous().view(-1, 1)

        # sample from memory bank
        m_x, m_y = self._sample_negative(queue)
        m_y = m_y.contiguous().view(-1, 1)

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, m_x.T),
                                          self.temperature)
        mask = torch.eq(anchor_y, m_y.T).float().cuda()
        # mask_s = mask_s.repeat(anchor_count, 1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # calculate neg sample
        neg_mask = 1 - mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        # exp_logits即自身, neg_logits即负样本之和
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-4)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_mean = loss.mean()

        return loss_mean

    def forward(self, f_s, label, queue):
        batch_size = f_s.shape[0]
        # student
        label = label.unsqueeze(1).float().clone()
        label = torch.nn.functional.interpolate(label,
                                                (f_s.shape[2], f_s.shape[3]), mode='nearest')
        label = label.contiguous().view(batch_size, -1)  # (B, 300)
        f_s = f_s.permute(0, 2, 3, 1)
        f_s = f_s.contiguous().view(f_s.shape[0], -1, f_s.shape[-1])  # (B, 300, 1024)
        f_s = nn.functional.normalize(f_s, p=2, dim=2)
        # sample
        loss = self._contrastive(f_s, label, queue=queue)
        return loss


class ContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, student, target):

        # seg_s = student['sem']
        # seg_t = teacher['sem']
        # embeding_T = student['embeding_T']
        embeding_S = student['embeding_S']
        pixel_queue = student['pixel_queue'].detach()

        # _, pre_s = torch.max(seg_s, 1)  # 获取预测图
        # _, pre_t = torch.max(seg_t, 1)  # 获取预测图
        loss_contrast = self.contrast_criterion(embeding_S, target, pixel_queue)

        return loss_contrast

if __name__ == '__main__':
    x = torch.randn(2, 128, 640, 480)
    embed = torch.randn(2, 128, 20, 15)
    pixel_queue = torch.randn(6, 300, 128)
    inputs = {'sem': x, 'embed': embed, 'pixel_queue': pixel_queue}
    y = torch.randn(2, 640, 480)
    model = ContrastLoss()
    loss = model(inputs, y)

