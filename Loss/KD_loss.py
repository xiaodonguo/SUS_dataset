import torch
import torch.nn.functional as F
import torch.nn as nn
def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def cosine(a, b):
    a = a.reshape(a.shape[0], a.shape[1], -1).transpose(2, 1)
    b = b.reshape(b.shape[0], b.shape[1], -1).transpose(2, 1)
    sqrt_a = (a**2).sum(dim=-1)**(1/2)
    sqrt_b = (b**2).sum(dim=-1)**(1/2)
    s = (a*b).sum(dim=-1)/(sqrt_a * sqrt_b)
    return s

def contrastive(pre_s, postive, negtive):
    # pre_s = torch.softmax(pre_s, dim=1)
    # postive = torch.softmax(postive, dim=1)
    # negtive = torch.softmax(negtive, dim=1)
    out = - torch.log(torch.exp(cosine(pre_s, postive))/(torch.exp(cosine(pre_s, postive)) + torch.exp(cosine(pre_s, negtive)))).mean()
    return out

def MSELoss(student, teacher):
    student = F.softmax(student, dim=1)
    teacher = F.softmax(teacher, dim=1)
    loss = (student - teacher).pow(2).sum(dim=1).mean()
    return loss

def kd_ce_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=1)).sum(dim=1).mean()
    return loss

class RelationDistillationLoss(nn.Module):
    def __init__(
        self,
        device,
        cl_num,
        fit_dim,
        temp=0.2
    ):
        super().__init__()
        self.device = device
        self.temp = temp
        self.num_classes = cl_num
        self.feat_dim =fit_dim
        self.temp = temp
        self.MSE = nn.MSELoss()

    def forward(self, features_s, features_t, labels):
        loss = torch.tensor(0.0, device=self.device)
        labels = F.interpolate(labels.unsqueeze(dim=1).double(), size=features_s.shape[2:], mode="nearest").long()
        cl_present = torch.unique(input=labels)
        features_s_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        features_t_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        for cl in cl_present:
            features_s_cl = features_s[
                (labels == cl).expand(-1, features_s.shape[1], -1, -1)
            ].view(features_s.shape[1], -1)   # [C, BxHxW]
            features_t_cl = features_t[
                (labels == cl).expand(-1, features_t.shape[1], -1, -1)
            ].view(features_t.shape[1], -1)   # [C, BxHxW]
            num_cl = torch.sum((labels == cl).expand(-1, features_s.shape[1], -1, -1)).item()
            features_s_cl = torch.sum(features_s_cl, dim=-1) / num_cl
            features_t_cl = torch.sum(features_t_cl, dim=-1) / num_cl
            features_s_mean[cl] = F.normalize(features_s_cl, p=2, dim=0)
            features_t_mean[cl] = F.normalize(features_t_cl, p=2, dim=0)

        features1_sim = torch.div(
            torch.matmul(features_s_mean, features_s_mean.T), self.temp
        )
        logits_mask = torch.scatter(
            torch.ones_like(features1_sim),
            1,
            torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
            0,
        )
        # logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
        # features1_sim = features1_sim - logits_max1.detach()
        row_size = features1_sim.size(0)
        logits1 = torch.exp(
            features1_sim[logits_mask.bool()].view(row_size, -1)
        ) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(
            dim=1, keepdim=True
        )

        features2_sim = torch.div(
            torch.matmul(features_t_mean, features_t_mean.T),
            self.temp,
        )
        # logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
        # features2_sim = features2_sim - logits_max2.detach()
        logits2 = torch.exp(
            features2_sim[logits_mask.bool()].view(row_size, -1)
        ) / torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(
            dim=1, keepdim=True
        )

        loss = (-logits2 * torch.log(logits1)).sum(1).mean()
        # loss = self.MSE(logits1, logits2)
        return loss

class Con_Deep(nn.Module):
    def __init__(
        self,
        device,
        cl_num,
        fit_dim,
        temp=0.1
    ):
        super().__init__()
        self.device = device
        self.temp = temp
        self.num_classes = cl_num
        self.feat_dim = fit_dim
        self.temp = temp
        self.MSE = nn.MSELoss()

    def forward(self, features_s, features_t, labels):
        labels = F.interpolate(labels.unsqueeze(dim=1).double(), size=features_s.shape[2:], mode="nearest").long()
        cl_present = torch.unique(input=labels)
        features_s_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        features_t_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        for cl in cl_present:
            features_s_cl = features_s[
                (labels == cl).expand(-1, features_s.shape[1], -1, -1)
            ].view(features_s.shape[1], -1)   # [C, BxHxW]
            features_t_cl = features_t[
                (labels == cl).expand(-1, features_t.shape[1], -1, -1)
            ].view(features_t.shape[1], -1)   # [C, BxHxW]
            num_cl = torch.sum((labels == cl)).item()
            features_s_cl = torch.sum(features_s_cl, dim=-1) / num_cl
            features_t_cl = torch.sum(features_t_cl, dim=-1) / num_cl
            features_s_mean[cl] = F.normalize(features_s_cl, p=2, dim=0)
            features_t_mean[cl] = F.normalize(features_t_cl, p=2, dim=0)

        logits1 = torch.div(
            torch.matmul(features_s_mean, features_t_mean.T), self.temp
        )
        mask = torch.zeros_like(logits1)
        for cl in cl_present:
            mask[cl, cl] = 1
        # logits_max, _ = torch.max(logits1, dim=1, keepdim=True)
        # logits1 = logits1 - logits_max
        logits2 = torch.exp(logits1[cl_present[:], :]).sum(1)  # 正样本+负样本

        logits1 = torch.exp(logits1[mask.bool()])  # 正样本
        loss = -torch.log(logits1 / logits2).mean()
        # loss = self.MSE(logits1, logits2)
        return loss

class Con_Shallow(nn.Module):
    def __init__(
        self,
        device,
        cl_num,
        temp=0.1
    ):
        super().__init__()
        self.device = device
        self.temp = temp
        self.num_classes = cl_num
        self.temp = temp
        self.MSE = nn.MSELoss()

    def forward(self, features_s, features_t, labels):
        B, C, H, W = features_s.shape
        labels = F.interpolate(labels.unsqueeze(dim=1).double(), size=features_s.shape[2:], mode="nearest").long()
        cl_present = torch.unique(input=labels)
        features_s_mean = torch.zeros(
            [self.num_classes, B*H*W], device=self.device
        )
        features_t_mean = torch.zeros(
            [self.num_classes, B*H*W], device=self.device
        )
        for cl in cl_present:
            mask_cl = (labels == cl).expand(-1, features_s.shape[1], -1, -1).int()
            features_s_cl = (features_s * mask_cl).view(-1, features_s.shape[1])   # [BxHxW, C]
            features_t_cl = (features_t * mask_cl).view(-1, features_t.shape[1])   # [BxHxW, C]
            # num_cl = torch.sum((labels == cl).expand(-1, features_s.shape[1], -1, -1)).item()
            features_s_cl = torch.sum(features_s_cl, dim=-1) / C
            features_t_cl = torch.sum(features_t_cl, dim=-1) / C
            features_s_mean[cl] = F.normalize(features_s_cl, p=2, dim=0)
            features_t_mean[cl] = F.normalize(features_t_cl, p=2, dim=0)

        logits1 = torch.div(
            torch.matmul(features_s_mean, features_t_mean.T), self.temp
        )
        mask = torch.zeros_like(logits1)
        for cl in cl_present:
            mask[cl, cl] = 1
        # logits_max, _ = torch.max(logits1, dim=1, keepdim=True)
        # logits1 = logits1 - logits_max
        logits2 = torch.exp(logits1[cl_present[:], :]).sum(1)  # 正样本+负样本

        logits1 = torch.exp(logits1[mask.bool()])  # 正样本
        loss = -torch.log(logits1 / logits2).mean()
        # loss = self.MSE(logits1, logits2)
        return loss

if __name__ == '__main__':
    feature_s = torch.randn(2, 64, 20, 15)
    feature_t = torch.randn(2, 64, 20, 15)
    shape = (2, 480, 640)
    label = torch.randint(0, 9, size=shape, dtype=torch.int)
    # relation_loss = Con_Shallow(0, 9)
    relation_loss = Con_Deep(0, 9, 64)
    loss = relation_loss(feature_s, feature_t, label)