import os
import shutil
import json
import time
from torch.cuda import amp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from Loss.KD_loss import MSELoss, kd_ce_loss
# from Loss.kd_losses.losses import KLDLoss
from proposed.teacher.teacher import Model
from Loss.contrast_KD_mem import ContrastLoss
import torch.nn.functional as F
setup_seed(33)



class trainLoss(nn.Module):

    def __init__(self, device, class_weight=None, ignore_index=-100, reduction='mean'):
        super(trainLoss, self).__init__()
        self.device = device
        self.class_weight_semantic = torch.from_numpy(np.array(
            [2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        self.class_weight_binary = torch.from_numpy(np.array([2.0197, 2.9765])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4584, 18.7187])).float()
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        # self.KLD = KLDLoss(transform_config={'loss_type': 'pixel'})
        self.mse = nn.MSELoss()
        self.con = ContrastLoss()
    def forward(self, s_logits, t_logits, targets, ep):

        semantic_gt, binary_gt, boundary_gt = targets
        t_sem, t_bound, t_bina = t_logits['sem'], t_logits['bound'], t_logits['bina']
        s_sem, s_bound, s_bina = s_logits['sem'], s_logits['bound'], s_logits['bina']
        # B, _, _, _ = t_logits[0].shape
        # KD Loss
        loss_binary = MSELoss(t_bina, s_bina) + MSELoss(t_bound, s_bound)
        loss_ce = kd_ce_loss(s_sem, t_sem)
        loss_response = loss_ce + loss_binary
        # loss_response = 0
        # loss_feature = self.mse(s_f[-1], t_f[-1])
        loss_feature = 0
        if ep > 0:
            loss_contrast = self.con(s_logits, targets[0])
            # loss_contrast = torch.tensor(0)
        else:
            loss_contrast = torch.tensor(0)
        loss_KD = loss_response + loss_contrast
        # Hard Loss
        loss1 = self.semantic_loss(s_sem, semantic_gt)
        loss2 = self.boundary_loss(s_bound, boundary_gt)
        loss3 = self.binary_loss(s_bina, binary_gt)
        loss_hard = loss1 + loss2 + loss3
        loss = loss_hard + loss_KD
        torch.cuda.empty_cache()
        return loss, loss_response, loss_feature, loss_contrast

class testLoss(nn.Module):

    def __init__(self, device, class_weight=None, ignore_index=-100, reduction='mean'):
        super(testLoss, self).__init__()
        self.device = device
        self.class_weight_semantic = torch.from_numpy(np.array(
            [2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        self.class_weight_binary = torch.from_numpy(np.array([2.0197, 2.9765])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4584, 18.7187])).float()
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        # self.KLD = KLDLoss(transform_config={'loss_type': 'pixel'})
        self.mse = nn.MSELoss()
        self.con = ContrastLoss()

    def forward(self, s_logits, t_logits, targets, ep):
        semantic_gt, binary_gt, boundary_gt = targets
        t_sem, t_bound, t_bina = t_logits['sem'], t_logits['bound'], t_logits['bina']
        s_sem, s_bound, s_bina = s_logits['sem'], s_logits['bound'], s_logits['bina']
        # B, _, _, _ = t_logits[0].shape
        # KD Loss
        loss_binary = MSELoss(t_bina, s_bina) + MSELoss(t_bound, s_bound)
        loss_ce = kd_ce_loss(s_sem, t_sem)
        loss_response = loss_ce + loss_binary
        # loss_response = 0
        # loss_feature = self.mse(s_f[-1], t_f[-1])
        loss_feature = torch.tensor(0)
        if ep > 0:
            loss_contrast = self.con(s_logits, targets[0])
        else:
            loss_contrast = torch.tensor(0)
        # Hard Loss
        loss = self.cross_entropy(s_sem, semantic_gt)
        torch.cuda.empty_cache()
        return loss, loss_response, loss_feature, loss_contrast

def _dequeue_and_enqueue(keys, labels, pixel_queue, pixel_queue_ptr):
    B, C, H, W = keys.shape
    labels = F.interpolate(labels.unsqueeze(dim=1).double(), size=keys.shape[2:], mode="nearest").long()
    for bs in range(B):
        this_feat = keys[bs].contiguous().view(C, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()
            num_pixels = idxs.shape[0]
            perm = torch.randperm(num_pixels)
            K = min(10, num_pixels)
            feat = this_feat[:, idxs[perm[:K]]].squeeze(-1)  # (256, K)
            # print(feat.shape)
            feat = torch.transpose(feat, 0, 1)
            ptr = int(pixel_queue_ptr[lb])
            # update memory bank (N, 300, 256)
            if ptr + K > args.pixel_queue:
                end_ptr = args.pixel_queue - ptr
                pixel_queue[lb, ptr:, :] = F.normalize(feat[:end_ptr, :], p=2, dim=1)
                pixel_queue[lb, :K-end_ptr, :] = F.normalize(feat[end_ptr:, :], p=2, dim=1)
                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % args.pixel_queue
            else:
                pixel_queue[lb, ptr:ptr+K, :] = F.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % args.pixel_queue

def run(args):
    # torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    ### multi-gpus
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    ###
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    S_model = get_model(cfg)
    print("==> S_model params: %.2fM" % (sum(p.numel() for p in S_model.parameters()) / 1e6))
    if (args.load is not None):
        S_model.load_state_dict(torch.load(args.load, map_location='cuda'))
        print('load model from ', args.load)
    ## multi-gpus
    S_model.to(gpu_ids[0])
    S_model = torch.nn.DataParallel(S_model, gpu_ids)
    T_model = Model(name='base')
    T_model.to(gpu_ids[0])
    # T_model = torch.nn.DataParallel(T_model, gpu_ids)
    print("==> T_model params: %.2fM" % (sum(p.numel() for p in T_model.parameters()) / 1e6))
    T_Weight = "/root/autodl-tmp/4090map/run/t_s_kd/teacher/145_model.pth"
    # T_Weight = "/root/autodl-tmp/seg/run/2024-07-15-20-46(SUS-ablation2)/200_model.pth"
    T_model.load_state_dict(torch.load(T_Weight, map_location='cuda'), strict=False)
    # T_model.load_state_dict(torch.load(T_Weight), strict=False)
    for p in T_model.parameters():
        p.stop_gradient = True
    T_model.eval()
    trainset, valset, _ = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
    # test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
    #                           pin_memory=True)

    params_list = S_model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    train_criterion = trainLoss(gpu_ids[0]).cuda()
    test_criterion = testLoss(gpu_ids[0]).cuda()
    # 指标 包含unlabel
    # train
    train_loss_meter = averageMeter()
    train_conloss_meter = averageMeter()
    train_featureloss_meter = averageMeter()
    train_responseloss_meter = averageMeter()
    # test
    test_loss_meter = averageMeter()
    test_conloss_meter = averageMeter()
    test_featureloss_meter = averageMeter()
    test_responseloss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])

    best_miou = 0


    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        S_model.train()
        T_model.eval()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].cuda()
                depth = sample['depth'].cuda()
                label = sample['label'].cuda()
                bound = sample['boundary'].cuda()
                binary_label = sample['binary'].cuda()
                targets = [label, bound, binary_label]
            else:
                image = sample['image'].cuda()
                depth = sample['depth'].cuda()
                label = sample['label'].cuda()
                bound = sample['bound'].cuda()
                binary_label = sample['binary_label'].cuda()
                targets = [label, binary_label, bound]
            with amp.autocast():
                with torch.no_grad():
                    output_T = T_model(image, depth)
                output_S = S_model(image, output_T['feature_T'], label)
                output_S['pixel_queue'] = S_model.module.pixel_queue
                output_S['pixel_queue_ptr'] = S_model.module.pixel_queue_ptr
                # loss, train_con, train_at, train_binary, train_ce = train_criterion(predict, predict_T, targets)
                loss, response_loss, feature_loss, con_loss = train_criterion(output_S, output_T, targets, ep)
            _dequeue_and_enqueue(output_S['embeding_T'].detach(), output_S['lb_key'], output_S['pixel_queue'], output_S['pixel_queue_ptr'])
            ####################################################
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()

            train_loss_meter.update(loss.item())
            train_conloss_meter.update(con_loss.item())
            train_featureloss_meter.update(feature_loss)
            train_responseloss_meter.update(response_loss.item())
        scheduler.step()

        with torch.no_grad():
            S_model.eval()  #告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(val_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].cuda()
                    depth = sample['depth'].cuda()
                    label = sample['label'].cuda()
                    bound = sample['boundary'].cuda()
                    binary_label = sample['binary'].cuda()
                    targets = [label, bound, binary_label]
                    output_T = T_model(image, depth)
                    output_S = S_model(image, output_T['feature_T'], label)
                    output_S['pixel_queue'] = S_model.module.pixel_queue
                    predict = output_S['sem']
                else:
                    image = sample['image'].cuda()
                    depth = sample['depth'].cuda()
                    label = sample['label'].cuda()
                    predict_S = S_model(image, depth)
                    output_T = T_model(image, depth, label)
                    predict = predict_S[0]
                loss, response_loss, feature_loss, con_loss = test_criterion(output_S, output_T, targets, ep)
                # loss_con, test_at, loss_binary, loss_ce = test_criterion(predict_S, predict_T, label)
                test_loss_meter.update(loss)
                test_conloss_meter.update(con_loss.item())
                test_featureloss_meter.update(feature_loss)
                test_responseloss_meter.update(response_loss.item())
                predict = predict.max(1)[1].cpu().numpy()  # [b,c,h,w] to [c, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        train_feature_loss = train_featureloss_meter.avg
        train_response_loss = train_responseloss_meter.avg
        train_con_loss = train_conloss_meter.avg
        test_loss = test_loss_meter.avg
        test_feature_loss = test_featureloss_meter.avg
        test_response_loss = test_responseloss_meter.avg
        test_con_loss = test_conloss_meter.avg


        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_mF1 = running_metrics_test.get_scores()[0]["F1-Score: "]
        test_avg = (test_macc + test_miou) / 2

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'response_loss={train_response_loss:.3f}/{test_response_loss:.3f}, '
                    f'feature_loss={train_feature_loss:.3f}/{test_feature_loss:.3f}, '
                    f'contrast_loss={train_con_loss:.3f}/{test_con_loss:.3f}, '
                    f'mAcc={test_macc:.3f}, '
                    f'miou={test_miou:.3f} '
                    f'mF1-score={test_mF1:.3f} '
                    )
        if test_miou > best_miou:
            best_miou = test_miou
            save_ckpt(logdir, S_model)

        if ep >=0.90 * cfg["epochs"]:
            name = f"{ep+1}" + "_"
            save_ckpt(logdir, S_model, name)

        if (ep + 1) % 50 == 0:
            name = f"{ep + 1}" + "_"
            save_ckpt(logdir, S_model, name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/root/autodl-tmp/4090map/configs/SUS_KD.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--pixel_queue", type=int, default=60,
                        help="warmup epoch for building contrastive memory bank")
    parser.add_argument("--gpu_ids", type=str, default='0', help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")
    parser.add_argument("--load", type=str, default=None, help="load weights from checkpoints")

    args = parser.parse_args()

    run(args)
