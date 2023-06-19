from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import numpy as np
import torch

class Encoder(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Encoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dimension),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class img2txt(nn.Module):
    def __init__(self, project_dim, cluster_num):
        super(img2txt, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(project_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.clustering = nn.Sequential(
            nn.Linear(256, cluster_num, bias=False),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


class txt2img(nn.Module):
    def __init__(self, project_dim, cluster_num):
        super(txt2img, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(project_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.clustering = nn.Sequential(
            nn.Linear(256, cluster_num, bias=False),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


class Net(nn.Module):
    def __init__(self, img_size, txt_size, embd_dim, cluster_num, project_dim):
        super(Net, self).__init__()
        self.img_size = img_size
        self.txt_size = txt_size
        self.embd_dim = embd_dim
        self.cluster_num = cluster_num
        self.project_dim = project_dim

        self.txt_encoder = Encoder(txt_size, self.embd_dim)
        self.img_encoder = Encoder(img_size, self.embd_dim)
        self.img2txt = img2txt(self.project_dim, self.cluster_num)
        self.txt2img = txt2img(self.project_dim, self.cluster_num)

        self.project_head = nn.Sequential(
            nn.Linear(self.embd_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.project_dim),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.clustering1 = nn.Sequential(
            nn.Linear(self.project_dim, self.cluster_num, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, img, txt):
        txt = self.txt_encoder(txt)
        img = self.img_encoder(img)

        img_pro = self.project_head(img)
        txt_pro = self.project_head(txt)

        img2txt_fea = self.img2txt(img_pro)
        img2txt_rec = self.txt2img(img2txt_fea)
        img2txt_cluster = self.img2txt.clustering(img2txt_fea)
        img_cluster = self.img2txt.clustering(img_pro)

        txt2img_fea = self.txt2img(txt_pro)
        txt2img_rec = self.img2txt(txt2img_fea)
        txt2img_cluster = self.txt2img.clustering(txt2img_fea)
        txt_cluster = self.txt2img.clustering(txt_pro)

        img_c = self.clustering1(img_pro)
        txt_c = self.clustering1(txt_pro)

        return img_pro, txt_pro, img_c, txt_c, [img2txt_fea, img2txt_rec, img2txt_cluster, img_cluster,
                                                txt2img_fea, txt2img_rec, txt2img_cluster, txt_cluster]


def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 75:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    try:
        argmaxes = np.nanargmax(CL, 0)
    except:
        argmaxes = np.argmax(CL, 0)
    newL = th.LongTensor(argmaxes)
    return newL


class CLoss1(nn.Module):

    def __init__(self, bs, tau=0.5, cos_sim=True, gpu=True, eps=1e-8):
        super(CLoss1, self).__init__()
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.gpu = gpu
        self.eps = eps

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_pos_and_neg_mask1(self, bs):
        zeros = torch.zeros((bs, bs), dtype=torch.uint8)
        eye = torch.eye(bs, dtype=torch.uint8)
        pos_mask = torch.cat([
            torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
        ], dim=1)
        neg_mask = (torch.ones(2 * bs, 2 * bs, dtype=torch.uint8) - torch.eye(
            2 * bs, dtype=torch.uint8))
        pos_mask = self.mask_type_transfer(pos_mask).to(self.device)
        neg_mask = self.mask_type_transfer(neg_mask).to(self.device)
        return pos_mask, neg_mask

    def mask_type_transfer(self, mask):
        mask = mask.type(torch.bool)
        return mask

    def get_pos_and_neg_mask(self, bs1, label1, label2, index_nearest1, index_nearest2):
        label1 = torch.argmax(label1, dim=1)
        label2 = torch.argmax(label2, dim=1)
        eye = torch.eye(bs1, dtype=torch.uint8).to(self.device)
        zeros = torch.zeros((bs1, bs1), dtype=torch.uint8).to(self.device)
        label1_, label2_ = self.mapL2toL1(label1, label2)
        label2_ = torch.tensor(label2_).to(self.device)
        index_nearest1 = torch.tensor(index_nearest1).to(self.device)
        index_nearest2 = torch.tensor(index_nearest2).to(self.device)

        label1 = label1_.unsqueeze(dim=1)
        label2 = label2_.unsqueeze(dim=1)
        pos_mask = label1.eq(label2.t()) | eye.type(torch.bool)
        pos_mask1 = pos_mask | index_nearest1
        pos_mask2 = pos_mask | index_nearest2
        pos_mask = torch.cat([
            torch.cat([zeros, pos_mask1], dim=0), torch.cat([pos_mask2, zeros], dim=0),
        ], dim=1).type(torch.bool)

        label_sum = torch.cat([label1_, label2_], 0)
        label_sum = label_sum.unsqueeze(dim=1)
        neg_mask = ~label_sum.eq(label_sum.t())
        return pos_mask, neg_mask

    def mapL2toL1(self, label1, label2):
        D = max(label1.max(), label2.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        try:
            for i in range(label1.size(0)):
                w[label1[i], label2[i]] += 1
        except:
            for i in range(label1.size):
                w[label1[i], label2[i]] += 1
        from scipy.optimize import linear_sum_assignment
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)

        map1 = {a2: a1 for a1, a2 in ind}
        try:
            label2 = [map1[l] for l in label2.cpu().numpy().squeeze()]
        except:
            label2 = [map1[l] for l in label2.squeeze()]
        return label1, label2

    def forward(self, zi, zj, label1, label2, index_nearest1, index_nearest2):
        bs1, bs2 = label1.shape[0], label2.shape[0]
        assert bs1 == bs2
        self.pos_mask, self.neg_mask = self.get_pos_and_neg_mask(bs1, label1, label2, index_nearest1, index_nearest2)
        z_all = torch.cat([label1, label2], dim=0)
        sim_mat = self.cosine_similarity(
            z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau  # s_(i,j)
        sim_mat = torch.log(sim_mat)
        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).clone())
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).clone())

        loss1 = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        bs = zi.shape[0]
        self.pos_mask, self.neg_mask = self.get_pos_and_neg_mask1(bs)
        z_all = torch.cat([zi, zj], dim=0)
        sim_mat = self.cosine_similarity(
        z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau

        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).view(2*bs).clone())
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).view(2*bs, -1).clone())

        loss2 = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()
        return loss1+ loss2


class Pseudo_Label_Loss(nn.Module):

    def __init__(self, bs, tau=0.5, gpu=True, eps=1e-8):
        super(Pseudo_Label_Loss, self).__init__()
        self.tau = tau
        self.gpu = gpu
        self.eps = eps
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_pos_and_neg_mask(self, bs1, label):
        eye = torch.eye(bs1, dtype=torch.uint8).to(self.device)
        label = label.unsqueeze(dim=1)
        pos_mask = label.eq(label.t()) ^ eye.type(torch.bool)
        neg_mask = ~pos_mask ^ eye.type(torch.bool)
        return pos_mask, neg_mask

    def forward(self, feature, label):
        label = torch.argmax(label, dim=1)
        bs1, bs2 = feature.shape[0], label.shape[0]
        assert bs1 == bs2
        self.pos_mask, self.neg_mask = self.get_pos_and_neg_mask(bs1, label)
        sim_mat = self.cosine_similarity(
            feature.unsqueeze(1), feature.unsqueeze(0)) / self.tau  # s_(i,j)
        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).clone())
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).clone())
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()/10

        return loss


