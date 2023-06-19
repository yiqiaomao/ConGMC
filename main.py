import argparse
import torch
import numpy as np
import utils1
from dataset import Dateset_mat, data_loder
from tqdm import trange
from model import Net, UD_constraint, CLoss1, Pseudo_Label_Loss
import torch.nn.functional as F
import warnings
import torch.nn as nn
import torch.distributions.normal as normal
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
data_name = 'youcook'
parser.add_argument("--dataset_root", default=data_name, type=str)   # nus 805 esp 80 flickr 1570 voc 910
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=512)
config = parser.parse_args()
config.max_ACC = 0
Closs1 = CLoss1(config.batch_size)
get_pseudo = Pseudo_Label_Loss(config.batch_size)
print(config.dataset_root)


def run():
    Dataset = Dateset_mat(config.dataset_root)
    dataset = Dataset.getdata()
    label = np.array(dataset[2]) - 1
    label = np.squeeze(label)
    cluster_num = max(label) + 1
    print("clustering number: ", cluster_num)
    img = torch.tensor(dataset[0], dtype=torch.float32).to(device)
    txt = torch.tensor(dataset[1], dtype=torch.float32).to(device)
    prior_loc = torch.zeros(config.batch_size, 256)
    prior_scale = torch.ones(config.batch_size, 256)
    prior = normal.Normal(prior_loc, prior_scale)
    max_ACC = 0

    criterion = torch.nn.CrossEntropyLoss().to(device)
    data = data_loder(config.batch_size)
    data.get_data(dataset)

    model = Net(img_size=img.size(1), txt_size=txt.size(1), embd_dim=512, cluster_num=cluster_num, project_dim=256)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in trange(config.num_epochs):
        model.train()
        model.zero_grad()
        optimiser.zero_grad()
        prior_sample = prior.sample().to(device)

        for data_ in data:
            img_ = nn.functional.normalize(data_[0], dim=1).to(device)
            txt_ = nn.functional.normalize(data_[1], dim=1).to(device)
            img_pro, txt_pro, img_c, txt_c, \
            [img2txt_fea, img2txt_rec, img2txt_cluster, img_cluster,
            txt2img_fea, txt2img_rec, txt2img_cluster, txt_cluster] \
                = model(img_, txt_)

            index_nearest1 = getNearest(img2txt_fea, txt_pro)
            index_nearest2 = getNearest(txt2img_fea, img_pro)
            try:
                skl1 = torch.nn.functional.kl_div(img_pro, prior_sample).to(device)
                skl2 = torch.nn.functional.kl_div(txt_pro, prior_sample).to(device)
            except:
                prior_loc = torch.zeros(img_.size(0), 256)
                prior_scale = torch.ones(img_.size(0), 256)
                prior = normal.Normal(prior_loc, prior_scale)
                prior_sample = prior.sample().to(device)
                skl1 = torch.nn.functional.kl_div(img_pro, prior_sample).to(device)
                skl2 = torch.nn.functional.kl_div(txt_pro, prior_sample).to(device)
            loss1 = Closs1(img_pro, txt_pro, img_c, txt_c, index_nearest1, index_nearest2)
            loss2 = skl1 + skl2
            loss3 = F.l1_loss(img2txt_rec, img_pro) + F.l1_loss(txt2img_rec, txt_pro) \
                    + criterion(img2txt_cluster, img_cluster.argmax(dim=-1)) + criterion(txt2img_cluster, txt_cluster.argmax(dim=-1))
            if epoch < 50:
                loss4 = 0
            else:
                loss4 = 0.1*(get_pseudo(img_pro, img_c) + get_pseudo(txt_pro, txt_c))

            if epoch % 5 == 0:
                with torch.no_grad():
                    UDC_img = UD_constraint(img_c).to(device)
                    UDC_txt = UD_constraint(txt_c).to(device)
                    beta1, beta2 = criterion(img_c, UDC_img), criterion(txt_c, UDC_txt)
                    loss_UDC = beta1 + beta2
            else:
                loss_UDC = 0

            loss = loss1 + loss2 + loss3 + loss4
            loss = loss + loss_UDC
            loss.backward()
            optimiser.step()

        if epoch % 10 == 0:
            model.eval()
            all_img_pro, all_txt_pro, all_img_c, all_txt_c, _ = model(img, txt)
            acc_1, nmi_1, ari1 = getACC_NMI(all_img_c, label)
            print('ACC %.4f NMI1: %.4f ARI1: %.4f' % (acc_1, nmi_1, ari1))



def getNearest(fea1, fea2):
    simi = torch.einsum("nd,md->nm", fea1, fea2)
    simi_b = torch.zeros(simi.shape)
    simi_b[(torch.arange(len(simi)).unsqueeze(1), torch.topk(simi, 1).indices)] = 1
    simi_b = simi_b>0
    return simi_b


def getACC_NMI(data1, label):
    pre_label = np.array(data1.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc1 = utils1.metrics.acc(pre_label, label)
    nmi1 = utils1.metrics.nmi(pre_label, label)
    ari1 = utils1.metrics.ari(pre_label, label)
    return acc1, nmi1, ari1


if __name__ == '__main__':
    run()

