
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from conv import graph_conv

def loss_bnd(img, gt_bnd):
    s = img.size(2)
    assert s==img.size(3)
    inv_idx = Variable(torch.arange(s-1,-1,-1).long().cuda())
    inv_idxr = Variable(torch.arange(s-3,-1,-1).long().cuda())
    pred_t = img[:,:,0,:]
    pred_t = pred_t.index_select(2,inv_idx)#    pred_t = Variable(torch.from_numpy(np.flip(pred_t.data.cpu().numpy(),2).copy()).cuda())
    pred_r = img[:,:,1:s-1,s-1]
    pred_r = pred_r.index_select(2,inv_idxr)#    pred_r = Variable(torch.from_numpy(np.flip(pred_r.data.cpu().numpy(),2).copy()).cuda())
    pred_b = img[:,:,s-1,:]
    pred_l = img[:,:,1:s-1,0]
    pred_bnd = torch.cat((pred_b, pred_r,pred_t, pred_l),2).transpose(1,2)
    ab=(torch.abs(Variable(gt_bnd.cuda(),requires_grad=False)-pred_bnd))
    loss = torch.sum(ab)/img.size(0)
    return loss


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def batch_pairwise_dist(a,b,faces):
    inf = 1000000000000
    x,y = a,b.contiguous().view(a.size(0),b.size(1),-1)
    bs, np_x, np_y= x.size(0), x.size(2), y.size(2)
    xx = torch.bmm(x.transpose(2,1), x)
    yy = torch.bmm(y.transpose(2,1), y)
    xy = torch.bmm(x.transpose(2,1), y)
    diag_indx = torch.arange(0, np_x).type(torch.cuda.LongTensor)
    diag_indy = torch.arange(0, np_y).type(torch.cuda.LongTensor)
    rx = Variable(torch.FloatTensor(bs,np_x).fill_(0).cuda())
    ry = Variable(torch.FloatTensor(bs,np_y).fill_(0).cuda())
    for i in range(bs):
        rx[i,:] = torch.diag(xx[i,:,:])
        ry[i,:] = torch.diag(yy[i,:,:])
    rx = rx.unsqueeze(2).expand_as(xy)
    ry = ry.unsqueeze(1).expand_as(xy)
    P = (rx + ry - 2*xy)
    for i in range(bs):
        np = torch.max(faces[i,:,:]).data[0]
        P[i,np:,:] = Variable(torch.FloatTensor(np_x-np,np_y).fill_(0).cuda())
    return P,bs,np_x,np_y

def loss_dis(x, img,faces):
    dis, bs, np_x, np_y= batch_pairwise_dist(x, img,faces)
    loss_x_p = torch.mean(torch.sum(torch.min(dis,2)[0].squeeze(),1))
    for i in range(bs):
        np = torch.max(faces[i,:,:]).data[0]
        dis[i,np:,:] = Variable(torch.FloatTensor(np_x-np,np_y).fill_(10000000).cuda())
    loss_img_p = torch.mean(torch.sum(torch.min(dis,1)[0].squeeze(),1))
    dis_tb = torch.norm( img[:,:,1:,:]- img[:,:,:-1,:],2,1)
    dis_b = torch.cat((dis_tb,dis_tb[:,:,-1,:].unsqueeze(2)),2)
    dis_t = torch.cat((dis_tb[:,:,0,:].unsqueeze(2),dis_tb),2)
    dis_lr = torch.norm( img[:,:,:,1:]- img[:,:,:,:-1],2,1)
    dis_r = torch.cat((dis_lr,dis_lr[:,:,:,-1].unsqueeze(3)),3)
    dis_l = torch.cat((dis_lr[:,:,:,0].unsqueeze(3),dis_lr),3)
    dis_local = torch.cat((dis_b,dis_t,dis_l,dis_r),1)
    gap_local = torch.max(dis_local,1)[0] - torch.min(dis_local,1)[0]
    loss_local = torch.mean(torch.sum(gap_local.contiguous().view(x.size(0),-1),1))
    return loss_x_p, loss_img_p, loss_local

