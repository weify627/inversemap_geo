
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
#from net_mapping import TriangleArea, Cosine

class graph_conv(nn.Module):
    def __init__(self, k, fin, fout,fout2,relu_=1,max_=1,bn_=1):
#out: (batchsize, npoints, nfeat)
        super(graph_conv, self).__init__()
        self.k = k
        self.fin = fin
        self.fout = fout
        self.conv = torch.nn.Conv2d(fin, fout, (1, 1 + self.k))
        self.bn = nn.BatchNorm2d(fout)
        if max_==1:
            self.mp = nn.MaxPool1d(k, stride = k)
            self.conv2 = nn.Conv1d(int(fout/k), fout2,1)
        else:
            self.conv2 = nn.Conv1d(fout, fout2,1)
        self.bn2 = nn.BatchNorm1d(fout2)
        self.relu_ = relu_
        self.max_ = max_
        self.bn_=bn_
    def forward(self, x, faces):
        nfaces = faces.size(1)
        batchsize = x.size(0)
        npoints = x.size(1)
        out = Variable( torch.FloatTensor(batchsize, npoints, self.k, self.fin).fill_(0).cuda())
        minv,_ = torch.kthvalue(faces.data.cpu(), 1) #, dim=2)
        midv,_ = torch.kthvalue(faces.data.cpu(), 2) #, dim=2)
        maxv,_ = torch.kthvalue(faces.data.cpu(), 3) #, dim=2)
        minv = minv.view(x.size(0),nfaces)
        midv = midv.view(x.size(0),nfaces)
        maxv = maxv.view(x.size(0),nfaces)
        for i in range(batchsize):
            adjacency = Variable( torch.FloatTensor(npoints, npoints).fill_(0).cuda())
            for j in range(nfaces):
                if minv[i,j]!=midv[i,j]:
                    adjacency[minv[i,j],midv[i,j]]=1
                    adjacency[minv[i,j],maxv[i,j]]=1
                    adjacency[midv[i,j],maxv[i,j]]=1#-adjacency[midv[i,j],maxv[i,j]]
                else:
                    break
            adjacency = adjacency+adjacency.transpose(0,1)
            neighbor, ind = adjacency.topk( self.k, 1)
            out[i,:,:,:] = torch.index_select( x[i,:,:],0, ind.contiguous().view(-1).detach()).contiguous().view( npoints, self.k, self.fin)
#            out[i,:,:,:] = torch.gather(x[i,:,:].contiguous().view(1,npoints,-1).repeat(npoints,1,1),1,ind.contiguous().view(npoints,self.k,1).repeat(1,1,self.fin))
            out[i,:,:,:] = neighbor.contiguous().view(npoints,self.k,1).repeat(1,1,self.fin) * out[i,:,:,:].clone()
        x = x.contiguous().view(batchsize,npoints,1,-1)
        out = torch.cat((x,out),2)
        out = out.transpose(2,3).transpose(1,2)
        out = F.relu(self.bn(self.conv(out)))
#        out = self.conv(out)
        #out = (batchsize,self.fout,npoints,1)
        out = out.squeeze().transpose(1,2)
#out = out.contiguous().view(batchsize, self.fout, npoints).transpose(1,2)
#       out = self.bn(out)
        #out = (batchsize,npoints,self.fout)
        if self.max_ :
            out = self.mp(out).transpose(1,2)
        #out = (batchsize,self.fin,npoints)
        else:
            out=out.transpose(1,2)
        out = ((self.conv2(out)))
        if self.bn_:
            out = self.bn2(out)
        if self.relu_ ==1:
            out = F.relu(out)
        #out = (batchsize,self.fout2,npoints)
        out = out.transpose(1,2)
        
        return out
'''
class MeshMap1(nn.Module):
    def __init__(self, degree, npoints=0, nfaces=0):
        super(MeshMap, self).__init__()
        self.degree = degree
        self.npoints = npoints
        self.nfaces = nfaces
        self.conv0 = nn.Conv1d(3,64,1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.gconv1 = graph_conv( degree, 64, 64*degree,128)

        self.gconv2 = graph_conv( degree, 128, 128*degree,128)
        self.gconv3 = graph_conv( degree, 128, 128*degree,256)
#
        self.gconv4 = graph_conv( degree, 256, 256*degree, 256)
#        self.gconv5 = graph_conv( degree, 256, 256*degree, 256)
#        self.gconv6 = graph_conv( degree, 256, 256*degree, 512)
#        self.gconv7 = graph_conv( degree, 512, 512*degree, 512)
        self.gconv8 = graph_conv( degree, 256, 256*degree, 64)
        self.gconv9 = graph_conv( degree, 64, 64*degree, 3)
        self.gconv10 = graph_conv( degree, 3, 3*degree, 2,relu=0)
    
    def forward( self, x, faces):
        x = F.relu(self.bn(self.conv0(x)))
        x = x.transpose(1,2)
        x = self.gconv1(x,faces)
        x = self.gconv2(x,faces)
        x = self.gconv3(x,faces)
        x = self.gconv4(x,faces)
#        x = self.gconv5(x,faces)
#        x = self.gconv6(x,faces)
#        x = self.gconv7(x,faces)
        x = self.gconv8(x,faces)
        x = self.gconv9(x,faces)
        x = self.gconv10(x,faces)
        area = TriangleArea(x,faces,1)
        cos = Cosine(x,faces,1)
        return area,cos,x,x
    
    
class MeshMap(nn.Module):
    def __init__(self, degree, npoints=0, nfaces=0):
        super(MeshMap, self).__init__()
        self.degree = degree
        self.npoints = npoints
        self.nfaces = nfaces
        self.conv0 = nn.Conv1d(3,64,1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        mpstride = 1
        self.gconv1 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv2 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv3 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv4 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv5 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv6 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv7 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv8 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv12 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv13 = graph_conv( degree, 64, 64*mpstride,64)
        self.gconv9 = graph_conv( degree, 64, 64*mpstride,128)

#
        self.gconv10 = graph_conv( degree, 128, 128*mpstride, 128)
#        self.gconv5 = graph_conv( degree, 256, 256*mpstride, 256)
#        self.gconv6 = graph_conv( degree, 256, 256*mpstride, 512)
#        self.gconv7 = graph_conv( degree, 512, 512*mpstride, 512)
        self.gconv11 = graph_conv( degree, 128, 128*mpstride, 64)
#        self.gconv9 = graph_conv( degree, 64, 64*degree, 3)
#        self.gconv10 = graph_conv( degree, 3, 3*degree, 2,relu=0)
        self.conv1 = nn.Conv1d(64,2,1)
#        self.bn1 = nn.BatchNorm1d(2)
    
    def forward( self, x, faces):
        x = F.relu(self.bn(self.conv0(x)))
        x = x.transpose(1,2)
        x = self.gconv1(x,faces)
        x = self.gconv2(x,faces)
        x = self.gconv3(x,faces)
        x = self.gconv4(x,faces)
        x = self.gconv5(x,faces)
        x = self.gconv6(x,faces)
        x = self.gconv7(x,faces)
        x = self.gconv8(x,faces)
        x = self.gconv12(x,faces)
        x = self.gconv13(x,faces)
        x = self.gconv9(x,faces)
        x = self.gconv10(x,faces)
        x = self.gconv11(x,faces)
        x = x.transpose(1,2)
        x = (self.conv1(x))
        x = x.transpose(1,2)
        area = TriangleArea(x,faces,1)
        cos = Cosine(x,faces,1)
        return area,cos,x,x
'''



