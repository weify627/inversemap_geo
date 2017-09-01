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
#used for v0 2d toy model in weekly report 2 
class STN2d(nn.Module):
    def __init__(self, num_points = 50):
        super(STN2d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,1]).astype(np.float32))).view(1,4).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 2, 2)
        return x

class STN3d(nn.Module):
    def __init__(self, dim=3,num_points = 500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim*dim)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dim=dim


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

#        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        iden = Variable(torch.eye(self.dim)).view(1,self.dim*self.dim).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


def cosine(s1,s2,is_v):
    cos=torch.sum(s1*s2,1) 
    cos=cos/(torch.norm(s1.add(1e-12),2,1)*torch.norm(s2.add(1e-12),2,1))
    return cos

#def Cosine(x,faces,is_variable):
#    faces=faces.long()
#    cos= (torch.zeros(x.size(0),3*faces.size(1)))
#    if is_variable:
#        cos = Variable(cos)
#    if x.is_cuda:
#        cos=cos.cuda()
#    for i in range(faces.size(0)):
#           v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
#           v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
#           v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
#        s12=v2-v1
#        s23=v3-v2
#        s13=v3-v1
#        cos1=cosine(s12,s13,is_variable)
#        cos2=cosine(s23,-s12,is_variable)
#        cos3=cosine(-s13,-s23,is_variable)
#        cos_cat=torch.cat((cos1,cos2,cos3),0)
#        cos[i,:]=cos_cat
#        return (cos)

#def TriangleSide(x,faces,is_variable):
#    faces=faces.long()
#    triangle_side= (torch.zeros(x.size(0),3*faces.size(1)))
#    if is_variable:
#        triangle_side = Variable(triangle_side)
#    if x.is_cuda:
#        triangle_side=triangle_side.cuda()
#    for i in range(faces.size(0)):
#           v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
#           v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
#           v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
#        s1=torch.nn.functional.pairwise_distance(v3,v2)
#        s2=torch.nn.functional.pairwise_distance(v3,v2)
#        s3=torch.nn.functional.pairwise_distance(v1,v2)
#        s_cat=torch.cat((s1,s2,s3),0)
#        triangle_side[i,:]=s_cat
#
#        return (triangle_side)
    
def TriangleArea(x,faces,is_variable):
    faces=faces.long()
    triangle_area= (torch.zeros(x.size(0),faces.size(1)))
    if is_variable:
        triangle_area = Variable(triangle_area)
    if x.is_cuda:
        triangle_area=triangle_area.cuda()
    for i in range(faces.size(0)):
        v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
        v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
        v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
        a_sub=(v2[:,1]-v3[:,1])
        a=(v1[:,0])*a_sub
        b=v2[:,0]*(v3[:,1]-v1[:,1])
        c=a+b
        d = (v3[:,0])*(v1[:,1]-v2[:,1])
        e=c+d
           
        triangle_area[i,:] = e*0.5
    return (triangle_area)

def TriangleArea3d(x,faces,is_variable):
    s1=TriangleArea(x[:,:,1:],faces,is_variable)
    if is_variable:
        v=torch.index_select(x[:,:,:],2,torch.LongTensor([2,0]).cuda())
    else:
        v=torch.index_select(x[:,:,:],2,torch.LongTensor([2,0]))

    s2=TriangleArea(v,faces,is_variable)
    s3=TriangleArea(x[:,:,0:2],faces,is_variable)
    triangle_area = 0.5*torch.sqrt(s1**2+s2**2+s3**2)
    return (triangle_area)

#def Circumference(x,faces,is_variable):
#    num_faces = faces.size(1)
#    num_points = x.size(1)
#    result = torch.zeros(x.size(0),1)
#    if is_variable:
#        result = Variable(result)
#    if x.is_cuda:
#        result=result.cuda()
#    adjacency= (torch.zeros(num_points,num_points))
#    minv,_ = torch.kthvalue(faces, 1) #, dim=2)
#    midv,_ = torch.kthvalue(faces, 2) #, dim=2)
#    maxv,_ = torch.kthvalue(faces, 3) #, dim=2)
#    minv = minv.view(x.size(0),num_faces)
#    midv = midv.view(x.size(0),num_faces)
#    maxv = maxv.view(x.size(0),num_faces)
#    for i in range(x.size(0)):
#        for j in range(num_faces):
#          if minv[i,j]!=midv[i,j]:
#        adjacency[minv[i,j],midv[i,j]]=1-adjacency[minv[i,j],midv[i,j]]
#        adjacency[minv[i,j],maxv[i,j]]=1-adjacency[minv[i,j],maxv[i,j]]
#        adjacency[midv[i,j],maxv[i,j]]=1-adjacency[midv[i,j],maxv[i,j]]
#        dis=0
#        for j in range(num_points):
#        dup = torch.LongTensor(num_points).fill_(j)
#        if is_variable:
#            dup = Variable(dup)
#        if x.is_cuda:
#            dup = dup.cuda()
#           s=torch.index_select(x[i,:,:],0, dup)
#        s=s-x[i,:,:]
#        mask=(adjacency[j,:]).view(-1,1)
#        mask=torch.cat((mask,mask),1)
#        if is_variable:
#            mask = Variable(mask)
#        if x.is_cuda:
#            mask=mask.cuda()
#        dis_j=torch.sum(torch.norm((s * mask).add(1e-12),2,1))
#        dis=dis+dis_j
#        result[i,0]=dis
#    return result

#def BoundaryFaceWeight(x,faces, is_variable, a):
#    num_faces = faces.size(1)
#    num_points = x.size(1)
#    weight = torch.FloatTensor(x.size(0),num_faces).fill_(1)
#    if is_variable:
#        weight = Variable(weight)
#    if x.is_cuda:
#        weight=weight.cuda()
#    
#    adjacency= (torch.zeros(num_points,num_points))
#    minv,_ = torch.kthvalue(faces, 1) #, dim=2)
#    midv,_ = torch.kthvalue(faces, 2) #, dim=2)
#    maxv,_ = torch.kthvalue(faces, 3) #, dim=2)
#    minv = minv.view(x.size(0),num_faces)
#    midv = midv.view(x.size(0),num_faces)
#    maxv = maxv.view(x.size(0),num_faces)
#    for i in range(x.size(0)):
#        for j in range(num_faces):
#          if minv[i,j]!=midv[i,j]:
#        adjacency[minv[i,j],midv[i,j]]=1-adjacency[minv[i,j],midv[i,j]]
#        adjacency[minv[i,j],maxv[i,j]]=1-adjacency[minv[i,j],maxv[i,j]]
#        adjacency[midv[i,j],maxv[i,j]]=1-adjacency[midv[i,j],maxv[i,j]]
#        for j in range(num_faces):
#            on_boundary=(adjacency[minv[i,j],midv[i,j]]==1) or (adjacency[minv[i,j],maxv[i,j]]==1) or (adjacency[midv[i,j],maxv[i,j]]==1)
#        if on_boundary:
#            weight[i,j] = a * weight[i,j]
#    return weight

#def BoundaryFaceWeightv2(x,faces,boundary, is_variable, a):
#    num_faces = faces.size(1)
#    num_points = x.size(1)
#    weight = torch.FloatTensor(x.size(0),num_faces).fill_(1)
#    if is_variable:
#        weight = Variable(weight)
#    if x.is_cuda:
#        weight=weight.cuda()
#    
#    mask = boundary.ge(0)
#    for i in range(x.size(0)):
#        b = torch.masked_select(boundary[i,:,:],mask[i,:,:])
#        for j in b :
#        weight[i,:]=weight[i,:] + a * Variable(torch.sum(torch.eq(faces[i,:,:],j),1).float().cuda())
#    return weight

'''
return larger number for those having smaller(>1 if z<0)z and smaller abs(x)+abs(y) (closer to the origin)
'''
def xyzArea(x,faces,base, is_variable): 
    faces=faces.long()
    weight= (torch.zeros(faces.size(0), faces.size(1)))
    v = (torch.zeros(faces.size()))
    if is_variable:
        weight = Variable(weight)
        v = Variable(v)
    if x.is_cuda:
        weight=weight.cuda()
        v=v.cuda()
    for i in range(faces.size(0)):
           v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
           v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
           v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
           v[i,:,:]= (v1+v2+v3)/3
    abxy = torch.abs(v[:,:,:2])
    a = base ** torch.mean(abxy.sum(2),1) 
    a = a.view(x.size(0),1)
    weight = a.expand_as(weight) / (base**abxy.sum(2))*torch.exp(-v[:,:,2])
    return (weight.detach())

def BarrierArea(area, is_variable, inf, s):
    mask1 = area.le(0)
    f1 = inf * mask1.float()
    mask2 = area.data.lt(s) & area.data.gt(0)
    f2 = 1/ ((area.pow(3)/(s**3)-3 * area.pow(2)/(s*s)+3*area/s).add(1e-6)) - 1
    result = f1.data + f2.data * mask2.float()
    if is_variable:
        result = Variable(result)
    if area.is_cuda:
        result=result.cuda()
    return result

def harArea(har,faces,a):
    s=torch.abs(TriangleArea(har,faces,0))
    s[:,0:int(torch.max(faces))] = a / s[:,0:int(torch.max(faces))].add(1e-9)
#    s = torch.FloatTensor(s.size()).fill_(1)+s.le(a)
    return s
def VertexperFace(x,faces):
    l = torch.FloatTensor(faces.size(0),faces.size(1)).fill_(0)
    l = Variable(l.cuda())
    for i in range(faces.size(0)):
           v1=torch.index_select(x[i,:,:],0,faces[i,:,0])
           v2=torch.index_select(x[i,:,:],0,faces[i,:,1])
           v3=torch.index_select(x[i,:,:],0,faces[i,:,2])
    return l


def BoundarySame(x1, x2, b1, b2):
    l = torch.FloatTensor(x1.size(0),1).fill_(0)
    l = Variable(l.cuda())
    mask1 = b1.ge(0)
    mask2 = b2.ge(0)
    for i in range(x1.size(0)):
        v1=torch.index_select(x1[i,:,:],0, Variable(torch.masked_select(b1[i,:,:],mask1[i,:,:]).cuda()))
        v2=torch.index_select(x2[i,:,:],0, Variable(torch.masked_select(b2[i,:,:],mask2[i,:,:]).cuda()))
        l[i] = torch.mean((v1-v2)**2)
    return l
def Margin(area):
    z=Variable(torch.FloatTensor(area.size()).zero_())
    if area.is_cuda:
            z = z.cuda()
    margin=torch.max(z,-area)
    return margin

#class LocalDescriptor(nn.Module):
#    def __init__(self, nnei):
#    super(LocalDescriptor, self).__init__()
#    self.nnei = nnei
#    self.neiconv1 = torch.nn.Conv2d(3,32,(1,1))
#    self.neibn1 = nn.BatchNorm2d(32)
#    self.neiconv2 = torch.nn.Conv2d(32,260,(1,1))
#    self.neibn2 = nn.BatchNorm2d(260)
#    self.neiconv3 = torch.nn.Conv2d(nnei,256,(1,8),stride=(1,4))
#    self.neibn3 = nn.BatchNorm2d(256)
#    self.mp1 = torch.nn.MaxPool2d((1,256))
#    self.neiconv4 = torch.nn.Conv2d(64,32,1)
#    self.neibn4 = nn.BatchNorm2d(32)
#
#    def forward(self,nei):
#    nei = F.relu(self.neibn1(self.neiconv1(nei)))
#    nei = F.relu(self.neibn2(self.neiconv2(nei)))
#    nei = nei.transpose(1,3)
#    nei = F.relu(self.neibn3(self.neiconv3(nei)))
#    nei = nei.transpose(1,3)
#    nei = self.mp1(nei)
#    nei = F.relu(self.neibn4(self.neiconv4(nei)))
#    nei = nei.view(nei.size(0),32,nei.size(2))
#        return nei

class PointNetfeat(nn.Module):
    def __init__(self, dim=3, fnum=0, nnei=0, num_points = 500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(dim+32*nnei,num_points = num_points)
        if nnei!=0:
            self.local = LocalDescriptor(nnei)
#        self.conv1 = torch.nn.Conv1d(dim+32*nnei, 64, 1)
        self.conv1 = torch.nn.Conv1d(dim+32*nnei, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        self.dim=dim
        self.nnei=nnei
        self.fnum=fnum
#        self.gconv = graph_conv(6,16,16*6,32,max_=0)       
#        self.gconv2 = graph_conv(6,32,32*6,64,max_=1)       
#        self.gconv3 = graph_conv(6,64,64*6,128,max_=1)       

    def forward(self, x, f, nei=0):
        batchsize = x.size()[0]
        if self.nnei!=0:
            nei = self.local(nei)
            x=torch.cat((x,nei),1)
        if self.fnum!=0:
            x=torch.cat((x,f),1)
#        if self.dim==3:
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
#        x = (self.bn1(self.conv1(x)))
#        x = x.transpose(1,2)
#        x = self.gconv(x,f)
#        x = self.gconv2(x,f)
#        x = x.transpose(1,2)
        pointfeat = x
#        x = x.transpose(1,2)
#        x = self.gconv3(x,f)
#        x = x.transpose(1,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, pointfeat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1)

class InvMap(nn.Module):
    def __init__(self,b_cls=0, nnei=0, num_points = 500, k = 2,dim=3):
        super(InvMap, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(dim=dim, nnei=nnei, num_points=num_points, global_feat=True)
        self.rep = nn.ReplicationPad2d((2,2,2,2))
        self.repfeat = nn.ReplicationPad2d((1,1,1,1))
        self.upconv1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1) #10
        #self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1) 
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  #20
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) #40
        self.conv0 = torch.nn.Conv2d(256+num_points, 256, 1,padding=0)#(bs,1024,1088)
        self.conv1 = torch.nn.Conv2d(256, 128, (3,3),padding=1)#(bs,1024,1088)
        self.conv2 = torch.nn.Conv2d(64, 32, (3,3),padding=1)
        self.conv3 = torch.nn.Conv2d(16, dim, (3,3),padding=1)
#        self.gconv = graph_conv(6,self.k,32,self.k,max_=0,relu_=0,bn_=0)       
#        self.gconv = graph_conv(6,128,32,self.k,max_=0,relu_=0,bn_=0)       
        self.bn1up = nn.BatchNorm2d(256)
        self.bn0 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2up = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3up = nn.BatchNorm2d(16)
#        self.bn3 = nn.BatchNorm2d(3)
#self.gconv = graph_conv(6,128,128,2,relu_=0,max_=0)       

    def forward(self, x,faces):
        batchsize = x.size()[0]
        x_ori = x
        x,feat = self.feat(x,faces)
        x = x.contiguous().view(batchsize,1024,1,1)
        x = self.rep(x)
        x = F.relu(self.bn1up(self.upconv1(x)))

        feat = feat.transpose(1,2).contiguous().view(batchsize,self.num_points,8,8)
        feat = self.repfeat(feat)
        x = torch.cat((x, feat),1)
        x = F.relu(self.bn0(self.conv0(x)))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2up(self.upconv2(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3up(self.upconv3(x)))
        x = self.conv3(x)
#	print(x.size())
#        x_cls = self.conv5(x)
#        x = x.transpose(2,1)
#        x = self.conv4(x)
#        x = x.transpose(2,1)
#        area = TriangleArea(x,faces,1)
#        cos = Cosine(x,faces,1)
##        xg = self.gconv(x,faces)
##        areag = TriangleArea(xg,faces,1)
##        cosg = Cosine(xg,faces,1)
#        x_cls = x_cls.transpose(2,1).contiguous()
#        x_cls = F.log_softmax(x_cls.view(-1,self.k))
#        x_cls = x_cls.view(batchsize, self.num_points, self.k)
#        areag,cosg,xg=area,cos,x
        
        return x
class PointNetDenseMap(nn.Module):
    def __init__(self,b_cls=0, nnei=0, num_points = 500, k = 2,dim=3):
        super(PointNetDenseMap, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(dim=dim, nnei=nnei, num_points=num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(128, self.k, 1)
#        self.gconv = graph_conv(6,self.k,32,self.k,max_=0,relu_=0,bn_=0)       
#        self.gconv = graph_conv(6,128,32,self.k,max_=0,relu_=0,bn_=0)       
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.b_cls = b_cls
#self.gconv = graph_conv(6,128,128,2,relu_=0,max_=0)       

    def forward(self, x, faces, edges,f,nei):
        batchsize = x.size()[0]
        x_ori = x
        x = self.feat(x,faces, nei)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_cls = self.conv5(x)
#        x = x.transpose(2,1)
        x = self.conv4(x)
        x = x.transpose(2,1)
        area = TriangleArea(x,faces,1)
        cos = Cosine(x,faces,1)
#        xg = self.gconv(x,faces)
#        areag = TriangleArea(xg,faces,1)
#        cosg = Cosine(xg,faces,1)
        x_cls = x_cls.transpose(2,1).contiguous()
        x_cls = F.log_softmax(x_cls.view(-1,self.k))
        x_cls = x_cls.view(batchsize, self.num_points, self.k)
        areag,cosg,xg=area,cos,x
        return area,cos,x,x_cls,xg,areag,cosg

# 128 x 128 transform
class Feats_STN3d(nn.Module):
    # for modelnet40, a 3d shape is with 2048 points 
    def __init__(self, num_points = 2500):    
        super(Feats_STN3d, self).__init__()
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.conv2 = nn.Conv1d(256, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points) 

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128*128)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # bz x 256 x 2048 
        x = F.relu(self.bn2(self.conv2(x))) # bz x 1024 x 2048
        x = self.mp1(x) # bz x 1024 x 1
        x = x.view(-1, 1024)

        x = F.relu(self.bn3(self.fc1(x))) # bz x 512 
        x = F.relu(self.bn4(self.fc2(x))) # bz x 256
        x = self.fc3(x) # bz x (128*128) 
        # identity transform
        # bz x (128*128)
        iden = Variable(torch.from_numpy(np.eye(128).astype(np.float32))).view(1,128*128).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 128, 128) # bz x 3 x 3 
        return x

class PointNetPartDenseCls(nn.Module):
    def __init__(self, dim=3, nnei=0,num_points = 2500, k = 2):
        super(PointNetPartDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.nnei=nnei
        # T1 
        self.stn1 = STN3d(dim=dim+32*nnei,num_points = num_points) # bz x 3 x 3, after transform => bz x 2048 x 3 

        self.conv1 = torch.nn.Conv1d(32*nnei+dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # T2 
        self.stn2 = Feats_STN3d(num_points = num_points)

        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 2048, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(2048)
        # pool layer 
        self.mp1 = torch.nn.MaxPool1d(num_points) 

        # MLP(256, 256, 128)
        self.conv7 = torch.nn.Conv1d(3024-16, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 256, 1)
        self.conv9 = torch.nn.Conv1d(256, 128, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)
        # last layer 
        self.conv10 = torch.nn.Conv1d(128, self.k, 1) # 50 
        self.bn10 = nn.BatchNorm1d(self.k)

    def forward(self, x): #, one_hot_labels):
        batch_size = x.size()[0]
        # T1 
        trans_1 = self.stn1(x) # regressing the transforming parameters using STN
        x = x.transpose(2,1) # bz x 2048 x 3 
        x = torch.bmm(x, trans_1) # (bz x 2048 x 3) x (bz x 3 x 3) 
        # change back 
        x = x.transpose(2,1) # bz x 3 x 2048
        out1 = F.relu(self.bn1(self.conv1(x))) # bz x 64 x 2048
        out2 = F.relu(self.bn2(self.conv2(out1))) # bz x 128 x 2048
        out3 = F.relu(self.bn3(self.conv3(out2))) # bz x 128 x 2048
        #######################################################################
        # T2, currently has bugs so now remove this temporately
        trans_2 = self.stn2(out3) # regressing the transforming parameters using STN
        out3_t = out3.transpose(2,1) # bz x 2048 x 128
        out3_trsf = torch.bmm(out3_t, trans_2) # (bz x 2048 x 128) x (bz x 128 x 3) 
        # change back 
        out3_trsf = out3_trsf.transpose(2,1) # bz x 128 x 2048

        out4 = F.relu(self.bn4(self.conv4(out3_trsf))) # bz x 128 x 2048
        out5 = F.relu(self.bn5(self.conv5(out4))) # bz x 512 x 2048 
        out6 = F.relu(self.bn6(self.conv6(out5))) # bz x 2048 x 2048
        out6 = self.mp1(out6) #  bz x 2048

        # concat out1, out2, ..., out5
        out6 = out6.view(-1, 2048, 1).repeat(1, 1, self.num_points)
        # out6 = x 
        # cetegories is 16
        # one_hot_labels: bz x 16
#        one_hot_labels = one_hot_labels.unsqueeze(2).repeat(1, 1, self.num_points)
        # 64 + 128 * 3 + 512 + 2048 + 16  
        # point_feats = torch.cat([out1, out2, out3, out4, out5, out6, one_hot_labels], 1)
        point_feats = torch.cat([out1, out2, out3, out4, out5, out6], 1)
        # Then feed point_feats to MLP(256, 256, 128) 
        mlp = F.relu(self.bn7(self.conv7(point_feats)))
        mlp = F.relu(self.bn8(self.conv8(mlp)))
        mlp = F.relu(self.bn9(self.conv9(mlp)))

        # last layer 
        pred_out = self.bn10(self.conv10(mlp)) # bz x 50(self.k) x 2048
        pred_out = pred_out.transpose(2,1)#.contiguous()
#        ipred_out = F.log_softmax(pred_out.view(-1,self.k))
        area=0
        cos=0
        return pred_out

'''
## Deprecated
#used for v1 in weekly report2 3d_ellipsoid
class PointNetMap(nn.Module):
    def __init__(self, num_points = 500,dim=3):
        super(PointNetMap, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 2, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2)
        self.num_points = num_points
    def forward(self, x, faces,is_test,s): 
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
    x = x.transpose(2,1)

    area = TriangleArea(x,faces,1)
    side = TriangleSide(x,faces,1)
    z=Variable(torch.FloatTensor(area.size()).zero_())
        if x.is_cuda:
            z = z.cuda()
    margin=torch.max(z,-area)
    margin_sum=torch.mean(margin.sum(1))
    cos = Cosine(x,faces,1)
        return area,side,cos,x,margin_sum



#class PointNetCls(nn.Module):
#    def __init__(self, num_points = 2500, k = 2):
#        super(PointNetCls, self).__init__()
#        self.num_points = num_points
#        self.feat = PointNetfeat(num_points, global_feat=True)
#        self.fc1 = nn.Linear(1024, 512)
#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, k)
#        self.bn1 = nn.BatchNorm1d(512)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.relu = nn.ReLU()
#    def forward(self, x):
#        x, trans = self.feat(x)
#        x = F.relu(self.bn1(self.fc1(x)))
#        x = F.relu(self.bn2(self.fc2(x)))
#        x = self.fc3(x)
#        return F.log_softmax(x), trans

if __name__ == '__main__':

    sim_data = Variable(torch.rand(32,2,50))
    faces=torch.LongTensor([[0,1,2],[4,5,6]])
    print(type(faces))
    pointfeat = PointNetMap()
    out, _ = pointfeat(sim_data,faces)
    print('map', out.size())
    '''
