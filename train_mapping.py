from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from loaddata import PartDataset
from net_mapping import InvMap,harArea, PointNetPartDenseCls, Margin,xyzArea, BarrierArea,  TriangleArea3d,PointNetDenseMap, TriangleArea
from lib_mapping import loss_bnd, loss_dis
import torch.nn.functional as F
from datetime import datetime
iterative=0
target_pts='' #'har'
cut,merge=0,1
w_bnd,w_x,w_img,w_local = 100,10,10,100
fnum,nneighbor = 0,0#3 
nepo=25 #define number of epoch for training
batchsize,lr=40,0.001 #0.001
if cut: data_name,npoints,nfaces='../real/data/real',550,1000
#else: data_name, npoints, nfaces='../real/data/real_half/1',704,1270 #'data/0.6z/'  #define dataset directory
#else: data_name, npoints, nfaces='../real/data/half3400/1',1840,3400 #'data/0.6z/'  #define dataset directory
else: data_name, npoints, nfaces='../../data/half3000/1',1868,3625 #'data/0.6z/'  #define dataset directory
#npoints=550 #550 #704 #693 #1030 real:511-545 nfaces=1000#1000 #1270 #1270 #2052 real:1000
nedges=321#290 #160 #155
res,sign_first,stn = 0,0,1 #1
area_norm = 3.14159
class loss:
    def __init__(self, base='N', basepara = 0, weight=1, w_method='N', weightpara = 0):
        self.base = base
        self.basepara = basepara * ( base !='N')
        self.weight = weight
        self.w_method = w_method
        self.weightpara = weightpara
    def summary(self):
        return self.base+str((self.basepara))+str((self.weight))+self.w_method+str(self.weightpara)
l_area = loss(base='N', basepara=6,weight=[1e12,0*1e4], w_method='N',weightpara=0.001)
l_area2 = loss(base='N', basepara=6,weight=[1e8,0*1e3], w_method='N',weightpara=0.001)
#l_sign = loss(base='margin',basepara=2e5,weight=20, w_method='boundary',weightpara=20) #margin=2e13
l_sign = loss(base='margin',basepara=2e5,weight=0*2e10, w_method='N',weightpara=20000) #weight=2000
l_sign2 = loss(base='margin',basepara=2e5,weight=0*200, w_method='har',weightpara=0.01) #weight=2000
#larger margin loss for centered faces?
w_cos,w_cos2=0*100,0*100 #100 #10
align,mannualalign = 0,0#True, True
s='g_2dgres'+'bnd'+str(w_bnd)+'x'+str(w_x)+'img'+str(w_img)+'local'+str(w_local) ##define key feature for this model
Mean=lambda x: (torch.sum(x)/ float(torch.ne(x,0).sum()))
MeanV=lambda x: (torch.sum(x)/ torch.ne(x.detach(),0).sum().float())
class Loss_compute:
    def __init__(self, epoch, num_batch): # mapper, optimizer,:
        mapper = InvMap(nnei=nneighbor,num_points = opt.num_points,dim=8)
        model=''
#        model='mapping/20170827_225435/mapping_model_g_2dgresbnd10x10img10local10_1.pth'
        if model!='':
            mapper.load_state_dict(torch.load(model))
            print(model+' loaded')
        self.optimizer=optim.Adam(mapper.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.mapper = mapper.cuda()
        self.num_batch = num_batch
        self.epoch = epoch
        self.criterion=torch.nn.MSELoss()
        self.criterion2 = nn.NLLLoss(weight=torch.FloatTensor([0.1,0.9]).cuda()) #0.055,0.945
        self.criterion_L1=nn.L1Loss()
        self.max_e,self.max_f,self.max_p=0,0,0
    def iter_do (self, i, data, train):   #train=1 train,2 test,0,val
        points, target, boundary, f_pts,npts,labels,gt_bnd =data #, harmonic_pts = data    
        points = points.transpose(2,1) 
        points, target,f_pts,npts = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False), Variable(f_pts.cuda()), Variable(npts.cuda())
        pred_img = self.mapper(points,target) #,target,boundary,f_pts, npts)
        l_x,l_img, l_local = loss_dis(points, pred_img,target)
        l_bnd = loss_bnd(pred_img, gt_bnd)

        loss =0
        loss +=  w_bnd*l_bnd + w_x *l_x + w_img * l_img + w_local * l_local

        if  np.isnan(loss.data.sum()):
          print(np.isnan(pred_img.sum()))
        assert  np.isnan(loss.data.sum())==False
        if train ==1:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i%10==0: #(int(num_batch/50)) ==0:
                print('[%d: %d/%d] mapper train loss:%.5f'%(epoch, i, num_batch, loss.data[0]))
                print('      %s: %.10f(weight:%.1f)' %(green('loss_bnd'),l_bnd.data[0],w_bnd))
                print('      %s: %.10f(weight:%.1f)' %(blue('loss_x'),l_x.data[0],w_x))
                print('      %s: %.10f(weight:%.1f)' %(yellow('loss_img'),l_img.data[0],w_img))
                print('      %s: %.10f(weight:%.1f)' %(red('loss_local'),l_local.data[0],w_local))
        return self.mapper,points, pred_img, target,gt_bnd

print(vars(l_area))#.__dict__
print(vars(l_sign))
print(vars(l_area2))#.__dict__
print(vars(l_sign2))
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=batchsize, help='input batch size')
parser.add_argument('--num_points', type=int, default=npoints, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=nepo, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='mapping',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
blue = lambda x:'\033[94m' + x + '\033[0m' 
green = lambda x:'\033[92m' + x + '\033[0m' 
red = lambda x:'\033[91m' + x + '\033[0m' 
yellow = lambda x:'\033[93m' + x + '\033[0m' 

now=datetime.now().strftime('%Y%m%d_%H%M%S')
opt = parser.parse_args()
print('time:%s, s_name:%s, nfaces:%d, data:%s'%(now,s,nfaces,data_name))

opt.manualSeed = random.randint(1, 10000) # fix seed
print(opt,"Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = data_name,merge=merge, cut=cut,harmonic = False,align = align,mannualalign=mannualalign,nneighbor= nneighbor,fnum = fnum,edge =1, nedges = nedges, classification = True, npoints = opt.num_points, nfaces = nfaces)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
if cut:
    data_name=data_name+'/test'
#if merge:
#    data_name=data_name[:-1]+'test/1'
test_dataset = PartDataset(root = data_name,merge=merge, cut=cut,harmonic = False,align = align,mannualalign=mannualalign,nneighbor = nneighbor,fnum = fnum,edge =1, nedges = nedges, classification = True, train = False, npoints = opt.num_points,nfaces = nfaces)
testdataloader = torch.utils.data.DataLoader(test_dataset,batch_size=opt.batchSize,shuffle=False, num_workers=int(opt.workers))
print(len(dataset), len(test_dataset))
try:
    os.makedirs('history/'+s)
except OSError:
    pass
try:
    os.makedirs(opt.outf)
except OSError:
    pass
os.makedirs('mapping/'+now)
os.makedirs('results/'+now)
target_file = open('history/'+s+'/log_'+s+now+'.txt', 'w')
if opt.model != '':
    mapper.load_state_dict(torch.load(opt.model))
    print(opt.model+' loaded')

num_batch = len(dataset)/opt.batchSize
loss_compute = Loss_compute(nepo, num_batch) #mapper, optimizer, n
c=1
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
#     if i<147:
#        continue
        mapper,_,_,_,bnd= loss_compute.iter_do(i, data,1)
##print train & test loss and save #bachisize test result every 100 iterations
        if i %int(num_batch/2) == 0:
            ori_f=open('results/'+now+'/'+s+'_input_'+str(c)+'.txt', 'w')
            map_f=open('results/'+now+'/'+s+'_output_'+str(c)+'.txt', 'w')
            face_f=open('results/'+now+'/'+s+'_face_'+str(c)+'.txt', 'w')
            bnd_f=open('results/'+now+'/'+s+'_bnd_'+str(c)+'.txt', 'w')
            c=c+1
            _, data = enumerate(testdataloader,0).__next__()
            
            _,points, pred_pts,target,bnd = loss_compute.iter_do(i, data,0)
            points = points.transpose(2,1) 
            for i in range(points.size(0)): 
                for j in range(points.size(1)): 
                    for k in range(points.size(2)): 
                        ori_f.write(str(points.data[i,j,k]))
                        ori_f.write(' ')
                    ori_f.write('\n')
                for j in range(pred_pts.size(1)): 
                    for k in range(pred_pts.size(2)): 
                      for l in range(pred_pts.size(3)): 
                        map_f.write(str(pred_pts.data[i,j,k,l]))
                        map_f.write('\n')
                    map_f.write('\n')
                for j in range(target.size(1)): 
                    for k in range(target.size(2)): 
                        face_f.write(str(target.data[i,j,k]))
                        face_f.write(' ')
                    face_f.write('\n')
                for j in range(bnd.size(1)): 
                    for k in range(bnd.size(2)): 
                        bnd_f.write(str(bnd[i,j,k]))
                        bnd_f.write(' ')
                    bnd_f.write('\n')
                  
            ori_f.close()
            map_f.close()
            face_f.close()
            bnd_f.close()
#save trained model for every epoch
    torch.save(mapper.state_dict(), '%s/%s/mapping_model_%s_%d.pth' % (opt.outf,now,s, epoch))

## test the final model
ori_f=open('results/'+now+'/'+s+'_input_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
map_f=open('results/'+now+'/'+s+'_output_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
face_f=open('results/'+now+'/'+s+'_face_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
bnd_f=open('results/'+now+'/'+s+'_bnd_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
printline = ('face saved as %s'%face_f.name)
print(printline)
target_file.write(printline)

for i,data in enumerate(testdataloader,0):
    if i*batchsize>300:
        break
    _,points, pred_pts, target,bnd= loss_compute.iter_do(i, data,2)
    points = points.transpose(2,1) 
    for i in range(points.size(0)): 
        for j in range(points.size(1)): 
            for k in range(points.size(2)): 
                ori_f.write(str(points.data[i,j,k]))
                ori_f.write(' ')
            ori_f.write('\n')
        for j in range(pred_pts.size(1)): 
            for k in range(pred_pts.size(2)): 
              for l in range(pred_pts.size(3)): 
                map_f.write(str(pred_pts.data[i,j,k,l]))
                map_f.write('\n')
            map_f.write('\n')
        for j in range(target.size(1)): 
            for k in range(target.size(2)): 
                face_f.write(str(target.data[i,j,k]))
                face_f.write(' ')
            face_f.write('\n')
        for j in range(bnd.size(1)): 
             for k in range(bnd.size(2)): 
                 bnd_f.write(str(bnd[i,j,k]))
                 bnd_f.write(' ')
             bnd_f.write('\n')
_, data = enumerate(dataloader,0).__next__()

_,points, pred_pts, target,bnd = loss_compute.iter_do(i, data,2)
points = points.transpose(2,1) 
for i in range(points.size(0)): 
    for j in range(points.size(1)): 
        for k in range(points.size(2)): 
            ori_f.write(str(points.data[i,j,k]))
            ori_f.write(' ')
        ori_f.write('\n')
    for j in range(pred_pts.size(1)): 
        for k in range(pred_pts.size(2)): 
          for l in range(pred_pts.size(3)): 
            map_f.write(str(pred_pts.data[i,j,k,l]))
            map_f.write('\n')
        map_f.write('\n')
    for j in range(target.size(1)): 
        for k in range(target.size(2)): 
            face_f.write(str(target.data[i,j,k]))
            face_f.write(' ')
        face_f.write('\n')
    for j in range(bnd.size(1)): 
         for k in range(bnd.size(2)): 
             bnd_f.write(str(bnd[i,j,k]))
             bnd_f.write(' ')
         bnd_f.write('\n')
ori_f.close()
map_f.close()
face_f.close()
bnd_f.close()
printline = ('input saved as %s'% ori_f.name)
print(printline)
target_file.write(printline)
printline = ('output saved as %s' % map_f.name)
print(printline)


target_file.close()
