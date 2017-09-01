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
from net_mapping import PointNetPartDenseCls, Margin,xyzArea, BarrierArea, BoundaryFaceWeightv2, BoundaryFaceWeight, TriangleArea3d,PointNetDenseMap, PointNetMap, TriangleArea, TriangleSide,Cosine
import torch.nn.functional as F
from datetime import datetime
dim=3
fnum=0 #15
w_harmonic =100000
nneighbor = 0#3 
nepo=25 #define number of epoch for training
batchsize=4
data_name='../real/data/real_half/1' #'data/0.6z/'  #define dataset directory
npoints=704 #693 #1030 real:511-545
nfaces=1270 #1270 #2052 real:1000
nedges=160 #155
res = 0
sign_first = 0 #1
class loss:
    def __init__(self, base='N', basepara = 0, weight=1, w_method='N', weightpara = 0):
	self.base = base
	self.basepara = basepara * ( base !='N')
	self.weight = weight
	self.w_method = w_method
	self.weightpara = weightpara
    def summary(self):
	return self.base+str((self.basepara))+str(int(self.weight))+self.w_method+str(self.weightpara)
#l_area = loss(base='xyz', basepara=4,weight=1e9, w_method='boundary',weightpara=15)
#l_sign = loss(base='barrier',basepara=2e5,weight=2e3, w_method='boundary') #margin=2e13
#l_area = loss(base='xyz', basepara=4,weight=0, w_method='boundary',weightpara=15)
#l_sign = loss(base='barrier',basepara=2e5,weight=2e9, w_method='boundary') #margin=2e13
l_area = loss(base='xyz', basepara=2,weight=1e9, w_method='boundary',weightpara=20)
l_sign = loss(base='margin',basepara=2e5,weight=20, w_method='boundary',weightpara=20) #margin=2e13
w_cos=100000 #10
align,mannualalign = True, True
s='stn2har_signfirst'+str(sign_first)+l_area.summary()+'_'+l_sign.summary()+'cos'+str(w_cos)  #+weighteddiff+str(weightco)+weightedarea+str(base)+str(w_area)+'_margin'+str(w_marginarea)+'barrier'+str(w_barrierarea)+'area_cos'+str(w_cos) #'0.6z_area' ##define key feature for this model
Mean=lambda x: (torch.sum(x)/ float(torch.ne(x,0).sum()))
MeanV=lambda x: (torch.sum(x)/ torch.ne(x.detach(),0).sum().float())
class Loss_compute:
    def __init__(self, epoch, num_batch): # mapper, optimizer,:
	mapper = PointNetDenseMap(res=res,nnei=nneighbor,num_points = opt.num_points,dim=3+fnum)
#	mapper = PointNetPartDenseCls(nnei=nneighbor,num_points = opt.num_points,dim=3+fnum)
	model=''
#	model = 'mapping/20170810_004615/mapping_model_har_signfirst0xyz21000000000boundary20_margin200000.020boundary20cos100000_9.pth'
	if model!='':
    		mapper.load_state_dict(torch.load(model))
		print(model+' loaded')
	optimizer=optim.Adam(mapper.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	mapper.cuda()
	self.mapper = mapper
	self.optimizer = optimizer
	self.num_batch = num_batch
	self.epoch = epoch
	self.criterion=torch.nn.MSELoss()

    def iter_do (self, i, data, train):   #train=1 train,2 test,0,val
        points, target, boundary, har_pts, f_pts,npts =data #, harmonic_pts = data	
#	points = torch.cat((points,f_pts[:,:,:fnum]),2)
        points, target,f_pts,npts = Variable(points.cuda()), Variable(target.cuda(),requires_grad=False), Variable(f_pts.cuda()), Variable(npts.cuda())
        points = points.transpose(2,1) 
	if fnum !=0:
            f_pts = f_pts.transpose(2,1) 
#        pred_area,pred_cos, pred_pts = self.mapper(points) #,target,boundary,f_pts, npts)
        pred_pts = self.mapper(points) #,target,boundary,f_pts, npts)
        if train == 2:
 	    return mapper, points, pred_pts, target,har_pts,f_pts 
        loss_harmonic = w_harmonic*self.criterion(pred_pts,Variable(har_pts.cuda(),requires_grad=False))
       	
### area
#	target_area = TriangleArea3d(points.data.transpose(2,1),target.data,0)
##(signed area)
#	if(l_area.base=='N'):
#		area = pred_area
#	elif(l_area.base=='xyz'): 
#		area = pred_area / xyzArea(points.transpose(2,1), target,l_area.basepara, 1)
#	else:
#		print('Unknown')
#	area_mult = Variable(torch.FloatTensor(pred_area.size()).fill_(0).cuda())
#	if(l_area.w_method=='N'):
#		weight_face = Variable(torch.FloatTensor(pred_area.size()).fill_(0).cuda())
#	elif(l_area.w_method=='boundary'):
#		weight_face = BoundaryFaceWeightv2(points.transpose(2,1),target.data.cpu(),boundary,1,l_area.weightpara)
#	elif(l_area.w_method=='xyz'):
#		weight_face = xyzArea(points.transpose(2,1), target,l_area.basepara, 1)
#	else:
#		print('Unknown')
#	ab=(torch.abs(Variable(target_area.cuda(),requires_grad=False)-area)) #**2 # not squared as MSE
#	area_mult = torch.addcmul(area_mult, 1,weight_face,ab )
## sign of area
#	if(l_sign.base=='N'):
#		sign = Variable(torch.FloatTensor(pred_area.size()).fill_(0).cuda())
#	elif(l_sign.base=='barrier'):
#		sign = BarrierArea(pred_area, 1, l_sign.basepara, 0.001*Mean(torch.abs(target_area)))	
#	elif(l_sign.base=='margin'):
#		sign = Margin(pred_area)
#	else:
#		print('Unknown')   #can add specific weight def for area sign
#	if(l_sign.w_method==l_area.w_method and l_sign.weightpara==l_area.weightpara):
#		weight_sign=weight_face
#	elif(l_sign.w_method=='N'):
#		weight_sign = Variable(torch.FloatTensor(pred_area.size()).fill_(1).cuda())
#	elif(l_sign.w_method=='boundary'):
#		weight_sign = BoundaryFaceWeightv2(points.transpose(2,1),target.data.cpu(),boundary,1,l_sign.weightpara)
#	elif(l_sign.w_method=='xyz'):
#		weight_sign = xyzArea(points.transpose(2,1), target,l_sign.basepara, 1)
#	else:
#		print('Unknown')
#	sign_mult = Variable(torch.FloatTensor(pred_area.size()).fill_(0).cuda())
#	sign_mult = torch.addcmul(sign_mult, 1,weight_sign, sign)
#
#	loss_sign = l_sign.weight * torch.mean(sign_mult) #MeanV(sign_mult) #torch.sum()
#	if sign_first:
#		mask = sign.le(0)
#		area_mult = torch.masked_select(area_mult,mask)
#	loss_area = l_area.weight * torch.mean(area_mult) #MeanV(area_mult)
#
##	pred_side = TriangleSide(points.data.transpose(2,1), pred_area.data,0)
##	target_side=TriangleSide(points.data.transpose(2,1),target.data,0)
##	loss_side = criterion(pred_side,Variable(target_side,requires_grad=False))
###cosine
#	target_cos=Cosine(points.data.transpose(2,1),target.data,0)
#	loss_cos = w_cos*self.criterion(pred_cos,Variable(target_cos,requires_grad=False))
#
#	loss = loss_area + loss_sign + loss_cos# + loss_harmonic# +loss_side
	loss = loss_harmonic
	if train ==1:
        	self.optimizer.zero_grad()
	        loss.backward()
	        self.optimizer.step()
		if i%700 ==0:
			print('[%d: %d/%d] train loss:%.5f'%(epoch, i, num_batch, loss.data[0]))
			print('harmonic loss:%.10f(weight:%.2f)'%(loss_harmonic.data[0],w_harmonic))
#			print('      %s: %.1f(weight:%.1f,basemethod:%s, w_method:%s),%s: %.5f(weight:%.1f), %s: %.1f(weight:%.1f, basemethod:%s)' %(blue('loss_area'),loss_area.data[0],l_area.weight,l_area.base,l_area.w_method, green('loss_cos'),loss_cos.data[0],w_cos,yellow('loss_sign'), loss_sign.data[0],l_sign.weight, l_sign.base ))
	return self.mapper,points, pred_pts, target,har_pts, f_pts 

print(vars(l_area))#.__dict__
print(vars(l_sign))
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

dataset = PartDataset(root = data_name, harmonic = True,align = align,mannualalign=mannualalign,nneighbor= nneighbor,fnum = fnum,edge =1, nedges = nedges, classification = True, npoints = opt.num_points, nfaces = nfaces)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataset = PartDataset(root = data_name, harmonic = True,align = align,mannualalign=mannualalign,nneighbor = nneighbor,fnum = fnum,edge =1, nedges = nedges, classification = True, train = False, npoints = opt.num_points,nfaces = nfaces)
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
	 mapper,_,_,_,_,_ = loss_compute.iter_do(i, data,1)
##print train & test loss and save #bachisize test result every 100 iterations
         if i % 100 == 0:
	   	ori_f=open('results/'+now+'/'+s+'_input_'+str(c)+'.txt', 'w')
	   	map_f=open('results/'+now+'/'+s+'_output_'+str(c)+'.txt', 'w')
	   	face_f=open('results/'+now+'/'+s+'_face_'+str(c)+'.txt', 'w')
	   	har_f=open('results/'+now+'/'+s+'_har_'+str(c)+'.txt', 'w')
	   	c=c+1
	   	j, data = enumerate(testdataloader,0).next()
	 	_,points, pred_pts, target,har_pts,f_pts = loss_compute.iter_do(i, data,0)
	   	points = points.transpose(2,1) 
		for i in range(points.size(0)): 
			for j in range(points.size(1)): 
				for k in range(points.size(2)): 
					ori_f.write(str(points.data[i,j,k]))
					ori_f.write(' ')
				ori_f.write('\n')
				for k in range(pred_pts.size(2)): 
					map_f.write(str(pred_pts.data[i,j,k]))
					map_f.write(' ')
				map_f.write('\n')
				for k in range(har_pts.size(2)): 
					har_f.write(str(har_pts[i,j,k]))
					har_f.write(' ')
				har_f.write('\n')
			for j in range(target.size(1)): 
				for k in range(target.size(2)): 
					face_f.write(str(target.data[i,j,k]))
					face_f.write(' ')
				face_f.write('\n')
	   	  	
		ori_f.close()
	   	map_f.close()
	   	face_f.close()
		har_f.close()
#save trained model for every epoch
    torch.save(mapper.state_dict(), '%s/%s/mapping_model_%s_%d.pth' % (opt.outf,now,s, epoch))

## test the final model
ori_f=open('results/'+now+'/'+s+'_input_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
map_f=open('results/'+now+'/'+s+'_output_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
face_f=open('results/'+now+'/'+s+'_face_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')
har_f=open('results/'+now+'/'+s+'_har_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w')

for _,data in enumerate(testdataloader,0):
	_,points, pred_pts, target,har_pts,f_pts = loss_compute.iter_do(i, data,2)
	points = points.transpose(2,1) 
	for i in range(points.size(0)): 
		for j in range(points.size(1)): 
			for k in range(points.size(2)): 
				ori_f.write(str(points.data[i,j,k]))
				ori_f.write(' ')
			ori_f.write('\n')
			for k in range(pred_pts.size(2)): 
				map_f.write(str(pred_pts.data[i,j,k]))
				map_f.write(' ')
			map_f.write('\n')
			for k in range(har_pts.size(2)): 
				har_f.write(str(har_pts[i,j,k]))
				har_f.write(' ')
			har_f.write('\n')
		for j in range(target.size(1)): 
			for k in range(target.size(2)): 
				face_f.write(str(target.data[i,j,k]))
				face_f.write(' ')
			face_f.write('\n')
ori_f.close()
map_f.close()
face_f.close()
har_f.close()
printline = ('input saved as %s'% ori_f.name)
print(printline)
target_file.write(printline)
printline = ('output saved as %s' % map_f.name)
print(printline)
target_file.write(printline)
printline = ('face saved as %s'%face_f.name)
print(printline)
target_file.write(printline)


target_file.close()
