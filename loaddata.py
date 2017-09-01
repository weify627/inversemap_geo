from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self, root,merge=True,cut =False ,harmonic = True, align=True,mannualalign=True,nneighbor=0, fnum=0,npoints = 1000,nfaces=2000, edge = 0, nedges = 50, classification = False, class_choice = None, train = True):
        self.mannualalign = mannualalign
        self.npoints = npoints
        self.nfaces = nfaces
        self.nedges = nedges
        self.root = root
        self.edge = edge
        self.fnum = fnum
        self.nneighbor = nneighbor
        self.cut = cut
        self.harmonic = harmonic
        self.meta = []
        self.datapath = []
        self.dim=8
        if align:
                dir_point = os.path.join(self.root, 'alignedpoints')
        else:
                dir_point = os.path.join(self.root, 'geodesic')
        if self.cut:
                dir_point = os.path.join(self.root, 'rotate_non0face_points')
#        dir_point = os.path.join(self.root, 'points')
        dir_face = os.path.join(self.root,  'faces')
        dir_edge = os.path.join(self.root,  'edges')
        if fnum!=0:
                dir_f = os.path.join(self.root,  'functionsOnMesh')
        if nneighbor!=0:
                dir_nei = os.path.join(self.root,  'neighbor')
        if harmonic:
                dir_har = os.path.join(self.root, 'harmonic')
                fns = sorted(os.listdir(dir_har))
        else:
                fns = sorted(os.listdir(dir_point))
        dir_bound = os.path.join(self.root,'gbound')
#        if train==1:
#            fns = fns[int(len(fns) * 0.05):] #            fns = fns[:int(len(fns) * 0.9)]
#        elif train==0:
#            fns = fns[:int(len(fns) * 0.1)]  #fns = fns[int(len(fns) * 0.9):]
#        else:
#            fns = fns
        if merge:
                fns = sorted(os.listdir(dir_point))
        count=0
        for fn in fns:
                count=count+1
                if train==False and count%50!=0:
                        continue
                token = (os.path.splitext(os.path.basename(fn))[0])
                if self.cut:
                        tokenf=token[:token.rfind('o')+2]
                else:
                        tokenf=token
                if merge:
                     tokenp=token
                     #token=token[:token.rfind('r')]
                     self.datapath.append((os.path.join(dir_point, tokenp + '.pts'), os.path.join(dir_face, token + '.fcs'), os.path.join(dir_edge, token + '.eds'),os.path.join(dir_bound, tokenp + '.pts')))#,os.path.join(dir_nei,token+'.pts')))
                     self.datapath.append((os.path.join(dir_point.replace("/1/","/2/"), tokenp + '.pts'), os.path.join(dir_face.replace("/1/","/2/"), token + '.fcs'), os.path.join(dir_edge.replace("/1/","/2/"), token + '.eds'),os.path.join(dir_bound.replace("/1/","/2/"), tokenp + '.pts')))#,os.path.join(dir_f, token + '.pts'),os.path.join(dir_nei,token+'.pts')))
        self.root = root.replace("3000","3400")
        if align:
                dir_point = os.path.join(self.root, 'alignedpoints')
        else:
                dir_point = os.path.join(self.root, 'geodesic')
        if self.cut:
                dir_point = os.path.join(self.root, 'rotate_non0face_points')
#        dir_point = os.path.join(self.root, 'points')
        dir_face = os.path.join(self.root,  'faces')
        dir_edge = os.path.join(self.root,  'edges')
        if fnum!=0:
                dir_f = os.path.join(self.root,  'functionsOnMesh')
        if nneighbor!=0:
                dir_nei = os.path.join(self.root,  'neighbor')
        if harmonic:
                dir_har = os.path.join(self.root, 'harmonic')
                fns = sorted(os.listdir(dir_har))
        else:
                fns = sorted(os.listdir(dir_point))
        dir_bound = os.path.join(self.root,'gbound')
#        if train==1:
#            fns = fns[int(len(fns) * 0.05):] #            fns = fns[:int(len(fns) * 0.9)]
#        elif train==0:
#            fns = fns[:int(len(fns) * 0.1)]  #fns = fns[int(len(fns) * 0.9):]
#        else:
#            fns = fns
        if merge:
            fns = sorted(os.listdir(dir_point))
        for fn in fns:
                count=count+1
                if train==False and count%50!=0:
                        continue
                token = (os.path.splitext(os.path.basename(fn))[0])
                if self.cut:
                        tokenf=token[:token.rfind('o')+2]
                else:
                        tokenf=token
                if merge:
                        tokenp=token
                        #token=token[:token.rfind('r')]
                        self.datapath.append((os.path.join(dir_point, tokenp + '.pts'), os.path.join(dir_face, token + '.fcs'), os.path.join(dir_edge, token + '.eds'),os.path.join(dir_bound, tokenp + '.pts')))#,os.path.join(dir_nei,token+'.pts')))
                        self.datapath.append((os.path.join(dir_point.replace("/1/","/2/"), tokenp + '.pts'), os.path.join(dir_face.replace("/1/","/2/"), token + '.fcs'), os.path.join(dir_edge.replace("/1/","/2/"), token + '.eds'),os.path.join(dir_bound.replace("/1/","/2/"), tokenp + '.pts')))#,os.path.join(dir_f, token + '.pts'),os.path.join(dir_nei,token+'.pts')))

    def __getitem__(self, index):
        theta = torch.rand(1)*6.28
#        R = torch.FloatTensor([[torch.cos(theta)[0],-torch.sin(theta)[0]],[torch.sin(theta)[0],torch.cos(theta)[0]]])
        R = torch.eye(8)
        R[0,0]=torch.cos(theta)[0]
        R[0,1]=-torch.sin(theta)[0]
        R[1,0]=torch.sin(theta)[0]
        R[1,1]=torch.cos(theta)[0]

        fn = self.datapath[index]
        pts = torch.FloatTensor(self.npoints,self.dim).fill_(0)   #max of 10000 cases:1028(over_0.8)500,520
        get_pts = np.loadtxt(fn[0]).astype(np.float32)
#        if fn[0].find('/2/')>0:
#               get_pts[:,2]=-get_pts[:,2]
        get_pts = torch.from_numpy(get_pts)
        pts_m = torch.mean(get_pts,0)
        pts_std = torch.std(get_pts,dim=0)
        get_pts=get_pts-pts_m.expand_as(get_pts)
        get_pts=get_pts/pts_std.expand_as(get_pts)
        pts[:get_pts.size(0),:]=get_pts

        faces = torch.LongTensor(self.nfaces,3).fill_(0) #max of 10000 cases: 2052(over_0.8)983, 987
        get_faces = np.loadtxt(fn[1]).astype(np.int64)
        get_faces = torch.from_numpy(get_faces)
        if fn[1].find('/2/')>0:
                faces[:get_faces.size(0),0]=get_faces[:,1]
                faces[:get_faces.size(0),1]=get_faces[:,0]
                faces[:get_faces.size(0),2]=get_faces[:,2]
        else:
                faces[:get_faces.size(0),:]=get_faces
#        faces[:get_faces.size(0),:]=get_faces
        edges = torch.LongTensor(self.nedges,2).fill_(-1)
        get_edges = np.loadtxt(fn[2]).astype(np.int64)
        get_edges = torch.from_numpy(get_edges)
        if self.cut:
                get_edges=get_edges-2
        edges[:get_edges.size(0),:]=get_edges
        if get_edges.size(0)>self.nedges:
                print('edge',get_edges.size(0))
        
        if self.mannualalign:
                tmp=(torch.mean(pts[get_edges.contiguous().view(-1).long()],0))[0,2] #.expand(get_edges.size(0)*2,3)
                for i in range(get_edges.size(0)):
                        pts[get_edges[i,0],2]=tmp
                        pts[get_edges[i,1],2]=tmp

        labels = torch.LongTensor(self.npoints,1).fill_(0)
        for i in range(len(get_edges.view(-1))):
                labels[(get_edges.view(-1))[i]]=1
        f_pts = torch.FloatTensor(self.npoints,self.fnum).fill_(0)
        npts = torch.FloatTensor(3,self.npoints,self.nneighbor).fill_(0)

        bnd = torch.FloatTensor(156,self.dim).fill_(0)   #max of 10000 cases:1028(over_0.8)500,520
        if fn[3].find('test')==-1:
            get_bnd = np.loadtxt(fn[3]).astype(np.float32)
            get_bnd = torch.from_numpy(get_bnd)
            bnd[:get_bnd.size(0),:]=get_bnd
#        if fn[3].find('/2/')!=-1:
#            bnd[:,2]=-bnd[:,2]

        pts = torch.mm(R,pts.transpose(0,1)).transpose(0,1)
        bnd = torch.mm(R,bnd.transpose(0,1)).transpose(0,1)

        Minxyz = torch.min(torch.cat((bnd.sum(1),bnd),1),0)[1][0][0]
        if Minxyz!=0:
                bnd = torch.cat((bnd[Minxyz:,:],bnd[0:Minxyz,:]),0)
        if self.cut or (self.fnum==0 and self.nneighbor==0):
                return pts, faces, edges, f_pts,npts,labels,bnd

        get_f = np.loadtxt(fn[4]).astype(np.float32)
        get_f = torch.from_numpy(get_f)
        if self.fnum!=0:
                f_pts[:get_f.size(0),:]=get_f[:,:self.fnum]

        if self.nneighbor!=0:
            get_npts = np.loadtxt(fn[5]).astype(np.float32)
            get_npts = torch.from_numpy(get_npts)
            for i in range(3):
                    get_npts[i*get_pts.size(0):(i+1)*get_pts.size(0),:]=get_npts[i*get_pts.size(0):(i+1)*get_pts.size(0),:]-pts_m[0,i]  
                    get_npts[i*get_pts.size(0):(i+1)*get_pts.size(0),:]=get_npts[i*get_pts.size(0):(i+1)*get_pts.size(0),:]/pts_std[0,i]
                    npts[i,:get_pts.size(0),:]=get_npts[i*get_pts.size(0):(i+1)*get_pts.size(0),:self.nneighbor]
        
        return pts, faces, edges,har_pts,f_pts,npts,labels

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'data',train=False)
    print(len(d))
    ps, seg = d[1]
    print(ps.size(), ps.type(), seg.size(),seg.type())
    print(ps,seg)
