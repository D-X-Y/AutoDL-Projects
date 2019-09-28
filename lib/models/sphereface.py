# SphereFace: Deep Hypersphere Embedding for Face Recognition
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def myphi(x,m):
  x = x * m
  return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
      x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
  def __init__(self, in_features, out_features, m = 4, phiflag=True):
    super(AngleLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.Tensor(in_features,out_features))
    self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
    self.phiflag = phiflag
    self.m = m
    self.mlambda = [
      lambda x: x**0,
      lambda x: x**1,
      lambda x: 2*x**2-1,
      lambda x: 4*x**3-3*x,
      lambda x: 8*x**4-8*x**2+1,
      lambda x: 16*x**5-20*x**3+5*x
    ]

  def forward(self, input):
    x = input   # size=(B,F)  F is feature len
    w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

    ww = w.renorm(2,1,1e-5).mul(1e5)
    xlen = x.pow(2).sum(1).pow(0.5) # size=B
    wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

    cos_theta = x.mm(ww) # size=(B,Classnum)
    cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
    cos_theta = cos_theta.clamp(-1,1)

    if self.phiflag:
      cos_m_theta = self.mlambda[self.m](cos_theta)
      with torch.no_grad():
        theta = cos_theta.acos()
      k = (self.m*theta/3.14159265).floor()
      n_one = k*0.0 - 1
      phi_theta = (n_one**k) * cos_m_theta - 2*k
    else:
      theta = cos_theta.acos()
      phi_theta = myphi(theta,self.m)
      phi_theta = phi_theta.clamp(-1*self.m,1)

    cos_theta = cos_theta * xlen.view(-1,1)
    phi_theta = phi_theta * xlen.view(-1,1)
    output = (cos_theta,phi_theta)
    return output # size=(B,Classnum,2)


class SphereFace20(nn.Module):
  def __init__(self, classnum=10574):
    super(SphereFace20, self).__init__()
    self.classnum = classnum
    #input = B*3*112*96
    self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
    self.relu1_1 = nn.PReLU(64)
    self.conv1_2 = nn.Conv2d(64,64,3,1,1)
    self.relu1_2 = nn.PReLU(64)
    self.conv1_3 = nn.Conv2d(64,64,3,1,1)
    self.relu1_3 = nn.PReLU(64)

    self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
    self.relu2_1 = nn.PReLU(128)
    self.conv2_2 = nn.Conv2d(128,128,3,1,1)
    self.relu2_2 = nn.PReLU(128)
    self.conv2_3 = nn.Conv2d(128,128,3,1,1)
    self.relu2_3 = nn.PReLU(128)

    self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
    self.relu2_4 = nn.PReLU(128)
    self.conv2_5 = nn.Conv2d(128,128,3,1,1)
    self.relu2_5 = nn.PReLU(128)


    self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
    self.relu3_1 = nn.PReLU(256)
    self.conv3_2 = nn.Conv2d(256,256,3,1,1)
    self.relu3_2 = nn.PReLU(256)
    self.conv3_3 = nn.Conv2d(256,256,3,1,1)
    self.relu3_3 = nn.PReLU(256)

    self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
    self.relu3_4 = nn.PReLU(256)
    self.conv3_5 = nn.Conv2d(256,256,3,1,1)
    self.relu3_5 = nn.PReLU(256)

    self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
    self.relu3_6 = nn.PReLU(256)
    self.conv3_7 = nn.Conv2d(256,256,3,1,1)
    self.relu3_7 = nn.PReLU(256)

    self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
    self.relu3_8 = nn.PReLU(256)
    self.conv3_9 = nn.Conv2d(256,256,3,1,1)
    self.relu3_9 = nn.PReLU(256)

    self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
    self.relu4_1 = nn.PReLU(512)
    self.conv4_2 = nn.Conv2d(512,512,3,1,1)
    self.relu4_2 = nn.PReLU(512)
    self.conv4_3 = nn.Conv2d(512,512,3,1,1)
    self.relu4_3 = nn.PReLU(512)

    self.fc5 = nn.Linear(512*7*6,512)
    self.fc6 = AngleLinear(512, self.classnum)


  def forward(self, x):
    x = self.relu1_1(self.conv1_1(x))
    x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

    x = self.relu2_1(self.conv2_1(x))
    x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
    x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

    x = self.relu3_1(self.conv3_1(x))
    x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
    x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
    x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
    x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

    x = self.relu4_1(self.conv4_1(x))
    x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

    x = x.view(x.size(0),-1)
    features = self.fc5(x)
    logits   = self.fc6(features)
    return features, logits
