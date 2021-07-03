#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys

# In[2]:


class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 14) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


net = Net()
print(net)


# In[3]:


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.load_state_dict(torch.load("try2.pth"))
net.eval()
net.to(device)


# In[4]:


import numpy as np
path=sys.argv[1]
myPicList= os.listdir(path)
images=[]
for y in (myPicList):
    try:
        img=cv2.imread(path + str('/')+y,0)
        img = cv2.resize(img,(150,50))
        images.append(img)
    except:
        pass
images = np.array(images)
predicted = []
for z in range (images.shape[0]):
    new=[]
    new.append(images[z][0:images.shape[1] , 0: int(images.shape[2]/3)])
    new.append(images[z][0:images.shape[1] , int(images.shape[2]/3) : int(2*int(images.shape[2]/3))])
    new.append(images[z][0:images.shape[1] , int(2*int(images.shape[2]/3)): images.shape[2]])
    new = torch.Tensor(new)
    new = (new/255)
    with torch.no_grad():
        for i in tqdm(range(len(new))):
            net_out = net(new[i].view(-1,1,50,50).to(device))[0] 
            predicted_class = torch.argmax(net_out)
            predicted.append(predicted_class)


# In[5]:


def arth(l):
    try:
        if l[0].item()>9:
            if l[0].item()== 10:
                return ("prefix",l[1].item()+l[2].item())
            if l[0].item()== 11:
                return ("prefix",l[1].item()-l[2].item())
            if l[0].item()== 12:
                return ("prefix",l[1].item()*l[2].item())
            if l[0].item()== 13:
                return ("prefix",l[1].item()/l[2].item())
        elif l[1].item()>9:
            if l[1].item()== 10:
                return ("infix",l[0].item()+l[2].item())
            if l[1].item()== 11:
                return ("infix",l[0].item()-l[2].item())
            if l[1].item()== 12:
                return ("infix",l[0].item()*l[2].item())
            if l[1].item()== 13:
                return ("infix",l[0].item()/l[2].item())
        elif l[2].item()>9:
            if l[2].item()== 10:
                return ("postfix",l[0].item()+l[1].item())
            if l[2].item()== 11:
                return ("postfix",l[0].item()-l[1].item())
            if l[2].item()== 12:
                return ("postfix",l[0].item()*l[1].item())
            if l[2].item()== 13:
                return ("postfix",l[0].item()/l[1].item())
        else:
            return ("dk",-100)
    except:
        if l[0].item()>9:
                return ("prefix",-100)
        elif l[1].item()>9:
            return ("infix",-100)
        elif l[2].item()>9:
            return ("postfix",-100)
        else:
            return ("dk",-100)


# In[6]:


outputs = []
outlab = []
for i in range(0,len(predicted),3):
    ans_label,answer = arth(predicted[i:i+3])
    answer = int(answer)
    outputs.append(answer)
    outlab.append(ans_label)


# In[7]:




# In[22]:


import pandas as pd
df = pd.DataFrame(outputs, index =myPicList,columns =['Result'])
df.to_csv('Operation_HAY_2.csv')
print("csv file is created")

# In[ ]:




