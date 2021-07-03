#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,datasets
import cv2
import os


# In[2]:


path=f'Data/augm/0'
myPicList= os.listdir(path)
img_0=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_0.append(img)
img_0 = np.array(img_0)


# In[3]:


np.random.shuffle(img_0)


# In[4]:


path=f'Data/augm/1'
myPicList= os.listdir(path)
img_1=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_1.append(img)
img_1 = np.array(img_1)
np.random.shuffle(img_1)


# In[5]:


path=f'Data/augm/2'
myPicList= os.listdir(path)
img_2=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_2.append(img)
img_2 = np.array(img_2)
np.random.shuffle(img_2)


# In[6]:


path=f'Data/augm/3'
myPicList= os.listdir(path)
img_3=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_3.append(img)
img_3 = np.array(img_3)
np.random.shuffle(img_3)


# In[7]:


path=f'Data/augm/4'
myPicList= os.listdir(path)
img_4=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_4.append(img)
img_4 = np.array(img_4)
np.random.shuffle(img_4)


# In[8]:


path=f'Data/augm/5'
myPicList= os.listdir(path)
img_5=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_5.append(img)
img_5 = np.array(img_5)
np.random.shuffle(img_5)


# In[9]:


path=f'Data/augm/6'
myPicList= os.listdir(path)
img_6=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_6.append(img)
img_6 = np.array(img_6)
np.random.shuffle(img_6)


# In[10]:


path=f'Data/augm/7'
myPicList= os.listdir(path)
img_7=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_7.append(img)
img_7 = np.array(img_7)
np.random.shuffle(img_7)


# In[11]:


path=f'Data/augm/8'
myPicList= os.listdir(path)
img_8=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_8.append(img)
img_8 = np.array(img_8)
np.random.shuffle(img_8)


# In[12]:


path=f'Data/augm/9'
myPicList= os.listdir(path)
img_9=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_9.append(img)
img_9 = np.array(img_9)
np.random.shuffle(img_9)


# ## 10 = +
# ## 11 = -
# ## 12 = x
# ## 13 = /

# In[13]:


path=f'Data/augm/+'
myPicList= os.listdir(path)
img_10=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_10.append(img)
img_10 = np.array(img_10)
np.random.shuffle(img_10)


# In[14]:


path=f'Data/augm/-'
myPicList= os.listdir(path)
img_11=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_11.append(img)
img_11 = np.array(img_11)
np.random.shuffle(img_11)


# In[15]:


path=f'Data/augm/mul'
myPicList= os.listdir(path)
img_12=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_12.append(img)
img_12 = np.array(img_12)
np.random.shuffle(img_12)


# In[16]:


path=f'Data/augm/div'
myPicList= os.listdir(path)
img_13=[]
for y in (myPicList):
    img=cv2.imread(path + str('/')+y,0)
    img = cv2.resize(img, (50,50))
    img_13.append(img)
img_13 = np.array(img_13)
np.random.shuffle(img_13)


# In[17]:


def image(n):
    if n==0:
        return img_0[:22000]
    if n==1:
        return img_1[:22000]
    if n==2:
        return img_2[:22000]
    if n==3:
        return img_3[:22000]
    if n==4:
        return img_4[:22000]
    if n==5:
        return img_5[:22000]
    if n==6:
        return img_6[:22000]
    if n==7:
        return img_7[:22000]
    if n==8:
        return img_8[:22000]
    if n==9:
        return img_9[:22000]
    if n==10:
        return img_10[:22000]
    if n==11:
        return img_11[:22000]
    if n==12:
        return img_12[:22000]
    if n==13:
        return img_13[:22000]
    
def image_test(n):
    if n==0:
        return img_0[22000:]
    if n==1:
        return img_1[22000:]
    if n==2:
        return img_2[22000:]
    if n==3:
        return img_3[22000:]
    if n==4:
        return img_4[22000:]
    if n==5:
        return img_5[22000:]
    if n==6:
        return img_6[22000:]
    if n==7:
        return img_7[22000:]
    if n==8:
        return img_8[22000:]
    if n==9:
        return img_9[22000:]
    if n==10:
        return img_10[22000:]
    if n==11:
        return img_11[22000:]
    if n==12:
        return img_12[22000:]
    if n==13:
        return img_13[22000:]


# In[18]:


img_train = image(0).copy()
img_test = image_test(0).copy()


# In[19]:


for i in range(1,14):
    img_train = np.concatenate((img_train,image(i)))
    img_test = np.concatenate((img_test,image_test(i)))


# In[20]:


img_train.shape
img_test.shape


# In[21]:


def image_label(n):
    return n*np.ones((1,22000),dtype="uint8")
def image_label_t(n):
    return n*np.ones((1,2200),dtype="uint8")


# In[22]:


img_train_label = image_label(0).copy()
img_test_label = image_label_t(0).copy()
for i in range(1,14):
    img_train_label = np.append(img_train_label,image_label(i))
    img_test_label = np.append(img_test_label,image_label_t(i))


# In[23]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[24]:


img_train,img_train_label = shuffle_in_unison(img_train,img_train_label)
img_test,img_test_label = shuffle_in_unison(img_test,img_test_label)


# In[25]:


del img_0
del img_1
del img_2
del img_3
del img_4
del img_5
del img_6
del img_7
del img_8
del img_9
del img_10
del img_11
del img_12
del img_13
import gc
gc.collect()


# In[26]:


import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[62]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) 
        self.fc2 = nn.Linear(512, 14) 

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
        x = x.view(-1, self._to_linear) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = Net()
print(net)


# In[28]:


img_train_label = np.array([np.eye(14)[img_train_label[i]] for i in range(len(img_train_label))])
img_test_label = np.array([np.eye(14)[img_test_label[i]] for i in range(len(img_test_label))])


# In[29]:


img_train_tensor,img_train_label = torch.from_numpy(img_train)/255,torch.from_numpy(img_train_label)
img_test_tensor,img_test_label = torch.from_numpy(img_test)/255,torch.from_numpy(img_test_label)
# img_train_tensor = img_train_tensor.to(device)
# img_train_label = img_train_label.to(device)
# img_test_tensor = img_test_tensor.to(device)
# img_test_label = img_test_label.to(device)


# In[63]:


net.to(device)
BATCH_SIZE = 112
EPOCHS = 3
import torch.optim as optim
from tqdm import tqdm
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(img_train_tensor), BATCH_SIZE)): 
        batch_X=img_train_tensor[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_Y=img_train_label[i:i+BATCH_SIZE]
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_Y.float())
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")


# In[77]:


from tqdm import tqdm
correct = 0
total = 0
predicted = []
with torch.no_grad():
    for i in tqdm(range(len(img_test_tensor))):
        real_class = img_test_label[i].to(device)
        real_class = torch.argmax(real_class)
        net_out = net(img_test_tensor[i].to(device).view(-1, 1, 50, 50))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)
        predicted.append(predicted_class)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))


# In[32]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage


# In[33]:


# path=f'SoML-50/data'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     try:
#         img=cv2.imread(path + str('/')+y,0)
#         images.append(img)
#     except:
#         pass
# images = np.array(images)
# final=[]
# #array_images.shape[0]
# for z in range (images.shape[0]):
#     new=[]
#     new.append(images[z][0:images.shape[1] , 0: int(images.shape[2]/3)])
#     new.append(images[z][0:images.shape[1] , int(images.shape[2]/3) : int(2*int(images.shape[2]/3))])
#     new.append(images[z][0:images.shape[1] , int(2*int(images.shape[2]/3)): images.shape[2]])
#     final.append(new)
# final=np.array(final)
# plt.imshow(final[1][1])


# In[34]:


# for i in range(len(final)):
#     for j in range(3):
#         cv2.imwrite(f'SoML-50/final/final_{i}{j}.jpg',final[i][j])


# In[35]:


prob = []
for i in range(50000):
    for j in range(3):
        im = cv2.imread(f'SoML-50/final/final_{i}{j}.jpg',0)
        im = cv2.resize(im,(50,50))
        prob.append(im)
prob = torch.Tensor(prob)
prob = (prob/255)


# In[64]:


from tqdm import tqdm
correct = 0
total = 0
predicted = []
with torch.no_grad():
    for i in tqdm(range(len(prob))):
        net_out = net(prob[i].to(device).view(-1, 1, 50, 50))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)
        predicted.append(predicted_class)


# In[65]:


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


# In[66]:


outputs = []
outlab = []
for i in range(0,len(predicted),3):
    ans_label,answer = arth(predicted[i:i+3])
    outputs.append(answer)
    outlab.append(ans_label)


# In[67]:


def num_pic(str):
    num = ""
    for i in range(0,len(str)):
        if str[i] == ".":
            return int(num)
            break
        else:
            num += str[i]


# In[68]:


import pandas as pd
data = pd.read_csv("SoML-50/annotations.csv")


# In[69]:


ans = data['Value']


# In[70]:


path=f'SoML-50/data'
myPicList= os.listdir(path)
correct = 0
total = 0
for i in range(len(outputs)):
    # output i corr to myPicList i
    predicted_class = outputs[i]
    pic = num_pic(myPicList[i])
    real_class = ans[pic-1]
    if predicted_class == real_class:
        correct += 1
    total += 1


# In[71]:


correct/total


# In[72]:


ans_l = data['Label']


# In[73]:


path=f'SoML-50/data'
myPicList= os.listdir(path)
correct_l = 0
total_l = 0
for i in range(len(outlab)):
    # output i corr to myPicList i
    predicted_class = outlab[i]
    pic = num_pic(myPicList[i])
    real_class = ans_l[pic-1]
    if predicted_class == real_class:
        correct_l += 1
    total_l += 1


# In[74]:


correct_l/total_l


# In[75]:


# PATH = "try2.pth"
#
#
# In[76]:
#
#
# torch.save(net.state_dict(), PATH)


# In[ ]:




