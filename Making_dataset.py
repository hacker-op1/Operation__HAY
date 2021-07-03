#!/usr/bin/env python
# coding: utf-8

# ## Please dont run this code as data is already created

# In[39]:


# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy import ndimage



# def rotation(image,angle):
#     if (image is not None):
#         a = image.copy()
#         a = ndimage.rotate(a, angle,cval=255)
#         return a

# path=f'SoML-50/fakedata'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)  
#     images.append(img)
# array_images = np.array(images)
    
# final=[]
# for z in range (array_images.shape[0]):
#     new=[]
#     new.append(images[z][0:array_images.shape[1] , 0: int(array_images.shape[2]/3)])
#     new.append(images[z][0:array_images.shape[1] , int(array_images.shape[2]/3) : int(2*int(array_images.shape[2]/3))])
#     new.append(images[z][0:array_images.shape[1] , int(2*int(array_images.shape[2]/3)): array_images.shape[2]])
#     final.append(new)
# final=np.array(final)

# for i in range(len(final)):
#     for j in range(3):
#         cv2.imwrite(f'Data/orig/mul/mul_{i+1}{j}.jpg',final[i][j])

# def move_left(img,x):
#     if x>=0:
#         a = img.copy()
#         for j in range(len(a)):
#             a[j] = np.append(a[j][x:],255*np.ones((1,x),dtype='uint'))
#         return a
#     else:
#         a = img.copy()
#         for j in range(len(a)):
#             a[j] = np.append(255*np.ones((1,-1*x),dtype='uint'),a[j][:x])
#         return a
    
# def move_up(img,x):
#     if x>=0:
#         a = img.copy()
#         n = len(a[0])
#         for i in range(x):
#             a = np.append(a[1:],255*np.ones((1,n),dtype='uint8'),0)
#         return a
#     else:
#         x = -1*x
#         a = img.copy()
#         n = len(a[0])
#         for i in range(x):
#             a = np.append(255*np.ones((1,n),dtype='uint8'),a[:-1],0)
#         return a


# In[35]:


# for l in range(10):
#     path=f'Data/orig/{l}'
#     myPicList= os.listdir(path)
#     images=[]
#     for y in (myPicList):
#         img=cv2.imread(path + str('/')+y,0)
#         images.append(img)
#     for k in range(len(images)):
#         for j in range(-5,6,10):
#             mov_img = move_up(images[k],j)
#             for i in range(-20,21,4):
#                 rotated_im = rotation(mov_img,i)
#                 cv2.imwrite(f'Data/augm/{l}/{l}_{k+1}_movup{j}_angle{i}.jpeg',rotated_im)


# In[41]:


# # for l in range(10):
# path=f'Data/orig/div'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)
#     images.append(img)
# for k in range(len(images)):
#     for j in range(-5,6,10):
#         mov_img = move_up(images[k],j)
#         for i in range(-20,21,4):
#             rotated_im = rotation(mov_img,i)
#             cv2.imwrite(f'Data/augm/div/div_{k+1}_movup{j}_angle{i}.jpeg',rotated_im)


# In[42]:


# path=f'Data/orig/mul'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)
#     images.append(img)
# for k in range(len(images)):
#     for j in range(-16,17,4):
#         mov_img = move_left(images[k],j)
#         for i in range(-20,21,4):
#             rotated_im = rotation(mov_img,i)
#             cv2.imwrite(f'Data/augm/mul/mul_{k+1}_mov{j}_angle{i}.jpeg',rotated_im)


# In[13]:


# path=f'Data/orig/mul'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)
#     images.append(img)
# for k in range(len(images)):
#     for j in range(-16,17,4):
#         mov_img = move_left(images[k],j)
#         for i in range(-20,21,4):
#             rotated_im = rotation(mov_img,i)
#             cv2.imwrite(f'Data/augm/mul/mul_{k+1}_mov{j}_angle{i}.jpeg',rotated_im)


# In[14]:


# path=f'Data/orig/div'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)
#     images.append(img)
# for k in range(len(images)):
#     for j in range(-16,17,4):
#         mov_img = move_left(images[k],j)
#         for i in range(-20,21,4):
#             rotated_im = rotation(mov_img,i)
#             cv2.imwrite(f'Data/augm/div/div_{k+1}_mov{j}_angle{i}.jpeg',rotated_im)


# In[6]:


# path=f'Data/orig/1'
# myPicList= os.listdir(path)
# images=[]
# for y in (myPicList):
#     img=cv2.imread(path + str('/')+y,0)
#     images.append(img)


# In[26]:


# plt.imshow(images[0])

