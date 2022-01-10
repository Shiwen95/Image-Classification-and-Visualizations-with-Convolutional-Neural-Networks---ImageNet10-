#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from imagenet10 import ImageNet10
from torch import nn
from torch import optim
from config import *
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import pandas as pd
import os

import matplotlib.pyplot as plt


# In[2]:


# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR+ dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)

data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# See what the dataframe now contains
print("Found", len(data_df), "images.")
# If you want to see the image meta data
print(data_df.head())

# Split the data into train and test sets and instantiate our new ImageNet10 objects.
train_split = 0.80  # Defines the ratio of train/valid data.

# valid_size = 1.0 - train_size
train_size = int(len(data_df) * train_split)

data_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

# See what you've loaded
print("len(dataset_train)", len(dataset_train))
print("len(dataset_valid)", len(dataset_valid))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))


# In[3]:


def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            running_loss += loss.item()
            n += 1
    return running_loss/n, correct/total


# In[4]:


#############################################################################################################
#
# Firstly, train model on only one batch of the training data, and part or all of the validation data
# 
#############################################################################################################


# In[5]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fcl=nn.Linear(3*128*128,10)
        
    def forward(self,x):
        x=x.reshape(x.size(0),-1)
        x=self.fcl(x)
        return x


# In[6]:


net = Classifier()
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
nepochs = 30

# Using only one batch of training set
iter_t = iter(train_loader)
img_t, class_t = next(iter_t)

statsrec = np.zeros((3,nepochs))

for epoch in range(nepochs):
    # training batch images.shape = 64*3*128*128
    optimizer.zero_grad()
    output_t = net(img_t)
    train_loss = loss_fn(output_t, class_t)

    # backward based on training loss
    train_loss.backward()
    optimizer.step()
    
    # get validation loss and accuracy
    valid_loss, accuracy = stats(valid_loader, net)
    
    statsrec[:,epoch] = (train_loss.item(), valid_loss, accuracy)
    print(f"epoch: {epoch+1} training loss: {train_loss.item(): .3f}  validation loss: {valid_loss: .3f} validation accuracy: {accuracy: .1%}")


# In[7]:

# graph the training loss and validation loss over epochs
fig, ax1 = plt.subplots()
plt.plot(statsrec[0], 'r', label = 'training loss', )
plt.plot(statsrec[1], 'g', label = 'validation loss' )
plt.legend(loc='center')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss, and validation accuracy')
ax2=ax1.twinx()
ax2.plot(statsrec[2], 'b', label = 'validation accuracy')
ax2.set_ylabel('accuracy')
plt.legend(loc='upper left')
plt.show()


# In[8]:


#########################################################################
#
#        Adjust the network
# 
#########################################################################


# In[9]:


class Classifier_cnn(nn.Module):
    def __init__(self):
        super(Classifier_cnn, self).__init__()

        self.Conv1 = nn.Conv2d(3, 5, kernel_size=5,padding=2,stride=4)  #64*5*32*32
        self.MaxP1 = nn.MaxPool2d(kernel_size=2,stride=2)               # 64*5*16*16
        self.fcl = nn.Linear(5*16*16, 10)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.MaxP1(x)
        x = x.view(-1, 5 * 16 * 16)
        x = self.fcl(x)
        return x


# In[10]:


net_cnn = Classifier_cnn()
optimizer_cnn = optim.SGD(net_cnn.parameters(), lr=0.01, momentum=0.9)

nepochs = 50
statsrec_cnn = np.zeros((3,nepochs))

for epoch in range(nepochs):
    
    # training batch images.shape = 64*3*128*128
    optimizer_cnn.zero_grad()
    output_cnn = net_cnn(img_t)
    train_loss_cnn = loss_fn(output_cnn, class_t)
    train_loss_cnn.backward()
    optimizer_cnn.step()
    
    #get validation loss and accuracy
    valid_loss_cnn, accuracy_cnn = stats(valid_loader, net_cnn)
    
    statsrec_cnn[:,epoch] = (train_loss_cnn.item(), valid_loss_cnn, accuracy_cnn)
    print(f"epoch: {epoch+1} training loss: {train_loss_cnn.item(): .3f}  validation loss: {valid_loss_cnn: .3f} validation accuracy: {accuracy_cnn: .1%}")


# In[11]:


fig, ax1 = plt.subplots()
plt.plot(statsrec_cnn[0], 'r', label = 'training loss', )
plt.plot(statsrec_cnn[1], 'g', label = 'validation loss' )
plt.legend(loc='center')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training & validation loss over training epochs, with modified architecture')
ax2=ax1.twinx()
ax2.plot(statsrec_cnn[2], 'b', label = 'validation accuracy')
ax2.set_ylabel('accuracy')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


###################################################################################
#
# Secondly, finetune the model and train on the complete training dataset 
# 
###################################################################################


# In[12]:


# Fine tune the network and use Dropout

class Classifier_ft(nn.Module):
    def __init__(self):
        super(Classifier_ft,self).__init__()
        
        self.Conv1=nn.Conv2d(3,20,kernel_size=5,padding=2,stride=4)   
        self.MaxP1=nn.MaxPool2d(kernel_size=2,stride=2)               
        self.Conv2=nn.Conv2d(20,40,kernel_size=5,padding=2,stride=2)   
        self.MaxP2=nn.MaxPool2d(kernel_size=2,stride=2)                  
        self.dout=nn.Dropout(0.5)
        self.fcl1=nn.Linear(40*4*4,100)
        self.fcl2=nn.Linear(100,10)
        
    def forward(self,x):
        x=F.relu(self.Conv1(x))
        x=self.MaxP1(x)
        x=F.relu(self.Conv2(x))
        x=self.MaxP2(x)
        x=x.view(-1,40*4*4)
        x=self.dout(x)
        x=self.fcl1(x)
        x=F.relu(x)
        x=self.dout(x)
        x=self.fcl2(x)
        return x


# In[13]:


#Using data augmentation

data_transform_aug = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

dataset_train_aug = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform_aug
)

train_loader_aug = torch.utils.data.DataLoader(
    dataset_train_aug,
    batch_size=64,
    shuffle=True,
    num_workers=2
)


# In[14]:


net_ft=Classifier_ft()
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(net_ft.parameters(),lr=0.005,momentum=0.9)

nepochs = 250
statsrec = np.zeros((3,nepochs))
max_accuracy=0.6

#Using complete training set
for epoch in range(nepochs): 

    train_loss = 0.0
    n = 0
    for i, data_t in enumerate(train_loader_aug, 0):
        n += 1
        img_t, classes_t = data_t
        
         # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        output_t = net_ft(img_t)
        loss_t = loss_fn(output_t, classes_t)
        loss_t.backward()
        optimizer.step()
    
        # accumulate loss
        train_loss += loss_t.item()

    ltrn = train_loss/n
    ltst, atst = stats(valid_loader, net_ft)
    statsrec[:,epoch] = (ltrn, ltst, atst)
    print(f"epoch: {epoch+1} training loss: {ltrn: .3f}  validation loss: {ltst: .3f} validation accuracy: {atst: .1%}")
    
    #Save best model
    val_accuracy = atst
    if val_accuracy > max_accuracy:
        max_accuracy = val_accuracy
        print("save model")
        torch.save(net_ft.state_dict(),'.../net_epoch{0} accuracy{1:.1%}.pth'.format(epoch,max_accuracy),_use_new_zipfile_serialization=False)


# In[15]:

# display the graph of training and validation loss over epochs to show the optimal number of training epochs

plt.figure()
fig, ax1 = plt.subplots()
plt.plot(statsrec[0], 'r', label = 'training loss', )
plt.plot(statsrec[1], 'g', label = 'validation loss' )
plt.legend(loc='center')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss, and validation accuracy')
ax2=ax1.twinx()
ax2.plot(statsrec[2], 'b', label = 'validation accuracy')
ax2.set_ylabel('accuracy')
plt.legend(loc='upper left')
plt.show()


# In[16]:


#Plot the confusion matrix

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float')    
    plt.imshow(cm, interpolation='nearest')    
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')


# In[17]:


cm_t=np.zeros((10,10))
cm_v=np.zeros((10,10))

#load the best net
best_net = Classifier_ft()
best_net.load_state_dict(torch.load('.../net_epoch136 accuracy62.6%.pth')) 
best_net.eval()

with torch.no_grad():
    for i, data_t in enumerate(train_loader, 0):
        img_t, classes_t = data_t
        output_t = best_net(img_t)
        _, predicted = torch.max(output_t.data, 1)
        for i,j in zip(classes_t,predicted):
            cm_t[i.data][j.data]+=1

    for i, data_v in enumerate(valid_loader, 0):
        img_v, classes_v = data_v
        output_v = best_net(img_v)
        _, predicted = torch.max(output_v.data, 1)
        for i,j in zip(classes_v,predicted):
            cm_v[i.data][j.data]+=1


# In[ ]:


print(cm_t)
'''
array([[607.,   3.,  17.,  64.,   7.,   1.,   5.,  16.,   6.,   1.],
       [ 24., 493.,  16.,  43.,  17.,   8.,  12.,   9.,  23.,  86.],
       [ 16.,   4., 616.,  11.,  32.,   4.,   9.,  24.,   4.,   1.],
       [ 77.,   7.,   5., 571.,  14.,   0.,  18.,   7.,  10.,   3.],
       [  3.,   7.,  17.,  13., 633.,  20.,  17.,   6.,  16.,   4.],
       [ 13.,  14.,  26.,  45.,  68., 413.,  74.,  24.,  18.,   4.],
       [ 28.,   9.,  23.,  44.,  99.,  56., 393.,  29.,  29.,   6.],
       [ 61.,  17.,  51.,  56.,  29.,   9.,  35., 415.,  26.,  11.],
       [ 14.,  14.,  17.,  47.,  78.,  34.,  29.,  36., 435.,  16.],
       [  4.,  59.,   6.,  16.,   5.,   3.,   3.,   6.,  11., 615.]])
'''


# In[ ]:


print(cm_v)
'''
array([[146.,   1.,   3.,  18.,   2.,   1.,   1.,   1.,   0.,   0.],
       [  6., 113.,   6.,   8.,   5.,   4.,   3.,   3.,   4.,  17.],
       [  5.,   1., 157.,   3.,   9.,   0.,   1.,   2.,   1.,   0.],
       [ 27.,   1.,   1., 144.,   3.,   0.,   3.,   4.,   5.,   0.],
       [  0.,   1.,   3.,   3., 147.,   2.,   0.,   2.,   5.,   1.],
       [  8.,   3.,   5.,  14.,  32., 100.,  17.,   7.,  12.,   3.],
       [  9.,   1.,   3.,  11.,  31.,  18.,  92.,   9.,   8.,   2.],
       [ 12.,   3.,   7.,  24.,   7.,   7.,  13., 108.,   9.,   0.],
       [  6.,   4.,   2.,  18.,  26.,   8.,  10.,   4.,  99.,   3.],
       [  1.,  13.,   0.,   1.,   3.,   0.,   0.,   1.,   1., 152.]])
'''


# In[18]:


plot_confusion_matrix(cm_t, labels_name=CLASS_LABELS, title='Training Set Confusion Matrix')


# In[19]:


plot_confusion_matrix(cm_v, labels_name=CLASS_LABELS, title='Validation Set Confusion Matrix')


# In[ ]:


#################################################################################################
#
#        At last, predict the unlabeled test data using best trained model 
# 
#################################################################################################


# In[20]:


# Gathers the test data
paths_test, classes_test = [], []
for entry in os.scandir(ROOT_DIR.replace('train_set','test_set')):
    if (entry.is_file()):
        paths_test.append(entry.path)
        classes_test.append(0)

paths_test.sort(key=None, reverse=False)

data_test = {
    'path': paths_test,
    'class': classes_test
}

data_df_test = pd.DataFrame(data_test, columns=['path', 'class'])
dataset_test = ImageNet10(
    df=data_df_test,
    transform=data_transform,
)

test_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("len(dataset_test)", len(dataset_test))
print("len(test_loader)", len(test_loader))
print(data_df_test)


# In[21]:


# Output prediction and store in a csv file

pci=[]
for img_test, classes_test in test_loader:
    
    output_test = best_net(img_test)
    _, predicted_test = torch.max(output_test.data, 1)
    pci.append(predicted_test.item())

paths_csv=[]
for path in range(1000):
    paths_csv.append(paths_test[path][-15:])
    
data_preds = {
    'image_name': paths_csv,
    'predicted_class_id': pci
}

df_preds = pd.DataFrame(data_preds, columns=['image_name','predicted_class_id'])

df_preds.to_csv('[mm20sy]_test_preds.csv',index=False,sep=',')

