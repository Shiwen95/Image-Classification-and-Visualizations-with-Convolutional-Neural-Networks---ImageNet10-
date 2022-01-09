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
    batch_size=80,
    shuffle=True,
    num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=500,
    shuffle=True,
    num_workers=0
)

# See what you've loaded
print("len(dataset_train)", len(dataset_train))
print("len(dataset_valid)", len(dataset_valid))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.Conv1 = nn.Conv2d(3, 5, kernel_size=3)  # 64*5*126*126
        self.MaxP1 = nn.MaxPool2d(kernel_size=2)  # 64*5*63*63
        self.Conv2 = nn.Conv2d(5, 10, kernel_size=3)  # 64*10*61*61
        self.MaxP2 = nn.MaxPool2d(kernel_size=2)  # 64*10*30*30
        self.Conv3 = nn.Conv2d(10, 20, kernel_size=3)  # 64*20*28*28
        self.MaxP3 = nn.MaxPool2d(kernel_size=2)  # 64*20*14*14
        self.Conv4 = nn.Conv2d(20, 40, kernel_size=3)  # 64*40*12*12
        self.MaxP4 = nn.MaxPool2d(kernel_size=2)  # 64*40*6*6
        self.fcl = nn.Linear(40 * 6 * 6, 10)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.MaxP1(x)
        x = F.relu(self.Conv2(x))
        x = self.MaxP2(x)
        x = F.relu(self.Conv3(x))
        x = self.MaxP3(x)
        x = F.relu(self.Conv4(x))
        x = self.MaxP4(x)
        x = x.view(-1, 40 * 6 * 6)
        x = self.fcl(x)
        return x


net = Classifier()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

nepochs = 100

# Using only one batch of training set
iter_t = iter(train_loader)
img_t, class_t = next(iter_t)

iter_v = iter(valid_loader)
img_v, class_v = next(iter_v)

losses1 = np.zeros(nepochs)
losses2 = np.zeros(nepochs)
accuracy= np.zeros(nepochs)

for epoch in range(nepochs):
    # training batch images.shape = 64*3*128*128
    optimizer.zero_grad()

    output_t = net(img_t)
    train_loss = loss_fn(output_t, class_t)

    output_v = net(img_v)
    valid_loss = loss_fn(output_v, class_v)

    #get accuracy
    correct = 0
    total = 0
    _, predicted = torch.max(output_v.data, 1)
    total += class_v.size(0)
    correct += (predicted == class_v).sum().item()
    accuracy[epoch] = correct/total*100

    # get training loss
    losses1[epoch] = train_loss
    print('epoch{}  training_loss is {}  '.format(epoch + 1, train_loss.item()), end='\t')

    # get validation loss
    losses2[epoch] = valid_loss
    print('validation_loss is {}'.format(valid_loss.item()))

    # backward based on training loss
    train_loss.backward()
    optimizer.step()

plt.plot(losses1, 'r', label='training loss', )
plt.plot(losses2, 'g', label='validation loss')
plt.plot(accuracy,'b', label='accuracy%')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training & validation loss over training epochs, with modified architecture')
plt.show()