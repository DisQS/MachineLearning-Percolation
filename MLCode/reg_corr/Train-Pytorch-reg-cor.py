#!/usr/bin/env python
# coding: utf-8

# ## Pytorch code percolation model with ResNet18
## parameter choices
                                                
# In[1]:
from __future__ import print_function, division
import os
import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torchsummary import summary
import time
import pickle
import copy
import random
import sklearn
import re
import sys                                           ##############    BE CAREFUL
sys.path.insert(0, '/home/p/phrhmb/Perco/MLCode')##########################################################
from MLtools import *                                ##########################################################
#####################################################################
print(sys.argv)
if ( len(sys.argv) == 8 ):
    #SEED = 101
    SEED = int(sys.argv[1])
    my_size= int(sys.argv[2])
    my_size_samp=int(sys.argv[3])
    my_validation_split= float(sys.argv[4])
    my_batch_size=int(sys.argv[5])
    my_num_epochs= int(sys.argv[6])
    flag=int(sys.argv[7])
else:
    print ('Number of', len(sys.argv), \
           'arguments is less than expected (2) --- ABORTING!')
    
print('--> defining parameters')
    
myseed=SEED
size= my_size
nimages= 100
size_samp=10000
validation_split= my_validation_split
batch_size=my_batch_size
num_epochs= my_num_epochs
training_set=0
validation_set=0
# everyone
myDATAPATH='../../../Data/L'+str(size)+'_rename/L'+str(size)+'_rename'
myDATAPATH_test='../../../Data/L'+str(size)+'_test_1000'
myCSV='../../../Data_csv/data_pkl_100_10000_shuffled_new_avg_int.csv'
myCSV_test='../../../Data_csv/data_pkl_100_test_redo_1000.csv'
dataname='Perco-data-reg-bw-int-corr-shuffled-L'+str(size)+'_'+str(size_samp)#'-100-s100'+'_'+str(size_samp)+'_s'+str(myseed)
#myCSV='../../../Data_csv/data_pkl_100_10000_int.csv'
#dataname='Perco-data-reg-bw-int-corr-L'+str(size)+'_'+str(size_samp)
#datapath='../L'+str(size)+'' 

    
print(dataname)#,"\n",datapath)
method='PyTorch-resnet34-pretrained-'+str(myseed)+'-e'+str(num_epochs)+'-bs'+str(batch_size)+'-s'+str(myseed)
modelname = 'Model_'+method+'_'+dataname+'_s'+str(myseed)
historyname = 'History_'+method+'_'+dataname+'_s'+str(myseed)+'.pkl'
print(method,"\n",modelname,"\n",historyname)

savepath = './'+dataname+'/'

try:
    os.mkdir(savepath)
except FileExistsError:
    pass

modelpath = savepath+modelname
historypath = savepath+historyname
print(savepath,modelpath,historypath)

#######################################################################################################
#print('######################')
print('--> defining seeds')
torch.manual_seed(myseed)
np.random.seed(myseed+1)
print('--> defining ML lib versions and devices')
print('torch version:',torch.__version__)
print('torchvision version:',torchvision.__version__)
print('sklearn version:', sklearn.__version__)
t=torch.Tensor()
print('current device: ', t.device, t.dtype, t.layout)
# switch to GPU if available
device=t.device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print('chosen device: ',device)
print('--> reading CSV data')
print(os.getcwd())
print(myCSV)
#whole_dataset =Dataset_csv_pkl_reg(csv_file=myCSV,
#                                     root_dir=myDATAPATH,size=size, classe_type='corr',
#                                array_type='bw',transform=transforms.ToTensor())
whole_dataset =Dataset_csv_pkl_reg(csv_file=myCSV,
                                     root_dir=myDATAPATH,size=size, classe_type='corr',
                                     array_type='bw',type_file='symlink',transform=transforms.ToTensor())
test_dataset =Dataset_csv_pkl_reg(csv_file=myCSV_test,
                                     root_dir=myDATAPATH_test,size=size, classe_type='corr',
                                     array_type='bw',transform=transforms.ToTensor())

os.getcwd()
data_size = len(whole_dataset)
# validation_split=0.1
split=int(np.floor(validation_split*data_size))
training=int(data_size-split)
# split the data into training and validation
training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))
print('--> loading training data')
train = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True)
print('--> loading validation data')
val = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        num_workers=16,
        shuffle=False)
test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=False)
print('--> defining classes/labels')
class_names =whole_dataset.classes
inputs,labels,paths=next(iter(train))
img_sizeX,img_sizeY= inputs.shape[-1],inputs.shape[-2]
num_of_train_samples = len(training_set) # total training samples
num_of_test_samples = len(validation_set) #total validation samples
steps_per_epoch = np.ceil(num_of_train_samples // batch_size)
number_classes = len(class_names)
print('number of samples in the training set:', num_of_train_samples)
print('number of samples in the validation set:', num_of_test_samples )
print('number of samples in a batch',len(train)) 
print('number of samples in a batch',len(val))
print('number of classes',number_classes )

print('--> building the CNN')
# ## building the CNN
resnet18_model=models.resnet34(pretrained=True, progress=True)
resnet18_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model=Resnet18_regression(resnet18_model=resnet18_model)
model = model.to(device)
print(model)
print('#############################')
print(summary(model, (1, 100,100)))
print('--> defining optimizer')
# defining the optimizer
optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# defining the loss function
criterion = nn.MSELoss(reduction='mean')
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

#the model is sent to the GPU
model = model.to(device)
model=model.double()


if flag==0:
    print('--> starting training epochs')
    print('number of epochs',num_epochs )
    print('batch_size',batch_size )
    print('number of classes',number_classes )
    base_model = train_reg_model(
        model,train,val,device, criterion, optimizer,num_epochs,exp_lr_scheduler,savepath, method,dataname,modelname,modelpath,
        batch_size,class_names )
else:
    print('--> loading saved model')
    checkpoint=torch.load(modelpath+'_epochs_19.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss=checkpoint['val loss']
    _loss=checkpoint['train loss']
    epochs=checkpoint['train epoch']
    model.eval()

str_classes_names=" ".join(str(x) for x in class_names)
print('--> storing regression predictions')
reg_prediction_cor(test,model,size,myseed,savepath,nb_classes=number_classes,data_type='test_redone')
#reg_prediction_cor(val,model,size,myseed,savepath,nb_classes=number_classes,data_type='val')
print('--> finished!')



