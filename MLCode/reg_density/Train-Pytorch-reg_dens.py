#!/usr/bin/env python
# coding: utf-8

# ## Pytorch code percolation model with ResNet18

# ## parameter choices

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
import time
import pickle
import copy
import sklearn
import re
import sys
sys.path.insert(0, '/home/physics/phsht/Projects/ML-Percolation')
from MLtools import *
##########################################################################
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
size_samp=my_size_samp
validation_split= my_validation_split
batch_size=my_batch_size
num_epochs= my_num_epochs
training_set=0
validation_set=0

myDATAPATH='../../Data/'
myCSV='../../Data_csv/'
dataname='Perco-data-reg-bw-int-density-L'+str(size)+'_'+str(size_samp)+'_s'+str(myseed)

    
print(dataname)#,"\n",datapath)
method='PyTorch-resnet18-pretrained-'+str(myseed)+'-e'+str(num_epochs)+'-bs'+str(batch_size)+'-s'+str(myseed)
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
print('--> defining seeds')
torch.manual_seed(myseed+1)
np.random.seed(myseed+2)
torch.cuda.manual_seed(myseed+3)
#torch.cuda.seed()
#torch.cuda.seed_all()
random.seed(myseed+4)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(myseed+5)
print('torch version:',torch.__version__)
print('sklearn version:', sklearn.__version__)
t=torch.Tensor()
print('current device: ', t.device, t.dtype, t.layout)

# switch to GPU if available
device=t.device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print('chosen device: ',device)

print(os.getcwd())


whole_dataset =Dataset_csv_pkl_reg(csv_file=myCSV+'data_pkl_'+str(size)+'_'+str(size_samp)+'.csv',
                                     root_dir=myDATAPATH+'L'+str(size)+'/',size=size, classe_type='density',
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
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True)
print('--> loading validation data')
val = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False)
print('--> defining classes/labels')
class_names =whole_dataset.classes
inputs,labels,paths=next(iter(train))
img_sizeX,img_sizeY= inputs.shape[-1],inputs.shape[-2]
num_of_train_samples = len(training_set) # total training samples
num_of_test_samples = len(validation_set) #total validation samples
steps_per_epoch = np.ceil(num_of_train_samples // batch_size)
number_classes = len(class_names)
print('--> protocolling set-up')
print('number of samples in the training set:', num_of_train_samples)
print('number of samples in the validation set:', num_of_test_samples )
print('number of samples in a batch',len(train)) 
print('number of samples in a batch',len(val))
print('number of classes',number_classes )


# ## building the CNN
print('--> building the CNN')
resnet18_model=models.resnet18(pretrained=True, progress=True)
resnet18_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = resnet18_model.fc.in_features
model_res=models.resnet18(pretrained=True, progress=True)
model_res.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model_res.fc.in_features
model_res.fc = nn.Linear(num_ftrs,number_classes)
model=Resnet18_regression(resnet18_model=resnet18_model)
model = model.to(device)
print('--> defining optimizer')
# defining the optimizer
optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# defining the loss function
criterion = nn.MSELoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

#the model is sent to the GPU
model = model.to(device)
model=model.float()
if flag==0:
    print('--> starting training epochs')
    print('number of epochs',num_epochs )
    print('batch_size',batch_size )
    print('number of classes',number_classes )
    base_model = train_reg_model(
        model,train,val,device, criterion, optimizer,num_epochs,exp_lr_scheduler,savepath,method,dataname,modelname,modelpath,
        batch_size,class_names)
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
reg_prediction_dens(dataloader,size,model,seed,nb_classes=31,data_type='val')
print('--> finished!')




