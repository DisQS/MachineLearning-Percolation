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
sys.path.insert(0, '/home/physics/phrhmb/Perco_data')
from MLtools import *


myDATAPATH='../../Data/'
myCSV='../../Data_csv/'
#########################################################################

class Resnet18_regression(nn.Module):
    def __init__(self,resnet18_model):
        super(Resnet18_regression, self).__init__()
        self.resnet18_model=resnet18_model
        num_ftrs = self.resnet18_model.fc.out_features
        self.new_layers=nn.Sequential(nn.ReLU(),
                                      nn.Linear(num_ftrs,256),
                                      nn.ReLU(),
                                      nn.Linear(256,64),
                                      nn.ReLU(),
                                      nn.Linear(64,1))
        
    def forward(self,x):
        x=self.resnet18_model(x)
        x=self.new_layers(x)
        return x

#############################################################################################
def classification_prediction(dataloader,size,model,whole_dataset):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]

    #model=model.to('cpu')
    header_l=['path','label','prediction']

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            #inputs=inputs.float()
            labels.numpy()
            
            predictions = model(inputs) #value of the output neurons
            _, pred= torch.max(predictions,1)
           
            for j in range(inputs.size()[0]):
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_labels=labels[j].detach().cpu().numpy()
                temp_preds=pred[j].detach().cpu().numpy()
                list_paths.append(temp_paths)
                list_labels.append(temp_labels)
                list_preds.append(temp_preds[0])
    
    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds} 

    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_reg_density_bw_int_'+str(size)+'_'+str(myseed)+'_val_avon.csv',index=False)
        
    return

#######################################################################################################
#print('######################')
#print(sys.argv)
if ( len(sys.argv) == 2 ):
    #SEED = 101
    SEED = int(sys.argv[1])


else:
    print ('Number of', len(sys.argv), \
           'arguments is less than expected (2) --- ABORTING!')
    
print('--> defining parameters')
myseed=SEED
size= 100
nimages= 100
img_sizeX= 100
img_sizeY= img_sizeX
size_samp=10000
validation_split= 0.1
batch_size=256
num_epochs= 20

dataname='Perco-data-reg-bw-int-density-L'+str(size)+'-'+str(nimages)+'-s'+str(img_sizeX)+'_'+str(size_samp)+'_s'+str(myseed)
#datapath='../L100' 

    
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
print('--> defining seeds'
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


whole_dataset =Dataset_csv_pkl_reg(csv_file=myCSV+'data_pkl_100_10000_reg_train_1_over_2.csv',
                                     root_dir=myDATAPATH+'L100/',size=100, classe_type='density',
                                array_type='bw',transform=transforms.ToTensor())


training_set=0
validation_set=0
os.getcwd()
data_size = len(whole_dataset)
# validation_split=0.1
split=int(np.floor(validation_split*data_size))
training=int(data_size-split)
# split the data into training and validation
training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))

train = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True)

val = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False)

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


# ## building the CNN
resnet18_model=models.resnet18(pretrained=True, progress=True)
resnet18_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = resnet18_model.fc.in_features
model_res=models.resnet18(pretrained=True, progress=True)
model_res.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model_res.fc.in_features
model_res.fc = nn.Linear(num_ftrs,number_classes)



model=Resnet18_regression(resnet18_model=resnet18_model)
model = model.to(device)

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
base_model = train_reg_model(
    model,train,val,device, criterion, optimizer,num_epochs,exp_lr_scheduler,savepath, method,dataname,modelname,modelpath,
    batch_size,class_names )

str_classes_names=" ".join(str(x) for x in class_names)


classification_prediction(val,100,model,whole_dataset)

csv_file=savepath+'predictions_reg_density_bw_int_'+str(size)+'_'+str(myseed)+'10000_val_avon.csv'




