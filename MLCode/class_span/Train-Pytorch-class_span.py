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
from torchsummary import summary
import pandas as pd
import time
import pickle
import random
import re
import copy
import sys
sys.path.insert(0, '/home/p/phrhmb/Perco/MLCode') # RAR
from MLtools import *
import sklearn

###########################################################################################
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
myDATAPATH='/home/p/phrhmb/Perco/Data/L'+str(size)+'_rename'
myTEST='/home/p/phrhmb/Perco/Data/L'+str(size)+'_test'
myCSV='/home/p/phrhmb/Perco/Data_csv/data_pkl_100_10000_p0.585_train_renamed_without_path.csv'
myCSV_val='/home/p/phrhmb/Perco/Data_csv/data_pkl_100_10000_p0.585_val_renamed_without_path.csv'
myCSV_test='/home/p/phrhmb/Perco/Data_csv/data_pkl_test_p0.585_10_1000_585.csv'
dataname='Perco-data-bw-very-hres-span-L'+str(size)+'_'+str(size_samp)+'_s'+str(myseed)
    
#############################################################################################
def density_as_func_proba_span_bis(csv_file,size_samp=10000,data_type='val',data_range=''):
    data=pd.read_csv(csv_file)
    density=data['density'].unique()
    density.sort()
    p_list=[]
    pred_1_1=[]
    pred_1_0=[]
    pred_0_1=[]
    pred_0_0=[]
    div_1_1=[]
    div_1_0=[]
    div_0_1=[]
    div_0_0=[]
    ml_Samp=[]
    list_nb_1=[]
    list_nb_0=[]

    class_percoSamp=data['density'].unique()
    len_percoSamp=len(class_percoSamp)
    percoSamp=[size_samp]*len_percoSamp
    
    for p in density:
        
        new_df= data[data['density']==p]
        nb_p=new_df['density'].count()
        new_df_1=new_df[new_df['label']==1]
        new_df_0=new_df[new_df['label']==0]
        nb_1=new_df[new_df['label']==1].count()['label']
        nb_0=new_df[new_df['label']==0].count()['label']
        new_df_1_1=new_df_1[new_df_1['prediction']==1]
        new_df_1_0=new_df_1[new_df_1['prediction']==0]
        new_df_0_1=new_df_0[new_df_0['prediction']==1]
        new_df_0_0=new_df_0[new_df_0['prediction']==0]
        nb_p_1_1=new_df_1_1['density'].count()
        nb_p_1_0=new_df_1_0['density'].count()
        nb_p_0_1=new_df_0_1['density'].count()
        nb_p_0_0=new_df_0_0['density'].count()


        div_1_1=nb_p_1_1#/nb_p
        div_1_0=nb_p_1_0#/nb_p
        div_0_1=nb_p_0_1#/nb_p
        div_0_0=nb_p_0_0#/nb_p
        p_list.append(p)
        pred_1_1.append(div_1_1)
        pred_1_0.append(div_1_0)
        pred_0_1.append(div_0_1)
        pred_0_0.append(div_0_0)
        list_nb_1.append(nb_1)
        list_nb_0.append(nb_0)
        ml_Samp.append(nb_p)
        print(ml_Samp)
    dict = {'density': p_list,'percoSamp':ml_Samp,'percoNS':list_nb_0,'percoS':list_nb_1,'SpS': pred_1_1, 'SpNS': pred_1_0,'NSpS': pred_0_1,'NSpNS': pred_0_0}
    print(len(p_list),len(percoSamp))
    print(p_list)
    print('##################')
    print(percoSamp)
    df = pd.DataFrame(dict)
    size=100
    myseed=0
    df.to_csv('class_span_bw_v_h_res_'+str(size)+'_'+str(size_samp)+'_'+str(myseed)+'_'+str(data_type)+'.csv',index=False)
   

    return p_list,pred_1_1,pred_1_0,pred_0_1,pred_0_0

####################################################################################################### 
print(dataname)#,"\n",datapath)
method='PyTorch-resnet18-Adam_001-pretrained-'+str(myseed)+'-e'+str(num_epochs)+'-bs'+str(batch_size)+'-s'+str(myseed)
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
print('--> defining ML lib versions and devices')
print('torch version:',torch.__version__)
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
train_dataset = Dataset_csv_pkl(csv_file=myCSV,
                                     root_dir=myDATAPATH, classe_type='span',
                                array_type='bw',type_file='symlink',data_type='pkl',transform=transforms.ToTensor())
val_dataset = Dataset_csv_pkl(csv_file=myCSV_val,
                                     root_dir=myDATAPATH, classe_type='span',
                                array_type='bw',type_file='symlink',data_type='pkl',transform=transforms.ToTensor())
test_dataset=Dataset_csv_pkl(csv_file=myCSV_test,
                                     root_dir=myTEST, classe_type='span',
                                array_type='bw',data_type='pkl',transform=transforms.ToTensor())

os.getcwd()
data_size = len(train_dataset)+len(val_dataset)
#split=int(np.floor(validation_split*data_size))
training=len(train_dataset)
# split the data into training and validation
#training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))
print('--> loading training data')
train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True)
print('--> loading validation data')
val = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False)
test= train = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True)
print('--> defining classes/labels')
class_names =val_dataset.classes
inputs,labels,paths=next(iter(train))
img_sizeX,img_sizeY= inputs.shape[-1],inputs.shape[-2]
num_of_train_samples = len(train_dataset) # total training samples
num_of_test_samples = len(val_dataset) #total validation samples
steps_per_epoch = np.ceil(num_of_train_samples // batch_size)
number_classes = len(class_names)
print('number of samples in the training set:', num_of_train_samples)
print('number of samples in the validation set:', num_of_test_samples )
print('number of samples in a batch',len(train)) 
print('number of samples in a batch',len(val))
print('number of classes',number_classes )


# ## building the CNN


model=models.resnet18(pretrained=True, progress=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features # number of input features of the last layer which is fully connected (fc)
#We modify the last layer in order to have nb_output: number_classes
model.fc=nn.Linear(num_ftrs, number_classes )
 #the model is sent to the GPU
model = model.to(device)
print(model)
print('#############################')
print(summary(model, (1, 100,100)))
# defining the optimizer
print('--> defining optimizer')
#optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# defining the loss function
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
print('--> starting training epochs')
print('number of epochs',num_epochs)
print('batch_size',batch_size)
print('number of classes',number_classes)
#the model is sent to the GPU
model = model.to(device)
model=model.double()
if flag==0:
	base_model = train_model(
   	 model,train,val,device, criterion, optimizer,num_epochs,exp_lr_scheduler,savepath, 	method,dataname,modelname,modelpath,
  	  batch_size,class_names )
else:
    print('--> loading saved model')
    print('--> loading saved model')
    checkpoint=torch.load(modelpath+'_epochs_19.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss=checkpoint['val loss']
    _loss=checkpoint['train loss']
    epochs=checkpoint['train epoch']
    model.eval()
print('--> computing/saving confusion matrices')
str_classes_names=" ".join(str(x) for x in class_names)
cm=simple_confusion_matrix(model,val,device,number_classes,class_names)
np.savetxt(savepath+method+'_'+dataname+'cm_val_best.txt',cm,fmt='%d')
cm_test=simple_confusion_matrix(model,test,device,number_classes,class_names)
np.savetxt(savepath+method+'_'+dataname+'cm_test_best.txt',cm_test,fmt='%d')
print('--> computing/saving classification outputs')
#classification_prediction_span(val,size,savepath,myseed,model)
classification_prediction_span(test,1000,savepath,myseed,model)
csv_file=savepath+'predictions_class_span_bw_v_h_res_'+str(size)+'_'+str(size_samp)+'_'+str(myseed)+'_val.csv'
csv_file2=savepath+'predictions_class_span_bw_v_h_res_'+str(size)+'_1000_'+str(myseed)+'_val.csv'
density_as_func_proba_span_bis(csv_file,size_samp)
density_as_func_proba_span_bis(csv_file2,1000)
#classification_prediction(val,size,model)

print('--> finished!')