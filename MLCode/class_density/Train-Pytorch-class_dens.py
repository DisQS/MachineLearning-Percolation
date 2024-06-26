#!/usr/bin/env python
# ## Pytorch code percolation model with ResNet18
from __future__ import print_function, division
import os
import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torchsummary import summary
import pandas as pd
import random
import sys
sys.path.insert(0, '/storage/disqs/ML-Percolation/MachineLearning-Percolation/MLCode')  #absolute path of MLtools file
from MLtools import *
import sklearn

###########################################################################################
print(sys.argv)
if ( len(sys.argv) == 8 ):
    print(sys.argv)
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
           'arguments is less than expected (8) --- ABORTING!')
print('--> defining parameters')

myseed=SEED
size= my_size
size_samp=my_size_samp
validation_split= my_validation_split
batch_size=my_batch_size
num_epochs= my_num_epochs
training_set=0
validation_set=0
#SET THE PATH FOR DIRECTORIES OF THE TRAINING/TEST DATASETS AND PATH TO CSV
##################################################################################################
myDATAPATH='../../Data/L'+str(size)+'/'
myTEST='../../Data/L'+str(size)+'_test'
myCSV='../../Data_csv/data_pkl_100_10000.csv'
myCSV_test='../../Data_csv/data_pkl_100_test_1000.csv'
####################################################################################################### 
print('--> defining files and directories')
dataname='Perco-data-bw-very-hres-density-L'+str(size)+'-'+str(nimages)+'_'+str(size_samp)+'_s'+str(myseed)
datapath=myDATAPATH     
print(dataname,"\n",datapath)
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
#print('######################')
print('--> defining seeds')
torch.manual_seed(myseed+1)
np.random.seed(myseed+2)
torch.cuda.manual_seed(myseed+3)
#torch.cuda.seed()
#torch.cuda.seed_all()
random.seed(myseed+4)

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
if flag==0:
    print('--> reading CSV data')
    print(os.getcwd())
    train_dataset = Dataset_csv_pkl(csv_file=myCSV,
                                         root_dir=myDATAPATH, classe_type='density',
                                     array_type='bw',type_file='normal',data_type='pkl',transform=transforms.ToTensor())

    test_dataset=Dataset_csv_pkl(csv_file=myCSV_test,
                                     root_dir=myTEST, classe_type='density',
                                array_type='bw',data_type='pkl',transform=transforms.ToTensor())

    os.getcwd()
    data_size = len(train_dataset)+len(val_dataset)
    split=int(np.floor(validation_split*data_size))
    training=len(train_dataset)
    # split the data into training and validation
    training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))
    print('--> loading training data')
    train = torch.utils.data.DataLoader(
            dataset=train_dataset,
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
    optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
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

    base_model = train_model(
   	 model,train,val,device, criterion, optimizer,num_epochs,exp_lr_scheduler,savepath, method,dataname,modelname,modelpath,
  	  batch_size,class_names )
else:
    print('--> reading CSV data')
    print('cwd',os.getcwd())
    test_dataset=Dataset_csv_pkl(csv_file=myCSV_test,
                                     root_dir=myTEST, classe_type='density',
                                array_type='bw',data_type='pkl',transform=transforms.ToTensor())

    os.getcwd()
   
    test = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=16,
            shuffle=True)
    print('--> defining classes/labels')
    class_names =test_dataset.classes
    inputs,labels,paths=next(iter(test))
    img_sizeX,img_sizeY= inputs.shape[-1],inputs.shape[-2]
    num_of_test_samples = len(test_dataset) #total validation samples
    steps_per_epoch = np.ceil(num_of_test_samples // batch_size)
    number_classes = len(class_names)
    print('number of samples in the test set:', num_of_test_samples )
    print('number of samples in a batch',len(test))
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
    optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
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

    print('--> loading saved model')
    temp_list_model=[files for files in os.listdir(savepath) if files.endswith('.pth') ]
    print(savepath)
    print(temp_list_model)
    print('#############################')
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    if len(temp_list_model)!=0:
        list_model=[savepath + files for files in temp_list_model]
        print('ok')
    else:
        print('no model saved')
    print(list_model[0])
    print(os.getcwd())
    list_model.sort(key=os.path.getctime)
    print(list_model)
    checkpoint=torch.load(list_model[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss=checkpoint['val loss']
    _loss=checkpoint['train loss']
    epochs=checkpoint['train epoch']
    model.eval()
size_samp=1000
str_classes_names=" ".join(str(x) for x in class_names)
print('--> computing/saving confusion matrices')
cm=simple_confusion_matrix(model,test,device,number_classes,class_names)
np.savetxt(savepath+method+'_'+dataname+'cm_test.txt',cm,fmt='%d',header=str_classes_names)

print('--> computing/saving classification outputs')
classification_prediction_dens(test,size,model,whole_dataset,savepath,31,'test')
csv_file=savepath+'predictions_class_density_bw_v_h_res_'+str(size)+'_'+str(myseed)+'_test.csv'
#density_as_func_proba_density(csv_file,size_samp)
#classification_prediction_bis(val,size,model)


print('--> finished!')
