import torchvision
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import csv
from IPython.display import Image
import pickle
import time
import os
import re
import itertools
import matplotlib.ticker as plticker
import time
#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
import copy
from tqdm import tqdm, trange
###################################################################################################
def train_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
"""
        Function for classification training.
        """
    start_epoch=0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    accuracy=[]
    _loss=[]
    val_accuracy=[]
    val_loss=[]
    epochs=[]
    val_epochs=[]
    number_classes=len(class_names)
    #if 'p' in class_names:
    #temp_class_names=[x.split('p')[1] for x in class_names]
    #    str_classes_names=" ".join(str(x) for x in temp_class_names)
    #else:
    temp_class_names=class_names
    str_classes_names=" ".join(str(x) for x in temp_class_names)
    temp_list_model=[files for files in os.listdir(savepath) if files.startswith(modelname) and files.endswith('.pth') ]
    print('#############################')
    print('modelpath',modelname)
    if len(temp_list_model)!=0:
        list_model=[savepath + files for files in temp_list_model]
        print(list_model[0])
        print(os.getcwd())
        list_model.sort(key=os.path.getctime) #sort list of saved model by oldest to more recent
        print(list_model)
        checkpoint=torch.load(list_model[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        best_model_wts = copy.deepcopy(model.state_dict()) #save best model
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        val_loss=checkpoint['val loss'] #load previous value of metrics
        accuracy=checkpoint['train acc']
        _loss=checkpoint['train loss']
        val_accuracy=checkpoint['val acc']
        val_loss=checkpoint['val loss']
        epochs=checkpoint['train epoch']
        best_loss=min(val_loss)
        start_epoch=max(epochs)+1
        print('Checkpoint found, training restarted at epoch: '+str(start_epoch))
    since=time.time()


    init=time.time()
    for epoch in range(start_epoch,num_epochs): #restart model at start_epoch
        print('Epoch {}/{}'.format(epoch+1,num_epochs)) 
        print('-' * 10)

    #two phases training and validating
        for phase in [train,val]:
            if phase == train:
                #print('Training', end=" ")
                len_dataset=len(train.dataset)
                name_phase='Training'
                model.train() # set the model to training mode
            else:
                #print('Validation', end=" ")
                len_dataset=len(val.dataset)
                name_phase='Validation'
                model.eval() # set the model to evaluation mode
            running_loss=0.0
            running_corrects=0.0

            # Here's where the training happens
            # print('--- iterating through data ...')

            for i, (inputs,labels,paths) in tqdm(enumerate(phase), total=int(len_dataset/phase.batch_size),desc=name_phase):
               # if inputs.shape[1]>4:
                    #inputs=inputs.unsqueeze(1)
                #inputs=inputs.double()
                #labels=labels.double()
                inputs=inputs.to(device)
                labels=labels.to(device)
                #paths=paths.to(device)
                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()
                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    outputs=outputs.double()
                    _, preds= torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                #still for the training phase we need to implement backword process and optimization
                    if phase==train:
                        loss.backward()
                        optimizer.step()
                # We want variables to hold the loss statistics
                #loss.item() extract the loss value as float then it is multiply by the batch size
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+= torch.sum(preds==labels.data)
            if phase == train:
                scheduler.step()

            if phase == train:
                epoch_loss= running_loss/len(phase.dataset)
                epoch_acc = running_corrects.double()/ len(phase.dataset)
                print('{} loss= {:4f}, accuracy= {:4f}'.format(
                    'Training result:', epoch_loss, epoch_acc))
                accuracy.append(epoch_acc)
                _loss.append(epoch_loss)
                epochs.append(epoch)

            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                epoch_acc = running_corrects.double()/len(val.dataset)
                print('{} val_loss= {:4f}, val_accuracy= {:4f}'.format(
                    'Validation result:', epoch_loss, epoch_acc))
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)
                #print('1:',val_loss)
                #print('1:',val_accuracy)
            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == val and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        cm=simple_confusion_matrix(model,val,device,number_classes,class_names)
        np.savetxt(savepath+method+'_'+dataname+'cm_val_'+str(epoch)+'.txt',cm,fmt='%d',header=str_classes_names,comments='')
        #print(epochs)
        #print(_loss)
        #print(accuracy)
        #print(val_loss)
        #print(val_accuracy)
        model.load_state_dict(best_model_wts)
        train_data=list(zip(epochs,_loss,accuracy,val_loss,val_accuracy))
        #print(train_data)
        header = '{0:^5s}   {1:^7s}   {2:^5s}   {3:^8s}   {4:^7s}'.format('epochs', 'loss', \
        'accuracy', 'val loss',   'val accuracy')
        filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
        np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f','  %.7f','     %.7f'])
        if time.time()-init>18000: #save at checkpoint
            torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train acc': accuracy,
            'val acc' : val_accuracy,
            'train loss': _loss,
            'val loss' : val_loss,
            'cm':cm}, modelpath+'_epochs_'+str(epoch)+'.pth')
            init=time.time()
            print('saved')
    torch.save({'train epoch': epochs, #save best model and metrics at the end of the training
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train acc': accuracy,
                'val acc' : val_accuracy,
                'train loss': _loss,
                'val loss' : val_loss,
                'cm':cm}, modelpath+'_best.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    train_data=list(zip(epochs,_loss,accuracy,val_loss,val_accuracy))
    header = '{0:^5s}   {1:^7s}   {2:^5s}   {3:^8s}   {4:^7s}'.format('epochs', 'loss', \
    'accuracy', 'val loss',   'val accuracy')
    filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f','  %.7f','     %.7f'])

    return model, accuracy, _loss, val_accuracy, val_loss, epochs, val_epochs
##########################################################################################
class Dataset_csv_pkl(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir,size=100,classe_type='span',array_type='clus',data_type='pkl',type_file='normal', transform=None):
        """
        Function to load pkl or imgsamples from information in csv.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            classe_type(string): Label trained between 'dens', 'span' and 'corr' or 'new'
            array_type(string): black and white type lattice ('bw') or with cluster number ('clus')
            data_type(string): Training with 'img' or 'pkl'
            type_file (string): if not normal symlink
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file
        self.size=size
        self.classe_type=classe_type
        self.array_type=array_type
        self.data_type=data_type
        self.type_file=type_file
        num_column=0

        if classe_type=='density': #column read in csv file, depends on classe~_type
            num_column=1
        elif classe_type=='corr':
            num_column=8
        elif classe_type=='new':
            num_column=9
            print(num_column)
        else:
            num_column=2
        classes=[]
        classe_order=[]
        self.num_column=num_column

        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[num_column])
                if classe_type=='corr':
                    classe_order.append(column[1])
        if classe_type=='corr':
            classes_zip=zip(classes,classe_order)
            df = pd.DataFrame(classes_zip,columns=['classes','classes order'])
        else:
             df = pd.DataFrame(classes,columns=['classes'])
        if classe_type=='density':
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            #classes=[p.strip('p') for p in classes_ordered]
            #classes=[float(p.strip('p')) for p in classes_ordered]
            classes=classes_ordered
            classes.sort()
        elif classe_type=='new':
            classes= list(df.drop_duplicates(keep="first")['classes'])
        elif classe_type=='corr':
            classes_ordered=df.sort_values('classes order').drop_duplicates(subset=['classes'], keep='first')
            classes_unprocessed=classes_ordered['classes'].tolist()
            classes = [str(round(float(num), 5)) for num in classes_unprocessed]
            print(classes)
        else:
            classes = list(df.drop_duplicates(keep="first")['classes'])
            #classes=[span for span in classes_ordered]
            #classes=[float(span) for span in classes_ordered]
            classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx=class_to_idx


    def _find_classes(self,root_dir,csv_file):
        if self.classe_type=='corr':
            num_column=8
        elif self.classe_type=='new':
            num_column=9
        elif self.classe_type=='density':
            num_column=1
        else:
            num_column=2
        classes=[]
        classe_order=[]

        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[num_column])
                if self.classe_type=='corr':
                    classe_order.append(column[1])
        if self.classe_type=='corr':
            classes_zip=zip(classes,classe_order)
            df = pd.DataFrame(classes_zip,columns=['classes','classes order'])
        else:
             df = pd.DataFrame(classes,columns=['classes'])

        if self.classe_type=='density':
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            #classes=[p.strip('p') for p in classes_ordered]
            #classes=[float(p.strip('p')) for p in classes_ordered]
            classes=classes_ordered
            classes.sort()
        elif self.classe_type=='new':
            classes= list(df.drop_duplicates(keep="first")['classes'])
        elif self.classe_type=='corr':
            classes_ordered=df.sort_values('classes order').drop_duplicates(subset=['classes'], keep='first')
            classes_unprocessed=classes_ordered['classes'].tolist()
            classes = [str(round(float(num), 5)) for num in classes_unprocessed]
        else:
            classes= list(df.drop_duplicates(keep="first")['classes'])
            #classes=[span for span in classes_ordered]
            #classes=[float(span) for span in classes_ordered]
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        #global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_idx=self.csv_data.iloc[idx, 0]
        #print('data_idx', data_idx)
        label_idx_0=self.csv_data.iloc[idx, 1]
        #print('label_idx_0',label_idx_0)
        label_idx_1=self.csv_data.iloc[idx, 2]
        #print('label_idx_1',label_idx_1)
        if self.type_file=='symlink':
            path = os.path.join(self.root_dir,
                                data_idx)
        else:
            path = os.path.join(self.root_dir,label_idx_0,data_idx)
        if self.data_type=='image':
            data = mpimg.imread(path)
        else:
            data_pkl=pickle.load(open(path,"rb"))
            data=data_pkl['cluster_pbc_norm']
        classes, self.class_to_idx=self._find_classes(self.root_dir,self.csv_file)
        if self.array_type=='bw' and self.data_type=='pkl':
            array=np.array([1 if x!=0 else x for x in data.flat]).reshape(self.size,self.size)
        else:
            array=data
            print(array)
        label1=str(self.csv_data.iloc[idx, self.num_column])
        if self.classe_type=='density':
            label = label1

        #print('ok6')
        elif self.classe_type=='corr':
            label=str(np.round(float(label1),decimals=5))
        elif self.classe_type=='new':
            label = label1
        else:
            label=label1
        #print(class_to_idx)

        label =self.class_to_idx[label]

        sample = {'data': array, 'labels': label, 'path':path}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        labels=sample['labels']
        paths=sample['path']
        #print(time.time()-start)
        return data, labels,paths
##########################################################################################
class MyImageFolder2(torchvision.datasets.ImageFolder):
    """Custom dataset that includes the paths of each sample. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(MyImageFolder2, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
##########################################################################################
class DatasetFolder2(torchvision.datasets.DatasetFolder):
    """Custom dataset that includes the paths of each sample. Extends
    torchvision.datasets.DatasetFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(DatasetFolder2, self).__getitem__(index)
        # the image file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

###########################################################################################
class Dataset_csv_img_p(torch.utils.data.Dataset): #classification of images for p
    def __init__(self, csv_file, root_dir, transform=None):
        self.img = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file
        classes=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[1])
        df = pd.DataFrame(classes, columns=['classes'])
        classes = list(df.drop_duplicates(keep="first")['classes'])
        classes.sort()
        self.classes = classes
        print(self.classes)
        self.class_to_idx={cls_name: i for i, cls_name in enumerate(self.classes)}

    def _find_classes(self, root_dir,csv_file):
        classes=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[1])
        df = pd.DataFrame(classes, columns=['classes'])
        classes = list(df.drop_duplicates(keep="first")['classes'])
        classes.sort()
        self.classes = classes
        class_to_idx={cls_name: i for i, cls_name in enumerate(self.classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            #print(idx)
        #print(idx)
        image_idx=self.img.iloc[idx, 0]
        label_idx=self.img.iloc[idx, 1]
       #print(self._find_classes(self.root_dir))
        path = os.path.join(self.root_dir,label_idx,
                                image_idx)
        image = mpimg.imread(path)
        label=self.img.iloc[idx, 1]
        #print(label)
        classes, class_to_idx=self._find_classes(self.root_dir,self.csv_file)
        label = class_to_idx[label]
        sample = {'image': image, 'labels': label, 'path':path}
        #print(sample)

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        images= sample['image']
        #img = transform.resize(images, (28, 28))
        labels=sample['labels']
        paths=sample['path']
        return images, labels,paths
###########################################################################################
class Dataset_csv_pkl_reg(torch.utils.data.Dataset): 
    def __init__(self, csv_file, root_dir,size,classe_type='span',array_type='clus',type_file='normal', transform=None):
        """
        Function to load pkl or img samples from information in csv. Here the label are NOT encoded.
        This class is used to load the dataset for a regression training.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            size (int): Size of the lattice 
            classe_type(string): Label trained between 'dens', 'span' and 'corr'
            array_type(string): black and white type lattice ('bw') or with cluster number ('clus')
            type_file (string): if not normal symlink
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = pd.read_csv(csv_file,error_bad_lines=False,header=None)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.size=size
        self.classe_type=classe_type
        self.array_type=array_type
        self.type_file=type_file
        num_column=0
        if classe_type=='density':
            num_column=1
        elif classe_type=='corr':
            num_column=7
        elif classe_type=='avg corr':
            num_column=8
        else:
            num_column=2
        classes=[]
        self.num_column=num_column
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[num_column])
        df = pd.DataFrame(classes, columns=['classes'])

        if classe_type=='density':
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            classes=[float(p.strip('p')) for p in classes_ordered]
            classes.sort()
        elif classe_type=='corr':
            classes_unprocessed = list(df['classes'])
            classes=[float(corr) for corr in classes_unprocessed]
        elif classe_type=='avg corr':
            classes_unprocessed = list(df['classes'])
            classes=[float(corr) for corr in classes_unprocessed]

        else:
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            classes=[float(span) for span in classes_ordered]
            classes.sort()
        self.classes = classes

    def _find_classes(self, root_dir,csv_file,classe_type):
        if classe_type=='corr':
            num_column=7
        elif classe_type=='density':
            num_column=1
        elif classe_type=='avg corr':
            num_column=8
        else:
            num_column=2
        classes=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[num_column])
        df = pd.DataFrame(classes, columns=['classes'])

        if classe_type=='density':
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            classes=[float(p.strip('p')) for p in classes_ordered]
            classes.sort()
        elif classe_type=='corr':
            classes_unprocessed = list(df['classes'])
            classes=[float(corr) for corr in classes_unprocessed]
        elif classe_type=='avg corr':
            classes_unprocessed = list(df['classes'])
            classes=[float(corr) for corr in classes_unprocessed]

        else:
            classes_ordered = list(df.drop_duplicates(keep="first")['classes'])
            classes=[float(span) for span in classes_ordered]
            classes.sort()
        self.classes = classes
        return classes

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_idx=self.img.iloc[idx, 0]
        label_idx_0=self.img.iloc[idx, 1]
        if self.type_file=='symlink':
            path = os.path.join(self.root_dir,
                                data_idx)
        else:
            path = os.path.join(self.root_dir,label_idx_0,
                                data_idx)
        data_pkl=pickle.load(open(path,"rb"))
        array=data_pkl['cluster_pbc_norm']
        if self.array_type=='bw':
            new_array=np.array([1 if x!=0 else x for x in array.flat]).reshape(self.size,self.size)
        else:
            new_array=array
        label1=self.img.iloc[idx, self.num_column]

        classes=self._find_classes(self.root_dir,self.csv_file,self.classe_type)

        if self.classe_type=='density':
            label = float(label1.strip('p'))
        else:
            label=float(label1)

        sample = {'data': new_array, 'labels': label, 'path':path}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        labels=sample['labels']
        paths=sample['path']
        return data, labels,paths

##########################################################################################
def train_reg_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
#Function for regression training
    best_loss = 20000.0
    _loss=[]
    val_loss=[]
    epochs=[]
    val_epochs=[]
    temp_list_model=[files for files in os.listdir(savepath) if files.startswith(modelname) and files.endswith('.pth') ]
    print('#############################')
    print('modelpath',modelname)
    if len(temp_list_model)!=0:
        list_model=[savepath + files for files in temp_list_model]
        print(list_model[0])
        print(os.getcwd())
        list_model.sort(key=os.path.getctime)
        print(list_model)
        checkpoint=torch.load(list_model[-1])

        model.load_state_dict(checkpoint['model_state_dict'])

        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss=checkpoint['val loss']

        _loss=checkpoint['train loss']

        val_loss=checkpoint['val loss']
        epochs=checkpoint['train epoch']
        best_loss=min(val_loss)
        start_epoch=max(epochs)+1
        print('Checkpoint found, training restarted at epoch: '+str(start_epoch))
    since=time.time()
    init=time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-' * 10)

    #two phases training and validating
        for phase in [train,val]:
            if phase == train:
                #print('Training', end=" ")
                len_dataset=len(train.dataset)
                name_phase='Training'
                model.train() # set the model to training mode
            else:
                #print('Validation', end=" ")
                len_dataset=len(val.dataset)
                name_phase='Validation'
                model.eval() # set the model to evaluation model

            running_loss=0.0
            running_corrects=0.0

            # Here's where the training happens
            # print('--- iterating through data ...')

            for i, (inputs,labels,paths) in tqdm(enumerate(phase), total=int(len_dataset/phase.batch_size),desc=name_phase):
                inputs=inputs.double()
                inputs=inputs.to(device)
                labels=labels.to(device)
                #labels = [_label.cuda() for _label in label]
                labels = labels.unsqueeze(1)
                #paths=paths.to(device)
                print(labels[0],labels[1], labels[2], labels[3], labels[4])
                print(paths[0],paths[1], paths[2], paths[3], paths[4])
                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()

                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    loss=criterion(outputs,labels.double())

                #still for the training phase we need to implement backword process and optimization

                    if phase==train:
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                #loss.item() extract the loss value as float then it is multiply by the batch size
                running_loss+=loss.item()*inputs.size(0)


            if phase == train:
                scheduler.step()

            if phase == train:
                epoch_loss= running_loss/len(phase.dataset)
                print('{} loss= {:4f}'.format(
                    'Training result:', epoch_loss))
                _loss.append(epoch_loss)
                epochs.append(epoch)

            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                print('{} val_loss= {:4f}'.format(
                     'Validation result:', epoch_loss))
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == val and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        if time.time()-init>=32400:
            model.load_state_dict(best_model_wts)
            train_data=list(zip(epochs,_loss,val_loss))
            header = '{0:^5s}   {1:^7s}   {2:^7s}'.format('epochs', 'loss', \
             'val loss')
            filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
            np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f'])

            torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train loss': _loss,
            'val loss' : val_loss }, modelpath+'_epochs_'+str(epoch)+'.pth')
            init=time.time()
            print('saved')

        print()
    epochs=[epochs[i]+1 for i in range(len(epochs))]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    train_data=list(zip(epochs,_loss,val_loss))
    header = '{0:^5s}   {1:^7s}     {2:^8s}   '.format('epochs', 'loss', \
     'val loss')
    filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f'])
    torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train loss': _loss,
            'val loss' : val_loss }, modelpath+'_epochs_'+str(epoch)+'.pth')

    return model, _loss, val_loss, epochs, val_epochs


##########################################################################################
class Resnet18_regression(nn.Module):
    def __init__(self,resnet18_model):
        super(Resnet18_regression, self).__init__()
        self.resnet18_model=resnet18_model
        num_ftrs = self.resnet18_model.fc.out_features
        self.new_layers=nn.Sequential(nn.Linear(num_ftrs,1))#nn.ReLU(),
                                      #nn.Linear(num_ftrs,256),
                                      #nn.ReLU(),
                                      #nn.Linear(256,64),
                                      #nn.Linear(64,1))
    def forward(self,x):
        x=self.resnet18_model(x)
        x=self.new_layers(x)
        return x

##########################################################################################
def imshow(inp, title=None):
    # torch convention gives [channel, height,width] but imshow gives [height,width,channel] hence the transpose
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(5,5))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated

##########################################################################################
def plot_save_loss(epochs, _loss, val_loss, savepath,method,dataname):
    fig=plt.figure()
    plt.plot(epochs,val_loss, label='val loss')
    plt.plot(epochs,_loss, label='training loss')
    plt.legend(loc='upper left')
    fig.savefig(savepath+method+'_'+dataname+'_loss'+'.png')
    plt.show()
##########################################################################################


def plot_save_accuracy(epochs, accuracy, val_accuracy, savepath,method,dataname):
    fig=plt.figure()
    plt.plot(epochs,val_accuracy, label='val accuracy')
    plt.plot(epochs,accuracy, label='training accuracy')
    plt.legend(loc='upper left')
    fig.savefig(savepath+method+'_'+dataname+'_accuracy'+'.png')
    plt.show()


##########################################################################################
def visualize_model(model,chosen_set,device,class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(chosen_set):
            if inputs.shape[1]>4:
                inputs=inputs.unsqueeze(1)
                inputs=inputs.float()
            inputs=inputs.to(device)
            labels=labels.to(device)

            outputs = model(inputs) #value of the output neurons
            _, preds = torch.max(outputs, 1) #gives the max value and stores in preds the neurons to which it belongs

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images//2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('predicted: {}; \n true label {}; \n path: {};'.format(class_names[preds[j]] ,
                                                                     class_names[labels[j]],paths[j])
                            )
                imshow(inputs.cpu().data[j],paths[j])

                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


##########################################################################################
def visualize_model_misclassified(model,chosen_set,device,class_names, num_images=6): #gives shows only the misclassified images
    was_training = model.training
    model.eval()
    images_handeled  = 0
    #fig = plt.figure()

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(chosen_set):

            inputs=inputs.to(device)
            labels=labels.to(device)
            inputs=inputs.float()

            outputs = model(inputs) #value of the output neurons
            _, preds = torch.max(outputs, 1) #gives the max value and stores in preds the neurons to which it belongs

            for j in range(inputs.size()[0]):
                if labels[j]!=preds[j]: # and abs(labels[j]-preds[j])>1:
                #print(inputs.size()[0])
                    images_handeled  += 1
                    ax = plt.subplot(num_images//2, 2, images_handeled )
                    ax.axis('off')
                    ax.set_title('predicted: {}; \n true label {}; \n path: {};'.format(class_names[preds[j]] ,
                                                                     class_names[labels[j]],paths[j])
                            )

                    filename, file_extension = os.path.splitext(paths[j])
                    image_s=filename+'_s.png'
                    #print(image_s)
                    #plt.figure(figsize=(20,20))
                    image = Image(image_s)
                    display(image)
                    plt.pause(0.001)

                    if images_handeled  == num_images:
                        model.train(mode=was_training)
                        return

##########################################################################################

def analysis(model,chosen_set,device,number_classes,class_names,savepath, method,dataname):
    class_correct = list(0. for i in range(number_classes))
    class_total = list(0. for i in range(number_classes))
    accuracy=list(0. for i in range(number_classes))
    average=list(0. for i in range(number_classes))
    total_avg=0
    with torch.no_grad():
        for i, (data) in enumerate(chosen_set):
            inputs=data[0]
            labels=data[1]
            #if inputs.shape[1]>4:
                #inputs=inputs.unsqueeze(1)
                #inputs=inputs.float()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            c = (preds == labels).squeeze()
            for i in range(inputs.size()[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_classes):
        average[i]=(class_correct[i] / class_total[i])*100
        total_avg+=average[i]

        print('Accuracy of %5s : %2d %%' % (class_names[i], average[i]))
    total_avg=total_avg/number_classes
    print(total_avg)
    print('Accuracy of the network on the  set: %2d %%' % (
        total_avg))
    plt.figure(figsize=(14,14))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)

    plt.plot(class_names,average)
    plt.savefig(savepath+method+'_'+dataname+'_classacc_'+'.png')

##########################################################################################
#@torch.no_grad()
def simple_confusion_matrix(model,loader,device,number_classes,class_names):
    with torch.no_grad():
        confusion_matrix = torch.zeros(number_classes, number_classes)
        for i, (data) in enumerate(loader):
            inputs=data[0]
            labels=data[1]
            if inputs.shape[1]>4:
                inputs=inputs.unsqueeze(1)
                inputs=inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for p,t in zip(preds.view(-1),labels.view(-1)):
		#confusion_matrix[t.long(), p.long()] += 1
                confusion_matrix[p.long(), t.long()] += 1
    return confusion_matrix

##########################################################################################
def confusion_matrix_torch(cm, target_names,savepath, method,dataname,cmap=None,title='Confusion Matrix'):
    #accuracy = np.trace(cm) / float(np.sum(cm))
    #misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig=plt.subplots(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title,fontsize=40)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=40)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=30, fontsize=35)
        plt.yticks(tick_marks, target_names,fontsize=35)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] == 0 or cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.rcParams.update({'font.size': 5})
    plt.ylim(len(cm)-0.5, -0.5)
    plt.xlabel('True label',fontsize=50)
    plt.ylabel('Predicted label',fontsize=50)
    plt.savefig(savepath+method+'_'+dataname+'.png')
    plt.show()
    return

######################################################################################################
def classification_prediction_cor(dataloader,dataset,size,model,seed,nb_classes=31,data_type='val'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
    list_p=[]
    model=model.to('cpu')
    header_l=['path','density','true label','prediction']
    class_to_idx=dataset.class_to_idx
    idx_to_class={v: k for k, v in class_to_idx.items()}
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            #inputs=inputs.float()
            labels.numpy()
            predictions = model(inputs) #value of the output neurons
            _, pred= torch.max(predictions,1)
            for j in range(inputs.size()[0]):
                p_occ=paths[j].split('_')[7]
                regex2 = re.compile('\d+\.\d+')
                p_reg=re.findall(regex2,p_occ)
                p=float(p_reg[0])
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_p=p
                temp_labels=labels[j].item()
                temp_preds=pred[j].numpy()
                temp_preds=int(temp_preds)
                temp_labels=int(temp_labels)
                real_pred=idx_to_class[temp_preds]
                real_label=idx_to_class[temp_labels]
                list_paths.append(temp_paths)
                list_p.append(temp_p)
                list_labels.append(real_label)
                list_preds.append(real_pred)

    dict = {'path':list_paths,'density':list_p,'label':list_labels,'prediction':list_preds}
    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_class_corr_bw_'+str(size)+'_'+str(myseed)+'_'+str(nb_classes)+'_'+str(data_type)+'_.csv',index=False)

    return
#############################################################################################
def reg_prediction_dens(dataloader,size,model,myseed,savepath,nb_classes=31,data_type='val'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
    model=model.to('cpu')
    header_l=['path','label','prediction']

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            #inputs=inputs.float()
            labels.numpy()
            predictions = model(inputs) #value of the output neurons
           # _, pred= torch.max(predictions,1)
            print('pred',predictions)

            for j in range(inputs.size()[0]):
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_labels=labels[j].detach().cpu().numpy()
                temp_preds=predictions[j].detach().cpu().numpy()
                print('temp_labels',temp_labels)
                print('temp_preds',temp_preds[0])
                list_paths.append(temp_paths)
                list_labels.append(temp_labels)
                list_preds.append(temp_preds[0])

    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds}

    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_reg_density_bw_int_'+str(size)+'_'+str(myseed)+'_sulis_'+str(data_type)+'.csv',index=False)

    return
#############################################################################################
def reg_prediction_cor(dataloader,model,size,myseed,savepath,nb_classes=31,data_type='val'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
    list_p=[]
    model=model.to('cpu')
    header_l=['path','density','true label','prediction']

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            #inputs=inputs.double()
            labels.numpy()
            predictions = model(inputs) #value of the output neurons

            for j in range(inputs.size()[0]):               
                #p_occ=paths[j].split('_')[7] #This section is commented out as symlink file do not have p in path
                #regex2 = re.compile('\d+\.\d+')
                #p_reg=re.findall(regex2,p_occ)
                ##print(p_reg)
                #p=float(p_reg[0])
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_labels=labels[j].detach().cpu().numpy()
                temp_preds=predictions[j].detach().cpu().numpy()
                print('temp_labels',temp_labels)
                print('temp_preds',temp_preds)
                list_paths.append(temp_paths)
                list_labels.append(temp_labels)
                list_preds.append(temp_preds[0])
                #list_p.append(p)


    #dict = {'path':list_paths,'density':list_p,'label':list_labels,'prediction':list_preds}
    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds}
    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_reg_corr_bw_int_'+str(size)+'_'+str(myseed)+'_'+str(nb_classes)+'_'+str(data_type)+'.csv',index=False)

    return
################################################################################################
def classification_prediction_dens(dataloader,size,seed,whole_dataset,savepath,nb_classes=31,data_type='val'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]

    #model=model.to('cpu')
    header_l=['path','label','prediction']
    class_to_idx=whole_dataset.class_to_idx
    idx_to_class={v: k for k, v in class_to_idx.items()}
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            #inputs=inputs.float()
            #labels.numpy()
            predictions = model(inputs) #value of the output neurons
            _, pred= torch.max(predictions,1)

            for j in range(inputs.size()[0]):
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_labels=labels[j].detach().cpu().numpy()
                temp_preds=pred[j].detach().cpu().numpy()
                temp_preds=int(temp_preds)
                temp_labels=int(temp_labels)
                real_pred=idx_to_class[temp_preds]
                real_label=idx_to_class[temp_labels]
                list_paths.append(temp_paths)
                list_labels.append(real_label)
                list_preds.append(real_pred)
    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds}
    df = pd.DataFrame(dict)
    df.to_csv(savepath+str(size)+'_'+str(myseed)+'_'+str(nb_classes)+'_'+str(data_type)+'_.csv',index=False)

    return
###################################################################################
def density_as_func_proba_density(csv_file,size_samp=10000):
    data=pd.read_csv(csv_file)
    density=data['true label'].unique()
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
    data_train=pd.read_csv('../../Data_csv/real_proportions_density_in_train_dataset_55_62_'+str(size_samp)+'.csv')

    class_percoSamp=data_train['density'].unique()
    len_percoSamp=len(class_percoSamp)
    percoSamp=[size_samp]*len_percoSamp
    percoNS=data_train['non_spanning']
    percoS=data_train['spanning']
    for p in density:
        new_df = data[data['density']==p]
        nb_p=new_df['density'].count()
        new_df_1=new_df[new_df['label']==1]
        new_df_0=new_df[new_df['label']==0]
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
        ml_Samp.append(nb_p)

    dict = {'density': p_list,'percoSamp':percoSamp,'percoNS':percoNS,'percoS':percoS,'mlSamp':ml_Samp,'SpS': pred_1_1, 'SpNS': pred_1_0,'NSpS': pred_0_1,'NSpNS': pred_0_0}

    df = pd.DataFrame(dict)
    df.to_csv(savepath+'class_density_bw_proba_'+str(size)+'_'+str(size_samp)+'_'+str(myseed)+'.csv',index=False)

    return p_list,pred_1_1,pred_1_0,pred_0_1,pred_0_0
#################################################################################
def classification_prediction_span(dataloader,size,size_samp,savepath,myseed,model,data_type='val'):
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
    list_p=[]
    model=model.to('cpu')
    header_l=['path','density','true label','prediction']
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):         
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            #inputs=inputs.float()
            labels.numpy()          
            predictions = model(inputs) #value of the output neurons
            _, pred= torch.max(predictions,1)
           
            for j in range(inputs.size()[0]):
                p_occ=paths[j].split('_')[7]      
                regex2 = re.compile('\d+\.\d+')
                p_reg=re.findall(regex2,p_occ)
                p=float(p_reg[0])
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_p=p
                temp_labels=labels[j].item()
                temp_preds=pred[j].numpy()
                list_paths.append(temp_paths)
                list_p.append(temp_p)
                list_labels.append(temp_labels)
                list_preds.append(temp_preds)
               
    dict = {'path':list_paths,'density':list_p,'label':list_labels,'prediction':list_preds} 

    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_class_span_bw_v_h_res_'+str(size)+'_'+str(size_samp)+'_'+str(myseed)+'_'+data_type+'.csv',index=False)
                       
    return

