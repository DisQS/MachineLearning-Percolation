import torchvision
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import csv
import pickle
import time
import os
#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
import copy
from tqdm import tqdm, trange

def train_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
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
        list_model.sort(key=os.path.getctime)
        print(list_model)
        checkpoint=torch.load(list_model[-1])
       
        model.load_state_dict(checkpoint['model_state_dict'])

        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss=checkpoint['val loss']
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
    for epoch in range(start_epoch,num_epochs):
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
    torch.save({'train epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train acc': accuracy,
                'val acc' : val_accuracy,
                'train loss': _loss,
                'val loss' : val_loss,
                'cm':cm}, modelpath+'.pth')
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
def tuning(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
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
        list_model.sort(key=os.path.getctime)
        print(list_model)
        checkpoint=torch.load(list_model[-1])
       
        model.load_state_dict(checkpoint['model_state_dict'])

        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss=checkpoint['val loss']
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
    for epoch in range(start_epoch,num_epochs):
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
            
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0   
            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                epoch_acc = running_corrects.double()/len(val.dataset)
                print('{} val_loss= {:4f}, val_accuracy= {:4f}'.format(
                    'Validation result:', epoch_loss, epoch_acc))
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)
                val_loss = running_loss
                val_steps+=1
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save({'train epoch': epochs,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'train acc': accuracy,
                         'val acc' : val_accuracy,
                         'train loss': _loss,
                         'val loss' : val_loss,
                         'cm':cm}, modelpath+'.pth')
                #tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

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
        #if time.time()-init>=32400:
        #    model.load_state_dict(best_model_wts)
        #    train_data=list(zip(epochs,_loss,accuracy,val_loss,val_accuracy))
        #    #print(train_data)
        #    header = '{0:^5s}   {1:^7s}   {2:^5s}   {3:^8s}   -{4:^7s}'.format('epochs', 'loss', \
        #    'accuracy', 'val loss',   'val accuracy')
        #    filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
        #    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f','  %.7f','     %.7f'])
    
        #    torch.save({'train epoch': epochs,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'train acc': accuracy,
        #    'val acc' : val_accuracy,
        #    'train loss': _loss,
        #    'val loss' : val_loss,
        #    'cm':cm}, modelpath+'_epochs_'+str(epoch)+'.pth')
        #    init=time.time()
        #    print('saved')

    
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
    def __init__(self, csv_file, root_dir,size=100,classe_type='span',array_type='clus',data_type='pkl', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
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
        num_column=0

        if classe_type=='density':
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
        global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_idx=self.csv_data.iloc[idx, 0]
        #print('data_idx', data_idx)
        label_idx_0=self.csv_data.iloc[idx, 1]
        #print('label_idx_0',label_idx_0)
        label_idx_1=self.csv_data.iloc[idx, 2]
        #print('label_idx_1',label_idx_1)
        path = os.path.join(self.root_dir,label_idx_0,
                                data_idx)
        if self.data_type=='image':
            data = mpimg.imread(path)
        else:
            data_pkl=pickle.load(open(path,"rb"))
            data=data_pkl['cluster_pbc_norm']
        classes, class_to_idx=self._find_classes(self.root_dir,self.csv_file)
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
        
        label = class_to_idx[label]

        sample = {'data': array, 'labels': label, 'path':path}
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        labels=sample['labels']
        paths=sample['path']
        #print(time.time()-start)
        return data, labels,paths

####################################################
class former_Dataset_csv_pkl(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir,size=100,classe_type='span',array_type='clus',data_type='pkl', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
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
        num_column=0

        if classe_type=='density':
            num_column=1
        elif classe_type=='corr':
            num_column=8
            
    
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
        #print(df)
        
        
        if classe_type=='density':
            classes_ordered = list(df.drop_duplicates(keep="first")['classes']) 
            classes=classes_ordered
            classes.sort()
           
        elif classe_type=='corr':
            classes_ordered=df.sort_values('classes order').drop_duplicates(subset=['classes'], keep='first')
            classes_unprocessed=classes_ordered['classes'].tolist()
            classes = [str(round(float(num), 5)) for num in classes_unprocessed]

            print(classes)

            
        else:
            classes_ordered = list(df.drop_duplicates(keep="first")['classes']) 
            classes=[span for span in classes_ordered]
            classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx=class_to_idx
    def _find_classes(self,root_dir,csv_file):    
        if self.classe_type=='corr':
            num_column=8
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
            classes=classes_ordered
            classes.sort()
           
        elif self.classe_type=='corr':
            classes_ordered=df.sort_values('classes order').drop_duplicates(subset=['classes'], keep='first')
            classes_unprocessed=classes_ordered['classes'].tolist()
            classes = [str(round(float(num), 5)) for num in classes_unprocessed]
            
        else:
            classes_ordered = list(df.drop_duplicates(keep="first")['classes']) 
            classes=[span for span in classes_ordered]
            classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
  
        data_idx=self.csv_data.iloc[idx, 0]
        label_idx_0=self.csv_data.iloc[idx, 1]
        label_idx_1=self.csv_data.iloc[idx, 2]
        
        path = os.path.join(self.root_dir,label_idx_0,
                                data_idx)
        if self.data_type=='image':
            data = mpimg.imread(path)
        elif self.classe_type=='span':
            data_pkl=pickle.load(open(path,"rb"))
            data=data_pkl['cluster_norm']
        else:
            data_pkl=pickle.load(open(path,"rb"))
            data=data_pkl['cluster_pbc_norm']
        classes, class_to_idx=self._find_classes(self.root_dir,self.csv_file)
        if self.array_type=='bw' and self.data_type=='pkl':
            array=np.array([1 if x!=0 else x for x in data.flat]).reshape(self.size,self.size)
        else:
            array=data

        label1=str(self.csv_data.iloc[idx, self.num_column])
        if self.classe_type=='density':
            label = label1
 
        if self.classe_type=='corr':
            label=str(np.round(float(label1),decimals=5))
        else:
            label=label1
        
        label = class_to_idx[label]

        sample = {'data': array, 'labels': label, 'path':path}
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        labels=sample['labels']
        paths=sample['path']
        return data, labels,paths
##########################################################################################

class MyImageFolder2(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
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
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
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
class Dataset_csv_img_p(torch.utils.data.Dataset):
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
###############################################################################
class Dataset_csv_pkl_multi(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file

        classes_p=[]
        classes_span=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes_p.append(column[1])
                classes_span.append(column[2])
        df_p = pd.DataFrame(classes_p, columns=['classes p'])
        df_span = pd.DataFrame(classes_span, columns=['classes span'])
        classes_p = list(df_p.drop_duplicates(keep="first")['classes p'])
        classes_span = list(df_span.drop_duplicates(keep="first")['classes span'])
        classes_p.sort()
        self.classes_p = classes_p
        classes_span.sort()
        self.classes_span = classes_span
        self.class_to_idx_p={cls_name: i for i, cls_name in enumerate(self.classes_p)}
        self.class_to_idx_span={cls_name: i for i, cls_name in enumerate(self.classes_span)}
       
        
    def _find_classes(self, root_dir,csv_file):    
        classes_p=[]
        classes_span=[]
        
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes_p.append(column[1])
                classes_span.append(column[2])
        df_p = pd.DataFrame(classes_p, columns=['classes p'])
        df_span = pd.DataFrame(classes_span, columns=['classes span'])
        classes_p = list(df_p.drop_duplicates(keep="first")['classes p'])
        classes_span = list(df_span.drop_duplicates(keep="first")['classes span'])
        classes_p.sort()
        self.classes_p = classes_p
        classes_span.sort()
        self.classes_span = classes_span
        class_to_idx_p={cls_name_p: i for i, cls_name_p in enumerate(self.classes_p)}
        class_to_idx_span={cls_name: j for j, cls_name in enumerate(self.classes_span)}
        return classes_p, classes_span, class_to_idx_p,class_to_idx_span

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_idx=self.data.iloc[idx, 0]
        label_idx_0=self.data.iloc[idx, 1]
        label_idx_1=self.data.iloc[idx, 2]
        path = os.path.join(self.root_dir,label_idx_0,
                                data_idx)
           
        data_pkl=pickle.load(open(path,"rb"))
        array=data_pkl['cluster_pbc_norm']
        new_array=np.array([1 if x!=0 else x for x in array.flat]).reshape(100,100)
        classes_p, classes_span, class_to_idx_p,class_to_idx_span =self._find_classes(self.root_dir,self.csv_file)
        label_p= class_to_idx_p[str(label_idx_0)]
        label_span = class_to_idx_span[str(label_idx_1)]
        sample = {'data': new_array, 'label_p': label_p,'label_span':label_span , 'path':path}
        #print(sample)
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        label_p=sample['label_p']
        label_span=sample['label_span']
        paths=sample['path']
        return data, label_p,label_span,paths
        
###############################################################################
class Dataset_csv_pkl_multi_clus(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file
        classes_p=[]
        classes_span=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes_p.append(column[1])
                classes_span.append(column[2])
        df_p = pd.DataFrame(classes_p, columns=['classes p'])
        df_span = pd.DataFrame(classes_span, columns=['classes span'])
        classes_p = list(df_p.drop_duplicates(keep="first")['classes p'])
        classes_span = list(df_span.drop_duplicates(keep="first")['classes span'])
        classes_p.sort()
        self.classes_p = classes_p
        classes_span.sort()
        self.classes_span = classes_span
        self.class_to_idx_p={cls_name: i for i, cls_name in enumerate(self.classes_p)}
        self.class_to_idx_span={cls_name: i for i, cls_name in enumerate(self.classes_span)}

    def _find_classes(self, root_dir,csv_file):    
        classes_p=[]
        classes_span=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes_p.append(column[1])
                classes_span.append(column[2])
        df_p = pd.DataFrame(classes_p, columns=['classes p'])
        df_span = pd.DataFrame(classes_span, columns=['classes span'])
        classes_p = list(df_p.drop_duplicates(keep="first")['classes p'])
        classes_span = list(df_span.drop_duplicates(keep="first")['classes span'])
        classes_p.sort()
        self.classes_p = classes_p
        classes_span.sort()
        self.classes_span = classes_span
        class_to_idx_p={cls_name_p: i for i, cls_name_p in enumerate(self.classes_p)}
        class_to_idx_span={cls_name: j for j, cls_name in enumerate(self.classes_span)}
        return classes_p, classes_span, class_to_idx_p,class_to_idx_span

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_idx=self.data.iloc[idx, 0]
        label_idx_0=self.data.iloc[idx, 1]
        label_idx_1=self.data.iloc[idx, 2]

        path = os.path.join(self.root_dir,label_idx_0,
                                data_idx)
        data_pkl=pickle.load(open(path,"rb"))
        array=data_pkl['cluster_pbc_norm']
        classes_p, classes_span, class_to_idx_p,class_to_idx_span =self._find_classes(self.root_dir,self.csv_file)
        label_p= class_to_idx_p[str(label_idx_0)]
        label_span = class_to_idx_span[str(label_idx_1)]

        sample = {'data': array, 'label_p': label_p,'label_span':label_span , 'path':path}
        #print(sample)
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        label_p=sample['label_p']
        label_span=sample['label_span']
        paths=sample['path']
        return data, label_p,label_span,paths
        
###########################################################################################
class Dataset_csv_pkl_reg(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir,size,classe_type='span',array_type='clus', transform=None):
        self.img = pd.read_csv(csv_file,error_bad_lines=False)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.size=size
        self.classe_type=classe_type
        self.array_type=array_type
        num_column=0

        if classe_type=='density':
            num_column=1
        elif classe_type=='corr':
            num_column=7
    
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
        
###########################################################################################

def raw_file_loader(width,input):
    functions=loadtext(input,comments="#", delimiter="n", unpack=False)
    functions=functions.reshape(width,width,width)
    functions=functions*functions
    return functions
##########################################################################################
def train_reg_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
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
                model.eval() # set the model to evaluation mode
          
            running_loss=0.0
            running_corrects=0.0
            
            # Here's where the training happens
            # print('--- iterating through data ...')
            
            for i, (inputs,labels,paths) in tqdm(enumerate(phase), total=int(len_dataset/phase.batch_size),desc=name_phase):
                
                
                
                inputs=inputs.float()

                inputs=inputs.to(device)
                labels=labels.to(device)
                #labels = [_label.cuda() for _label in label]
                labels = labels.unsqueeze(1)
                #paths=paths.to(device)
                                
                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()
                                
                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    loss=criterion(outputs,labels.float())
                
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
def train_model_multi(model,train,val,device,criterion1,criterion2, optimizer, num_epochs, scheduler,savepath, method,dataname, batch_size):
    #global accuracy, _loss, val_accuracy, val_loss, epochs, val_epochs 
    since=time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_1 = 0.0
    best_acc_2 = 0.0
    accuracy=[]
    accuracy_1=[]
    accuracy_2=[]
    _loss=[]
    _loss_1=[]
    _loss_2=[]
    val_accuracy=[]
    val_accuracy_1=[]
    val_accuracy_2=[]
    val_loss=[]
    val_loss_1=[]
    val_loss_2=[]
    epochs=[]
    val_epochs=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-' * 10)
        
    #two phases training and validating
        for phase in [train,val]:
            if phase == train:
                print('Training', end=" ")
                model.train() # set the model to training mode
            else:
                print('Validation', end=" ")
                model.eval() # set the model to evaluation mode
                
            batches= len(list(enumerate(phase)))
            print('with', batches, 'batches')
            running_loss=0.0
            running_loss_1=0.0
            running_loss_2=0.0
            running_corrects=0.0
            running_corrects_1=0.0
            running_corrects_2=0.0
            
            # Here's where the training happens
            # print('--- iterating through data ...')
            
            for i, (inputs,labels_1,labels_2,paths) in enumerate(phase):
                
                
                print(i*100//batches, '%', end="\r", flush=True)

               # if inputs.shape[1]>4:
                    #inputs=inputs.unsqueeze(1)
                    #inputs=inputs.float()

                inputs=inputs.to(device)

                labels_1=labels_1.to(device)
                labels_2=labels_2.to(device)
                #paths=paths.to(device)
                                
                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()
                                
                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    labels1=outputs['label1']
                    labels2=outputs['label2']
                    #print(type(outputs))
                    _, preds1= torch.max(labels1,1)
                    __, preds2= torch.max(labels2,1)

                    loss_1=criterion1(labels1,labels_1)
                    loss_2=criterion2(labels2,labels_2)
                    loss=loss_1+loss_2
                #still for the training phase we need to implement backword process and optimization
                
                    if phase==train:
                        loss.backward()
                        optimizer.step()
                        
                # We want variables to hold the loss statistics
                #loss.item() extract the loss value as float then it is multiply by the batch size
                running_loss_1+=loss_1.item()*inputs.size(0)
                running_loss_2+=loss_2.item()*inputs.size(0)
                running_loss=running_loss_1+running_loss_2
                running_corrects_1+= torch.sum(preds1==labels_1.data)
                running_corrects_2+= torch.sum(preds2==labels_2.data)
                running_corrects+= running_corrects_1 + running_corrects_2
            if phase == train:
                scheduler.step()            
            
            if phase == train:
                epoch_loss_1= running_loss_1/len(phase.dataset)
                epoch_loss_2= running_loss_2/len(phase.dataset)
                epoch_loss=epoch_loss_1 + epoch_loss_2
                
                epoch_acc_1 = running_corrects_1.double()/ len(phase.dataset)
                epoch_acc_2 = running_corrects_2.double()/ len(phase.dataset)
                epoch_acc= epoch_acc_1 +  epoch_acc_2

                print('{} loss 1= {:4f}, accuracy 1= {:4f}'.format(
                    'Training result:', epoch_loss_1, epoch_acc_1))

                print('{} loss 2= {:4f}, accuracy 2= {:4f}'.format(
                    'Training result:', epoch_loss_2, epoch_acc_2))

                print('{} loss= {:4f}, accuracy= {:4f}'.format(
                    'Training result:', epoch_loss, epoch_acc))

                accuracy_1.append(epoch_acc_1)
                _loss_1.append(epoch_loss_1)

                accuracy_2.append(epoch_acc_2)
                _loss_2.append(epoch_loss_2)

                accuracy.append(epoch_acc)
                _loss.append(epoch_loss)
                epochs.append(epoch)
                
            if phase == val:
                epoch_loss_1= running_loss_1/len(val.dataset)
                epoch_loss_2= running_loss_2/len(val.dataset)
                epoch_loss=epoch_loss_1 + epoch_loss_2


                epoch_acc_1 = running_corrects_1.double()/len(val.dataset)
                epoch_acc_2 = running_corrects_2.double()/len(val.dataset)
                epoch_acc= epoch_acc_1 +  epoch_acc_2

                print('{} val_loss 1= {:4f}, val_accuracy 1= {:4f}'.format(
                    'Validation result:', epoch_loss_1, epoch_acc_1))

                print('{} val_loss 2= {:4f}, val_accuracy 2= {:4f}'.format(
                    'Validation result:', epoch_loss_2, epoch_acc_2))
                
                print('{} val_loss= {:4f}, val_accuracy= {:4f}'.format(
                    'Validation result:', epoch_loss, epoch_acc))
                

                val_accuracy_1.append(epoch_acc_1)
                val_loss_1.append(epoch_loss_1)

                val_accuracy_2.append(epoch_acc_2)
                val_loss_2.append(epoch_loss_2)

                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)
                
            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == val and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    epochs=[epochs[i]+1 for i in range(len(epochs))]
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    train_data=list(zip(epochs,_loss,_loss_1,_loss_2,accuracy,accuracy_1,\
                        accuracy_2,val_loss,val_loss_1,val_loss_2,val_accuracy,val_accuracy_1,val_accuracy_2))
    header = '{0:^5s}   {1:^7s}  {2:^7s}  {3:^7s} {4:^5s}  {5:^5s}\
  {6:^5s}  {7:^8s}  {8:^8s}  {9:^8s}\
  {10:^7s}  {11:^7s}  {12:^7s}'.format('epochs', 'loss', 'loss 1', 'loss 2', \
    'accuracy','accuracy 1', 'accuracy 2', 'val loss',\
                                       'val loss 1', 'val loss 2', 'val accuracy', 'val accuracy 1', 'val accuracy 2')
    filename=savepath+method+'_'+dataname+'_loss'+'.txt'
    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f', '   %.7f',\
 '    %.7f', '  %.7f','  %.7f','  %.7f',\
'     %.7f', '  %.7f','  %.7f','  %.7f','     %.7f'])
    

    
    return model, accuracy,accuracy_1,\
                        accuracy_2, _loss, _loss_1,_loss_2, val_accuracy,val_accuracy_1,\
val_accuracy_2, val_loss, val_loss_1, val_loss_2, epochs, val_epochs        

##########################################################################################

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv3d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm3d(intermediate_channels)
        self.conv2 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(intermediate_channels)
        self.conv3 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm3d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm3d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    
def ResNet18(data_channel=3, num_classes=1000):
    return ResNet(block, [2, 2, 2, 2], data_channel, num_classes)

def ResNet34(data_channel=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], data_channel, num_classes)

def ResNet50(data_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], data_channel, num_classes)


def ResNet101(data_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], data_channel, num_classes)


def ResNet152(data_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], data_channel, num_classes)


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
    import re
    from IPython.display import Image

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
                    image_a=filename+'_s.png'
                    #print(image_a)
                    
                    #plt.figure(figsize=(20,20))
                    
                    image = Image(image_a)
                   
                    display(image)

                    plt.pause(0.001) 
                
                    if images_handeled  == num_images:
                        model.train(mode=was_training)
                        return 
        
        model.train(mode=was_training)      
        
        
        
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
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import matplotlib.ticker as plticker
    import time



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
def confusion_matrix_seaborn(cm,target,savepath, method,dataname):
    import seaborn as sn  
    cm_matrix = cm
    cm_labels = target
    df_cm = pd.DataFrame(cm_matrix,cm_labels,cm_labels)
    sn.set(rc={'figure.figsize':(20,18)},font_scale = 5)
    sn.heatmap(df_cm,annot=True,linewidths=2,cmap="Blues")
    plt.xlabel('True label', fontsize = 40) # x-axis label with fontsize 15
    plt.ylabel('Predicted label', fontsize = 40) # y-axis label with fontsize 15

    plt.savefig(savepath+method+'_'+dataname+'.pdf')
    return
