import os
import numpy
import csv
import re

def generate_csv_image(data_dir,data_type=None):    
    
    if data_type=='h_res':
        regex2 = re.compile('\d+\.\d+')
        directory=[d.name for d in os.scandir(data_dir) if d.is_dir() if 0.55<=float(re.findall(regex2,d.name)[0])<=0.66]    
        filename='_h_res.csv'
    else:
        directory=[d.name for d in os.scandir(data_dir) if d.is_dir()]
        filename='.csv'
        
        
    for p in directory:
        list_files=os.listdir(data_dir+'/'+p)
        print(data_dir+'/'+p)
        list_s = [a for a in list_files if "_s.png" in a]
        list_a = [x for x in list_files if "_a.png" in x]
        list_b = [y for y in list_files if "_b.png" in y]
        list_n = [z for z in list_files if "_n.png" in z]
        
        len_s=len(list_s)
        len_a=len(list_a)
        len_b=len(list_b)
        len_n=len(list_n)
        
        p_a=[p]*len_a
        p_b=[p]*len_b
        p_n=[p]*len_n
        p_s=[p]*len_s
       
        data_a=list(zip(list_a,p_a))
        if 'image_a'+filename in os.listdir('.'):
            with open('image_a'+filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_a)
        else:
            with open('image_a'+filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_a)
        
        data_b=list(zip(list_b,p_b))  
        if 'image_b'+filename in os.listdir('.'):
            with open('image_b'+filename, 'a', newline='') as g:
                writer_b = csv.writer(g)
                writer_b.writerows(data_b)
        else:
            with open('image_b'+filename, 'w', newline='') as g:
                writer_b = csv.writer(g)
                writer_b.writerows(data_b)
            
        data_n=list(zip(list_n,p_n))
        if 'image_n'+filename in os.listdir('.'):
            with open('image_n'+filename, 'a', newline='') as h:
                writer_n = csv.writer(h)
                writer_n.writerows(data_n)
        else:
            with open('image_n'+filename, 'w', newline='') as h:
                writer_n = csv.writer(h)
                writer_n.writerows(data_n)
            
        data_s=list(zip(list_s,p_s)) 
        if 'image_s'+filename in os.listdir('.'):
            with open('image_s'+filename, 'a', newline='') as i:
                writer_s = csv.writer(i)
                writer_s.writerows(data_s)
        else:
            with open('image_s'+filename, 'w', newline='') as i:
                writer_s = csv.writer(i)
                writer_s.writerows(data_s)
            
    return
#############################################################################

def generate_csv_data(data_dir,data_type=None):    
    directory=[d.name for d in os.scandir(data_dir) if d.is_dir()]
    if data_type=='h_res':
        regex2 = re.compile('\d+\.\d+')
        directory=[d.name for d in os.scandir(data_dir) if d.is_dir() if 0.55<=float(re.findall(regex2,d.name)[0])<=0.66]    
        filename='data_pkl_h_res.csv'
    else:
        directory=[d.name for d in os.scandir(data_dir) if d.is_dir()]
        filename='data_pkl.csv'
        
        
    for p in directory:
        list_files=os.listdir(data_dir+'/'+p)
        print(data_dir+'/'+p)
        list_pkl = [a for a in list_files if ".pkl" in a]        
        len_pkl=len(list_pkl)       
        p_list=[p]*len_pkl
       
        data_pkl=list(zip(list_pkl,p_list))
        if filename in os.listdir('.'):
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_pkl)
        else:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_pkl)
        
        
            
    return
