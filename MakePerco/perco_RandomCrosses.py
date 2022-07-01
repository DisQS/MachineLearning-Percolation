#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter, OrderedDict
import random
import os
import binascii
import sys
import time 
import datetime
import copy

np.set_printoptions(threshold=sys.maxsize)

# ###############################################################################
# percolation tool library from (hopefully) correct path
# ###############################################################################

machine= "csc"
user= "phsht"

# adding Folder_2 to the system path
#print(sys.path)

if machine!="avon":
    print("--- NOT working on avon")
    if user=="phsht":
        print(" --- working for phsht")
        #sys.path.insert(0,'../PyCode')
        sys.path.insert(0,'/storage/disqs/ML-Percolation/MachineLearning-Percolation/PyCode')
        #print(sys.path)
else:
    print("--- working on avon")
    if user=="phsht":
        print(machine, user)
        print(" --- working for phsht")
        sys.path.insert(0, '../PyCode')    

#print(sys.path)
from libPerco import *

###############################################################################
def percolation(im,p,size_sys,seed,nb_hlines,hthick,nb_vlines,vthick,nb_updlines,updthick,nb_downdlines,downdthick,typemod):

    import pickle

    # the seed has alreayd been checked to see if it's ok
    # but it also means we should only use it for im=1
    print('percolation: new percolation starting with seed=', seed)
    np.random.seed(seed)
    random.seed(seed)

    #create_directory('percolating')   #if we want to classify them inside each density directory
    #create_directory('not_percolating')
    
    for t in range(im):
        print('--- working in image', t, datetime.datetime.now())
        start2=time.time()
        lattice_para=lattice_config(size_sys,seed,p)

        
        occupied = lattice_para[0].nonzero()
        print('occupied',len(list(zip(*occupied))))
        n_clusters_pbc = 0
        n_clusters = 0
        cluster_pbc=np.zeros((size_sys,size_sys), dtype=int)-1
        cluster=np.zeros((size_sys,size_sys), dtype=int)-1
        sizes_pbc = Counter()
        sizes = Counter()
#         print('cluster')
        print("Hoshen-Kopelman PBS START", datetime.datetime.now())
        for i, j in zip(*occupied):
            if cluster_pbc[i,j] == -1:
                cluster_matrix_pbc(i, j, n_clusters_pbc, size_sys,lattice_para[0],cluster_pbc, sizes_pbc)

                n_clusters_pbc += 1
        print("Hoshen-Kopelman PBC END", datetime.datetime.now())
        
        print("Hoshen-Kopelman HW START", datetime.datetime.now())
        for i, j in zip(*occupied):
            if cluster[i,j] == -1:
                cluster_matrix(i, j, n_clusters, size_sys,lattice_para[0],cluster, sizes)

                n_clusters += 1
                #print(n_clusters)
        print("Hoshen-Kopelman HW END", datetime.datetime.now())
        
        print("PBC characterization", datetime.datetime.now())
            
        order_pbc=OrderedDict(sizes_pbc.most_common())
                  
        classification_pbc=list(order_pbc)
        #print(order)
        
        numbers_pbc = np.arange(0,n_clusters_pbc)
        weight_pbc=-np.sort(-(numbers_pbc))
        k_pbc=list(zip(classification_pbc,weight_pbc))
        
        correspondance_pbc=sorted(k_pbc, key = lambda t: t[0])
        unzip_pbc=list(zip(*correspondance_pbc))
        new_mapping_pbc=unzip_pbc[1]
        
        
        print("HW characterization", datetime.datetime.now())
        
        order=OrderedDict(sizes.most_common())
                  
        classification=list(order)
        
        
        numbers = np.arange(0,n_clusters)
        weight=-np.sort(-(numbers))
        k=list(zip(classification,weight))
        
        correspondance=sorted(k, key = lambda t: t[0])
        unzip=list(zip(*correspondance))
        
        new_mapping=unzip[1]

        print("PBC coloring", datetime.datetime.now())

        print('n clusters pbc',n_clusters_pbc)
        print('n clusters',n_clusters)
        
        # here starts the cross
        #print('init cluster',cluster)
        coord_hlines=[]
        thick_hlines=[]
        coord_vlines=[]
        thick_vlines=[]
        up_coord_dlines=[]
        up_thick_dlines=[]
        down_coord_dlines=[]
        down_thick_dlines=[]
        temp_cluster_pbc=copy.copy(cluster_pbc) #np.zeros((size_sys,size_sys))
        temp_cluster=copy.copy(cluster)
        if typemod>=0:
            mod=0
        if typemod<0:
            mod=-1
        #print('#################################################')
        #print('pbc before mod',temp_cluster_pbc)
        #print('hw before mod',temp_cluster)
        for hline in range(nb_hlines):
            thline=hthick
            hline=random.randint(0,(size_sys-1)-(thline-1))
                        
            coord_hlines.append(hline)
            thick_hlines.append(thline)
            
            #PBC -----------------------------------
            #print('PBC: len before h-line modification',len(cluster_pbc))
           # print('init',cluster_pbc)
            #print('hline',hline)
            for line in range(thline):
                temp_cluster_pbc[hline+line,:]=[mod]*len(temp_cluster_pbc[0])
                temp_cluster[hline+line,:]=[mod]*len(temp_cluster[0])
                #print('temp pbc',temp_cluster_pbc)
                #print('temp hw',temp_cluster)
            
#             print('PBC: len after h-line modification, size_blank_pbc', size_blank_pbc)
            
#             #HW -----------------------------------
#             print('HW: len before h-line modification',len(cluster))
#             print('init',cluster)
        
           
#             print('cluster_blank',cluster_blank)
#             print('HW: len after h-line modification, size_blank_pbc', size_blank)
            
        for vline in range(nb_vlines):
            tvline=vthick
            vline=random.randint(0,(size_sys-1)-(tvline-1))
            
            coord_vlines.append(vline)
            thick_vlines.append(tvline)
            #print('#######################################vline')
            #print('vline',vline)
            #PBC -----------------------------------
            #print('PBC: len before v-line modification',len(cluster_pbc))
            for line in range(tvline):
                temp_cluster_pbc[:,vline+line]=[mod]*len(temp_cluster_pbc)
                temp_cluster[:,vline+line]=[mod]*len(temp_cluster)
                #print('temp pbc',temp_cluster_pbc)
                #print('temp hw',temp_cluster)
        
        for up_dline in range(nb_updlines):
            up_tdline=updthick
            up_dline=random.randint(0,(size_sys-1)-(up_tdline-1))
            up_coord_dlines.append(up_dline)
            up_thick_dlines.append(up_tdline)
            #print('#######################################vline')
            #print('up_dline',up_dline)
            for line in range(up_tdline):
                #np.fill_diagonal(temp_cluster_pbc[:,up_dline+line:], mod)
                #np.fill_diagonal(temp_cluster[:,up_dline+line:], mod)
                temp_cluster_pbc.ravel()[up_dline+line:max(0,temp_cluster_pbc.shape[1]-up_dline+line)*temp_cluster_pbc.shape[1]:temp_cluster_pbc.shape[1]+1]=mod
                temp_cluster.ravel()[up_dline+line:max(0,temp_cluster.shape[1]-up_dline+line)*temp_cluster.shape[1]:temp_cluster.shape[1]+1]=mod
               # print('temp pbc',temp_cluster_pbc)
               # print('temp hw',temp_cluster)
        for down_dline in range(nb_downdlines):
            down_tdline=downdthick
            down_dline=random.randint(0,(size_sys-1)-(down_tdline-1))
            down_coord_dlines.append(down_dline)
            down_thick_dlines.append(down_tdline)
            #print('#######################################vline')
            #print('down_dline',down_dline)
            for line in range(down_tdline):
                #np.fill_diagonal(np.fliplr(temp_cluster_pbc[down_dline:,:]),mod)
                #np.fill_diagonal(np.fliplr(temp_cluster[down_dline:,:]),mod)
                temp_cluster_pbc.ravel()[down_dline+line:(down_dline+line)*(temp_cluster_pbc.shape[1]+1):temp_cluster_pbc.shape[1]-1]=mod
                temp_cluster.ravel()[down_dline+line:(down_dline+line)*(temp_cluster.shape[1]+1):temp_cluster.shape[1]-1]=mod
                #print('temp pbc',temp_cluster_pbc)
                #print('temp hw',temp_cluster)
        cluster_blank_pbc= temp_cluster_pbc
        cluster_blank=temp_cluster
            
#             print('cluster_blank_pbc',cluster_blank_pbc)
#             size_blank_pbc=len(cluster_blank_pbc)
#             print('PBC: len after v-line modification, size_blank_pbc', size_blank_pbc)
            
#             #HW -----------------------------------
#             print('HW: len before v-line modification',len(cluster))
                        
#             print('HW: len after v-line modification, size_blank_pbc', size_blank)
        
############################################ For PBC
#        print('before pbc',len(cluster_pbc))
#        
#        cluster_pbc=np.delete(cluster_pbc,mid,1)
#        cluster_pbc=np.delete(cluster_pbc,mid,0)
#        col=np.zeros((len(cluster_pbc),thline), dtype='int')-1
#     
#        print(col)
#        print( len(cluster_pbc))
#        cluster_blank_pbc=np.c_[cluster_pbc[:,:mid],col,cluster_pbc[:,mid:]]
#        row=np.zeros((thline,len(cluster_blank_pbc[0])), dtype='int')-1
#
#        print(len(row),row)
#
#        cluster_blank_pbc=np.r_[cluster_blank_pbc[:mid,:],row,cluster_blank_pbc[mid:,:]]
#        size_blank_pbc=len(cluster_blank_pbc)
#        print('size_blank_pbc', size_blank_pbc)
#
############################################ For HW

#        print('before',len(cluster))
#        cluster=np.delete(cluster,mid,1)
#        cluster=np.delete(cluster,mid,0)
#        print( len(cluster))
#        cluster_blank=np.c_[cluster[:,:mid],col,cluster[:,mid:]]
#        row=np.zeros((thline,len(cluster_blank[0])), dtype='int')-1
#        cluster_blank=np.r_[cluster_blank[:mid,:],row,cluster_blank[mid:,:]]
#        size_blank=len(cluster_blank)
#        print('size_blank', size_blank)
################################################
        #print('mapping',new_mapping)
        cluster_blank_pbc_int= np.array([new_mapping_pbc[v]+1 \
                                         if not v == -1 else 0 for v in cluster_blank_pbc.flat]).reshape(size_sys,size_sys)
        cluster_blank_pbc_norm = np.array([(new_mapping_pbc[v]+1)/n_clusters_pbc \
                                           if not v == -1 else 0 for v in cluster_blank_pbc.flat]).reshape(size_sys,size_sys)
        cluster_blank_pbc_nan= np.array([new_mapping_pbc[v]+1 \
                                         if not v == -1 else np.nan for v in cluster_blank_pbc.flat]).reshape(size_sys,size_sys)      
        
        cluster_pbc_nan= np.array([new_mapping_pbc[v]+1 \
                            if not v == -1 else np.nan for v in cluster_pbc.flat]).reshape(size_sys,size_sys)
       # print("HW coloring", datetime.datetime.now())
        
        cluster_blank_int= np.array([new_mapping[v]+1 \
                                     if not v == -1 else 0 for v in cluster_blank.flat]).reshape(size_sys,size_sys)
        cluster_blank_norm = np.array([(new_mapping[v]+1)/n_clusters \
                                       if not v == -1 else 0 for v in cluster_blank.flat]).reshape(size_sys,size_sys)
        cluster_blank_nan = np.array([new_mapping[v]+1 \
                                      if not v == -1 else np.nan for v in cluster_blank.flat]).reshape(size_sys,size_sys)
        cluster_nan = np.array([new_mapping[v]+1\
                           if not v == -1 else np.nan for v in cluster.flat]).reshape(size_sys,size_sys)

        p1=len(list(zip(*cluster_blank_int.nonzero()))) / (size_sys**2)
        print("density change p=",p,"->",p1)
        
        all_sizes_pbc = Counter(list(sizes_pbc.values()))
        
        #get size of largest cluster
        if n_clusters_pbc !=0:
            max_size_pbc = max(all_sizes_pbc.keys())
        
        occ=len(list(zip(*cluster_blank_int.nonzero())))
        start4=time.time()
        proba_largest=(max_size_pbc/(size_sys**2))**2

        square_proba=p*p

       
        print('max clus', max_size_pbc)
        end4=time.time()-start4

        occ=len(list(zip(*cluster_blank_int.nonzero())))

        top=cluster_nan[0][:]
        bottom=cluster_nan[-1][:]
        left=cluster_nan[:,0]
        right=cluster_nan[:,-1]
       
        top_bot_inter=set(x for x in cluster_nan[0][:]).intersection(set(y for y in cluster_nan[-1][:]))
        sides_inter=set(w for w in cluster_nan[:,0]).intersection(set(z for z in cluster_nan[:,-1]))
        
        #print('inter top',top_bot_inter,'inter side',sides_inter)
        #print('cluster_nan',cluster_nan)
        HWTB=0
        HWLR=0
        PBCTB=0
        PBCLR=0
        
        size_side_spanning_pbc=0
        size_top_spanning_pbc=0
        size_top_side_spanning_pbc=0
        rgba1=0
        rgba2=0
        init_span=0
        if (top_bot_inter!=set() or sides_inter!=set()):
            init_span=1
            top_pbc=set(x for x in cluster_blank_pbc_nan[0][:]).intersection(set(y for y in cluster_blank_pbc_nan[-1][:]))
            side_pbc=set(w for w in cluster_blank_pbc_nan[:,0]).intersection(set(z for z in cluster_blank_pbc_nan[:,-1]))
            
            union_top_side_spanning_pbc= side_pbc.union(top_pbc)
            
            
            if top_bot_inter!=0 and sides_inter!=0:
                HWTB=1
                HWLR=1
                PBCTB=pbc_percolation(top_bot_inter,top,bottom,PBCTB)
               
                
                PBCLR=pbc_percolation(sides_inter,left,right,PBCLR)
                
 
            elif top_bot_inter!=0:
                HWTB=1
                PBCTB=pbc_percolation(top_bot_inter,top,bottom,PBCTB)

  
            else:
                HWLR=1
                PBCLR=pbc_percolation(sides_inter,left,right,PBCLR)

        else:
            pass

        top_bot_inter_mod=set(x for x in cluster_blank_nan[0][:]).intersection(set(y for y in cluster_blank_nan[-1][:]))
        side_inter_mod=set(w for w in cluster_blank_nan[:,0]).intersection(set(z for z in cluster_blank_nan[:,-1]))
        if (top_bot_inter_mod!=set() or side_inter_mod!=set()):
            span_mod=1
        else:
            span_mod=0

   
        filename='pc_'+str(span_mod)+'_'+str(init_span)+'_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__pi'+str(p)+'__pm'+str(p1)+'_L'+str(size_sys)+'_s'+str(seed)+ \
                '_nc'+str(n_clusters_pbc)+'_smc'+str(max_size_pbc)+'_n'+str(n_clusters_pbc)

                
                
        text_file=open(filename+'.txt', "w+")
        text_file.write('Total number of cluster= '+ repr(n_clusters)+'\n')
        text_file.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size_pbc)+'\n')
        text_file.write('Number of clusters with given size= ' +repr(sizes)+"\n")
        text_file.close()
        data_pkl = {'cluster_pbc_int' : cluster_blank_pbc_int ,
                    'cluster_pbc_norm' : cluster_blank_pbc_norm,
                    'n_clusters_pbc':n_clusters_pbc,
                    'cluster_int':cluster_blank_int,
                    'cluster_norm':cluster_blank_norm,
                    'proba largest' : proba_largest,
                    'square proba':square_proba,
                    'size max cluster':max_size_pbc,
                    'pi':p,
                    'pf':p1,
                    'nb_vlines':nb_vlines,
                    'nb_hlines':nb_hlines,
                    'coord_hlines':coord_hlines,
                    'thick_hlines':thick_hlines,
                    'coord_vlines':coord_vlines,
                    'thick_vlines':thick_vlines}
            
        pkl_file=open(filename+'.pkl', "wb")
        pickle.dump(data_pkl ,pkl_file)
        pkl_file.close()
            
        end2=time.time()-start2
        print(end2)

    return cluster,n_clusters,cluster_blank_pbc,n_clusters_pbc, sizes_pbc,lattice_para[0],new_mapping,order,order_pbc,occ,sizes,cluster_blank_int, top_bot_inter,sides_inter,cluster_blank_nan



#################################################################################################################################
def percolation_density(number_configs,perco_list,lattice_size,hlines,hthick,vlines,vthick,updlines,updthick,downdlines,downdthick,typemod):  
    import os
    
    import time
    
    dens=[]
    start1= time.time()
    configs_wanted=number_configs
    #seed=int(binascii.hexlify(os.urandom(4)),16)
    for p in perco_list:
        print('percolation_density: working on p=', p)

        configs_created=0
        
        if os.path.exists('p'+str(p)) and len(os.listdir('p'+str(p)))!=0:
            print('A directory '+'p='+str(p)+' already exists')
        else:
            create_directory('p'+str(p))
        
        os.chdir('p'+str(p))

        max_seed,seeds_existing,configs_existing=check_name('.')
        print('--- found seeds=', seeds_existing)

        if configs_wanted >= configs_existing:  #nbre_images:
            configs_tomake = configs_wanted - configs_existing   #nbre_images
            print('percolation_density: configs existing, wanted, tomake=',configs_existing, configs_wanted, configs_tomake)
            while configs_tomake > 0:
                seed=int(binascii.hexlify(os.urandom(4)),16)
                if seed in seeds_existing: #seed_list:
                    print('Image with seed = ',seed, 'already exists')
                    while seed in seeds_existing:
                        seed=int(binascii.hexlify(os.urandom(4)),16)
                    seeds_existing.append(seed)
                    print('--- NEW seed ', seed, ' scheduled to be made')
                    # now we have a good seed, let's percolate
                perco_calcul= percolation(1,p,lattice_size,seed,hlines,hthick,vlines,vthick,updlines,updthick,downdlines,downdthick,typemod)
                configs_created+=1
                configs_tomake-=1
                print('--- NEW configuration', seed,' was created')

            os.chdir('..')
            if configs_created!=0:
                print("-->",configs_created, 'new images were created')
        
    end1=time.time()
    total_time=end1-start1
    print("Images generated in : ", total_time, "seconds")
    return 

####################################################################################################################
def main():
    if ( len(sys.argv) == 15 ):
        #SEED = int(sys.argv[1])
        lattice_size = int(sys.argv[1])
        perco_init = int(sys.argv[2]) 
        perco_final = int(sys.argv[3])
        perco_inc = int(sys.argv[4])
        number_configs = int(sys.argv[5])
        hlines=int(sys.argv[6])
        hthick=int(sys.argv[7])
        vlines=int(sys.argv[8])
        vthick=int(sys.argv[9])
        updlines=int(sys.argv[10])
        updthick=int(sys.argv[11])
        downdlines=int(sys.argv[12])
        downdthick=int(sys.argv[13])
        typemod=int(sys.argv[14])


        print("perco_RandomCrosses:", lattice_size, perco_init,perco_final,perco_inc, number_configs)
        perco_list=[val/10000 for val in range(perco_init,perco_final+1,perco_inc)]
        #print(range(perco_init,perco_final+1,perco_inc))
        #print(perco_list)

        # %%
        percolation_density(number_configs,perco_list,lattice_size,hlines,hthick,vlines,vthick,updlines,updthick,downdlines,downdthick,typemod) 
        #1: number of images for a given p
        #2:list of p
        #3: side length of the square lattice
        #4: seed 

    else:
        print ('Number of', len(sys.argv), \
               'arguments is less than expected (15) --- ABORTING!')
        print ('Usage: python '+sys.argv[0],\
               '  size p_initial*10000 p_final*10000 dp*10000 number_of_configurations')
        print ('Argument List:', str(sys.argv))        
        return

main()
