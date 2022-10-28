import numpy as np
from collections import Counter, OrderedDict
import random
import os
import binascii
import sys
import time 
import datetime

np.set_printoptions(threshold=sys.maxsize)

###############################################################################
def check_name(path):
    import re
    results=[]
    seeds=[]
   
    for files in os.listdir(path):
        if files.endswith('.txt'):
            results.append(files)
            seed_sys=files.split('_')[12]      
            regex3 = re.compile('\d+')
            seed_sys_reg=re.findall(regex3,seed_sys)
            seeds.append(int(seed_sys_reg[0]))
            
				    
    nbre_files=len(results)
    if len(seeds)>0:
        max_seed=max(seeds)
    else:
        max_seed=0

    print('check: max_seed, nbre_file =', max_seed, nbre_files)
    return max_seed, seeds, nbre_files


###############################################################################
def create_directory(path):
    import os
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory failed, existing file with the same name : "+str(path))
    else:
        print ("Successfully created the directory : "+str(path))
    
    return
###############################################################################


def lattice_config(size,seed,p):
    print("lattice_config START", datetime.datetime.now())
    seed1=np.random.seed(seed)
    number_occupied=(size*size)*p
    
    lattice=np.zeros((size,size), dtype=int)
    occupied=0
   
    while occupied < number_occupied:
        
        i=random.randint(0,size-1)
        j=random.randint(0,size-1)
        if lattice[i,j]==0:
            lattice[i,j]=1
            occupied+=1

    print("lattice_config END", datetime.datetime.now())
    return lattice, number_occupied

###############################################################################       

def cluster_matrix(x, y, n_clusters, N, lattice,cluster, sizes):
    stack=[(x,y)]

    while len(stack)>0:

        x,y=stack.pop(-1)

        if lattice[x,y] == 1 and cluster[x,y]==-1:
            cluster[x,y] = n_clusters
            sizes[n_clusters] += 1     #augmente le nombre de site du cluster
            if y+1 < N:
                stack.append((x,y+1))

            if y-1 >= 0:

                stack.append((x,y-1))

            if x+1 < N:

                stack.append((x+1,y))

            if x-1 >= 0:
                stack.append((x-1,y))


################################################################

def cluster_matrix_pbc(x, y, n_clusters, N, lattice,cluster, sizes):
    stack=[(x,y)]

    while len(stack)>0:

        x,y=stack.pop(-1)

        if lattice[x%N,y%N] == 1 and cluster[x%N,y%N]==-1:
            cluster[x%N,y%N] = n_clusters
            sizes[n_clusters] += 1     #augmente le nombre de site du cluster

            stack.append((x,(y+1%N)))

            stack.append((x,(y-1)%N))

            stack.append(((x+1)%N,y))

            stack.append(((x-1)%N,y))


###############################################################################
def pbc_percolation(boundary_set,boundary_array_1, boundary_array_2,PBC):
    cluster_number=0
    for element in boundary_set:
        cluster_number=element

        new_array=np.array([v if  v == cluster_number else 0 for v in boundary_array_1])

        coord_non_zero=new_array.nonzero()[0]

        for arg in coord_non_zero:

            if  boundary_array_2[arg]!=0:
                PBC=1
                
    return PBC

###############################################################################
def size_spanning_cluster_old(spanning_set, cluster):
    
    size_spanning=0
    nnz1=cluster.nonzero()
    for k in range(len(spanning_set)):
        for i, j in zip(*nnz1):
            if cluster[i,j] ==list(spanning_set)[k]:
                size_spanning+=1
                
    return size_spanning
    
##############################################################################
def size_spanning_cluster(spanning_set,cluster,cluster_pbc_int):
   
    size=[]
    size_spanning=0
    span=list(spanning_set)
    n_cluster_pbc_int, counts = np.unique(cluster_pbc_int, return_counts=True)
    zip_lists=list(zip( n_cluster_pbc_int, counts))
    for k in range(len(spanning_set)):
        for i in range(len(zip_lists)):
            if zip_lists[i][0]==span[k]:
                size.append(zip_lists[i][1])
    if len(size)>1:
        size_spanning=max(size)
        
    return size_spanning



#################################################################################
def mapping_cluster(cluster_nan,cluster_pbc_int,n_perco):

    mapping = dict(zip(cluster_nan.flat,cluster_pbc_int.flat))
    keys_mapping=mapping.keys()
    values_mapping=mapping.values()
    v=[p for p in values_mapping if p!=0]
    k=list(filter(lambda v: v==v, keys_mapping))
    zip_keys_values=list(zip(k,v))
    new_n=[]
    
    for i in range(0,len(zip_keys_values)):
        if (zip_keys_values[i][0] in n_perco) and (zip_keys_values[i][1] not in new_n):
            new_n.append(zip_keys_values[i][1])

    return new_n
    
