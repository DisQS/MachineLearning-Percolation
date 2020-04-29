# %%
"""
# Generation of percolation's datasets ( 16 avril 2020) 
# (reduced memory usage compare to the first version) 
"""



###############################################################################

import matplotlib.pyplot as pl
import numpy as np
from collections import Counter
from random import random
import os
#Pour voir les matrices  en entiers sans troncature
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

###############################################################################
"""
Function giving the intersection set of two list given in input
"""
def list_dir(path):
    for name in os.listdir(path):
        yield name

###############################################################################
def check_name(path):
    #print(L)
    import re
    c=0
    N=os.listdir(path)
    #print(N)
    nbre_images=len(N) #V
    del(N)
    global max_seed
   
    result=[0]*nbre_images
    Z=[0]*nbre_images

    B=(name for name in os.listdir(path))

    for c in range(nbre_images):
          
        A=next(B).split('_')[5]      
        #print(A)
    
        regex1 = re.compile('\d+')
        result[c]=re.findall(regex1,A)
        c+=1
    #print(result)
    for j in range(len(result)):
        #print(j)
        Z[j]=int(result[j][0])
        j+=1
    #print(Z)
    max_seed=max(Z)
   
    #print(max_seed)
    
    return max_seed
    
###############################################################################
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def intersection1(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

###############################################################################
def create_directory(path):
    import os
    #print(os.getcwd())

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory failed, existing file with the same name : "+str(path))
    else:
        print ("Successfully created the directory : "+str(path))
    
    return

###############################################################################
def fill(x, y, n_clusters, N, A, cluster, sizes):
    
    stack = [(x,y)] #last occupied site

    while len(stack) > 0:
        x, y = stack.pop(-1)

        if A[x,y] == 1 and cluster[x,y] == 0:
            cluster[x,y] = n_clusters+1
            sizes[n_clusters+1] += 1     #augmente le nombre de site du cluster

            if y+1 < N:
                stack.append((x,y+1))
            if y-1 >= 0:
                stack.append((x,y-1))

            if x+1 < N:
                stack.append((x+1,y))
            if x-1 >= 0:
                stack.append((x-1,y))
        

###############################################################################
"""
Function generating im image at density p with N=LxL sites.
"""

###############################################################################
def size_spanning_cluster(spanning_set, cluster):
    global size_spanning
    size_spanning=0
    nnz1=cluster.nonzero()
    for k in range(len(spanning_set)):
        for i, j in zip(*nnz1):
            if cluster[i,j] ==list(spanning_set)[k]:
                size_spanning+=1
    
###############################################################################
def percolation(im,p,L,seed):
    my_dpi=96 # DPI of the monitor
    import time
    import pickle
    start=time.time()
    #create_directory('percolating')   #if we want to classify them inside each density directory
    #create_directory('not_percolating')
               
    for t in range(im):
        seed1=np.random.seed(seed)
        # site occupation matrix
        A = np.random.binomial(1,p,size=(L,L))
        nnz = A.nonzero()
        n_clusters = 0
        cluster = np.zeros((L,L),dtype=int) 
        sizes = Counter()
         
        for i, j in zip(*nnz):
            if cluster[i,j] == 0:
                fill(i,j, n_clusters, L, A, cluster, sizes)
                n_clusters += 1
        new_mapping = np.arange(1,n_clusters+1)
        np.random.shuffle(new_mapping)
        cluster1= np.array([new_mapping[v-1] \
                            if not v == 0 else 0 for v in cluster.flat]).reshape(L,L)
        cluster = np.array([new_mapping[v-1]/(n_clusters) \
                            if not v == 0 else np.nan for v in cluster.flat]).reshape(L,L)
            
        all_sizes = Counter(list(sizes.values()))

        # get size of largest cluster
        if n_clusters !=0:
            max_size = max(all_sizes.keys())

        #create new figures
        #fig = pl.figure()
        fig =pl.figure(figsize=(133/my_dpi, 133/my_dpi), dpi=my_dpi)
        pl.axis('off')
        pl.imshow(cluster,cmap='nipy_spectral')
        cmap2 = pl.cm.get_cmap('nipy_spectral')

#        first_line=(x for x in cluster[0][:])
#        last_line=(y for y in cluster[-1][:])
#        first_column=(w for w in cluster[:,0])
#        last_column=(z for z in cluster[:,-1])

        top=set(x for x in cluster[0][:]).intersection(set(y for y in cluster[-1][:]))
        side=set(w for w in cluster[:,0]).intersection(set(z for z in cluster[:,-1]))
        
#        if np.nan in top:
#            top.remove(np.nan)
#        if np.nan in side:
#            side.remove(np.nan)

        if (top!=set() or side!=set()):
            #os.chdir('percolating')
            fig.savefig('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+\
                        '_top_'+str(top)+'_side_'+str(side)+'_size_max_clus'+str(max_size)+\
                        '.png',bbox_inches='tight', pad_inches = 0,dpi=my_dpi)
            size_side_spanning=0
            size_top_spanning=0
            rgba1=0
            rgba2=0
            
            if side!=set() and top!=set():
                size_spanning_cluster(side,cluster)
                size_side_spanning= size_spanning
                rgba2=cmap2(next(iter(side)))
                size_spanning_cluster(top,cluster)
                size_top_spanning=size_spanning
                rgba1=cmap2(next(iter(top)))
                
            elif side!=set():
                size_spanning_cluster(side,cluster)
                size_side_spanning= size_spanning
                rgba2=cmap2(next(iter(side)))  #rgb color tuple + alpha
                 
            else:
                size_spanning_cluster(top,cluster)
                size_top_spanning=size_spanning
                rgba1=cmap2(next(iter(top)))

            rgba1=cmap2(list(top)) #rgb color tuple + alpha
            rgba2=cmap2(list(side))

            f=open('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_top_'+str(top)+\
                   '_side_'+str(side)+'_size_max_clus'+str(max_size)+'.txt', "w+")
            f.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            f.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            f.write('Number of clusters with given size= ' +repr(sizes)+"\n")
            f.write('Spanning cluster top-bottom = '+ repr(top)+' = '+repr(size_top_spanning)+"\n")
            f.write('Spanning cluster side-side= '+ repr(side)+ ' = '+repr(size_side_spanning)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba1)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba2)+"\n")
            f.close()
            
            h= open('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_top_'+str(top)+\
                   '_side_'+str(side)+'_size_max_clus'+str(max_size)+'.pkl', "wb")
            pickle.dump(cluster1,h)
            h.close()
            
            #os.chdir('..')
        else:
            #os.chdir('not_percolating')
            fig.savefig('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.png', bbox_inches='tight',\
                        pad_inches = 0,dpi=my_dpi)
            g=open('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.txt', "w+")
            g.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            g.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            g.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            g.close()
            
            
            i=open('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.pkl', "wb")
            pickle.dump(cluster1,i)
            i.close()
            #os.chdir('..')

        pl.close('all')
        seed+=1
    end=time.time()
    execution=round(end-start,4)
    print(execution, 'seconds')

    return

###############################################################################
def percolation_density(im,M,L,seed):
    import os
    create_directory('L'+str(L))
    os.chdir('L'+str(L))
    import time
    start1= time.time()
    for p in M:
        new_im=0
        if os.path.exists('p'+str(p)):
            print('The file '+'p '+str(p)+' already exists')
            #print ("Creation of the directory failed")
            check_name('p'+str(p))
            print('The file already exist with max seed=',max_seed)
            os.chdir('p'+str(p))
            if max_seed <= im:
                new_im=(im- max_seed)
                seed=1+max_seed
            percolation(new_im,p,L,seed)
            os.chdir('..')
            if new_im!=0:
                print(new_im, 'new images were created')
        
        else:
            create_directory('p'+str(p))
            os.chdir('p'+str(p))
            percolation(im,p,L,seed)
            os.chdir('..')
    os.chdir('..')  
    end1=time.time()
    total_time=end1-start1
    print("Images generated in : ", total_time, "seconds")
    return
###############################################################################

###############################################################################

            
if ( len(sys.argv) == 7 ):
    #SEED = 101
    SEED = int(sys.argv[1])
    lattice_size = int(sys.argv[2])
    perco_init = int(sys.argv[3]) 
    perco_final = int(sys.argv[4])
    perco_inc = int(sys.argv[5])
    number_configs = int(sys.argv[6])

    perco_list=[val/10000 for val in range(perco_init,perco_final,perco_inc)]
            
    # %%
    percolation_density(number_configs,perco_list,lattice_size,SEED) 
    #1: number of images for a given p
    #2:list of p
    #3: side length of the square lattice
    #4: seed 

else:
    print ('Number of', len(sys.argv), \
           'arguments is less than expected (6) --- ABORTING!')
    print ('Usage: python '+sys.argv[0],\
           ' seed size p_initial*10000 p_final*10000 dp*10000 number_of_configurations')
    #print ('Argument List:', str(sys.argv))        
    

## %%
#create_directory('images_perco_density_new')
#os.chdir('images_perco_density_new')

## %%
#print(os.getcwd())

## %%
#M=[x/1000 for x in range(0,1000,50)]
#M=M[1:]
#N=[x/10000 for x in range(5920,5942,4)]
#O=M+N

