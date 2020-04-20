# %%
"""
# Generation of percolation's datasets ( 16 avril 2020) 
# (reduced memory usage compare to the first version) 
"""



# %%
import matplotlib.pyplot as pl
import numpy as np
from collections import Counter
from random import random
import os
#Pour voir les matrices  en entiers sans troncature
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# %%
"""
Function giving the intersection set of two list given in input
"""

# %%
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def intersection1(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


# %%
def create_directory(path):
    import os
    print(os.getcwd())

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory failed, existing file with the same name")
    else:
        print ("Successfully created the directory")
    
    return


# %%
def fill4(x, y, n_clusters, N, A, cluster, sizes):

    stack = [(x,y)] #last occupied site

    while len(stack) > 0:
        x, y = stack.pop(-1)

        if A[x,y] == 1 and cluster[x,y] == -1:
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

# %%
"""
Function generating im image at density p with N=LxL sites.
"""

# %%
def percolation(im,p,L,seed):
    my_dpi=96
    import time
    start=time.time()
    #create_directory('percolating')   #if we want to classify them inside each density directory
    #create_directory('not_percolating')
    
     
           
    for t in range(im):
        seed1=np.random.seed(seed)
        # site occupation matrix
        A = np.random.binomial(1,p,size=(L,L))
        nnz = A.nonzero()
        n_clusters = 0
        cluster = np.zeros((L,L),dtype=int) - 1
        sizes = Counter()
         
        for i, j in zip(*nnz):
            if cluster[i,j] < 0:
                fill4(i,j, n_clusters, L, A, cluster, sizes)
                n_clusters += 1
        new_mapping = np.arange(0,n_clusters)
        np.random.shuffle(new_mapping)
        cluster = np.array([new_mapping[v] if not v == -1 else np.nan for v in cluster.flat]).reshape(L,L)

            
        all_sizes = Counter(list(sizes.values()))

        # get size of largest cluster
        if n_clusters !=0:
            max_size = max(all_sizes.keys())

        #create new figures
        #fig = pl.figure()
        fig =pl.figure(figsize=(L/my_dpi, L/my_dpi), dpi=my_dpi)
        pl.axis('off')
        pl.imshow(cluster,cmap='nipy_spectral')


       
        first_line=(x for x in cluster[0,])
        last_line=(y for y in cluster[len(cluster[0])-1,:])
        first_column=(w for w in cluster[:,0])
        last_column=(z for z in cluster[:,len(cluster)-1])


        top=set(first_line).intersection(set(last_line))
        side=set(first_column).intersection(set(last_column))

        if (top!=set() or side!=set()):
            #os.chdir('percolating')
            fig.savefig('pc_1__p'+str(p)+'_L'+str(L)+'_N'+str(im)+'size_max_clus'+str(max_size)+'_s'+str(seed)+'.png',dpi=my_dpi)
            #os.chdir('..')
        else:
            #os.chdir('not_percolating')
            fig.savefig('pc_0__p'+str(p)+'_L'+str(L)+'_N'+str(im)+'_s'+str(seed)+'_.png',dpi=my_dpi)
            #os.chdir('..')

        pl.close('all')
        seed+=1
    end=time.time()
    execution=round(end-start,4)
    print(execution, 'seconds')

    return



# %%
def percolation_density(im,M,L,seed):
    import os
    create_directory('L'+str(L)+'_N'+str(im))
    os.chdir('L'+str(L)+'_N'+str(im))
    for p in M:
        if os.path.exists('p'+str(p)):
            print('The file '+'p '+str(p)+' already exists')
            print ("Creation of the directory failed")
            continue
        else:
            create_directory('p'+str(p))
            os.chdir('p'+str(p))
            percolation(im,p,L,seed)
            os.chdir('..')
    os.chdir('..')     
    return

            
        
    

# %%
create_directory('images_perco_density')
os.chdir('images_perco_density')

# %%
print(os.getcwd())

# %%
M=[x/1000 for x in range(0,1000,100)]
M=M[1:]
N=[x/10000 for x in range(5920,5942,4)]



# %%
percolation_density(10,M,100,1) 
#1: number of images for a given p
#2:list of p
#3: side length of the square lattice
#4: seed 
