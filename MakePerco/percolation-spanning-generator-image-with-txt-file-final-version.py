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
def list_dir(path):
    for name in os.listdir(path):
        yield name


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
    

# %%
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def intersection1(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


# %%
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


# %%
def fill(x, y, n_clusters, N, A, cluster, sizes):
    
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
    my_dpi=96 # DPI of the monitor
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
                fill(i,j, n_clusters, L, A, cluster, sizes)
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
        
        if np.nan in top:
            top.remove(np.nan)
        if np.nan in side:
            side.remove(np.nan)

        if (top!=set() or side!=set()):
            #os.chdir('percolating')
            fig.savefig('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_top_'+str(top)+'_side_'+str(side)+'_size_max_clus'+str(max_size)+'.png',bbox_inches='tight', pad_inches = 0,dpi=my_dpi)
            rgba1=cmap2(list(top)) #rgb color  code with alpha
            rgba2=cmap2(list(side))
            f=open('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_top_'+str(top)+'_side_'+str(side)+'_size_max_clus'+str(max_size)+'.txt', "w+")
            f.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            f.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            f.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            f.write('Spanning cluster top-bottom = '+ repr(top)+"\n")
            f.write('Spanning cluster side-side= '+ repr(side)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba1)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba2)+"\n")
            f.close()

#            np.savetxt('pc_1__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_size_max_clus'+str(max_size)+'.txt',cluster, newline="\n",fmt='%.5e',
#                      header='density:'+str(p)+', size of each cluster:'+str(all_sizes)+'max cluster'+str(max_size))
            #os.chdir('..')
        else:
            #os.chdir('not_percolating')
            fig.savefig('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.png', bbox_inches='tight',pad_inches = 0,dpi=my_dpi)
            g=open('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.txt', "w+")
            g.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            g.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            g.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            g.close()
            #g.write('number of cluster=', n_clusters)
#            np.savetxt('pc_0__p'+str(p)+'_L'+str(L)+'_s'+str(seed)+'_.txt',cluster, newline="\n",fmt='%.5e',
#                      header='density:'+str(p)+', size of each cluster:'+str(all_sizes)+'max cluster'+str(max_size))
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
        #os.chdir('..')
            #continue
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

            
        
    

# %%
create_directory('images_perco_density')
os.chdir('images_perco_density')

# %%
print(os.getcwd())

# %%
M=[x/1000 for x in range(0,1000,50)]
M=M[1:]
N=[x/10000 for x in range(5920,5942,4)]
O=M+N


# %%
percolation_density(100,O,100,1) 
#1: number of images for a given p
#2:list of p
#3: side length of the square lattice
#4: seed 