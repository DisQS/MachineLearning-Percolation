###############################################################################
"""
# Generation of percolation's datasets ( 16 avril 2020) 
# (reduced memory usage compare to the first version) 
"""

###############################################################################

import matplotlib.pyplot as pl
import numpy as np
from collections import Counter
import random
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
    global seed_list, max_seed, nbre_images
    import re
    c=0
    N=os.listdir(path)
    #print(N)
    nbre_file=len(N) #V
    del(N)
    
    result=[0]*nbre_file
    seed_list=[0]*nbre_file

    B=(name for name in os.listdir(path))

    for c in range(nbre_file):
          
        A=next(B).split('_')[9]      
        #print(A)
    
        regex1 = re.compile('\d+')
        result[c]=re.findall(regex1,A)
        c+=1
    #print(result)
    for j in range(len(result)):
        #print(j)
        seed_list[j]=int(result[j][0])
        j+=1
        
    max_seed=max(seed_list)
    nbre_images=nbre_file/3
    
            
    #print(Z)
   
    #print(max_seed)
    
    return max_seed

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


def lattice_config(size,seed,p):
    global lattice, number_occupied
    seed1=np.random.seed(seed)
    number_occupied=(size*size)*p
    #print(random, type(random))
    lattice=np.zeros((size,size), dtype=int)
    while len(list(zip(*lattice.nonzero())))<number_occupied:
        #print(len(list(zip(*lattice.nonzero()))))
        i=random.randint(0,size-1)
        j=random.randint(0,size-1)
        if lattice[i,j]==0:
            lattice[i,j]=1
#        for k in range(size):
#            for l in range(size):
#                if (k%(size-1)==0):
#                    lattice[k,l]=lattice[0,l]
#                if (l%(size-1)==0):
#                    lattice[k,l]=lattice[k,0]
#        #print(lattice)
#        if len(list(zip(*lattice.nonzero())))>number_occupied:
#            #print(len(list(zip(*lattice.nonzero()))))
#            num_occ=len(list(zip(*lattice.nonzero())))
#            diff=num_occ-len(list(zip(*lattice.nonzero())))
#            for m in range(diff):
#                num_occ=len(list(zip(*lattice.nonzero())))
#                choice=random.randint(0,num_occ-1)
#                coord_occ=list(zip(*lattice.nonzero()))
#                #print(coord_occ[choice])
#                if (coord_occ[choice][0]!=0 and coord_occ[choice][0]!=size-1 and\
#                    coord_occ[choice][1]!=0 and coord_occ[choice][1]!=size-1):
#                    coord_occ.pop(choice)
#                    lattice[coord_occ[choice]]==0
#                

    return
#def lattice_config(size,seed,p):   ####old version
#    global lattice, number_occupied
#    seed1=np.random.seed(seed)
#    number_occupied=(size*size)*p
#    lattice=np.zeros((size,size), dtype=int)
#    occ_site=lattice.nonzero()
#    coord_occ=list(zip(*occ_site))
#    while len(coord_occ)<number_occupied:
#        i=random.randint(0,size-1)
#        j=random.randint(0,size-1)
#        if lattice[i,j]==0:
#            lattice[i,j]=1
#        for k in range(size):
#            for l in range(size):
#                if (i%(lattice_size-1)==0 and j%lattice_size==0):
#                    lattice[i,j]=lattice[0,0]
#                elif (i%(lattice_size-1)==0):
#                    lattice[i,j]=lattice[0,j]
#                elif (j%(lattice_size-1)==0):
#                    lattice[i,j]=lattice[i,0]
#        if len(coord_occ)>number_occupied:
#            diff=len(lattice.nonzero())-len(coord_occ)
#            for m in range(diff):
#                choice=random.randint(0,len(coord_occ)-1)
#                if (coord_occ[choice][0]!=0 and coord_occ[choice][0]!=size-1 and\
#                    coord_occ[choice][1]!=0 and coord_occ[choice][1]!=size-1):
#                    coord_occ.pop(choice)
#                    lattice[coord_occ[choice]]==0
#                
                
#        j=random.randint(0,(size**2)-1)
#        #print(count)
#        if lattice[j]==0:
#            lattice[j]=1
#            if lattice[j]==1:
#                count+=1
#        if lattice.nonzero()>number_occupied:
#            
#    lattice=lattice.reshape(size,size)
#
#    return
###############################################################################       
def color_point(cluster,top,side):  
    global quarter_b,three_quarter_b,anti_quarter_b,center_b,anti_three_quarter_b 
 
    mid_diag=int(np.ceil(len(cluster/2)))-1
    quarter_diag=int(np.ceil(len(cluster)/3))-1  
    center=cluster[mid_diag][mid_diag]
    quarter=cluster[quarter_diag][quarter_diag]
    three_quarter=cluster[-(quarter_diag+1)][-(quarter_diag+1)]
    anti_quarter=cluster[(quarter_diag)][-(quarter_diag+1)]
    anti_three_quarter=cluster[-(quarter_diag+1)][quarter_diag]
    center_b=0
    quarter_b=0
    three_quarter_b=0
    anti_quarter_b=0
    anti_three_quarter_b=0  
    if top!=set():
        for i in range(len(top)):
            num_cluster=list(top)[i]
            if center==num_cluster:
                center_b=1
            if quarter_b==num_cluster:
                quarter_b=1
            
            if three_quarter==num_cluster:
                three_quarter_b=1
            
            if anti_quarter==num_cluster:
                anti_quarter_b=1
            
            if anti_three_quarter==num_cluster:
                anti_three_quarter_b=1

    if side!=set():
         for j in range(len(side)):
             num_cluster=list(side)[j]
             if center==num_cluster:
                 center_b=1
             if quarter==num_cluster:
                 quarter_b=1
            
             if three_quarter==num_cluster:
                 three_quarter_b=1
            
             if anti_quarter==num_cluster:
                 anti_quarter_b=1
            
             if anti_three_quarter==num_cluster:
                 anti_three_quarter_b=1

    return
    

        
###############################################################################
def fill_cluster(x, y, n_clusters, N, lattice,cluster, sizes):
    
    stack = [(x,y)] #last occupied site

    while len(stack) > 0:
        x, y = stack.pop(-1)

        if lattice[x,y] == 1 and cluster[x,y]==0:
            cluster[x,y] = n_clusters+1
            sizes[n_clusters] += 1     #augmente le nombre de site du cluster

            if y+1 < N:
                stack.append((x,y+1))
            if y-1 >= 0:
                stack.append((x,y-1))

            if x+1 < N:
                stack.append((x+1,y))
            if x-1 >= 0:
                stack.append((x-1,y))


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
def percolation(im,p,size_sys,seed):
    my_dpi=96 # DPI of the monitor
    import pickle
    import numpy as np

    #create_directory('percolating')   #if we want to classify them inside each density directory
    #create_directory('not_percolating')
    
    for t in range(im):
        lattice_config(size_sys,seed,p)
        occupied = lattice.nonzero()
        n_clusters = 0
        cluster=np.zeros((size_sys,size_sys), dtype=int)
        sizes = Counter()
        for i, j in zip(*occupied):
            if cluster[i,j] == 0:
                fill_cluster(i, j, n_clusters, size_sys,lattice,cluster, sizes)
                n_clusters += 1
        #print(lattice)

        new_mapping = np.arange(1,n_clusters+1)
        np.random.shuffle(new_mapping)
        cluster1= np.array([new_mapping[v-1] \
                            if not v == 0 else 0 for v in cluster.flat]).reshape(size_sys,size_sys)
        cluster2 = np.array([new_mapping[v-1]/n_clusters \
                            if not v == 0 else np.nan for v in cluster.flat]).reshape(size_sys,size_sys)
        cluster3 = np.array([new_mapping[v-1]\
                            if not v == 0 else np.nan for v in cluster.flat]).reshape(size_sys,size_sys)
    
        all_sizes = Counter(list(sizes.values()))
        
        #get size of largest cluster
        if n_clusters !=0:
            max_size = max(all_sizes.keys())
            
        if size_sys==100:
            size_im=131
        else:
            size_im=size_sys+ (size_sys/100)*30

        
       
        fig =pl.figure(figsize=((size_im/my_dpi), (size_im/my_dpi)), dpi=my_dpi)
        pl.axis('off')
        pl.imshow(cluster2,cmap='nipy_spectral')
        cmap2 = pl.cm.get_cmap('nipy_spectral')
        occupied=len(list(zip(*cluster1.nonzero())))
       
        top=set(x for x in cluster3[0][:]).intersection(set(y for y in cluster3[-1][:]))
        side=set(w for w in cluster3[:,0]).intersection(set(z for z in cluster3[:,-1]))
        
        
        HWTB=0
        HWLR=0
        PBTB=0
        PBLR=0
         
        if (top!=set() or side!=set()):
            #os.chdir('percolating') 

            color_point(cluster3,top,side)  #,center_b=0,quarter_b=0,three_quarter_b=0,anti_quarter_b=0,anti_three_quarter_b=0)
            T=cluster3[0][:]
            B=cluster3[-1][:]
            L=cluster3[:,0]
            R=cluster3[:,-1]
            
            if np.array_equal(T,B):
                PBTB=1
            if np.array_equal(L,R):
                PBLR=1
            del(T)
            del(B)
            del(L)
            del(R)
                
                
            
           
            
            
            size_side_spanning=0
            size_top_spanning=0
            
           
            
            rgba1=0
            rgba2=0
            
            size_side_spanning=0
            size_top_spanning=0
            rgba1=0
            rgba2=0
            
            if side!=set() and top!=set():
                size_spanning_cluster(side,cluster2)
                size_side_spanning= size_spanning
                rgba2=cmap2(next(iter(side)))
                size_spanning_cluster(top,cluster2)
                size_top_spanning=size_spanning
                rgba1=cmap2(next(iter(top)))
                HWTB=1
                HWLR=1
                
            elif side!=set():
                size_spanning_cluster(side,cluster2)
                size_side_spanning= size_spanning
                rgba2=cmap2(next(iter(side)))  #rgb color tuple + alpha
                HWLR=1
                 
            else:
                size_spanning_cluster(top,cluster2)
                size_top_spanning=size_spanning
                rgba1=cmap2(next(iter(top)))
                HWTB=1
                
                
            filename1='pc_1_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBTB)+'_'+str(PBLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
            '_'+str(center_b)+'_'+str(quarter_b)+'_'+\
            str(three_quarter_b)+'_'+str(anti_quarter_b)+'_'+str(anti_three_quarter_b)+\
                        '_top_'+str(top)+'_side_'+str(side)+'_size_max_clus'+str(max_size)+\
                        '_occ_'+str(occupied)+'_'
                       
            
            fig.savefig(filename1+'.png',bbox_inches='tight', pad_inches = 0,dpi=my_dpi)
                
                
            f=open(filename1+'.txt', "w+")
            f.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            f.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            f.write('Number of clusters with given size= ' +repr(sizes)+"\n")
            f.write('Spanning cluster top-bottom = '+ repr(top)+' = '+repr(size_top_spanning)+"\n")
            f.write('Spanning cluster side-side= '+ repr(side)+ ' = '+repr(size_side_spanning)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba1)+"\n")
            f.write('color of the spanning cluster = '+repr(rgba2)+"\n")
            f.close()
            
            

      
            h= open(filename1+'.pkl', "wb")
            pickle.dump(cluster1,h)
            h.close()
            
            #os.chdir('..')
        else:
            #os.chdir('not_percolating')
            filename0='pc_0_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBTB)+'_'+str(PBLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                        '_occ_'+str(occupied)+'_'
            
            fig.savefig(filename0+'_.png', bbox_inches='tight',\
                        pad_inches = 0,dpi=my_dpi)
            
            
            
            g=open(filename0+'.txt', "w+")
            g.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            g.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            g.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            g.close()
            
            i=open(filename0+'.pkl', "wb")
            pickle.dump(cluster1,i)
            i.close()
            #os.chdir('..')

        pl.close('all')
        seed+=1

    return


###############################################################################
def percolation_density(number_configs,perco_list,lattice_size,seed):
    import os
    #create_directory('L'+str(L))
    #os.chdir('L'+str(L))
    import time
    start1= time.time()
    seed_ini=seed
    im_ini=number_configs
    for p in perco_list:
        new_im=0
        seed=seed_ini
        im=im_ini
        
        if os.path.exists('p'+str(p)) and len(os.listdir('p'+str(p)))!=0:
            print('A directory '+'p='+str(p)+' already exists')
            #print ("Creation of the directory failed")
            check_name('p'+str(p))
            print('A file already exist with max seed=',max_seed)
            os.chdir('p'+str(p))
            if im>= nbre_images:
                im=im-nbre_images
                while im > 0:
                    if seed in seed_list:
                        print('Image with seed = ',seed, 'already exists')
                        seed+=1
                    else:
                        percolation(1,p,lattice_size,seed)
                        im-=1
                        new_im+=1
                        seed+=1
                        print('NEW image with seed = ',seed, 'was created')
            else:
                print('The directory already contains ', nbre_images,\
                      ' images, please choose a higher number of configurations.')
                    
            os.chdir('..')
            if new_im!=0:
                print("-->",new_im, 'new images were created')
        
        else:
            create_directory('p'+str(p))
            os.chdir('p'+str(p))
            percolation(im,p,lattice_size,seed_ini)
            os.chdir('..')
            print(im, 'new images were created')
    #os.chdir('..')  
    end1=time.time()
    total_time=end1-start1
    print("Images generated in : ", total_time, "seconds")
    return

###############################################################################
            
if ( len(sys.argv) == 7 ):
    #SEED = 101
    SEED = int(sys.argv[1])
    lattice_size = int(sys.argv[2])
    perco_init = int(sys.argv[3]) 
    perco_final = int(sys.argv[4])
    perco_inc = int(sys.argv[5])
    number_configs = int(sys.argv[6])

    perco_list=[val/10000 for val in range(perco_init,perco_final+1,perco_inc)]
            
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
##create_directory('images_perco_density_new')
##os.chdir('images_perco_density_new')
#
### %%
##print(os.getcwd())

## %%
#M=[x/1000 for x in range(0,1000,50)]
#M=M[1:]
#N=[x/10000 for x in range(5920,5942,4)]
#O=M+N

