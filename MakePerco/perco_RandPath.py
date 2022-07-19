import numpy as np
from collections import Counter, OrderedDict
import random
import os
import imageio
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
        if files.endswith('info.txt'):
            results.append(files)
            seed_sys=files.split('_')[9]      
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
    j=size//2
    i=0
    lattice[i,j]=1
    while occupied < np.round_(number_occupied) and i<(size-1):
        #print(lattice)
        proba=random.randint(0,1)
        #print('j',j)
        if (proba==0 and j>0):
            j-=1
            lattice[i,j]=1
            lattice[i+1,j]=1
            i+=1
            proba=0
        elif (proba==0 and j<0):
            j+=1
            lattice[i,j]=1
            lattice[i+1,j]=1
            i+=1
            proba=0
        elif (proba==1 and j<size-1):
            j+=1
            lattice[i,j]=1
            lattice[i+1,j]=1
            i+=1
            proba=0
        elif (proba==1 and j>size-1):
            j-=1
            lattice[i,j]=1
            lattice[i+1,j]=1
            i+=1
            proba=0
    occupied=len(list(zip(*lattice.nonzero())))
    print('###################################################################')
    print('lattice after random path',lattice)
    #print('occ',occupied)   
    if np.round_(occupied)< np.round_(number_occupied):
        i=0
        j=0
        while np.round_(occupied)< np.round_(number_occupied):
            i=random.randint(0,size-1)
            j=random.randint(0,size-1)
            if lattice[i,j]==0:
                lattice[i,j]=1
                occupied+=1
    occupied_2=len(list(zip(*lattice.nonzero())))
    print('occ1',occupied_2)
    print("lattice_config END", datetime.datetime.now())
    print('lattice after random path+add',lattice)
    return lattice, occupied_2

###############################################################################       
def color_point(cluster,top,side):  
     
 
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

    return quarter_b,three_quarter_b,anti_quarter_b,center_b,anti_three_quarter_b
    

################################################################################
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
    
    
###############################################################################                        
                        
def correlation_l(distance_to_site,Correlation_func, size_sys,proba_largest,p):
    
    from math import sqrt 
    start=time.time()
    correlation=0
    sq_corr_l=0
    denom=0
    correlation_large=0
    denom_large=0
    for i in range(len(distance_to_site)):
        corr_val=Correlation_func[i]-proba_largest
        if corr_val>=0 and corr_val>=10**(-8):
            correlation+= ((distance_to_site[i]**2)*(Correlation_func[i]))#-proba_largest))
            correlation_large+= ((distance_to_site[i]**2)*(Correlation_func[i]-proba_largest))
        
                    
            denom+=(Correlation_func[i])#-proba_largest)
            denom_large+=(Correlation_func[i]-proba_largest)
        
        else:
            break
    sq_corr_l=(correlation_large/(6*denom_large))
        

    if sq_corr_l==0:
        correlation_length=0
    elif sq_corr_l<0:
        correlation_length=0

    else:

        correlation_length=sqrt(sq_corr_l)
    end=time.time()-start
    print('corre_length_end',end)
    return correlation_length
  

    
###############################################################################
def percolation(im,p,size_sys,seed):

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
        nb_occupied=len(list(zip(*lattice_para[0].nonzero())))
        expected=p*size_sys*size_sys
        print('nb_occupied',nb_occupied,'expected',expected)
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
        
        
        
        
        cluster_pbc_int= np.array([new_mapping_pbc[v]+1 \
                            if not v == -1 else 0 for v in cluster_pbc.flat]).reshape(size_sys,size_sys)
        cluster_pbc_norm = np.array([(new_mapping_pbc[v]+1)/n_clusters_pbc \
                            if not v == -1 else 0 for v in cluster_pbc.flat]).reshape(size_sys,size_sys)
        cluster_pbc_nan= np.array([new_mapping_pbc[v]+1 \
                            if not v == -1 else np.nan for v in cluster_pbc.flat]).reshape(size_sys,size_sys)
        
        
        print("HW coloring", datetime.datetime.now())
        
        cluster_int= np.array([new_mapping[v]+1 \
                            if not v == -1 else 0 for v in cluster.flat]).reshape(size_sys,size_sys)
        cluster_norm = np.array([(new_mapping[v]+1)/n_clusters \
                            if not v == -1 else 0 for v in cluster.flat]).reshape(size_sys,size_sys)
        cluster_nan = np.array([new_mapping[v]+1\
                           if not v == -1 else np.nan for v in cluster.flat]).reshape(size_sys,size_sys)

        
        
        all_sizes_pbc = Counter(list(sizes_pbc.values()))
        
        #get size of largest cluster
        if n_clusters_pbc !=0:
            max_size_pbc = max(all_sizes_pbc.keys())
        
        occ=len(list(zip(*cluster_int.nonzero())))
        start4=time.time()
        proba_largest=(max_size_pbc/(size_sys**2))**2

        square_proba=p*p

       
        print('max clus', max_size_pbc)
        
        
 
        end4=time.time()-start4

        
        
            
        occ=len(list(zip(*cluster_int.nonzero())))

        
        top=cluster_nan[0][:]
        bottom=cluster_nan[-1][:]
        left=cluster_nan[:,0]
        right=cluster_nan[:,-1]
       
        top_bot_inter=set(x for x in cluster_nan[0][:]).intersection(set(y for y in cluster_nan[-1][:]))
        sides_inter=set(w for w in cluster_nan[:,0]).intersection(set(z for z in cluster_nan[:,-1]))
        
        
        HWTB=0
        HWLR=0
        PBCTB=0
        PBCLR=0
        
        size_side_spanning_pbc=0
        size_top_spanning_pbc=0
        size_top_side_spanning_pbc=0
        rgba1=0
        rgba2=0
         
        if (top_bot_inter!=set() or sides_inter!=set()):
            
            top_pbc=set(x for x in cluster_pbc_nan[0][:]).intersection(set(y for y in cluster_pbc_nan[-1][:]))
            side_pbc=set(w for w in cluster_pbc_nan[:,0]).intersection(set(z for z in cluster_pbc_nan[:,-1]))
            
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

 
   
            filename1='pc_1_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                        '_nc'+str(n_clusters_pbc)+'_smc'+str(max_size_pbc)+'_n'+str(n_clusters_pbc)

                
            print(filename1)  
            text_file1=open(filename1+'.txt', "w+")
            text_file1.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            text_file1.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size_pbc)+'\n')
            text_file1.write('Number of clusters with given size= ' +repr(sizes)+"\n")
            text_file1.close()

            
            data_pkl1 = {'cluster_pbc_int' : cluster_pbc_int ,
                       'cluster_pbc_norm' : cluster_pbc_norm,
                       'n_clusters_pbc':n_clusters_pbc,
                       'cluster_int':cluster_int,
                       'cluster_norm':cluster_norm,
                       'proba largest' : proba_largest,
                        'square proba':square_proba,
                        'size max cluster':max_size_pbc}
            
            pkl_file1= open(filename1+'.pkl', "wb")
            pickle.dump(data_pkl1,pkl_file1)
            pkl_file1.close()


        else:
            
            filename0='pc_0_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                        '_nc'+str(n_clusters_pbc)+'_smc'+str(max_size_pbc)+'_n'+str(n_clusters_pbc)
          
            
            print(filename0)   
            text_file0=open(filename0+'.txt', "w+")
            text_file0.write('Total number of cluster= '+ repr(n_clusters_pbc)+'\n')
            text_file0.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size_pbc)+'\n')
            text_file0.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            text_file0.close()
            
            
            
            data_pkl0 = {'cluster_pbc_int' : cluster_pbc_int ,
                       'cluster_pbc_norm' : cluster_pbc_norm,
                       'n_clusters_pbc':n_clusters_pbc,
                       'cluster_int':cluster_int,
                       'cluster_norm':cluster_norm,
                       'proba largest' : proba_largest,
                        'square proba':square_proba,
                        'size max cluster':max_size_pbc}
            
            pkl_file0=open(filename0+'.pkl', "wb")
            pickle.dump(data_pkl0 ,pkl_file0)
            pkl_file0.close()
            
        end2=time.time()-start2
        print(end2)

    return cluster,n_clusters,cluster_pbc,n_clusters_pbc, sizes_pbc,lattice_para[0],new_mapping,order,order_pbc,occ,\
sizes,cluster_int, top_bot_inter,sides_inter,cluster_nan



#################################################################################################################################
def percolation_density(number_configs,perco_list,lattice_size):  
    import os
    
    import time
    
    dens=[]
    start1= time.time()
    configs_wanted=number_configs
    #seed=int(binascii.hexlify(os.urandom(4)),16)
    for p in perco_list:
        print('percolation_density: workin on p=', p)

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
                perco_calcul= percolation(1,p,lattice_size,seed) 
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
if ( len(sys.argv) == 6 ):
    #SEED = 101
    #SEED = int(sys.argv[1])
    lattice_size = int(sys.argv[1])
    perco_init = int(sys.argv[2]) 
    perco_final = int(sys.argv[3])
    perco_inc = int(sys.argv[4])
    number_configs = int(sys.argv[5])

    perco_list=[val/10000 for val in range(perco_init,perco_final+1,perco_inc)]
            
    # %%
    percolation_density(number_configs,perco_list,lattice_size) 
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
