import numpy as np
from collections import Counter, OrderedDict
import random
import itertools
import os
import imageio
import binascii
import sys
import time 
import datetime
from operator import itemgetter
np.set_printoptions(threshold=sys.maxsize)

###############################################################################
def check_name(path): #check seeds in file, returns max value and nb of files.
    import re
    results=[]
    seeds=[]
   
    for files in os.listdir(path):
        if files.endswith('.txt'):
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
    while occupied < number_occupied:       
        i=random.randint(0,size-1)
        j=random.randint(0,size-1)
        if lattice[i,j]==0:
            lattice[i,j]=1
            occupied+=1

    print("lattice_config END", datetime.datetime.now())
    return lattice, number_occupied

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
def cluster_matrix(x, y, n_clusters, N, lattice,cluster, sizes): #labelling of clusters HW
    stack=[(x,y)]
    while len(stack)>0:
        x,y=stack.pop(-1)
        if lattice[x,y] == 1 and cluster[x,y]==-1:
            cluster[x,y] = n_clusters
            sizes[n_clusters] += 1     #count nb of clusters
            if y+1 < N:
                stack.append((x,y+1))
            if y-1 >= 0:
                stack.append((x,y-1))
            if x+1 < N:
                stack.append((x+1,y))
            if x-1 >= 0:
                stack.append((x-1,y))
################################################################
def cluster_matrix_pbc(x, y, n_clusters, N, lattice,cluster, sizes): #labelling of clusters PBC
    stack=[(x,y)]
    while len(stack)>0:
        x,y=stack.pop(-1)
        if lattice[x%N,y%N] == 1 and cluster[x%N,y%N]==-1:
            cluster[x%N,y%N] = n_clusters
            sizes[n_clusters] += 1     #count nb of clusters
            stack.append((x,(y+1%N))) #modulo for PBC and wrapping of clusters
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
def span_wrap_check(p,cluster,cluster_pbc,n_clusters,n_clusters_pbc,sizes,sizes_pbc,new_mapping, new_mapping_pbc,size_sys): 
    # variable initialisation
    HWTB=0
    HWLR=0
    SPAN=0
    n_span_top=0
    list_span_tuple=0
    tuple_span=0
    n_span_sides=0
    list_span_top=[]
    list_span_side=[]
    size_span=0
    proba_max_span=0
    total_nsites_span=0
    proba_tot_span=0
    PBCTB=0 #identifier in path, if 1 wraps top-bottom
    PBCLR=0   #identifier in path, if 1 wraps left-right
    n_span_top_wrap=0 #will store number of cluster wrapping top-bottom
    tuple_span_wrap=0 #will store label of cluster wrapping 
    n_span_sides_wrap=0 #will store number of cluster wrapping top-bottom
    list_span_tuple_wrap=0
    total_nsites_span_wrap=0
    old_size_max_wrap=0
    list_span_top_wrap=[]
    list_span_side_wrap=[]
    size_span_wrap=0
    proba_span_wrap=0
    proba_tot_span_wrap=0
    proba_max_span_wrap=0
    max_size_pbc=0
    #########################################################Creation of intermediate matrices with new mapping for HW and PBC
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
    ###################################################################################################
    
    size_all=Counter(cluster_int.flat) #count number of cluster in lattice with OBC
    del size_all[0]
    
    size_all_pbc=Counter(cluster_pbc_int.flat) #count number of cluster in lattice with OBC
    del size_all_pbc[0]
    
    all_sizes_pbc = Counter(list(sizes_pbc.values())) #give list of clusters and associatd size
        
    #get size of largest cluster
    if n_clusters_pbc !=0:
        max_size_pbc = max(all_sizes_pbc.keys())       
  
    start4=time.time()
    proba_largest_pbc=(max_size_pbc/(size_sys**2))**2 #get proba to be in the largest cluster for pbc
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
    ########################################################### Creation of super lattice and check wrap
    temp=np.concatenate((cluster_int,cluster_int,cluster_int), axis=0) 
    wrap=np.concatenate((temp,temp,temp), axis=1) #creates super lattice of size_sys^3*size_sys^3
    lattice_wrap= np.array([1
                            if not v == 0 else v for v in wrap.flat]).reshape(size_sys*3,size_sys*3) #creation of superlattice
    occupied_wrap = lattice_wrap.nonzero() #retrieve occupied sites
    temp_n_clusters_wrap= 0
    cluster_wrap=np.zeros((size_sys*3,size_sys*3), dtype=int)-1
    sizes_wrap = Counter()
    print("Hoshen-Kopelman PBS START", datetime.datetime.now()) #labelling of super matrice with HW
    for i, j in zip(*occupied_wrap):
        if cluster_wrap[i,j] == -1:
            cluster_matrix(i, j, temp_n_clusters_wrap, size_sys*3,lattice_wrap,cluster_wrap, sizes_wrap)
            temp_n_clusters_wrap += 1
       
    order_wrap=OrderedDict(sizes_wrap.most_common())
    classification_wrap=list(order_wrap)
    numbers_wrap = np.arange(0,temp_n_clusters_wrap)
    weight_wrap=-np.sort(-(numbers_wrap))
    k_wrap=list(zip(classification_wrap,weight_wrap))
    correspondance_wrap=sorted(k_wrap, key = lambda t: t[0])
    unzip_wrap=list(zip(*correspondance_wrap))
    new_mapping_wrap=unzip_wrap[1] 
        
   # cluster_nan = np.array([element
  #                     if not element==0 else np.nan for element in cluster_int.flat]).reshape(size_sys,size_sys) #replace 0 by np.nan
   # cluster_norm = np.array([element/n_clusters
    #                   if not element==0 else element for element in cluster_int.flat]).reshape(size_sys,size_sys) #normalise the cluster's' labels
        
    wrap_cluster_nan = np.array([new_mapping_wrap[v]+1\
                       if not v == -1 else np.nan for v in cluster_wrap.flat]).reshape(size_sys*3,size_sys*3) #replace 0 by np.nan for super lattice
    temp_wrap_cluster_int = np.array([(new_mapping_wrap[v]+1) \
                        if not v == -1 else 0 for v in cluster_wrap.flat]).reshape(size_sys*3,size_sys*3) #normalise the cluster labels for super lattice
    wrap_cluster_int=temp_wrap_cluster_int[size_sys:size_sys*2,size_sys:size_sys*2] #retrieve new lattice with pbc
        
    top_bot_inter_wrap=set(x for x in wrap_cluster_nan[0][:]).intersection(set(y for y in wrap_cluster_nan[-1][:]))
    sides_inter_wrap=set(w for w in wrap_cluster_nan[:,0]).intersection(set(z for z in wrap_cluster_nan[:,-1]))
    size_all_wrap=Counter(cluster_int.flat) #count number of cluster in lattice with HW      
    del size_all_wrap[0]
    n_clusters_wrap=len(size_all_wrap.keys())
    temp_corresp_wrap=[list(zip(ele1,ele2)) for ele1,ele2 in list(zip( wrap_cluster_int,cluster_pbc_int))] #correspondance label wrap_cluster_int-cluster_pbc_int
    merge_corresp_wrap=list(itertools.chain.from_iterable(temp_corresp_wrap))
    corresp_wrap=Counter(merge_corresp_wrap)
    print(corresp_wrap)
    print('##############')
    #[item for item in a if item[0] == 1]
    if (top_bot_inter!=set() or sides_inter!=set()): 
        print('SPAN')
        print('top_bot_inter',top_bot_inter,'sides_inter',sides_inter)
        SPAN=1
        if top_bot_inter!=set():
            print(top_bot_inter)
            HWTB=1
            n_span_top=[ele for ele in top_bot_inter]
            print(n_span_top)
            size_top=[size_all[ele] for ele in n_span_top]
            list_span_top=list(zip(n_span_top,size_top))
            print(list_span_top)
            temp_size_span=max(size_top)
            print(temp_size_span)
               
        if sides_inter!=set():
            print(sides_inter)
            HWLR=1  
            n_span_sides=[ele for ele in sides_inter]
            print(n_span_sides)
            size_sides=[size_all[ele] for ele in n_span_sides]
            list_span_side=list(zip(n_span_sides,size_sides))
            temp_size_span=max(size_sides)
            print(temp_size_span)
        temp_list_span_tuple=list_span_top+list_span_side
        list_span_tuple=list(set(temp_list_span_tuple))
        tuple_span=max(list_span_tuple, key=itemgetter(1))
        total_nsites_span=sum([element[1] for element in list_span_tuple])
        proba_tot_span=(total_nsites_span/(size_sys**2))**2
        print('list_span_tuple',list_span_tuple)
        print('max tuple span',tuple_span)
        print('total_nsites_span',total_nsites_span)
        size_max_span=tuple_span[1]
        proba_max_span=(size_span/(size_sys**2))**2
    else:
        print('Not span')
    
####################################################### Check for wrapping
    if (top_bot_inter_wrap!=set() or sides_inter_wrap!=set()): 
        print('SPAN')
        print('top_bot_inter_wrap',top_bot_inter_wrap,'sides_inter_wrap',sides_inter_wrap)
        cluster_wrap_nan =wrap_cluster_nan[size_sys:size_sys*2,size_sys:size_sys*2]
        size_all_wrap=Counter(cluster_wrap_nan.flat)
        if top_bot_inter_wrap!=set():
            print(top_bot_inter_wrap)
            PBCTB=1
            n_span_top_wrap=[ele for ele in top_bot_inter_wrap]
            print(n_span_top_wrap)
            size_top_wrap=[size_all_wrap[ele] for ele in n_span_top_wrap]
            list_span_top_wrap=list(zip(n_span_top_wrap,size_top_wrap))
            print(list_span_top_wrap)
            temp_size_span_wrap=max(size_top_wrap)
            print('temp_size span top_wrap',temp_size_span_wrap)
                
        if sides_inter_wrap!=set():
            print(sides_inter_wrap)
            PBCLR=1  
            n_span_sides_wrap=[ele for ele in sides_inter_wrap]
            print(n_span_sides_wrap)
            size_sides_wrap=[size_all_wrap[ele] for ele in n_span_sides_wrap]
            list_span_side_wrap=list(zip(n_span_sides_wrap,size_sides_wrap))
            temp_size_span_wrap=max(size_sides_wrap)
            print('temp_size span sides_wrap',temp_size_span_wrap)
        temp_list_span_tuple_wrap=list_span_top_wrap+list_span_side_wrap
        list_span_tuple_wrap=list(set(temp_list_span_tuple_wrap))
        tuple_span_wrap=max(list_span_tuple_wrap, key=itemgetter(1))
        total_nsites_span_wrap=sum([tuple_span_wrap[1] for tuple_span_wrap in list_span_tuple_wrap])
        proba_tot_span_wrap=(total_nsites_span_wrap/(size_sys**2))**2
        print('list_span_tuple_wrap',list_span_tuple_wrap)
        print('max tuple span_wrap',tuple_span_wrap)
        print('total_nsites_span_wrap',total_nsites_span_wrap)
            
        max_size_span_wrap=tuple_span_wrap[1]
        proba_max_span_wrap=(max_size_span_wrap/(size_sys**2))**2
    else:
        print('Not wrap')
    print(SPAN)
    return cluster_pbc_int, cluster_pbc_norm,cluster_int,cluster_nan,cluster_norm,wrap_cluster_int,occ,top_bot_inter,sides_inter,list_span_side_wrap,list_span_top_wrap,SPAN, HWLR, HWTB, PBCLR, PBCTB, size_span_wrap, proba_max_span_wrap,proba_tot_span_wrap,proba_max_span, proba_tot_span, size_span, max_size_pbc
    

##################################################################################
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
        cluster_pbc_int, cluster_pbc_norm,cluster_int,cluster_nan,cluster_norm,wrap_cluster_int,occ,top_bot_inter,sides_inter,list_span_side_wrap,list_span_top_wrap,SPAN,\
       HWLR, HWTB, PBCLR, PBCTB, size_span_wrap, proba_max_span_wrap,proba_tot_span_wrap,proba_max_span, proba_tot_span, size_span, max_size_pbc=span_wrap_check(p,cluster,cluster_pbc,n_clusters,n_clusters_pbc,sizes,sizes_pbc,new_mapping, new_mapping_pbc,size_sys)
          
        filename='pc_'+str(SPAN)+'_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                    '_npbc'+str(n_clusters_pbc)+'_smc'+str(max_size_pbc)+'_nc'+str(n_clusters)
        text_file=open(filename+'.txt', "w+")
        text_file.write('Total number of cluster pbc= '+ repr(n_clusters_pbc)+'\n')
        text_file.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size_pbc)+'\n')
        text_file.write('spanning cluster top to bottom)= ' +repr(top_bot_inter)+"\n")
        text_file.write('spanning cluster left to right)= ' +repr(sides_inter)+"\n")
        text_file.write('wrapping cluster top to bottom)= ' +repr(list_span_top_wrap)+"\n")
        text_file.write('wrapping cluster left to right)= ' +repr(list_span_side_wrap)+"\n")
        text_file.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
        text_file.close()
        data_pkl = {'cluster_pbc_int' : cluster_pbc_int ,
                   'cluster_pbc_norm' : cluster_pbc_norm,
                   'n_clusters_pbc':n_clusters_pbc,
                   'wrap_cluster_int': wrap_cluster_int,
                   'proba_max_span_wrap':proba_max_span_wrap,
                   'proba_tot_span_wrap':proba_tot_span_wrap,
                   'cluster_int':cluster_int,
                   'cluster_norm':cluster_norm,
                   'proba_max_span':proba_max_span,
                   'proba_tot_span':proba_tot_span,
                   'n_clusters':n_clusters}
            
        pkl_file=open(filename+'.pkl', "wb")
        pickle.dump(data_pkl ,pkl_file)
        pkl_file.close()
            
        end2=time.time()-start2
        print(end2)

    return cluster,n_clusters,cluster_pbc,n_clusters_pbc, sizes_pbc,lattice_para[0],new_mapping,order,order_pbc,occ,\
sizes,cluster_int,cluster_nan

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
