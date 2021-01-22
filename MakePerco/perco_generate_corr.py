

import matplotlib.pyplot as pl
import numpy as np
from collections import Counter, OrderedDict
import random
import os
import imageio
#Pour voir les matrices  en entiers sans troncature
import sys
import numpy
import time 
import datetime

numpy.set_printoptions(threshold=sys.maxsize)

###############################################################################
def check_name(path):
    
    import re
    c=0
    N=os.listdir(path)
    
    nbre_file=len(N) 
    del(N)
    
    result=[0]*nbre_file
    seed_list=[0]*nbre_file

    B=(name for name in os.listdir(path))

    for c in range(nbre_file):
          
        A=next(B).split('_')[9]      
       
    
        regex1 = re.compile('\d+')
        result[c]=re.findall(regex1,A)
        c+=1
    
    for j in range(len(result)):
        
        seed_list[j]=int(result[j][0])
        j+=1
        
    max_seed=max(seed_list)
    nbre_images=nbre_file/5
    
   
    
    return max_seed, seed_list, nbre_images

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
        print('clus_n',cluster_number)
        new_array=np.array([v if  v == cluster_number else 0 for v in boundary_array_1])
        print('new_array',new_array)
        coord_non_zero=new_array.nonzero()[0]
        print('coord_non_z',coord_non_zero)
        for arg in coord_non_zero:
            print('arg',arg)
            print('bound_array',boundary_array_2)
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
    n_cluster_pbc_int, counts = numpy.unique(cluster_pbc_int, return_counts=True)
    zip_lists=list(zip( n_cluster_pbc_int, counts))
    for k in range(len(spanning_set)):
        for i in range(len(zip_lists)):
            if zip_lists[i][0]==span[k]:
                size.append(zip_lists[i][1])
    if len(size)>1:
        size_spanning=max(size)
        
    return size_spanning

##############################################################################
def correlation_function_pbc_new_diagonal(lattice,l,n_clusters,seed):
    from math import sqrt

    start50=time.time()

    square_distance=np.zeros(l**2)
    corr2=np.zeros(l**2)
    corr_max_cluster=np.zeros(l**2)
    
    y_func=np.zeros(l**2)
    div_pbc=np.zeros(l**2)
    x_pbc_max_cluster=np.zeros(l**2)
    div_pbc_max_cluster=np.zeros(l**2)

    print('start correlation function',datetime.datetime.now())

    
    new_occupied = lattice.nonzero()
    new_occ=zip(*new_occupied)

    for y0 in range(l):
        for x0 in range(l):
            for y1 in range(y0,l):
                if y1==y0:

                    for x1 in range(x0,l):

                        x_distance=abs(x0-x1)
                        y_distance=abs(y0-y1)
                        
  

                        
                        if x_distance>(l/2):
                            x_distance=abs(l-x_distance) 
                        if y_distance>(l/2):
                            y_distance=abs(l-y_distance) 

                
                        distance=x_distance**2+ y_distance**2       
                        
                        div_pbc[distance]+=1
                        if x0!=x1 or y0!=y1:
                            div_pbc[distance]+=1
                    
                        if square_distance[distance]!=distance:
                            square_distance[distance]=distance
                        
                        
                        
                        if lattice[y0,x0]!=0 and lattice[y1,x1]!=0 and lattice[y0,x0]==lattice[y1,x1] : 
                            corr2[distance]+=1

                            if x0!=x1 or y0!=y1:
                                corr2[distance]+=1
                        if lattice[y0,x0]==n_clusters:
                            div_pbc_max_cluster[distance]+=1
                            if x0!=x1 or y0!=y1:
                                div_pbc_max_cluster[distance]+=1

                            if lattice[y0,x0]!=0 and  lattice[y1,x1]!=0 and lattice[y0,x0]==lattice[y1,x1]: 
                                corr_max_cluster[distance]+=1
                                if x0!=x1 or y0!=y1:
                                    corr_max_cluster[distance]+=1

                    
                else:

                    for x1 in range(l): 

                        x_distance=abs(x0-x1)
                        y_distance=abs(y0-y1)
                        

                        if x_distance>(l/2):
                            x_distance=abs(l-x_distance)

                        
                        if y_distance>(l/2):
                            y_distance=abs(l-y_distance) 

                
                        distance= x_distance**2+ y_distance**2
   
                        
                        div_pbc[distance]+=1
                        if x0!=x1 or y0!=y1:
                            div_pbc[distance]+=1
                                      
                        if square_distance[distance]!=distance:
                            square_distance[distance]=distance

                                      
                                    
                        if lattice[y0,x0]!=0 and lattice[y1,x1]!=0 and lattice[y0,x0]==lattice[y1,x1]: 
                            corr2[distance]+=1

                            if x0!=x1 or y0!=y1:
                                corr2[distance]+=1
                        if lattice[y0,x0]==n_clusters:
                            div_pbc_max_cluster[distance]+=1
                            if x0!=x1 or y0!=y1:
                                div_pbc_max_cluster[distance]+=1

                            if lattice[y0,x0]!=0 and lattice[y1,x1]!=0 and lattice[y0,x0]==lattice[y1,x1]: 
                                corr_max_cluster[distance]+=1
                                if x0!=x1 or y0!=y1:
                                    corr_max_cluster[distance]+=1

        
    end50=time.time()-start50
    print('end correlation function',end50)
                        

    length=len(div_pbc)
    print('test')
                                      

    
    sqrt_distance=[sqrt(square_distance[h]) for h in range(len(square_distance))]
    sqrt_distance=np.array(sqrt_distance)
    print('len sqrt distance',len(sqrt_distance))
    
    
    average=[o/l for o,l in zip(corr2,div_pbc) ]
    average=np.array(average)
    index_max_corr=np.max(np.nonzero(corr2))
    corr3=corr2[:index_max_corr+1]
    index = np.where(corr3 == 0)[0]  #np.argwhere(np.isnan(W))
    correlation_value=np.delete(average,index)
    new_corr_before_average=np.delete(corr3,index)
    new_distance=np.delete(sqrt_distance,index)
    new_distance=np.delete(sqrt_distance,np.argwhere(sqrt_distance==0))
    
    new_distance=np.insert(new_distance, 0, 0)
    

    index_max=np.max(np.nonzero(new_corr_before_average))
    len_zero=len(new_distance)-(index_max+1)
    new_average=np.concatenate((correlation_value[:index_max+1],np.zeros(len_zero)))
    new_corr_before_average=np.concatenate((new_corr_before_average[:index_max+1],np.zeros(len_zero)))
    correlation_value=np.concatenate((correlation_value[:index_max+1],np.zeros(len_zero)))
    
    
    
    
    average_largest=[o/l for o,l in zip(corr_max_cluster,div_pbc) ]
    average_largest=np.array(average_largest)
    index_max_corr_largest=np.max(np.nonzero(corr_max_cluster))
    corr3_largest=corr_max_cluster[:index_max_corr_largest+1]
    
    index_largest = np.where(corr3_largest == 0)[0]  #np.argwhere(np.isnan(W))
    correlation_value_largest=np.delete(average_largest,index_largest)
    new_corr_before_average_largest=np.delete(corr3_largest,index_largest)
    new_distance_largest=np.delete(sqrt_distance,index_largest)
    new_distance_largest=np.delete(sqrt_distance,np.argwhere(sqrt_distance==0))
    
    new_distance_largest=np.insert(new_distance_largest, 0, 0)
    

    index_max_largest=np.max(np.nonzero(new_corr_before_average_largest))
    len_zero_largest=len(new_distance_largest)-(index_max_largest+1)
    new_average_largest=np.concatenate((correlation_value_largest[:index_max_largest+1],np.zeros(len_zero_largest)))
    new_corr_before_average_largest=np.concatenate((new_corr_before_average_largest[:index_max_largest+1],np.zeros(len_zero_largest)))
    correlation_value_largest=np.concatenate((correlation_value_largest[:index_max_largest+1],np.zeros(len_zero_largest)))
    

    
    end6=time.time()-start50
    print('end correlation function',end6)

    
    return new_distance,correlation_value, new_distance_largest,correlation_value_largest, new_corr_before_average
#################################################################################
def size_max(order_pbc,n_clusters):
    
    L=order_pbc
    values=[]
    keys=[]

    for k, v in L.items():
        if k!=-1:
            keys.append(k)
            values.append(v)
    max_clus=max(values)
    max_n_clus=n_clusters
    
    return max_clus, max_n_clus


################################################################################
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
    my_dpi=96 # DPI of the monitor
    
    global cluster_nan
   
    import pickle
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
#     print('debut')

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
#                 cluster_matrix_pbc(i, j, n_clusters, size_sys,lattice,cluster, sizes)
                n_clusters_pbc += 1
        print("Hoshen-Kopelman PBC END", datetime.datetime.now())
        
        print("Hoshen-Kopelman HW START", datetime.datetime.now())
        for i, j in zip(*occupied):
            if cluster[i,j] == -1:
                cluster_matrix(i, j, n_clusters, size_sys,lattice_para[0],cluster, sizes)
#                 cluster_matrix_pbc(i, j, n_clusters, size_sys,lattice,cluster, sizes)
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
        #print(unzip)
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

        #print(cluster_int)
        #print(cluster_pbc_int)
        
        
        print("correlation calculations", datetime.datetime.now())
        
        occ=len(list(zip(*cluster_int.nonzero())))
        start4=time.time()
        size_maxi=size_max(order_pbc,n_clusters)
        proba_largest= (size_maxi[0]/(size_sys**2))**2
        print('first correlation')
        corr_value=correlation_function_pbc_new_diagonal(cluster_pbc_int,size_sys,n_clusters_pbc,seed)
        print('end first correlation')
        
        square_proba=p*p
        
        
        
        
        print('proba largest', proba_largest,'\n')
        print('occ',occ,'\n')
        print('max_clus',size_maxi[0],'\n')
        
       
        
        
        pl.plot(corr_value[0],corr_value[1],label='g(r) every clusters')
        pl.plot(corr_value[2],corr_value[3], label='largest cluster')
        pl.axhline(y=square_proba, color='g', linestyle='--')
        pl.axhline(y=proba_largest, color='grey', linestyle='--')
        pl.legend(loc='best')
        pl.xlabel('distance r')
        pl.ylabel('correlation function g(r)')
        pl.title('Correlation function for system '+str(size_sys)+' at density '+str(p))
        pl.savefig('pc_________s'+str(seed)+'corr_func_'+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+'_'+'.png')
      
    
        corr_proba_largest=corr_value[1]-proba_largest
        

    
        
        
        end4=time.time()-start4
        
#         print('fin_corre', datetime.datetime.now())
        
        all_sizes = Counter(list(sizes.values()))
        
        #get size of largest cluster
        if n_clusters !=0:
            max_size = max(all_sizes.keys())
            

       
        fig =pl.figure()
        pl.axis('off')
        pl.imshow(cluster_pbc_norm,cmap='Greys')
        

        occ=len(list(zip(*cluster_int.nonzero())))

        cmap2 = pl.cm.get_cmap('Greys')
        
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

                
            
               
            

           

            correlation_len=correlation_l(corr_value[0],corr_value[1], size_sys,proba_largest,p) 
           
            
            
            
          
                
            filename1='pc_1_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                        '_top_'+str(top_bot_inter)+'_side_'+str(sides_inter)+'_size_max_clus'+str(max_size)+\
                        '_occ_'+str(occ)+'_'
                       

            pl.imsave(filename1+'.png', cluster_pbc_norm,cmap='Greys')
            
            

            z=open(filename1+'corr_length'+'.txt', "w+")
            z.write('\n'+repr(correlation_len)+'\n')
            z.close()
                
                
            f=open(filename1+'.txt', "w+")
            f.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            f.write('Size of the largest cluster (number of site occupied)= '+ repr(max_size)+'\n')
            f.write('Number of clusters with given size= ' +repr(sizes)+"\n")
#             f.write('Spanning cluster top-bottom = '+ repr(top)+' = '+repr(size_top_spanning)+"\n")
#             f.write('Spanning cluster side-side= '+ repr(side)+ ' = '+repr(size_side_spanning)+"\n")

            f.close()
            
            

      
            h= open(filename1+'cluster_pbc_int'+'.pkl', "wb")
            pickle.dump(cluster_pbc_int,h)
            h.close()
            

        else:
            size_max(order_pbc,n_clusters)
            
            correlation_len=correlation_l(corr_value[0],corr_value[1], size_sys,proba_largest,p)
            
            
            filename0='pc_0_'+str(HWTB)+'_'+str(HWLR)+'_'+str(PBCTB)+'_'+str(PBCLR)+'__p'+str(p)+'_L'+str(size_sys)+'_s'+str(seed)+\
                        '_size_max_clus'+str(size_maxi[0])+'_n'+str(size_maxi[1])+'_occ_'+str(occ)+'_'
            

            pl.imsave(filename0+'.png', cluster_pbc_norm,cmap='Greys')
            
            
            
            z=open(filename0+'corr_length'+'.txt', "w+")
            z.write('\n'+repr(correlation_len)+'\n')
            z.close()
            
            

            
            
            g=open(filename0+'.txt', "w+")
            g.write('Total number of cluster= '+ repr(n_clusters)+'\n')
            g.write('Size of the largest cluster (number of site occupied)= '+ repr(size_maxi[0])+'\n')
            g.write('Sizes of each clusters (number associated to the cluster: number of occupied sites)= ' +repr(sizes)+"\n")
            g.close()
            
            i=open(filename0+'cluster_pbc_int'+'.pkl', "wb")
            pickle.dump(cluster_pbc_int,i)
            i.close()
            #os.chdir('..')

        pl.close('all')
        seed+=1
        end2=time.time()-start2
        print(end2)

    return cluster,n_clusters,cluster_pbc,n_clusters_pbc, sizes_pbc,lattice_para[0],new_mapping,order,order_pbc,occ,\
sizes,cluster_int, top_bot_inter,sides_inter,cluster_nan,



#################################################################################################################################
def percolation_density(number_configs,perco_list,lattice_size,seed):
    import os
    #create_directory('L'+str(L))
    #os.chdir('L'+str(L))
    import time
#     correlation=[]
#     correlation_func=[]
    dens=[]
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
            check=check_name('p'+str(p))
            print('A file already exist with max seed=',check[0])  #max_seed)
            os.chdir('p'+str(p))
            if im>= check[2]:  #nbre_images:
                im=im-check[2]   #nbre_images
                while im > 0:
                    if seed in check[1]: #seed_list:
                        print('Image with seed = ',seed, 'already exists')
                        seed+=1
                    else:
                        dens.append(p)
                        perco_calcul=percolation(1,p,lattice_size,seed) 
                        perco_calcul
                        new_im+=1
                        seed+=1
                        print('NEW image with seed = ',seed, 'was created')
            else:
                print('The directory already contains ', check[2],\
                      ' images, please choose a higher number of configurations.')
                    
            os.chdir('..')
            if new_im!=0:
                print("-->",new_im, 'new images were created')
        
        else:
            create_directory('p'+str(p)+'new_diagonal')
            os.chdir('p'+str(p)+'new_diagonal')
            percolation(im,p,lattice_size,seed_ini)
            os.chdir('..')
            print(im, 'new images were created')
#     os.chdir('..')  
    end1=time.time()
    total_time=end1-start1
    print("Images generated in : ", total_time, "seconds")
    return 

####################################################################################################################
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

   



