import numpy as np
from collections import Counter, OrderedDict
import random
import os
import imageio
import binascii
import sys
import time 
import datetime
import pickle
import re
from math import sqrt 
    
def correlation_function_pbc(filename_data):
    filename, file_extension = os.path.splitext(filename_data) #split the basename from the extension
    if filename+'.cor' in os.listdir('.'):
        print('--- perco_generate_corr():', filename, ' already exists --- skipping!')
        return
    else:
        print('--- perco_generate_corr():', filename, ' is now being made!')
        # make an empty file to ensure the file exists already before the computation
        corr_value_zipped=np.zeros(10)
        header = '{0:^5s}   {1:^7s}   {2:^7s} {3:^5s}   {4:^7s} '.format('distance', 'Corr func','Corr func-proba largest','distance','Corr func largest')
        np.savetxt(filename+'.cor',np.zeros(10), header=header, fmt=['    %.7f  '])

        # actual data generation
        data=pickle.load(open(filename_data,"rb")) 
        lattice=data['cluster_pbc_int'] #cluster_pbc_int
        #square_proba=data['square proba']
        proba_largest=data['proba largest']
        #n_clusters=data['n_clusters_pbc']
        #We collect the probability of occupation p, the seed and the size of the system from the filename
        L_size=filename.split('_')[8]      
        regex1 = re.compile('\d+')
        size_sys_reg=re.findall(regex1,L_size)
        l=int(size_sys_reg[0])

        p_occ=filename.split('_')[7]      
        regex2 = re.compile('\d+\.\d+')
        p_reg=re.findall(regex2,p_occ)
        p=float(p_reg[0])

        seed_sys=filename.split('_')[9]      
        regex3 = re.compile('\d+')
        seed_sys_reg=re.findall(regex3,seed_sys)
        seed=int(seed_sys_reg[0])
        
        n_clus_sys=filename.split('_')[10]     
        regex4 = re.compile('\d+')
        n_clus_sys_reg=re.findall(regex4,n_clus_sys)
        n_clusters=int(n_clus_sys_reg[0])

        square_proba=p*p
		
        start50=time.time()
        #initialization of the array 
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

        for y0 in range(l):   # we loop over the system
            for x0 in range(l):  # x0,y0: coordinate of our origine point
                for y1 in range(y0,l): #y1: coordinate of the 2nd point
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
                                square_distance[distance]=distance #add to array containing all the square distances found in the lattice
                            # correlation function for all clusters
                            if lattice[y0,x0]!=0 and lattice[y1,x1]!=0 and lattice[y0,x0]==lattice[y1,x1] : 
                                corr2[distance]+=1
                                if x0!=x1 or y0!=y1:
                                    corr2[distance]+=1
                            # correlation fcn for largest cluster
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

        #data processing
        sqrt_distance=[sqrt(square_distance[h]) for h in range(len(square_distance))] 
        sqrt_distance=np.array(sqrt_distance)
        print('len sqrt distance',len(sqrt_distance))

        average=np.divide(corr2,div_pbc, out=np.zeros_like(corr2), where=div_pbc!=0)# computation of the probabilities
        index_max_corr=np.max(np.nonzero(corr2)) #find the index of the max non zero element in corr2
        corr3=corr2[:index_max_corr+1] #cut the array after index_max_corr
        index = np.where(corr3 == 0)[0]  #np.argwhere(np.isnan(W)) #find the index of the zero elements in corr3
        correlation_value=np.delete(average,index) #delete zero elements in average
        new_corr_before_average=np.delete(corr3,index)# delete zero elements in corr3
        new_distance=np.delete(sqrt_distance,index)#delete zero elements in sqrt_distance
        new_distance=np.delete(sqrt_distance,np.argwhere(sqrt_distance==0))

        new_distance=np.insert(new_distance, 0, 0)#add back zero at the begining

        index_max=np.max(np.nonzero(new_corr_before_average))
        len_zero=len(new_distance)-(index_max+1)
        new_average=np.concatenate((correlation_value[:index_max+1],np.zeros(len_zero)))
        new_corr_before_average=np.concatenate((new_corr_before_average[:index_max+1],np.zeros(len_zero)))
        correlation_value=np.concatenate((correlation_value[:index_max+1],np.zeros(len_zero)))
        #same process as before but for the largest cluster
        average_largest=np.divide(corr_max_cluster,div_pbc, out=np.zeros_like(corr_max_cluster), where=div_pbc!=0)
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
        corr_proba_largest=correlation_value-proba_largest 

        # now prepare data the write into actual file
        corr_value_zipped=list(zip(new_distance,correlation_value, corr_proba_largest,new_distance_largest,correlation_value_largest))
        header = '{0:^5s}   {1:^7s}   {2:^7s} {3:^5s}   {4:^7s} '.format('distance', 'Corr func','Corr func-proba largest','distance','Corr func largest')
        np.savetxt(filename+'.cor',corr_value_zipped, header=header, fmt=['    %.7f  ','    %.7f  ','    %.7f  ','  %.7f','  %.7f'])
        end6=time.time()-start50
        print('end correlation function',end6)

    return new_distance,correlation_value, new_distance_largest,correlation_value_largest, new_corr_before_average
  ################################################################################################################
def correlation_l(filename_pkl):
    from math import sqrt
    start=time.time()
    correlation=0
    sq_corr_l=0
    denom=0
    correlation_large=0
    denom_large=0
    filename=filename_pkl.rsplit('.', 1)[0]
    print(filename)    
    corr_data= np.loadtxt(filename+'.cor', unpack=True)

    pkl_file=open(filename_pkl,"rb")
    dictionary=pickle.load(pkl_file) #load data from dictionary
    pkl_file.close()
    #We collect the probability of occupation p, the seed and the size of the system from the filename
    p_occ=filename.split('_')[7]      
    regex2 = re.compile('\d+\.\d+')
    p_reg=re.findall(regex2,p_occ)
    p=float(p_reg[0])
    seed_sys=filename.split('_')[9]      
    regex3 = re.compile('\d+')
    seed_sys_reg=re.findall(regex3,seed_sys)
    seed=int(seed_sys_reg[0])
    L_sys=filename.split('_')[8]      
    regex4 = re.compile('\d+')
    L_sys_reg=re.findall(regex4,L_sys)
    sys_size=int(L_sys_reg[0])    
    proba_largest=dictionary['proba largest']
    filename_cl='corlen_L'+str(sys_size)+'_p'+str(p)+'.cl' #name of the file containing the correlation lengths
   
    if filename_cl in os.listdir('.'): #check if the file exists in the directory and a correlation length associated to our current seed exists
        seeds=np.loadtxt(filename_cl,unpack=True)[0]
        if type(seeds)==np.ndarray and seed in seeds:
            print('the correlation length associated to this seed already exists')
            return
        elif type(seeds)==np.float64 and seed==seeds:
            print('the correlation length associated to this seed already exists')
            return
        else:
            for i in range(len(corr_data[0])):
                corr_value=corr_data[1][i]-proba_largest
                if corr_value>=0 and corr_value>=10**(-8):
                    correlation_large+= ((corr_data[0][i]**2)*(corr_data[1][i]-proba_largest))              
                    denom_large+=(corr_data[1][i]-proba_largest)
                else:
                    break
            sq_corr_l=(correlation_large/(6*denom_large))
            if sq_corr_l==0:
                correlation_length=0
            elif sq_corr_l<0:
                correlation_length=0
            else:
                correlation_length=sqrt(sq_corr_l)
            with open(filename_cl, "a+") as file_cl:
                file_cl.write("\n"+str(seed)+'    '+ str(p)+'    ' + str(correlation_length))
                file_cl.close()   
            return
    else:
        for i in range(len(corr_data[0])):
            
            corr_value=corr_data[1][i]-proba_largest
            if corr_value>=0 and corr_value>=10**(-8):
                correlation_large+= ((corr_data[0][i]**2)*(corr_data[1][i]-proba_largest))              
                denom_large+=(corr_data[1][i]-proba_largest)
            else:
                break
        sq_corr_l=(correlation_large/(6*denom_large))
    if sq_corr_l==0:
        correlation_length=0
    elif sq_corr_l<0:
        correlation_length=0
    else:
        correlation_length=sqrt(sq_corr_l)
    with open(filename_cl, "a+") as file_cl:
        file_cl.write(str(seed)+'    '+ str(p)+'    ' + str(correlation_length))
        file_cl.close()        
    end=time.time()-start
    print('corre_length_end',end)
    return correlation_length

##########################################################  
def average_correlation_l(filename_cl):
    sum_corr_l=0
    div=0
    #We collect the probability of occupation p and the size of the system from the filename
    filename=filename_cl.rsplit('.', 1)[0]
    p_occ=filename.split('_')[2]     
    regex2 = re.compile('\d+\.\d+')
    p_reg=re.findall(regex2,p_occ)
    p=float(p_reg[0])
    L_sys=filename.split('_')[1]     
    regex4 = re.compile('\d+')
    L_sys_reg=re.findall(regex4,L_sys)
    sys_size=int(L_sys_reg[0])
    
    corr_data= np.loadtxt(filename_cl,unpack=True) #load the corlen file
    seeds=corr_data[0]
    p_occs=corr_data[1]
    corr_lengths=corr_data[2]

    for corr_l in corr_lengths:
        if corr_l!=0:
            sum_corr_l+=corr_l
            div+=1
    average_correlation_length=sum_corr_l/div

    txt_file=open('avg_corrlen_L'+str(sys_size)+'_p'+str(p)+'.acl', "w+")
    txt_file.write(repr(average_correlation_length))
    txt_file.close()

    return                
#########################################################################################################################
option_select= int(sys.argv[1])
filename_pkl= str(sys.argv[2])
filename_corr= str(sys.argv[3])

directory=os.getcwd()
p=directory.rsplit('/', 1)[1]
L=directory.rsplit('/', 2)[1]

# 1 = 001 = cf
# 2 = 010 = cl, assumes that cf exists
# 4 = 100 = al, assumes cl exists
# 3 = 011 = cf, cl
# 6 = 110 = al,cl
# 7 = 111 = cf, cl, al

if option_select==1 :
    correlation_function_pbc(filename_pkl)
elif option_select==2 :
    correlation_l(filename_corr)
elif option_select==3 :
    correlation_function_pbc(filename_pkl)
    correlation_l(filename_corr)
elif option_select==4 :
    average_correlation_l('corlen_'+L+'_'+p+'.cl')
elif option_select==6 :
    correlation_l(filename_corr)
    average_correlation_l('corlen_'+L+'_'+p+'.cl')
else:
    correlation_function_pbc(filename_pkl)
    correlation_l(filename_corr)
    average_correlation_l('corlen_'+L+'_'+p+'.cl')

