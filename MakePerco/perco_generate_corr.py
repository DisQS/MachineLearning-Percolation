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


def correlation_function_pbc(filename_data):
    from math import sqrt
    filename, file_extension = os.path.splitext(filename_data)
    if filename+'_corr_func'+'.txt' in os.listdir('.'):
        return
    else:
        data=pickle.load(open(filename_data,"rb"))

        lattice=data['cluster_pbc_int'] #cluster_pbc_int
        square_proba=data['square proba']
        proba_largest=data['proba largest']
        n_clusters=data['n_clusters_pbc']






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

        n_clus_sys=filename.split('_')[11]      
        regex4 = re.compile('\d+')
        n_clus_sys_reg=re.findall(regex4,n_clus_sys)
        n_clusters=int(n_clus_sys_reg[0])

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

        average=np.divide(corr2,div_pbc, out=np.zeros_like(corr2), where=div_pbc!=0)
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

        corr_value_zipped=list(zip(new_distance,correlation_value, corr_proba_largest,new_distance_largest,correlation_value_largest))
        header = '{0:^5s}   {1:^7s}   {2:^7s} {3:^5s}   {4:^7s} '.format('distance', 'Corr func','Corr func-proba largest','distance','Corr func largest')
        np.savetxt(filename+'_corr_func'+'.txt',corr_value_zipped, header=header, fmt=['    %.7f  ','    %.7f  ','    %.7f  ','  %.7f','  %.7f'])





        end6=time.time()-start50
        print('end correlation function',end6)

    
    return new_distance,correlation_value, new_distance_largest,correlation_value_largest, new_corr_before_average
    
    
    
#########################################################################################################################
#if ( len(sys.argv) == 1 ):
#directory=str(sys.argv[1])
filename= str(sys.argv[1])

 
correlation_function_pbc(filename)


#else:
#    print ('Number of', len(sys.argv), \
#           'arguments is less than expected (1) --- ABORTING!')
#    print ('Usage: python '+sys.argv[0],\
 #          ' filename')
    #print ('Argument List:', str(sys.argv)) 
