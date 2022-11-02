import matplotlib.pyplot as plt
import os
import re
import pickle
import sys 
import numpy as np
from collections import Counter, OrderedDict
from PIL import Image

def plot_im_lattice(filename_pkl):
    
    data=pickle.load(open(filename_pkl,"rb"))
    cluster_norm=data['cluster_norm']
    cluster_int=data['cluster_int']
    n_clusters=data['n_clusters_pbc']
    
    filename, file_extension = os.path.splitext(filename_pkl)

    L_size=filename.split('_')[8]   
    #print(L_size)   
    regex1 = re.compile('\d+')
    size_sys_reg=re.findall(regex1,L_size)
    size=int(size_sys_reg[0])

    print('n_cluster',n_clusters)
    print('###############################################################')
    if (filename+'_n.pdf' and filename+'_s.pdf' and filename+'_b.pdf' and filename+'_a.pdf' and filename+'_bw.pdf')  in os.listdir('.'):
        return


    if filename+'_bw.pdf' not in os.listdir('.'):
        cluster_bw=np.array([1 if x!=0 else x for x in cluster_int.flat]).reshape(size,size)
        cluster_bw=np.where(cluster_bw==0,2,cluster_bw)
        cluster_bw=np.where(cluster_bw==1,0,cluster_bw)
        cluster_bw=np.where(cluster_bw==2,1,cluster_bw)
        
        data = (cluster_bw * 255).astype(np.uint8)
        Image.fromarray(data).save(filename+'_bw.pdf')

    if filename+'_n.pdf' not in os.listdir('.'):

        fig=plt.figure()
        plt.axis('off')
        plt.imshow(cluster_norm,cmap='Greys')
        plt.imsave(filename+'_n.pdf', cluster_norm,cmap='Greys')
        plt.close('all')
        
    if filename+'_s.pdf' not in os.listdir('.'): 
        # reshuffle greys random 
        new_mapping = np.arange(1,n_clusters+1)
        np.random.shuffle(new_mapping)
        reshuffle= np.array([new_mapping[v-1]/n_clusters \
                            if not v == 0 else 0 for v in cluster_int.flat]).reshape(size,size)
        fig=plt.figure()
        plt.axis('off')
        plt.imshow(reshuffle,cmap='Greys')
        plt.imsave(filename+'_s.pdf',reshuffle,cmap='Greys')
        plt.close('all')

    if filename+'_b.pdf' not in os.listdir('.'):
        # reshuffle greys random with largest cluster BLACK
        new_mapping_largest = np.arange(1,n_clusters)
        #print('OLD UNcomplete',new_mapping_largest, 'n_clusters=',n_clusters)
        np.random.shuffle(new_mapping_largest)
        #print('new UNcomplete',new_mapping_largest)
        new_mapping_largest=np.append(new_mapping_largest,[n_clusters])
        #print('new complete',new_mapping_largest)
        reshuffle_largest= np.array([new_mapping_largest[v-1]/n_clusters \
                                     if not (v == 0)  else 0 for v in cluster_int.flat]).reshape(size,size)
        fig=plt.figure()
        plt.axis('off')
        plt.imshow(reshuffle_largest,cmap='Greys')
        plt.imsave(filename+'_b.pdf', reshuffle_largest,cmap='Greys')
        plt.close('all') 
    

    #if filename+'_a.pdf' not in os.listdir('.'):
    #    # reshuffle greys according to cluster size with largest cluster BLACK
    #    ratio={}
    #    occ_num=[]
    #    inc=0
    #    num_clus, number_sites = np.unique(cluster_int, return_counts=True)
    #    original_map=dict(zip(num_clus, number_sites))
    #    counter=Counter(original_map.values())
        
    #    for key,value in counter.items():
    #        occ_num.append((key,value))  
    #    for num in range(len(occ_num)):
    #        if occ_num[num][1]>1:
    #            ratio.update({occ_num[num][0]:1/occ_num[num][1]})
    #    several=list(ratio.keys())
    #    mapping_values=list(original_map.values())
    #    new_mapping_values=np.zeros(len(mapping_values))
    #    for value in range(len(mapping_values)):
    #        if mapping_values[value] in several and mapping_values[value]==mapping_values[value-1]:
    #            p=ratio[mapping_values[value]]
    #            new_mapping_values[value]=mapping_values[value]+p*inc
    #            inc+=1
    #        elif mapping_values[value] in several and mapping_values[value]!=mapping_values[value-1]:
    #            inc=0
    #            p=ratio[mapping_values[value]]
    #            new_mapping_values[value]=mapping_values[value]+p*inc
    #            inc+=1
    #        else:
    #            new_mapping_values[value]=mapping_values[value]
    #            a_mapping= np.array([new_mapping_values[v]/n_clusters \
    #                        if not v == 0 else 0 for v in cluster_int.flat]).reshape(size,size)
    #            print('new_mapping',new_mapping_values,'value',value,new_mapping_values)
    #    fig=plt.figure()
    #    plt.axis('off')
    #    plt.imshow(a_mapping,cmap='Greys')
    #    plt.imsave(filename+'_a.pdf', a_mapping,cmap='Greys')
    #    plt.close('all')




    
    
    return
    
def plot_corr_funcs(filename_corr, filename_pkl): 
    corr_data  = np.loadtxt(filename_corr, unpack=True)
    data=pickle.load(open(filename_pkl,"rb"))
    square_proba=data['square proba']
    proba_largest=data['proba largest']

    filename, file_extension = os.path.splitext(filename_corr)
    if filename+'_im_lattice.pdf' in os.listdir('.'):
        return
    else:
        L_size=filename.split('_')[8]      
        regex1 = re.compile('\d+')
        size_sys_reg=re.findall(regex1,L_size)
        size_sys=int(size_sys_reg[0])
    
        p_occ=filename.split('_')[7]      
        regex2 = re.compile('\d+\.\d+')
        p_reg=re.findall(regex2,p_occ)
        p=float(p_reg[0])
    
        plt.plot(corr_data[0],corr_data[1],label='g(r) every clusters')
        plt.plot(corr_data[3],corr_data[4], label='largest cluster')
        plt.axhline(y=square_proba, color='g', linestyle='--')
        plt.axhline(y=proba_largest, color='grey', linestyle='--')
        plt.legend(loc='best')
        plt.xlabel('distance r')
        plt.ylabel('correlation function g(r)')
        plt.title('Correlation function for system '+str(size_sys)+' at density '+str(p))
        plt.savefig(filename+'.pdf')
        plt.close('all')
    
        plt.plot(corr_data[0],corr_data[1],label='g(r) every clusters')
        plt.plot(corr_data[3],corr_data[4], label='largest cluster')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel('distance r')
        plt.ylabel('correlation function g(r)')
        plt.title('Correlation function (log scale) for system '+str(size_sys)+' at density '+str(p))
        plt.savefig(filename+'log'+'.pdf')
        plt.close('all')
        return
#############################################################
option_select= int(sys.argv[1])
filename_pkl= str(sys.argv[2])
filename_corr= str(sys.argv[3])

if option_select==0 :
    plot_im_lattice(filename_pkl)
elif option_select==1 :
    plot_corr_funcs(filename_corr, filename_pkl)
else:
    plot_im_lattice(filename_pkl)
    plot_corr_funcs(filename_corr, filename_pkl)
