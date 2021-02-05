import matplotlib.pyplot as plt
import os
import re
import pickle

def plot_im_lattice(filename_pkl):
    
    data=pickle.load(open(filename_pkl,"rb"))
    cluster_pbc_norm=data['cluster_pbc_norm']
    filename, file_extension = os.path.splitext(filename_pkl)
    fig=plt.figure()
    plt.axis('off')
    plt.imshow(cluster_pbc_norm,cmap='Greys')
    plt.imsave(filename+'_im_lattice.png', cluster_pbc_norm,cmap='Greys')
    plt.close('all')      
    return
    
    
    
    
def plot_corr_funcs(filename_corr, filename_pkl): 
    corr_data  = np.loadtxt(filename_corr, unpack=True)
    data=pickle.load(open(filename_pkl,"rb"))
    square_proba=data['square proba']
    proba_largest=data['proba largest']

    filename, file_extension = os.path.splitext(filename_corr)
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
    plt.savefig(filename+'.png')
    plt.close('all')
    
    plt.plot(corr_data[0],corr_data[1],label='g(r) every clusters')
    plt.plot(corr_data[3],corr_data[4], label='largest cluster')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.xlabel('distance r')
    plt.ylabel('correlation function g(r)')
    plt.title('Correlation function (log scale) for system '+str(size_sys)+' at density '+str(p))
    plt.savefig(filename+'log'+'.png')
    plt.close('all')
    return
#############################################################################################
filename_pkl= str(sys.argv[1])
filename_corr= str(sys.argv[2])




plot_im_lattice(filename_pkl)

plot_corr_funcs(filename_corr, filename_pkl)
