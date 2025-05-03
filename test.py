from DLVOlib.params import DLVO_parameters
from DLVOlib.cornerplot import cornerplot
import matplotlib.pyplot as plt

datafile = 'data/127_B_3,12um_0mgg_27-03-25_moyenne_force_distance.txt'
params = DLVO_parameters(datafile,cutoff=0,force_err=0.5,rerun=False)
params.get_parameters()

cornerplot(params.chains_alpha,params.chains_beta,params.chains_k).plot(f'$\\alpha$',f'$\\beta$',f'$k$')
    

