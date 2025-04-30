from DLVOlib.params import DLVO_parameters

datafile = 'data/127_B_3,12um_0mgg_27-03-25_moyenne_force_distance.txt'
params = DLVO_parameters(datafile,cutoff=0,force_err=0.5,rerun=False)

params.get_parameters()
params.compare_model()