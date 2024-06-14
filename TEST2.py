#%%
#Importing project packages and required libraries
import numpy as np
import spike_find.pgas.pgas_bound as pgas
from spike_find.syn_gen import synth_gen
import matplotlib.pyplot as plt
import scipy.io as sio

import os
print("Current directory: {}".format(os.getcwd()))
from spike_find.cascade2p import checks, utils, cascade
print("\nChecks for packages:")
checks.check_packages()

def open_Janelia_1(j_path):
    all_data = sio.loadmat(j_path)
    data = all_data['dff']
    time = all_data['time_stamps']

    return time, data


"""Test a run of the particle Gibbs sampler to extract cell parameters"""
# First we'll load in the original data as a numpy array
filename = os.path.join("gt_data","jGCaMP8f_ANM471994_cell01.mat")
time,data = open_Janelia_1(filename)
time1 = np.float64(time[0,1000:3040-1])
time1 = time1.copy()
data1 = np.float64(data[0,1000:3040-1])
data1 = data1.copy()

# Run the particle Gibbs sampler to extract cell parameters
## Setting up parameters for the particle gibbs sampler
#data_file="tests/sample_data/LineScan-11302022-0954-009_0_data_poisson.dat"
constants_file="tests/sample_data/constants_GCaMP3_soma.json"
output_folder="sample_data/output"
column=1
tag="cRNN_no_relu_janelia"
model_choice = "cRNN"
niter=100
gtSpike_file="tests/sample_data/stimtimes_poisson_counts.dat"
maxlen=1000
Gparam_file="src/spike_find/pgas/20230525_gold.dat"
verbose=1

analyzer = pgas.Analyzer(
    time=time1,
    data=data1,
    constants_file=constants_file,
    output_folder=output_folder,
    column=column,
    tag=tag,
    niter=niter,
    append=False,
    verbose=verbose,
    gtSpike_file=gtSpike_file,
    has_gtspikes=True,
    maxlen=maxlen, 
    Gparam_file=Gparam_file
)

## Run the sampler
analyzer.run()

## Return and print the output
final_params = analyzer.get_final_params()
print(final_params)

# Create synthetic data
## Load parameters into the GCaMP model to use for synthetic data creation
Cparams = final_params
Gparams = np.loadtxt(Gparam_file)
gcamp = pgas.GCaMP(Gparams,final_params)

## Generate synthetic data from the PGAS-derived cell paramters 
synth = synth_gen.synth_gen(plot_on=False,GCaMP_model=gcamp,cell_params=Cparams,tag=tag)
synth.generate()

# Train a cascade model using the synthetic dataset
## First get the data and noise level we're training against
# data = np.loadtxt(data_file).transpose()
# time = data[0,:]
# fluo_data = data[1,:]
# frame_rate = 1/np.mean(np.diff(data[0,:]))
# noise_level = utils.calculate_noise_levels(fluo_data,frame_rate)

# ## Set configurations file for sample training
# syn_tag = "default_trunc"
# synthetic_training_dataset = f"synth_{syn_tag}"
# cfg = dict( 
#     model_name = tag,    # Model name (and name of the save folder)
#     sampling_rate = 30,    # Sampling rate in Hz (round to next integer)
    
#     training_datasets = [
#             synthetic_training_dataset
#                         ],
    
#     noise_levels = [noise for noise in range(2,3)],#[int(np.ceil(noise_level)+1)],  # int values of noise values (do not use numpy here => representer error!)
    
#     smoothing = 0.05,     # std of Gaussian smoothing in time (sec)
#     causal_kernel = 0,   # causal ground truth smoothing kernel
#     verbose = 1,
#           )
# ## save parameter as config.yaml file - TODO: make cascade overwrite configs on this call
# print(cfg['noise_levels'])
# #cfg["loss_function"] = "binary_crossentropy"
# cascade.create_model_folder( cfg )

# ## Train a model based on config contents
# from spike_find.cascade2p import models
# model_name = cfg['model_name']
# cascade.train_model( model_name, model_type=models.choose_model(model_choice) )

# ## Use trained model to perform inference on the original dataset
# from spike_find.cascade2p.utils_discrete_spikes import infer_discrete_spikes
# spike_prob = cascade.predict(model_name, np.reshape(fluo_data, (1, len(fluo_data))))
# discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob,model_name)

# # Separate Python file that organizes model names 
# ## Saving routine
# save_path = os.path.join("sample_data/output",f"{tag}_output.mat")
# sio.savemat(save_path,{'spike_prob':spike_prob,'spike_time_estimates':spike_time_estimates,'time':time,'dff':fluo_data,'cfg':cfg})


# Pearson's correlation with spike probs and convolved ground truth spike times