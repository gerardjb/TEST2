#Importing project packages and required libraries
import numpy as np
import pgas.pgas_bound as pgas
from syn_gen import synth_gen
import matplotlib.pyplot as plt

import os
print("Current directory: {}".format(os.getcwd()))
from cascade2p import checks, utils, cascade
print("\nChecks for packages:")
checks.check_packages()

# Run the particle Gibbs sampler to extract cell parameters
## Setting up parameters for the particle gibbs sampler
data_file="sample_data/LineScan-11302022-0954-009_0_data_poisson.dat"
constants_file="sample_data/constants_GCaMP3_soma.json"
output_folder="sample_data/output"
column=1
tag="default"
niter=100
gtSpike_file="sample_data/stimtimes_poisson_counts.dat"
maxlen=2000
Gparam_file="pgas/20230525_gold.dat"
verbose=1

analyzer = pgas.Analyzer(
    data_file=data_file,
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
print("Final params returned to python: ", final_params)
print(f"Final params dtype =  {final_params.dtype}")

# Create synthetic data
## Load parameters into the GCaMP model to use for synthetic data creation
Cparams = final_params
Gparams = np.loadtxt(Gparam_file)
gcamp = pgas.GCaMP(Gparams,final_params)

## Generate synthetic data from the PGAS-derived cell paramters 
synth = synth_gen.synth_gen(plot_on=False,GCaMP_model=gcamp,cell_params=Cparams)
synth.generate()

# Train a cascade model using the synthetic dataset
## First get the data and noise level we're training against
data = np.loadtxt(data_file).transpose()
time = data[0,:]
fluo_data = data[1,:]
frame_rate = 1/np.mean(np.diff(data[0,:]))
noise_level = utils.calculate_noise_levels(fluo_data,frame_rate)

## Set configurations file for sample training
synthetic_training_dataset = f"synth_{tag}"
cfg = dict( 
    model_name = tag,    # Model name (and name of the save folder)
    sampling_rate = 30,    # Sampling rate in Hz (round to next integer)
    
    training_datasets = [
            synthetic_training_dataset
                        ],
    
    noise_levels = [noise for noise in range(2,3)],#[int(np.ceil(noise_level)+1)],  # int values of noise values (do not use numpy here => representer error!)
    
    smoothing = 0.05,     # std of Gaussian smoothing in time (sec)
    causal_kernel = 0,   # causal ground truth smoothing kernel
    verbose = 1,
          )
## save parameter as config.yaml file - TODO: make cascade overwrite configs on this call
print(cfg['noise_levels'])
cascade.create_model_folder( cfg )

## Train a model based on config contents
model_name = cfg['model_name']
cascade.train_model( model_name )

## Use trained model to perform inference on the original dataset
'''
Actually, this block takes a really long time for poor models - skipping for now but could be a good place to revisit
from cascade2p.utils_discrete_spikes import infer_discrete_spikes
spike_prob = cascade.predict( model_name, dff )
discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob,model_name)
'''
## Saving routine
import scipy.io as sio
save_path = os.join.path("sample_data","sample_TEST_output.mat")
sio.savemat(save_path,{'spike_prob':spike_prob,'time':time,'dff':dff,'cfg':cfg})#removed ,'spike_time_estimates':spike_time_estimates (see above)
