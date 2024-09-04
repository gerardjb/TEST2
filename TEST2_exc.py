#%%
#Importing project packages and required libraries
import numpy as np
import build.pgas_bound as pgas
from src.spike_find.syn_gen import synth_gen
import matplotlib.pyplot as plt
import scipy.io as sio

import os
print("Current directory: {}".format(os.getcwd()))
from src.spike_find.cascade2p import checks, utils, cascade
print("\nChecks for packages:")
checks.check_packages()

#Setting flags for what to calculate on this run
recalc_Cparams = True
recalc_synth = False
retrain_and_infer = False

#Utility-type methods
def spike_times_2_binary(spike_times,time_stamps):
    # initialize the binary vector
    binary_vector = np.zeros(len(time_stamps), dtype=int)

    # get event times within the time_stamps ends
    good_spike_times = spike_times[(spike_times >= time_stamps[0]) & (spike_times <= time_stamps[-1])]
    
    # Find the nearest element in 'a' that is less than the elements in 'b'
    for event_time in good_spike_times:
        # Find indices where 'a' is less than 'event_time'
        valid_indices = np.where(time_stamps < event_time)[0]
        if valid_indices.size > 0:
            nearest_index = valid_indices[-1]  # Taking the last valid index
            binary_vector[nearest_index] += 1

    return binary_vector

# For opening the janelia datasets
def open_Janelia_1(j_path):
    all_data = sio.loadmat(j_path)
    dff = all_data['dff']
    time_stamps = all_data['time_stamps']
    spike_times = all_data['ap_times'] 

    return time_stamps, dff, spike_times

def calculate_standardized_noise(dff,frame_rate):
    noise_levels = np.nanmedian(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100     # scale noise levels to percent

"""Test a run of the particle Gibbs sampler to extract cell parameters"""
# First we'll load in the original data as a numpy array
#janelia_file = "jGCaMP8f_ANM471993_cell03" #"jGCaMP8f_ANM478349_cell06" #
#filename = os.path.join("gt_data",janelia_file+".mat")
janelia_file = 'test'
filename = "/scratch/gpfs/gerardjb/TEST2/sample_data/output/sim4fig3/test_exc.mat"

time,data,spike_times = open_Janelia_1(filename)
time1 = np.float64(time[0,:]) #,2000:4080-1])
time1 = time1.copy()
data1 = np.float64(data[0,:]) #,2000:4080-1])
data1 = data1.copy()
#binary_spikes = np.float64(spike_times_2_binary(spike_times,time1))
binary_spikes = np.float64([0])

# Run the particle Gibbs sampler to extract cell parameters
## Setting up parameters for the particle gibbs sampler
#data_file="tests/sample_data/LineScan-11302022-0954-009_0_data_poisson.dat"
constants_file="tests/sample_data/constants_GCaMP3_soma.json"
output_folder="sample_data/output"
column=1
tag="default_janelia_sample_excitatory" #inhibitory" #excitatory"
model_choice = "cRNN_no_relu"
syn_tag = "cRNN_no_relu_janelia"
niter=500
gtSpikes = binary_spikes
maxlen=1000
Gparam_file="src/spike_find/pgas/20230525_gold.dat"
verbose=1
##Now we need to test if we have already calculated parameters for this file
final_params_fname = os.path.join("sample_data","output",janelia_file+".dat")

if not os.path.isfile(final_params_fname) and recalc_Cparams:
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
        gtSpikes=gtSpikes,
        has_gtspikes=False,
        maxlen=maxlen, 
        Gparam_file=Gparam_file,
        seed=2
    )

    ## Run the sampler
    analyzer.run()

    ## Return and print the output
    final_params = analyzer.get_final_params()
    parameter_samples = analyzer.get_parameter_estimates()
    ## Save these for use in the future for server disconnect
    np.save(final_params_fname,final_params)
else:
    final_params = np.load(final_params_fname+".npy")
    print("last sample")
    print(final_params)
#Opening the saved parameter samples for use as Cparams
param_sample_file = os.path.join("sample_data","output","param_samples_"+tag+".dat")
parameter_samples = np.loadtxt(param_sample_file,delimiter=',',skiprows=1)
burnin = 100
parameter_samples = parameter_samples[burnin:,0:6]
print("mean of samples")
print(np.mean(parameter_samples,axis=0))


# Only recompute the synthetic data if necessary
if recalc_synth:
    # Create synthetic data
    ## Load parameters into the GCaMP model to use for synthetic data creation
    Cparams = np.mean(parameter_samples,axis=0)
    Gparams = np.loadtxt(Gparam_file)
    gcamp = pgas.GCaMP(Gparams,final_params)

    ## Generate synthetic data from the PGAS-derived cell paramters 
    # Now making broader spike pulls
    nominal_rates = np.array([1,1.1,1.5,2,2.5,3,3.5,4,4.5,5])#
    for rate in nominal_rates:
        synth = synth_gen.synth_gen(plot_on=False,GCaMP_model=gcamp,\
            spike_rate=rate,cell_params=Cparams,tag=tag,use_noise=True)
        synth.generate()

if retrain_and_infer:
    ## Train a cascade model using the synthetic dataset
    # First get the data and noise level we're training against
    #data = np.loadtxt(data_file).transpose()
    time_stamps = time[0,:]
    fluo_data = data[0,:]
    frame_rate = 1/np.mean(np.diff(data[0,:]))
    noise_level = utils.calculate_noise_levels(fluo_data,frame_rate)

    # ## Set configurations file for sample training
    synthetic_training_dataset = os.path.join(f"synth_{tag}")
    cfg = dict( 
        model_name = tag,    # Model name (and name of the save folder)
        sampling_rate = 30,    # Sampling rate in Hz (round to next integer)
        
        training_datasets = [
                synthetic_training_dataset
                            ],
        
        noise_levels = [noise for noise in range(2,3)],#[int(np.ceil(noise_level)+1)]
        
        smoothing = 0.05,     # std of Gaussian smoothing in time (sec)
        causal_kernel = 0,   # causal ground truth smoothing kernel
        verbose = 1,
            )
    # ## save parameter as config.yaml file - TODO: make cascade overwrite configs on this call
    print(cfg['noise_levels'])
    # #cfg["loss_function"] = "binary_crossentropy"
    cascade.create_model_folder( cfg )

    ## Train a model based on config contents
    from spike_find.cascade2p import models
    model_name = cfg['model_name']
    cascade.train_model( model_name, model_type=models.choose_model(model_choice) )

    # ## Use trained model to perform inference on the original dataset
    from spike_find.cascade2p.utils_discrete_spikes import infer_discrete_spikes
    spike_prob = cascade.predict(model_name, np.reshape(fluo_data, (1, len(fluo_data))))
    '''
    For now turning this part off
    '''
    #spike_prob = cascade.predict(model_name, fluo_data) # for multidimensional_data
    #discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob,model_name)

    # Separate Python file that organizes model names 
    ## Saving routine
    save_dir = "sample_data/predictions"
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,f"{tag}_{model_choice}_output.mat")
    sio.savemat(save_path,{'spike_prob':spike_prob,'time_stamps':time_stamps,'dff':fluo_data,'cfg':cfg})#'spike_time_estimates':spike_time_estimates, <- this is for the discrete spike time estimates
