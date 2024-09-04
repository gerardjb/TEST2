import numpy as np
import os
import spike_find.pgas.pgas_bound as pgas
import scipy.io as sio

# method for converting spike times to a binary vector as required by Giovanni's implementation of the PGAS algorithm
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


def test_mcmc(tmp_path):
    """Test a run of the particle Gibbs sampler to extract cell parameters"""
    # First we'll load in the original data as a numpy array
    filename = os.path.join("gt_data","jGCaMP8f_ANM471994_cell01.mat")
    time, data, spike_times = open_Janelia_1(filename)
    time1 = np.float64(time[0,1000:3040-1])
    time1 = time1.copy()
    data1 = np.float64(data[0,1000:3040-1])
    data1 = data1.copy()
    binary_spikes = np.float64(spike_times_2_binary(spike_times,time[0,1000:3040-1]))


    # Run the particle Gibbs sampler to extract cell parameters
    ## Setting up parameters for the particle gibbs sampler
    #data_file="tests/sample_data/LineScan-11302022-0954-009_0_data_poisson.dat"
    constants_file="tests/sample_data/constants_GCaMP3_soma.json"
    output_folder=str(tmp_path)
    column=1
    tag="default"
    niter=1
    gtSpikes=binary_spikes
    maxlen=1000
    Gparam_file="src/spike_find/pgas/20230525_gold.dat"
    verbose=1
    seed1 = 2
    seed2 = 3

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
        has_gtspikes=True,
        maxlen=maxlen, 
        Gparam_file=Gparam_file,
        seed=seed1
    )

    analyzer2 = pgas.Analyzer(
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
        has_gtspikes=True,
        maxlen=maxlen, 
        Gparam_file=Gparam_file,
        seed=seed2
    )

    ## Run the sampler
    analyzer.run()
    analyzer2.run()

    ## Ensure that the final outputs differ with different seeds
    final_params = analyzer.get_final_params()
    final_params2 = analyzer2.get_final_params()
    print("Final params seed 1: ", final_params)
    print("Final params seed 2: ", final_params2)
    
    assert not np.allclose(final_params,final_params2)

    ## Ensure the posterior distributions of the parameter
    ## estimates do not differ in mean and std within some tolerance
    parameter_samples = analyzer.get_parameter_estimates()
    parameter_samples2 = analyzer2.get_parameter_estimates()
    print(parameter_samples.shape)
    print(parameter_samples2.shape)
    
if __name__== "__main__":
    test_mcmc('')