import numpy as np
import os
import spike_find.pgas.pgas_bound as pgas
import scipy.io as sio

def open_Janelia_1(j_path):
    all_data = sio.loadmat(j_path)
    data = all_data['dff']
    time = all_data['time_stamps']

    return time, data

def test_mcmc(tmp_path):
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
    output_folder=str(tmp_path)
    column=1
    tag="default"
    niter=1
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
    print("Final params returned to python: ", final_params)
    print(f"Final params dtype =  {final_params.dtype}")

    np.testing.assert_allclose(final_params, 
                               [3.90363469e-05, 1.11873255e+03, 1.04594802e-05, 
                                3.63424808e+00, 6.55528532e-02, 1.36894643e-02])
    
if __name__== "__main__":
    test_mcmc('')