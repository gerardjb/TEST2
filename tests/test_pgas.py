import numpy as np
import spike_find.pgas.pgas_bound as pgas


def test_mcmc(tmp_path):
    """Test a run of the particle Gibbs sampler to extract cell parameters"""

    # Run the particle Gibbs sampler to extract cell parameters
    ## Setting up parameters for the particle gibbs sampler
    data_file="tests/sample_data/LineScan-11302022-0954-009_0_data_poisson.dat"
    constants_file="tests/sample_data/constants_GCaMP3_soma.json"
    output_folder=str(tmp_path)
    column=1
    tag="default"
    niter=5
    gtSpike_file="tests/sample_data/stimtimes_poisson_counts.dat"
    maxlen=2000
    Gparam_file="src/spike_find/pgas/20230525_gold.dat"
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

    np.testing.assert_allclose(final_params, 
                               [3.90363469e-05, 1.11873255e+03, 1.04594802e-05, 
                                3.63424808e+00, 6.55528532e-02, 1.36894643e-02])