#include <iostream>
#include <iomanip>
#include "include/particle.h"
#include "include/constants.h"
#include <fstream>
#include <sys/stat.h>
#include "include/utils.h"
#include <getopt.h>
#include "include/Analyzer.h"

using namespace std;

 Analyzer::Analyzer(const arma::vec& time, const arma::vec& data, const std::string& constants_file, const std::string& output_folder,
             unsigned int column, const std::string& tag, unsigned int niter, const std::string& trainedPriorFile,
             bool append, unsigned int trim, bool verbose, const std::string& gtSpike_file,
             bool has_trained_priors, bool has_gtspikes, unsigned int maxlen, const std::string& Gparam_file)
        : time(time), data(data), constants_file(constants_file), output_folder(output_folder), column(column), tag(tag),
          niter(niter), trainedPriorFile(trainedPriorFile), append(append), trim(trim), verbose(verbose),
          gtSpike_file(gtSpike_file), has_trained_priors(has_trained_priors), has_gtspikes(has_gtspikes),
          maxlen(maxlen), Gparam_file(Gparam_file) {
            cout<<"time[2]_in_constructor="<<time[2]<<endl;
          }



void Analyzer::run() {
    // Other init type stuff that was needed
    int existing_samples=0;
    cout<<"time(2)="<<time(2)<<endl;
	cout<<"time(1)="<<time(1)<<endl;
    cout<<"time(0)="<<time(0)<<endl;
    cout<<"time.n_elem="<<time.n_elem<<endl;
    // Original main function code here, replace argc/argv handling with member variables
    constpar constants(constants_file);
    constants.output_folder = output_folder;

    cout << "constants: " << constants_file << endl;
    cout << "output_folder: " << output_folder << endl;
    cout << "using column " << column << endl;
    cout << "using tsmode " << constants.TSMode << endl;

    ofstream parsamples;
    ofstream trajsamples;
    ofstream logp;
    istringstream last_params;

    struct stat sb;
    if (stat(output_folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            printf("Error creating directory!");
            exit(1);
        }
    }

    if (append) {
        ifstream ps(output_folder + "/param_samples_" + tag + ".dat");
        string line;        
        while (ps >> std::ws && std::getline(ps, line)) existing_samples++;

        if (existing_samples == 0) {
            cerr << "empty existing file!!" << endl;
            exit(1);
        }
        
        last_params.str(line); 
        ps.close();

        parsamples.open(output_folder + "/param_samples_" + tag + ".dat", std::ios_base::app);
        trajsamples.open(output_folder + "/traj_samples_" + tag + ".dat", std::ios_base::app);
        logp.open(output_folder + "/logp_" + tag + ".dat", std::ios_base::app);
    } else {
        logp.open(output_folder + "/logp_" + tag + ".dat");
    }

    param testpar(output_folder + "/param_samples_" + tag + ".dat");
    param testpar2(output_folder + "/param_samples_" + tag + ".dat");
		
    // loading the ground truth spikes
    arma::vec gtSpikes; 
    if (has_gtspikes) {
        constants.KNOWN_SPIKES = true;
        gtSpikes.load(gtSpike_file, arma::raw_ascii);
    }

    if (has_trained_priors) {
        string dum;
        cout << "Using trained priors:" << endl;
        // update the constants
        ifstream trainedPrior(trainedPriorFile);
        trainedPrior >> dum >> constants.G_tot_mean >> constants.G_tot_sd;
        trainedPrior >> dum >> constants.gamma_mean >> constants.gamma_sd;
        trainedPrior >> dum >> constants.DCaT_mean >> constants.DCaT_sd;
        trainedPrior >> dum >> constants.Rf_mean >> constants.Rf_sd;
        trainedPrior >> dum >> constants.alpha_sigma2 >> constants.beta_sigma2;
        trainedPrior >> dum >> constants.alpha_rate_b0 >> constants.beta_rate_b0;
        trainedPrior >> dum >> constants.alpha_rate_b1 >> constants.beta_rate_b1;
        trainedPrior >> dum >> constants.alpha_w01 >> constants.beta_w01;
        trainedPrior >> dum >> constants.alpha_w10 >> constants.beta_w10;
        trainedPrior.close();
    }

    if (niter > 0) {
        constants.niter = niter;
    }

    // Initialize the sampler (this will also reset the scales, that's why we need to initialize after we update the constants)
    // note the different constructors for SMC class here - one expects Analyzer to be called with a filename, the other takes data passed in directly
    cout<<"time(1)="<<time(1)<<endl;
    cout<<"time(0)="<<time(0)<<endl;
    SMC sampler(time, data, column, constants, false, 0, maxlen, Gparam_file);
    cout<<"line 112_Analyzer"<<endl;
    // Initialize the trajectory
    cout<<"sampler.TIME = "<<sampler.TIME<<endl;
    Trajectory traj_sam1(sampler.TIME, ""), traj_sam2(sampler.TIME, output_folder + "/traj_samples_" + tag + ".dat");
    cout<<"sampler.TIME = "<<sampler.TIME<<endl;
    for (unsigned int t = 0; t < sampler.TIME; ++t) {
        traj_sam1.B(t) = 0;
        traj_sam1.burst(t) = 0;
        traj_sam1.C(t) = 0;
        traj_sam1.S(t) = 0;
        if (has_gtspikes) traj_sam1.S(t) = gtSpikes(t);
        traj_sam1.Y(t) = 0;
        //cout<<t<<endl;
    }
    cout<<"line 124_Analyzer"<<endl;
    // set initial parameters 
    if (append) {
        cout << "Use parameters from previous analysis" << endl;
        vector<string> parse_params;
        while (last_params.good()) {
            string substr;
            getline(last_params, substr, ',');
            parse_params.push_back(substr);
        }

        testpar.G_tot = stod(parse_params[0]);
        testpar.gamma = stod(parse_params[1]);
        testpar.DCaT = stod(parse_params[2]);
        testpar.Rf = stod(parse_params[3]);
        testpar.sigma2 = stod(parse_params[4]);
        testpar.r0 = stod(parse_params[5]);
        testpar.r1 = stod(parse_params[6]);
        testpar.wbb[0] = stod(parse_params[7]);
        testpar.wbb[1] = stod(parse_params[8]);
    } else {
        cout << "Draw new parameters" << endl;
        testpar.r0 = constants.alpha_rate_b0 / constants.beta_rate_b0;
        testpar.r1 = constants.alpha_rate_b1 / constants.beta_rate_b1;
        testpar.wbb[0] = constants.alpha_w01 / constants.beta_w01;
        testpar.wbb[1] = constants.alpha_w10 / constants.beta_w10;
        testpar.sigma2 = constants.beta_sigma2 / (constants.alpha_sigma2 + 1); // the mode of the inverse gamma
        testpar.G_tot = constants.G_tot_mean;
        testpar.gamma = constants.gamma_mean;
        testpar.DCaT = constants.DCaT_mean;
        testpar.Rf = constants.Rf_mean;
        testpar.gam_in = constants.gam_in_mean;
        testpar.gam_out = constants.gam_out_mean;
    }
		cout<<"Constants params G_tot: "<<constants.G_tot_mean<<endl;
    cout << "start MCMC loop" << endl;
		

    for (unsigned int i = (existing_samples - 1) * trim + 1; i < constants.niter * trim; i++) {
        sampler.PGAS(testpar, traj_sam1, traj_sam2);

        traj_sam1 = traj_sam2;
        for (unsigned int k = 0; k < 10; k++) {
            if (constants.SAMPLE_PARAMETERS) {
                sampler.sampleParameters(testpar, testpar2, traj_sam2);
            } else {
                testpar2 = testpar;
            }

            testpar = testpar2;
        }
				

        if (i % trim == 0) {
            testpar.write(parsamples, constants.sampling_frequency);
            traj_sam2.write(trajsamples, i / trim);
            logp << testpar.logPrior(constants) + traj_sam2.logp(&testpar, &constants) << endl;
        }
        if (verbose) cout << "iteration:" << setw(5) << i << ", spikes: " << setw(5) << arma::accu(traj_sam1.S) << '\r' << flush;
    }

    parsamples.close();
    trajsamples.close();

    // Save last param set for direct comparison b/t compiled c++ and python binding version
    ofstream lastParams(output_folder + "/last_params_" + tag + ".dat");
    lastParams << testpar.r0 << " " << testpar.r1 << " " << testpar.wbb[0] << " " << testpar.wbb[1] << " " << testpar.sigma2 << " "
               << testpar.G_tot << " " << testpar.gamma << ' ' << testpar.DCaT << ' ' << testpar.Rf << endl;
							 
		// Populate final_params after calculations
    final_params.push_back(testpar.G_tot);
    final_params.push_back(testpar.gamma);
    final_params.push_back(testpar.DCaT);
    final_params.push_back(testpar.Rf);
    final_params.push_back(testpar.gam_in);
    final_params.push_back(testpar.gam_out);
}
