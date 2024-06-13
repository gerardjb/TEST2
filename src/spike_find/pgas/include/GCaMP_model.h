#ifndef GCAMPMODEL_H
#define GCAMPMODEL_H
#include<armadillo>

using namespace std;

enum TimeStepMode {FIXED, FIXEDLA};

class GCaMP {
public:
    GCaMP(double G_tot,double gamma, double DCaT, double Rf, double gam_in, double gam_out, string Gparam_file="");
		GCaMP(string Gparam_file, string Cparam_file);
		GCaMP(double* Gparams_in, const double* Cparams_in);
		
		void initial_setup();
    void setParams(double G_tot,double gamma, double DCaT, double Rf, double gam_in, double gam_out);
    
    void evolve(double deltat, int ns);
    void fixedStep(double deltat, int ns);
    void fixedStep_LA(double deltat, int ns);
    double fixedStep_LA_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out);
    void evolve_test(double deltat, int ns);
    void evolve(double deltat, int ns, const arma::vec& state);
    void evolve_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out, double & dff_out);

    arma::vec steady_state(double c0);
    void init();
    double flux(double ca, const arma::vec& G);
    double flux_konN_konC(double konN, double konC, const arma::vec& G);
    void setGmat(double ca);
    void fillGmat(arma::mat::fixed<9,9> & Gmatrix, double ca);
    void setGmat_konN_konC(arma::mat::fixed<9,9> & Gmatrix, double konN, double konC);
    void setState(double ca);
    void setTimeStepMode(TimeStepMode);
    double getDFF();
    double getDFF(const arma::vec&);
    double getAmplitude();

    arma::vec::fixed<12> state;
    arma::vec::fixed<12> initial_state;
    double DFF;
    
    double* Gparams;
    size_t size_Gparams = 14;
		double* Cparams;
    size_t size_Cparams = 6;
    arma::mat::fixed<9,9> Gmat;
    arma::uvec::fixed<5> brightStates;

    int step_count=0;

    // Adaptive time step
    TimeStepMode TSMode=FIXED;
		
		// Methods for making and retrieving GCaMP simulations via python bindings
		void integrateOverTime(const arma::vec& time_vect, const arma::vec& spike_times);
		void integrateOverTime2(const arma::vec& time_vect, const arma::vec& spike_times);
		// Method to retrieve the stored DFF values
    const arma::vec& getDFFValues() const;

private:
    // parameters that are allowed to vary
    double G_tot;
    double gamma;
    double DCaT;
    double Rf;

    // parameters that can be fixed
    // indicator 
    double konN,   koffN;
		double konC,   koffC;
		double konPN,  koffPN;
		double konPN2, koffPN2;
		double konPC,  koffPC;
		double konPC2,  koffPC2;

    //calcium
    const double c0     = 5e-8;
    const double FWHM   = 2.8e-4;
    double sigma2_calcium_spike;
    double gam_in;
    double gam_out;
    
    // buffer
    const double koff_B = 1e4;
    const double kon_B  = 1e8;
    const double B_tot  = 0.004;
    double BCa0;
    double kapB;

    // fluorescence
    const double csat   = 1e-2;

    // dFF normalization
    double G0, Gsat, Ginit;
		
		// For python output
		arma::vec DFF_values; 

};

#endif
