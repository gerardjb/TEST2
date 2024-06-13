#include"include/GCaMP_model.h"
#include"include/utils.h"

using namespace std;

// Constructor used by the PGAS particles
GCaMP::GCaMP(double p1, double p2, double p3, double p4, double p5, double p6, string Gparam_file) {
  // Set passed-in parameters
  G_tot = p1;
  gamma = p2;
  DCaT = p3;
  Rf = p4;
  gam_in = p5;
  gam_out = p6;

  // Load Gparams from file
  if (Gparam_file.empty()) {
    throw std::invalid_argument("GParam file cannot be empty");
  }
  Gparams = utils::read_vector_from_file(Gparam_file, size_Gparams);// with custom file reader to replace arma::raw_ascii load method

  // Call initial_setup to complete object initialization
  initial_setup();
}

// Constructor for passing files
GCaMP::GCaMP(string Gparam_file, string Cparam_file) {
	cout<<"Got to the string, string GCaMP constructor method"<<endl;
	
  // Load Gparams from file or default values
  if (Gparam_file.empty()) {
      static const double default_Gparams[] = {
        4.0320349221094061E-7,
        74.94798544278855,
        1.7376760772790556,
        3.62856271417812E-8,
        0.6056939075038793,
        1.1024894240446206,
        30.699056808075316,
        311.52950002130206,
        0.15520110541159166,
        43.466019961996466,
        47.325861294367463,
        368.84356191597425,
        0.71415314204711966,
        31.860437304985595
      };
      copy(std::begin(default_Gparams), std::end(default_Gparams), Gparams);

  } else {
		cout<<"Gparam_file = "<<Gparam_file<<endl;
    Gparams = utils::read_vector_from_file(Gparam_file, size_Gparams);
  }
	
	//Loading from cparams file (assumes order G_tot, gamma, DCaT, Rf, gam_in, gom_out
  if (Cparam_file.empty()) {
    // Use default values (as in the original constructor)
    G_tot = 1e-5;
    gamma = 1000;
    DCaT = 1e-5;
		Rf = 5;
		gam_in = 40;
		gam_out = 4;
  } else {
		Cparams = utils::read_vector_from_file(Cparam_file, size_Cparams);
		G_tot = Cparams[0];
    gamma = Cparams[1];
    DCaT = Cparams[2];
		Rf = Cparams[3];
		gam_in = Cparams[4];
		gam_out = Cparams[5];
	}
	
	cout<<"G_tot = "<<G_tot<<"; gam_out = "<<gam_out<<endl;
	cout<<"G1 = "<<Gparams[0]<<"; G6 = "<<Gparams[5]<<endl;
	
  // Call initial_setup to complete object initialization
  initial_setup();
}

// Passing in the parameters as numeric types
GCaMP::GCaMP(double* Gparams_in, const double* Cparams_in) {
  // Set member variables from passed-in arguments
	G_tot = Cparams_in[0];
  gamma = Cparams_in[1];
  DCaT = Cparams_in[2];
  Rf = Cparams_in[3];
  gam_in = Cparams_in[4];
  gam_out = Cparams_in[5];
  Gparams = Gparams_in;  // Assuming Gparams is already a vector of appropriate size
	
	cout<<"G_tot = "<<G_tot<<"; gam_out = "<<gam_out<<endl;
	cout<<"G1 = "<<Gparams[0]<<"; G6 = "<<Gparams[5]<<endl;
  // Call initial_setup to complete object initialization
  initial_setup();
}


void GCaMP::initial_setup() {
  brightStates = {2, 3, 5, 6, 8};

  Gparams[0] = Gparams[1] / pow(Gparams[0], Gparams[2]);
  Gparams[3] = Gparams[4] / pow(Gparams[3], Gparams[5]);
	
	
  kapB = B_tot / (koff_B / kon_B);

  BCa0 = kon_B * c0 / (kon_B * c0 + koff_B) * B_tot;

  konN = Gparams[0] * pow(c0, Gparams[2]); koffN = Gparams[1];
  konC = Gparams[3] * pow(c0, Gparams[5]); koffC = Gparams[4];

  konPN = Gparams[6]; koffPN = Gparams[7];
  konPC = Gparams[8]; koffPC = Gparams[9];

  konPN2 = Gparams[10]; koffPN2 = Gparams[11];
  konPC2 = Gparams[12]; koffPC2 = Gparams[13];

  sigma2_calcium_spike = pow(FWHM, 2) / (8 * log(2));
	
	cout<<"kon_N = "<<konN<<"; koffPN2 = "<<koffPN2<<endl;

  setState(c0);

  double* G0_v = steady_state(0);
  double* Gsat_v = steady_state(csat);
  double* Ginit_v = steady_state(c0);

  G0 = utils::accum(G0_v,brightStates);
  Gsat = utils::accum(Gsat_v,brightStates);
  Ginit = utils::accum(Ginit_v,brightStates);

  DFF = 0;
}


void GCaMP::setParams(double p1, double p2, double p3, double p4, double p5, double p6){
    G_tot = p1;
    gamma = p2;
    DCaT  = p3;
    Rf    = p4;
    gam_in = p5;
    gam_out = p6;

    double* G0_v    = steady_state(0);
    double* Gsat_v  = steady_state(csat);
    double* Ginit_v = steady_state(c0);

    G0     =  utils::accum(G0_v,brightStates);
    Gsat   =  utils::accum(Gsat_v,brightStates);
    Ginit  =  utils::accum(Ginit_v,brightStates);
   
}


void GCaMP::setGmat_konN_konC(arma::mat::fixed<9,9> & Gmatrix, double konN, double konC){

  Gmatrix(0, 0) = -(konN+konC);
  Gmatrix(1, 0) = konN;
  Gmatrix(1, 1) = -(koffN+konPN+konC);
  Gmatrix(2, 2) = -(konC+koffPN);
  Gmatrix(3, 2) = konC;
  Gmatrix(4, 0) = konC;
  Gmatrix(4, 4) = -(koffC+konPC+konN);
  Gmatrix(5, 5) = -(koffPC+konN);
  Gmatrix(6, 5) = konN;
  Gmatrix(7, 1) = konC;
  Gmatrix(7, 4) = konN;

}

void GCaMP::setGmat(double ca){

  if (Gmat == nullptr) {
      Gmat = new double[Gmat_size];
  }

	konN   = Gparams[0]*pow(ca,Gparams[2]);
  konC   = Gparams[3]*pow(ca,Gparams[5]);

// note: because only 2 parameters need to be updated when changing calcium level, perhaps here we can optimize something

//	Gmat = {{-(konN+konC),koffN,koffPN,0,koffC,koffX2,0,0,0},
//            {konN,-(koffN+konPN+konC),0,0,0,0,koffPC,koffC,0},
//            {0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC},
//            {0,0,konC,-(koffC+konX+koffPN),0,0,0,konPN,0},
//            {konC,0,0,koffPN,-(koffC+konPC+konN),koffPC,0,koffN,0},
//            {0,0,0,0,konPC,-(koffPC+konN+koffX2),koffN,0,koffPN},
//            {0,0,0,0,0,konN,-(koffN+konPN+koffPC),konX,0},
//            {0,konC,0,0,konN,0,0,-(koffC+koffN+konX+konPN),0},
//            {0,0,0,konX,0,0,konPN,0,-(koffPN+koffPC)}};

	std::vector<double> G_temp = {-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0,
		    konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0,
			0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2,
			0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0,
			konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0,
			0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2,
			0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0,
			0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0,
			0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)};

  std:copy(G_temp.begin(),G_temp.end(),Gmat);

}

void GCaMP::fillGmat(double & Gmatrix, double ca){

	konN   = Gparams[0]*pow(ca,Gparams[2]);
  konC   = Gparams[3]*pow(ca,Gparams[5]);

	std::vector<double> G_temp = {-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0,
		    konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0,
			0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2,
			0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0,
			konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0,
			0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2,
			0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0,
			0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0,
			0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)};

  std:copy(G_temp.begin(),G_temp.end(),Gmatrix);

}

double GCaMP::flux(double ca, const arma::vec& G){

	konN   = Gparams[0]*pow(ca,Gparams[2]);
  konC   = Gparams[3]*pow(ca,Gparams[5]);

  // Calculate fluxes per binding site
  double N = -(konN*(G(0) + G(4) + G(5))) +
                koffN*(G(1) + G(6)+ G(7))  + 
                koffPN*G(2)                + 
                koffPN2*(G(3) + G(8));

  double C = -(konC *(G(0) + G(1) + G(2))) +
                koffC*(G(4) + G(3) + G(7))  + 
                koffPC*(G(5))+ 
                koffPC2*(G(6)+G(8));

  int Csites=2;
  return 2*N + Csites*C;
}

double GCaMP::flux_konN_konC(double konN, double konC, const arma::vec& G){

  // Calculate fluxes per binding site
  double N = -(konN*(G(0) + G(4) + G(5))) +
                koffN*(G(1) + G(6)+ G(7))  + 
                koffPN*G(2)                + 
                koffPN2*(G(3) + G(8));

  double C = -(konC *(G(0) + G(1) + G(2))) +
                koffC*(G(4) + G(3) + G(7))  + 
                koffPC*(G(5))+ 
                koffPC2*(G(6)+G(8));

  int Csites=2;
  return 2*N + Csites*C;
}



double* GCaMP::steady_state(double c0){
	setGmat(c0);

  arma::mat Gmat_arma = utils::double_to_arma_mat(Gmat, 9, 9) ;

	arma::mat u = arma::null(Gmat_arma);
	arma::vec ss_arma = u.col(0);
	ss_arma = ss_arma/arma::accu(ss_arma)*G_tot;
    arma::vec b_c = {BCa0,c0,c0};
    ss_arma = arma::join_vert(ss_arma, b_c);
  
  double* ss = utils::arma_mat_to_double(ss_arma, 9, 9);

  return ss;
/* //Here I was attempting to replace the null space and normalization functions with dependecy free ones. I borrowed a utils method for that,
  //but for now I'm dropping that idea.
  // Find null space of Gmat (example implementation)
  std::vector<double> u = utils::null_space(& Gmat, Gmat_size,Gmat_size);

  double* ss = new double[9];
  for (size_t i = 0; i < 9; ++i) {
    ss[i] = u[i * 9];
  }

  // Normalize ss
  double sum_ss = std::accumulate(ss, ss + 9, 0.0);
  for (size_t i = 0; i < 9; ++i) {
    ss[i] = ss[i] / sum_ss * G_tot;
  }

  // Append BCa0, c0, c0 to ss
  double* result = new double[12];
  std::memcpy(result, ss, 9 * sizeof(double));
  result[9] = BCa0;
  result[10] = c0;
  result[11] = c0;

  delete[] ss;
  return result;*/
}

void GCaMP::init(){
    setState(c0);
    DFF=0;
}

void GCaMP::setState(double ca){
    double* is = steady_state(ca);
    state = is;
    initial_state=is;
}

void GCaMP::setTimeStepMode(TimeStepMode tsm){
    TSMode=tsm;
}

void GCaMP::evolve(double deltat, int ns, const arma::vec& s){
    state = s;
    evolve(deltat,ns);
}

void GCaMP::evolve(double deltat, int ns){
    switch(TSMode){
        case FIXED:
            fixedStep(deltat,ns);
            break;
        case FIXEDLA:
            fixedStep_LA(deltat,ns);
            break;
    }
}

void GCaMP::evolve_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out, double & dff_out){
    switch(TSMode){
        case FIXED:
            fixedStep(deltat,ns);
            break;
        case FIXEDLA:
            dff_out = fixedStep_LA_threadsafe(deltat,ns,state_in,state_out);
            break;
    }
}


void GCaMP::fixedStep(double deltat, int ns){

    double* G = new state(arma::span(0,8));

    double BCa = state(9);
    double dBCa_dt;

    double Ca    = state(10);
    double Ca_in = state(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux;

    double fine_timestep=3e-6;
    double dt;

    arma::vec timesteps = arma::regspace(0,fine_timestep,deltat);
    arma::vec calcium_input = DCaT/sqrt(2*arma::datum::pi*sigma2_calcium_spike)*exp(-0.5/sigma2_calcium_spike*pow(deltat/2-timesteps,2));
    
    for(unsigned int i=1;i<timesteps.n_elem;++i){
        setGmat(Ca);
        double* dG_dt = new double[];
        utils::unrolled_matrix_vector_multiplication(Gmat, 9, 9, G, dG_dt);// Gmat*G;
        Gflux = flux(Ca,G);

        dBCa_dt = kon_B*(B_tot-BCa)*Ca - koff_B*BCa;
        dCa_dt  = -gamma*(Ca-c0)  
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0)  //intra-compartmental exchange
            + Gflux - dBCa_dt + ns*calcium_input(i);
        dCa_in_dt = gam_in*(Ca - c0) - gam_out*(Ca_in - c0);

        dt  = timesteps(i)-timesteps(i-1);
        
        G   = G   +  dt*dG_dt;
        BCa = BCa +  dt*dBCa_dt;
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in+= dt*dCa_in_dt;


        step_count+=1;
    }

    for(unsigned int i=0;i<9;i++) state(i) = G(i);
    state(9)  = BCa;
    state(10) = Ca;

    DFF = (arma::accu(G(brightStates)) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));                                
}

void GCaMP::fixedStep_LA(double deltat, int ns){

    arma::vec G = state(arma::span(0,8));

    double BCa = state(9);
    double dBCa_dt;

    double Ca  = state(10);
    double Ca_in = state(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux, Cflux;
    double finedt=100e-6;
    double dt;

    arma::vec timesteps = arma::regspace(0,finedt,deltat);
    double calcium_input;
    
    for(unsigned int i=1;i<timesteps.n_elem;++i){

        calcium_input = (i==1) ? ns*DCaT/finedt : 0;

        double logCa = log(Ca);
	      double konN = Gparams[0]*exp(Gparams[2]*logCa);
        double konC = Gparams[3]*exp(Gparams[5]*logCa);

        setGmat_konN_konC(Gmat, konN, konC);
        arma::vec dG_dt  = Gmat*G;
        Gflux = flux_konN_konC(konN, konC, G);

        Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(kapB + 1);

        //cout<<timesteps(i)<<endl;
        //cout<<"BCa = "<<BCa<<"; d_dt:"<<dBCa_dt<<", "<<calcium_input*kapB/(kapB+1)<<endl;
        //cout<<"Ca  = "<<Ca<<" ; d_dt:"<<dCa_dt<<", "<<calcium_input*1/(kapB+1)<<endl;
        //cout<<"------------------------"<<endl;

        dt  = timesteps(i)-timesteps(i-1); 
        G   = G   +  dt*dG_dt;
        BCa = BCa +  dt*(dBCa_dt+kapB/(kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
    }

    for(unsigned int i=0;i<9;i++) state(i) = G(i);
    state(9)  = BCa;
    state(10) = Ca;
    state(11) = Ca_in;

    DFF = (arma::accu(G(brightStates)) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));                                
}

double GCaMP::fixedStep_LA_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out){

    arma::vec G = state_in(arma::span(0,8));
    arma::mat::fixed<9,9> Gmatrix;
    
    double BCa = state_in(9);
    double dBCa_dt;

    double Ca  = state_in(10);
    double Ca_in = state_in(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux, Cflux;
    double finedt=100e-6;
    double dt;

    arma::vec timesteps = arma::regspace(0,finedt,deltat);
    double calcium_input;

    // Intiliaze the Gmatrix
    fillGmat(Gmatrix, Ca);

    for(unsigned int i=1;i<timesteps.n_elem;++i){

        calcium_input = (i==1) ? ns*DCaT/finedt : 0;

        double logCa = log(Ca);
	      double konN = Gparams[0]*exp(Gparams[2]*logCa);
        double konC = Gparams[3]*exp(Gparams[5]*logCa);

        setGmat_konN_konC(Gmatrix, konN, konC);
        arma::vec dG_dt  = Gmatrix*G;
        Gflux = flux_konN_konC(konN, konC, G);

        Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(kapB + 1);

        //cout<<timesteps(i)<<endl;
        //cout<<"BCa = "<<BCa<<"; d_dt:"<<dBCa_dt<<", "<<calcium_input*kapB/(kapB+1)<<endl;
        //cout<<"Ca  = "<<Ca<<" ; d_dt:"<<dCa_dt<<", "<<calcium_input*1/(kapB+1)<<endl;
        //cout<<"------------------------"<<endl;

        dt  = timesteps(i)-timesteps(i-1); 
        G   = G   +  dt*dG_dt;
        BCa = BCa +  dt*(dBCa_dt+kapB/(kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
    }

    for(unsigned int i=0;i<9;i++) state_out(i) = G(i);
    state_out(9)  = BCa;
    state_out(10) = Ca;
    state_out(11) = Ca_in;

    double DFF_out = (arma::accu(G(brightStates)) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));

    return DFF_out;
}


// Currently, this method is only called via python bindings
void GCaMP::integrateOverTime(const arma::vec& time_vect, const arma::vec& spike_times) {

		DFF_values.clear();  // Clear previous DFF values
		
		double fine_dt = 100e-6; // Hard code this near instability edge for hardest G/Cparams

		arma::vec G = state(arma::span(0, 8));
    double BCa = state(9);
    double Ca = state(10);
    double Ca_in = state(11);
    double dBCa_dt, dCa_dt, dCa_in_dt, Gflux, Cflux;

    // Generate timesteps, prep calcium vect
    arma::vec timesteps = arma::regspace(time_vect(0), fine_dt, time_vect(time_vect.n_elem - 1));
    arma::vec calcium_input(timesteps.n_elem, arma::fill::zeros);

		// Temporary vector to store intermediate DFF values
    arma::vec temp_DFF_values(timesteps.n_elem);
    int spike_counter=0;
    // Find all timesteps that are within fine_timestep of one of the elements of spike_times
    for (double spike_time : spike_times) {
        arma::uvec indices = arma::find(arma::abs(timesteps - spike_time) <= fine_dt/2);
        calcium_input(indices).fill(DCaT/fine_dt);
        spike_counter+=indices.n_elem;
    }
		cout<<"N spike = "<<spike_counter<<"; total Ca = "<<arma::accu(calcium_input)<<endl;

    for (unsigned int i = 1; i < timesteps.n_elem; ++i) {
				
				/*
					Giovanni's implementation
				*/
				setGmat(Ca);
				arma::vec dG_dt = Gmat*G;
				Gflux = flux(Ca,G);
				
				Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input(i)*1/(kapB + 1);
						
				G   = G   +  fine_dt*dG_dt;
        BCa = BCa +  fine_dt*(dBCa_dt+kapB/(kapB+1)*calcium_input(i));
        Ca  = Ca  +  fine_dt*dCa_dt;
        Ca_in = Ca_in + fine_dt*dCa_in_dt;

        // Store DFF value at each timestep
        temp_DFF_values(i) = (arma::accu(G(brightStates)) - Ginit) /
                             (Ginit - G0 + (Gsat - G0) / (Rf - 1));
    }

    // Interpolate back onto original time vector
		//cout<<"Size of timesteps = "<<timesteps.n_elem<<"; Size of DFF_values = "<<DFF_values.n_elem<<endl;
		arma::interp1(timesteps, temp_DFF_values, time_vect, DFF_values, "linear");
		//cout<<"Size of DFF_values after interp1 = "<<DFF_values.n_elem<<endl;
}

//
void GCaMP::integrateOverTime2(const arma::vec& time_vect, const arma::vec& spike_times) {
    
		DFF_values.clear();  // Clear previous DFF values
		
		double fine_dt = 100e-6; // Hard code this near instability edge for hardest G/Cparams

		arma::vec G = state(arma::span(0, 8));
    double BCa = state(9);
    double Ca = state(10);
    double Ca_in = state(11);
    double dBCa_dt, dCa_dt, dCa_in_dt, Gflux, Cflux_out, Cflux_in;

    // Generate timesteps, prep calcium vect
    arma::vec timesteps = arma::regspace(time_vect(0), fine_dt, time_vect(time_vect.n_elem - 1));
    arma::vec calcium_input(timesteps.n_elem, arma::fill::zeros);

		// Temporary vector to store intermediate DFF values
    arma::vec temp_DFF_values(timesteps.n_elem);

    // Find all timesteps that are within fine_timestep of one of the elements of spike_times
    for (double spike_time : spike_times) {
        arma::uvec indices = arma::find(arma::abs(timesteps - spike_time) <= fine_dt/2);
        calcium_input(indices).fill(DCaT/fine_dt);
    }

    for (unsigned int i = 1; i < timesteps.n_elem; ++i) {
				
				/*
					Updated version with proper internal cal handling
				*/
				setGmat(Ca);
				arma::vec dG_dt = Gmat*G;
				Gflux = flux(Ca,G);
				
				Cflux_out = -gamma*(Ca-c0) - Gflux;
				Cflux_in = - gam_in*(Ca-c0) + gam_out*(Ca-c0);
        dBCa_dt = (Cflux_out + Cflux_in + calcium_input(i))*kapB/(kapB + 1);
        dCa_in_dt = -Cflux_in*kapB/(1+kapB);
				dCa_dt  = (Cflux_in + Cflux_out + calcium_input(i))*1/(kapB+1);
						
				G   = G   +  fine_dt*dG_dt;
        BCa = BCa +  fine_dt*dBCa_dt;
        Ca  = Ca  +  fine_dt*dCa_dt;
        Ca_in = Ca_in + fine_dt*dCa_in_dt;

        // Store DFF value at each timestep
        temp_DFF_values(i) = (arma::accu(G(brightStates)) - Ginit) /
                             (Ginit - G0 + (Gsat - G0) / (Rf - 1));
    }

    // Interpolate back onto original time vector
		//cout<<"Size of timesteps = "<<timesteps.n_elem<<"; Size of temp_DFF_values = "<<temp_DFF_values.n_elem<<endl;
		arma::interp1(timesteps, temp_DFF_values, time_vect, DFF_values, "linear");
		//cout<<"Size of DFF_values after interp1 = "<<DFF_values.n_elem<<endl;
}


// Getter with python bindings
const arma::vec& GCaMP::getDFFValues() const {
    return DFF_values;
}


double GCaMP::getDFF(){
    arma::vec G = state(brightStates);
    return (arma::accu(G) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));
}

double GCaMP::getDFF(const arma::vec& s){
    arma::vec G = s(brightStates);
    return (arma::accu(G) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));
}

double GCaMP::getAmplitude(){
    init();

    arma::uvec indices = {20};
    arma::ivec spikes(100);
    spikes.elem(indices).ones();
    arma::vec resp(100);

    for(unsigned int i=0;i<100;i++){
        evolve(1e-3,spikes(i));
        resp(i)=DFF;
    }

    return(resp.max());
}