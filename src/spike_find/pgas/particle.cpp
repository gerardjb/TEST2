#include"include/particle.h"
#include"include/utils.h"
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_math.h>
#include<fstream>
#include<string>
#include<ctime>
#include"include/GCaMP_model.h"
#include<iomanip>

// A helper function that generates a discrete random variable from a uniform sample.
// This function lets us pre-generate noise samples for the move and weight steps
// Ahead of time to aid in reproducibility.
#include <gsl/gsl_randist.h>
size_t gsl_ran_discrete_from_uniform(const gsl_ran_discrete_t *g, double u)
{

    #define KNUTH_CONVENTION 1

    size_t c=0;
    double f;
#if KNUTH_CONVENTION
    c = (u*(g->K));
#else
    u *= g->K;
    c = u;
    u -= c;
#endif
    f = (g->F)[c];
    /* fprintf(stderr,"c,f,u: %d %.4f %f\n",c,f,u); */
    if (f == 1.0) return c;

    if (u < f) {
        return c;
    }
    else {
        return (g->A)[c];
    }
}

using namespace std;

double fixedStep_LA_kernel(
    double deltat, int ns,
    const double * state_in, 
    double * state_out,
    const GCaMP_params & p)
{

    double G[9];

    for(int i=0;i<9;i++) G[i] = state_in[i];

    double BCa = state_in[9];
    double dBCa_dt;

    double Ca  = state_in[10];
    double Ca_in = state_in[11];
    double dCa_dt, dCa_in_dt;
    
    double Gflux, Cflux;
    double finedt=100e-6;
    double dt;

    double calcium_input;

    double konN,   koffN;
    double konC,   koffC;
    double konPN,  koffPN;
    double konPN2, koffPN2;
    double konPC,  koffPC;
    double konPC2,  koffPC2;

    // Load from the params struct
    koffN = p.koffN;
    koffC = p.koffC;
    konPN = p.konPN;
    koffPN = p.koffPN;
    konPN2 = p.konPN2;
    koffPN2 = p.koffPN2;
    konPC = p.konPC;
    koffPC = p.koffPC;
    konPC2 = p.konPC2;
    koffPC2 = p.koffPC2;
    
    // Calculate konN and konC
    double logCa = log(Ca);
    konN = p.Gparams[0]*exp(p.Gparams[2]*logCa);
    konC = p.Gparams[3]*exp(p.Gparams[5]*logCa);
  
    // Intiliaze the Gmatrix
	double Gmatrix[9][9] = {{-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0},
        {konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0},
        {0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2},
        {0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0},
        {konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0},
        {0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2},
        {0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0},
        {0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0},
        {0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)}};

    double dG_dt[9];
    double old_t = 0;
    double t = finedt;
    int tstep_i = 1;
    int n_steps = floor(deltat/finedt);
    while(t <= (n_steps*finedt)) {

        calcium_input = (tstep_i==1) ? ns*p.DCaT/finedt : 0;

        double logCa = log(Ca);
	    konN = p.Gparams[0]*exp(p.Gparams[2]*logCa);
        konC = p.Gparams[3]*exp(p.Gparams[5]*logCa);

        // Update the G matrix
        Gmatrix[0][0] = -(konN+konC);
        Gmatrix[1][0] = konN;
        Gmatrix[1][1] = -(koffN+konPN+konC);
        Gmatrix[2][2] = -(konC+koffPN);
        Gmatrix[3][2] = konC;
        Gmatrix[4][0] = konC;
        Gmatrix[4][4] = -(koffC+konPC+konN);
        Gmatrix[5][5] = -(koffPC+konN);
        Gmatrix[6][5] = konN;
        Gmatrix[7][1] = konC;
        Gmatrix[7][4] = konN;
       
        // arma::vec dG_dt  = Gmatrix*G;
        // Implement matrix vector product
        for(int i=0;i<9;i++){
            dG_dt[i] = 0;
            for(int j=0;j<9;j++) dG_dt[i] += Gmatrix[i][j]*G[j];
        }

        // Gflux = flux_konN_konC(konN, konC, G);
        // Calculate fluxes per binding site
        double N = -(konN*(G[0] + G[4] + G[5])) +
                koffN*(G[1] + G[6]+ G[7])  + 
                koffPN*G[2]                + 
                koffPN2*(G[3] + G[8]);

        double C = -(konC *(G[0] + G[1] + G[2])) +
                koffC*(G[4] + G[3] + G[7])  + 
                koffPC*(G[5])+ 
                koffPC2*(G[6]+G[8]);

        int Csites=2;
        Gflux = 2*N + Csites*C;

        Cflux   = -p.gamma*(Ca-p.c0) + Gflux;
        dBCa_dt = Cflux*p.kapB/(p.kapB + 1);
        dCa_in_dt = p.gam_in*(Ca-p.c0) - p.gam_out*(Ca_in - p.c0);

        dCa_dt  = -p.gamma*(Ca-p.c0) // pump out
            -p.gam_in*(Ca - p.c0) + p.gam_out*(Ca_in - p.c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(p.kapB + 1);

        dt  = t-old_t;
        
        // G   = G   +  dt*dG_dt;
        for(int i=0;i<9;i++) G[i] = G[i] + dt*dG_dt[i];

        BCa = BCa +  dt*(dBCa_dt+p.kapB/(p.kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
        old_t = t; 
        t = finedt*tstep_i;
        tstep_i++;
    }

    for(unsigned int i=0;i<9;i++) state_out[i] = G[i];
    state_out[9]  = BCa;
    state_out[10] = Ca;
    state_out[11] = Ca_in;

    int brightStates[5] = {2, 3, 5, 6, 8};
    double brightStatesSum = 0.0;
    for(unsigned int i=0;i<5;i++) brightStatesSum += G[brightStates[i]];

    double DFF_out = (brightStatesSum - p.Ginit)/(p.Ginit-p.G0+(p.Gsat-p.G0)/(p.Rf-1));

    return DFF_out;
}


ParticleArray::ParticleArray(int N, int T) : N(N), T(T) 
{

    B = MatrixType("B",N,T);
    burst = IntMatrixType("burst",N,T);
    C = Kokkos::View<double**[12], MemSpace>("C",N,T);
    S = IntMatrixType("S",N,T);
    logWeight = MatrixType("logWeight",N,T);
    ancestor = IntMatrixType("ancestor",N,T);
    index = IntMatrixType("index",N,T);

    B_h = Kokkos::create_mirror_view(B);
    burst_h = Kokkos::create_mirror_view(burst);
    C_h = Kokkos::create_mirror_view(C);
    S_h = Kokkos::create_mirror_view(S);
    logWeight_h = Kokkos::create_mirror_view(logWeight);
    ancestor_h = Kokkos::create_mirror_view(ancestor);
    index_h = Kokkos::create_mirror_view(index);

    // Intermediate arrays for the move and weight steps
    log_probs = MatrixType("log_probs",N,maxspikes*2);
    log_probs_h = Kokkos::create_mirror_view(log_probs);

    probs = MatrixType("probs",N,maxspikes*2);
    probs_h = Kokkos::create_mirror_view(probs);

    discrete = IntVectorType("discrete",N);
    discrete_h = Kokkos::create_mirror_view(discrete);

    logW = VectorType("logW",N);
    logW_h = Kokkos::create_mirror_view(logW);

    ar_logW = VectorType("ar_logW",N);
    ar_logW_h = Kokkos::create_mirror_view(ar_logW);
}


void ParticleArray::set_particle(int t, int idx, const Particle &p)
{
    B_h(idx,t) = p.B;
    burst_h(idx,t) = p.burst;
    for(int i=0;i<12;i++) C_h(idx,t,i) = p.C(i);
    S_h(idx,t) = p.S;
    logWeight_h(idx,t) = p.logWeight;
    ancestor_h(idx,t) = p.ancestor;
    index_h(idx,t) = p.index;
}

void ParticleArray::get_particle(int t, int idx, Particle &p)
{
    p.B = B_h(idx,t);
    p.burst = burst_h(idx,t);
    for(int i=0;i<12;i++) p.C(i) = C_h(idx,t,i);
    p.S = S_h(idx,t);
    p.logWeight = logWeight_h(idx,t);
    p.ancestor = ancestor_h(idx,t);
    p.index = index_h(idx,t);
}

void ParticleArray::copy_to_device()
{
    Kokkos::deep_copy(B,B_h);
    Kokkos::deep_copy(burst,burst_h);
    Kokkos::deep_copy(C,C_h);
    Kokkos::deep_copy(S,S_h);
    Kokkos::deep_copy(logWeight,logWeight_h);
    Kokkos::deep_copy(ancestor,ancestor_h);
    Kokkos::deep_copy(index,index_h);
}

void ParticleArray::copy_to_host()
{
    Kokkos::deep_copy(B_h,B);
    Kokkos::deep_copy(burst_h,burst);
    Kokkos::deep_copy(C_h,C);
    Kokkos::deep_copy(S_h,S);
    Kokkos::deep_copy(logWeight_h,logWeight);
    Kokkos::deep_copy(ancestor_h,ancestor);
    Kokkos::deep_copy(index_h,index);
}

bool ParticleArray::check_particle_system(int t, std::vector<Particle*> & particleSystem)
{
    copy_to_host();

    bool die = false;

    // Check that the particleArray and particleSystem are the same
    for(int i=0;i<N;i++){
        Particle p_cpu = particleSystem[t][i];
        Particle p_gpu;
        get_particle(t, i, p_gpu);
        if (p_cpu.S != p_gpu.S) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU S=" << p_cpu.S <<  " GPU S=" << p_gpu.S << endl;
            die = true;
        }
        if (fabs(p_cpu.B - p_gpu.B) > 1e-10){
            cout << "TIME=" << t << "  Particle(" << i << "): CPU B=" << p_cpu.B <<  " GPU B=" << p_gpu.B << endl;
            die = true;
        }
        if (p_cpu.burst != p_gpu.burst) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU burst=" << p_cpu.burst <<  " GPU burst=" << p_gpu.burst << endl;
            die = true;
        }
        for(int j=0;j<12;j++){
            if (fabs(p_cpu.C(j) - p_gpu.C(j)) > 1e-10) {
                cout << "TIME=" << t << "  Particle(" << i << "): CPU C(" << j << ")=" << p_cpu.C(j) <<  " GPU C(" << j << ")=" << p_gpu.C(j) << endl;
                die = true;
            }
        }
        if (p_cpu.ancestor != p_gpu.ancestor) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU ancestor=" << p_cpu.ancestor <<  " GPU ancestor=" << p_gpu.ancestor << endl;
            die = true;
        }
        if (fabs(p_cpu.logWeight - p_gpu.logWeight) > 1e-10) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU logWeight=" << p_cpu.logWeight <<  " GPU logWeight=" << p_gpu.logWeight << endl;
            die = true;
        }

        if (die)
            break;
    }

    return !die;
}

double ParticleArray::logf(
    int part_idx_in, int t_in, 
    int part_idx_out, int t_out, 
    const param &par, constpar *constants) 
{
    double lf = 0;
    double dt = 1.0/constants->sampling_frequency;
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    double rate[2] = {par.r0*dt,par.r1*dt};

    int pin_burst = burst(part_idx_in, t_in);
    double pin_B = B(part_idx_in, t_in);
    int pin_S = S(part_idx_in, t_in);
    
    int pout_burst = burst(part_idx_out, t_out);
    double pout_B = B(part_idx_out, t_out);
    int pout_S = S(part_idx_out, t_out);

    lf += log(W[pin_burst][pout_burst]) +
        pout_S*log(rate[pout_burst])-rate[pout_burst] - log(tgamma(pout_S+1)) +
        -0.5*pow((pout_B-pin_B)/(constants->bm_sigma*sqrt(dt)),2);

    return(lf);
}

void ParticleArray::calc_ancestor_resampling(
    int t, const param &par, constpar *constants)
{
    // Retrieve weights and calculate ancestor resampling weights
    Kokkos::parallel_for("calc_ancestor_weights",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {
                
                double lf = 0;
                double dt = 1.0/constants->sampling_frequency;
                double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
                    {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
                double rate[2] = {par.r0*dt,par.r1*dt};

                int part_idx_in = part_idx;
                int t_in = t-1;
                int part_idx_out = 0;
                int t_out = t;

                int pin_burst = burst(part_idx_in, t_in);
                double pin_B = B(part_idx_in, t_in);
                int pin_S = S(part_idx_in, t_in);
                
                int pout_burst = burst(part_idx_out, t_out);
                double pout_B = B(part_idx_out, t_out);
                int pout_S = S(part_idx_out, t_out);

                lf += log(W[pin_burst][pout_burst]) +
                    pout_S*log(rate[pout_burst])-rate[pout_burst] - log(tgamma(pout_S+1)) +
                    -0.5*pow((pout_B-pin_B)/(constants->bm_sigma*sqrt(dt)),2);
                
                logW(part_idx) = logWeight(part_idx, t-1);
                ar_logW(part_idx) = logWeight(part_idx, t-1) + lf;
                
            });
    Kokkos::fence();

    // Copy back to host (can remove this later when random number generation is done on GPU)
    Kokkos::deep_copy(logW_h, logW);
    Kokkos::deep_copy(ar_logW_h, ar_logW);
}

void ParticleArray::move_and_weight(
    int t,
    VectorType y, 
    const param &par, constpar *constants, 
    VectorType g_noise, std::vector<double> & u_noise,
    const GCaMP_params & params)
{

    double dt = 1.0/constants->sampling_frequency;

    double rate[2] = {par.r0*dt,par.r1*dt};
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    
    double z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};

    Kokkos::parallel_for("evolve_for_maxspikes",
		Kokkos::RangePolicy<ExecSpace>(0, N*maxspikes*2),
            KOKKOS_CLASS_LAMBDA(const int idx) {
                int particle_idx = idx / (maxspikes*2);
                int spike_idx = idx % (maxspikes*2);
                int b = floor(spike_idx/maxspikes);
                int ns    = spike_idx % maxspikes;
                
                int a = ancestor(particle_idx, t);
                int parent_b = burst(a, t-1);
                double parent_B = B(a, t-1);
                
                // Evolve
                // model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
                double state_out[12];
                double state_in[12];
                for(int i=0;i<12;i++) state_in[i] = C(a, t-1, i);
                double ct = fixedStep_LA_kernel(dt, ns, state_in, state_out, params);

                double log_prob_tmp = log(W[parent_b][b]);
                log_prob_tmp += ns*log(rate[b]) - log(tgamma(ns+1)) - rate[b];
                log_prob_tmp += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y(t)-ct-parent_B,2);
                log_probs(particle_idx, spike_idx) = log_prob_tmp;

            });

    // // Wait for kernel to complete
    Kokkos::fence();

    // Calclulate w and Z
    Kokkos::parallel_for("w_and_Z",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {
                                
                // Get the maximum log weight for the particle
                double max_log_prob = -INFINITY;
                for(int i=0;i<2*maxspikes;i++) 
                    if (log_probs(part_idx,i) > max_log_prob) 
                        max_log_prob = log_probs(part_idx, i);
                    
                // Compute w (probs) and Z
                double Z_tmp = 0;
                for(int i=0;i<2*maxspikes;i++) {
                    probs(part_idx, i) = exp(log_probs(part_idx, i)-max_log_prob);
                    Z_tmp += probs(part_idx, i);
                }

                Z_tmp = Z_tmp*exp(max_log_prob);
                logWeight(part_idx, t) = log(Z_tmp);

            });

    // // Wait for kernel to complete
    Kokkos::fence();

    // Copy probabilities to host so we can generate the discrete random variables
    // Will do this on GPU in the future after testing
    IntVectorType discrete = IntVectorType("discrete", N);
    IntVectorType::HostMirror discrete_h = Kokkos::create_mirror_view(discrete);

    MatrixType::HostMirror probs_h = Kokkos::create_mirror_view(probs);
    Kokkos::deep_copy(probs_h, probs);
    for(int i=0;i<N;i++) {
        double probs_tmp[2*maxspikes];
        for(int j=0;j<2*maxspikes;j++) probs_tmp[j] = probs_h(i,j);
        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2*maxspikes, probs_tmp);
        discrete_h(i) = gsl_ran_discrete_from_uniform(rdisc, u_noise[i]);
        gsl_ran_discrete_free(rdisc);
    }
    Kokkos::deep_copy(discrete, discrete_h);

    // Move the particles
    Kokkos::parallel_for("move_particles",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {
                // move particle if not already set
                int idx = discrete(part_idx);

                if(part_idx != 0){
                    burst(part_idx, t) = floor(idx/maxspikes);
                    S(part_idx, t) = idx%maxspikes;

                    double g_noise_val = g_noise(part_idx);
                    double parent_B = B(ancestor(part_idx, t), t-1);

                    B(part_idx, t) = B(ancestor(part_idx, t), t-1) + g_noise(part_idx);
                }
                
                // model->evolve_threadsafe(dt, part.S, parent.C, state_out, ct);
                // part.C     = state_out;
                double state_out[12];
                double state_in[12];
                for(int i=0;i<12;i++) state_in[i] = C(ancestor(part_idx, t), t-1, i);
                double ct = fixedStep_LA_kernel(dt, S(part_idx, t), state_in, state_out, params);
                for(int i=0;i<12;i++) C(part_idx, t, i) = state_out[i];

            });

    Kokkos::fence();

}



// PARTICLE CLASS
// -----------------------------------------------------------
//
Particle::Particle(){
    C.set_size(12);
    B=0;
    burst=0;
    S=0;
    ancestor=-1;
    logWeight=0;
}

Particle& Particle::operator=(const Particle& p){
    B=p.B;
    burst=p.burst;
    C=p.C;
    S=p.S;
    logWeight=p.logWeight;
    return *this;
}

void Particle::print(){
    cout<<"    C    : "<<C(0)<<endl
        <<"    B    : "<<B<<endl
        <<"    burst: "<<burst<<endl
        <<"    S    : "<<S<<endl;
}

// Trajectory class
// ------------------------------------------------------------
//
Trajectory::Trajectory(unsigned int s, string fname){

    filename=fname;
    size=s;
    B.resize(s);
    burst.resize(s);
    C.resize(s);
    S.resize(s);
    S.zeros();
    Y.resize(s);

}

void Trajectory::simulate(gsl_rng* rng, const param& par, const constpar *constants){

    GCaMP model(par.G_tot, par.gamma, par.DCaT, par.Rf, par.gam_in, par.gam_out);

    double dt=1.0/constants->sampling_frequency;
    double rate[2] = {par.r0*dt,par.r1*dt};

    B(0) = 0;//gsl_ran_gaussian(rng,constants->bm_sigma);
    burst(0) = 0;
    S(0) = gsl_ran_poisson(rng,par.r0*dt);
    Y(0) = 0+B(0)+gsl_ran_gaussian(rng,sqrt(par.sigma2));
    
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};

    for(unsigned int t=1;t<size;t++){

        burst(t) = (gsl_rng_uniform(rng) < W[burst(t-1)][0]) ? 0 : 1;

        B(t)     = B(t-1)+gsl_ran_gaussian(rng,constants->bm_sigma*sqrt(dt)); 
        S(t)     = gsl_ran_poisson(rng,rate[burst(t)]);
        
        model.evolve(dt,S(t));

        C(t)     = model.DFF;
        Y(t)     = model.DFF+B(t)+gsl_ran_gaussian(rng,sqrt(par.sigma2)); 
    }
}

void Trajectory::write(ofstream &outfile, unsigned int index){
    
    if(!outfile.is_open()){
        outfile.open(filename);
        outfile<<"index,burst,B,S,C,Y"<<endl;
    }

    for(unsigned int i=0;i<size;++i){
        outfile<<index<<','
            <<burst(i)<<','
            <<B(i)<<','
            <<S(i)<<','
            <<C(i)<<','
            <<Y(i)<<endl;
    }

}

Trajectory& Trajectory::operator=(const Trajectory& traj){
        this->burst = traj.burst;
        this->B     = traj.B;
        this->S     = traj.S;
        this->C     = traj.C;
        this->Y     = traj.Y;

        return *this ;
}

double Trajectory::logp(const param* p, const constpar* constants){
    double logp = -0.5 * pow( B(0)/1e-4, 2) -0.5 * pow( (Y(0) - B(0)), 2)/p->sigma2 ;
    double dt = 1.0/constants->sampling_frequency;
    double W[2][2] = {{1-p->wbb[0]*dt, p->wbb[0]*dt},
        {p->wbb[1]*dt, 1-p->wbb[1]*dt}};
    double rate[2] = {p->r0*dt,p->r1*dt};

    for(unsigned int i=1;i<size;++i){
        
        logp += log(W[burst(i-1)][burst(i)]) +
                S(i)*log(rate[burst(i)])-rate[burst(i)] - log(gsl_sf_gamma(S(i)+1)) +
                -0.5*pow((B(i)-B(i-1))/(constants->bm_sigma*sqrt(dt)),2) +
                -0.5 * pow( (Y(i) - C(i) - B(i)), 2)/p->sigma2;

    }

    return logp;

}

        

// SMC class
// -------------------------------------------------------------------
//
SMC::SMC(string filename, int index, constpar& cst, bool has_header, int seed, unsigned int maxlen, string Gparam_file){
		
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (seed==0) ? cst.seed : seed);
		
    model = new GCaMP(cst.G_tot_mean,cst.gamma_mean,cst.DCaT_mean,cst.Rf_mean,cst.gam_in_mean, cst.gam_out_mean, Gparam_file);
    model->setTimeStepMode((TimeStepMode)cst.TSMode);

    arma::field <string> header;

    if(has_header){
        tracemat.load(arma::csv_name(filename, header));
    } else {
        tracemat.load(filename);
    }
    if(maxlen>0 && maxlen<tracemat.n_rows) tracemat = tracemat.rows(0,maxlen);

    data_time  = tracemat.col(0);
    data_y    = tracemat.col(index);
    constants = &cst;

    // The time units here are assumed to be seconds
    constants->sampling_frequency = 1.0 / (tracemat(1, 0)-tracemat(0,0));
    cout << "setting sampling frequency to: "<<constants->sampling_frequency << endl;
    constants->set_time_scales();

    // set number of particles
    nparticles=constants->nparticles;
    TIME = tracemat.n_rows;
    cout<<"nparticles: "<<nparticles<<endl;
    cout<<"TIME      : "<<TIME<<endl;

    // set particle system
    particleSystem.resize(TIME);
    for(unsigned int t=0;t<TIME;++t){
        particleSystem[t] = new Particle[nparticles];
        for(unsigned int j=0;j<nparticles;j++) particleSystem[t][j].index=j;
    }

}

SMC::SMC(arma::vec &Y, constpar& cst, bool v){

    verbose=v;

    constants = &cst;

    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng,constants->seed);

    data_time  = arma::regspace(0,Y.n_elem,1.0/constants->sampling_frequency);
    data_y    = Y;

    // The time units here are assumed to be seconds
    if(verbose) cout << "setting sampling frequency to: "<<constants->sampling_frequency << endl;
    constants->set_time_scales();

    // set number of particles
    nparticles=constants->nparticles;
    TIME = Y.n_rows;

    // set particle system
    particleSystem.resize(TIME);
    for(unsigned int t=0;t<TIME;++t){
        particleSystem[t] = new Particle[nparticles];
        for(unsigned int j=0;j<nparticles;j++) particleSystem[t][j].index=j;
    }
}


void SMC::rmu(Particle &p, double y, const param &par, bool set=false){

    if(!set){
        // draw state if not set

        p.burst = (gsl_rng_uniform(rng) > 0.5 ) ? 1 : 0;

        if(!constants->KNOWN_SPIKES){
            double rate=(p.burst==0) ? par.r0 : par.r1;
            p.S     = 0; //gsl_ran_poisson(rng, rate/constants->sampling_frequency);
        }

        p.B     = gsl_ran_gaussian(rng,1e-4);
    }

    p.C=model->initial_state;

    p.logWeight=-0.5*log(2*M_PI*par.sigma2)-0.5/par.sigma2*pow(y-0-p.B,2); // from g(y|x) only
                
}
double SMC::logf(const Particle &pin, const Particle &pout, const param &par){
    double lf = 0;
    double dt = 1.0/constants->sampling_frequency;
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    double rate[2] = {par.r0*dt,par.r1*dt};


    lf += log(W[pin.burst][pout.burst]) +
        pout.S*log(rate[pout.burst])-rate[pout.burst] - log(gsl_sf_gamma(pout.S+1)) +
        -0.5*pow((pout.B-pin.B)/(constants->bm_sigma*sqrt(dt)),2);

    return(lf);
}

// Here part.S is supposed to be given.
//
void SMC::move_and_weight_GTS(Particle &part, const Particle& parent, double y, const param &par, double g_noise, double u_noise, bool set=false){

    double probs[2];
    double log_probs[2];
    double Z;
    double ct;
    unsigned int burst;
    unsigned int ns=part.S;

    double dt = 1.0/constants->sampling_frequency;

    double rate[2] = {par.r0*dt,par.r1*dt};
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};

    double z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};
    double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));
    
    arma::vec state_out(12);
    model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
    
    log_probs[0] = log(W[parent.burst][0]);
    log_probs[0] += ns*log(rate[0]) - log(gsl_sf_gamma(ns+1)) -rate[0];
    log_probs[0] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2);
    log_probs[1] = log(W[parent.burst][1]);
    log_probs[1] += ns*log(rate[1]) - log(gsl_sf_gamma(ns+1)) -rate[1];
    log_probs[1] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2);

    utils::w_from_logW(log_probs,probs,2);
    Z=utils::Z_from_logW(log_probs,2);

    part.logWeight = log(Z);

    if(!set){
        // move particle if not already set
        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2, probs);
        int idx = gsl_ran_discrete_from_uniform(rdisc, u_noise);

        part.burst = idx;
        part.B     = z[0]*parent.B+z[1]*(y-parent.C(0)) + g_noise;

        gsl_ran_discrete_free(rdisc);
    }
    
    // part.C     = model->state;
    part.C = state_out;

}

void SMC::move_and_weight(Particle &part, const Particle& parent, double y, const param &par, double g_noise, double u_noise, bool set=false){

    const int maxspikes = 2;  // The number of spikes goes from 0 to maxspikes-1
    double probs[2*maxspikes];
    double log_probs[2*maxspikes];
    double Z;
    double ct;
    unsigned int burst, ns;
    double dt = 1.0/constants->sampling_frequency;

    double rate[2] = {par.r0*dt,par.r1*dt};
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    
    double z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};
    
    arma::vec state_out(12);
    for(unsigned int i=0;i<2*maxspikes;i++){

        burst = floor(i/maxspikes);
        ns    = i%maxspikes;

        model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
        
        log_probs[i] = log(W[parent.burst][burst]);
        log_probs[i] += ns*log(rate[burst]) - log(tgamma(ns+1)) -rate[burst];
        log_probs[i] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2);        
    } 

    utils::w_from_logW(log_probs,probs,2*maxspikes);
    Z=utils::Z_from_logW(log_probs,2*maxspikes);

    part.logWeight = log(Z);

    if(!set){
        // move particle if not already set
        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2*maxspikes, probs);
        int idx = gsl_ran_discrete_from_uniform(rdisc, u_noise);
        
        part.burst = floor(idx/maxspikes);
        part.S     = idx%maxspikes;

        //part.B     = z[0]*parent.B+z[1]*(y-model->getDFF(parent.C)) + gsl_ran_gaussian(rng,sigma_B_posterior);
        part.B     = parent.B + g_noise;

        gsl_ran_discrete_free(rdisc);
    }
    
    model->evolve_threadsafe(dt, part.S, parent.C, state_out, ct);
    part.C     = state_out;

}

void SMC::sampleParameters(const param &pin, param &pout, Trajectory &traj){

    // Sampling the posterior burst transition rates
    //
    int counts[2][2] = {{0,0},{0,0}};

    for(unsigned int t=1;t<traj.size;t++){
        counts[traj.burst(t-1)][traj.burst(t)]++;
    }
    double dt = 1.0/constants->sampling_frequency;

    double alpha_w01_post = constants->alpha_w01 + counts[0][1];
    double beta_w01_post  = constants->beta_w01  + counts[0][0]*dt;
    double alpha_w10_post = constants->alpha_w10 + counts[1][0];
    double beta_w10_post  = constants->beta_w10  + counts[1][1]*dt;

    pout.wbb[0] = gsl_ran_gamma(rng,alpha_w01_post,1.0/beta_w01_post);
    pout.wbb[1] = gsl_ran_gamma(rng,alpha_w10_post,1.0/beta_w10_post);

    // Sampling the posterior firing rates
    //
    arma::uvec which_b0 = arma::find(traj.burst==0);
    double alpha_rate_b0_post = constants->alpha_rate_b0 + arma::accu(traj.S(which_b0));
    double beta_rate_b0_post  = constants->beta_rate_b0  + which_b0.n_elem*dt;

    arma::uvec which_b1 = arma::find(traj.burst==1);
    double alpha_rate_b1_post = constants->alpha_rate_b1 + arma::accu(traj.S(which_b1));
    double beta_rate_b1_post  = constants->beta_rate_b1  + which_b1.n_elem*dt;

    pout.r0 = gsl_ran_gamma(rng,alpha_rate_b0_post,1.0/beta_rate_b0_post);
    pout.r1 = gsl_ran_gamma(rng,alpha_rate_b1_post,1.0/beta_rate_b1_post);

    // Sampling standard deviation
    //
    double res = pow(arma::norm(data_y - traj.B - traj.C),2);
    pout.sigma2 = 1.0 / gsl_ran_gamma(rng, constants->alpha_sigma2 + traj.size / 2.0, 1.0 / (constants->beta_sigma2 + 0.5 * res));
    //pout.sigma2=pin.sigma2;

    // Sampling Amax, rise and decay times

    unsigned int sampled_variable;
    sampled_variable=gsl_rng_uniform_int(rng,6);

    unsigned int var_selec[6] = {0,0,0,0,0,0}; 
    var_selec[sampled_variable] = 1;

    if(constants->KNOWN_SPIKES){
        var_selec[0]=1;
        var_selec[1]=1;
        var_selec[2]=1;
        var_selec[3]=1;
        var_selec[4]=1;
        var_selec[5]=1;
    }

    // Sampling parameter set using gaussian moves
    //double G_tot_test = pin.G_tot + var_selec[0]*gsl_ran_gaussian(rng,constants->G_tot_prop_sd);
    //double gamma_test = pin.gamma + var_selec[1]*gsl_ran_gaussian(rng,constants->gamma_prop_sd);
    //double DCaT_test  = pin.DCaT  + var_selec[2]*gsl_ran_gaussian(rng,constants->DCaT_prop_sd);
    //double Rf_test    = pin.Rf    + var_selec[3]*gsl_ran_gaussian(rng,constants->Rf_prop_sd);
    //double gam_in_test  = pin.gam_in  + var_selec[4]*gsl_ran_gaussian(rng,constants->gam_in_prop_sd);
    //double gam_out_test = pin.gam_out + var_selec[5]*gsl_ran_gaussian(rng,constants->gam_out_prop_sd);

    // Sampling parameters using gamma multiplicative proposals
    
    double proposal_factors[6] = {1,1,1,1,1,1};
    for(unsigned int i=0; i<6; i++){
        proposal_factors[i] = gsl_ran_gamma(rng,1000,0.001);
    }

    double G_tot_test = pin.G_tot*((1-var_selec[0])+var_selec[0]*proposal_factors[0]);
    double gamma_test = pin.gamma*((1-var_selec[1])+var_selec[1]*proposal_factors[1]);
    double DCaT_test  = pin.DCaT*((1-var_selec[2])+var_selec[2]*proposal_factors[2]);
    double Rf_test    = pin.Rf *((1-var_selec[3])+var_selec[3]*proposal_factors[3]);
    double gam_in_test  = pin.gam_in *((1-var_selec[4])+var_selec[4]*proposal_factors[4]);
    double gam_out_test = pin.gam_out*((1-var_selec[5])+var_selec[5]*proposal_factors[5]);
    
    arma::vec partest={G_tot_test,gamma_test,DCaT_test,Rf_test,gam_in_test,gam_out_test};
    string parnames[] = {"G_tot", "gamma","DCaT","Rf","gam_in","gam_out"};
    arma::vec C_test(traj.size);
    double res_test;

    // resetting the GCaMP model
    model->setParams(G_tot_test,gamma_test,DCaT_test,Rf_test,gam_in_test, gam_out_test);
    model->init();

    res_test=0;
    for(unsigned int t=0;t<traj.size;t++) {
        model->evolve(dt,traj.S(t));
        res_test += pow(data_y(t)-traj.B(t)-model->DFF,2);
        C_test(t) = model->DFF;
    }

    double log_alpha_MH = -0.5*pow((G_tot_test-constants->G_tot_mean)/constants->G_tot_sd,2) + 0.5*pow((pin.G_tot-constants->G_tot_mean)/constants->G_tot_sd,2) 
                          -0.5*pow((gamma_test-constants->gamma_mean)/constants->gamma_sd,2) + 0.5*pow((pin.gamma-constants->gamma_mean)/constants->gamma_sd,2)
                          -0.5*pow((DCaT_test-constants->DCaT_mean)/constants->DCaT_sd,2)   + 0.5*pow((pin.DCaT-constants->DCaT_mean)/constants->DCaT_sd,2)
                          -0.5*pow((Rf_test-constants->Rf_mean)/constants->Rf_sd,2)+ 0.5*pow((pin.Rf-constants->Rf_mean)/constants->Rf_sd,2) 
                          -0.5*pow((gam_in_test-constants->gam_in_mean)/constants->gam_in_sd,2)+ 0.5*pow((pin.gam_in-constants->gam_in_mean)/constants->gam_in_sd,2) 
                          -0.5*pow((gam_out_test-constants->gam_out_mean)/constants->gam_out_sd,2)+ 0.5*pow((pin.gam_out-constants->gam_out_mean)/constants->gam_out_sd,2) 
                          -0.5*(res_test)/pout.sigma2 + 0.5*(res/pout.sigma2);

    // add proposal factors
    for (unsigned int i=0; i<6;i++){
        //log_alpha_MH += stats::gsl_gamma_logpdf(1.0/proposal_factors[i],100,0.01) - stats::gsl_gamma_logpdf(proposal_factors[i],100,0.01);
    }

    cout<<"var: "<<setw(6)<<parnames[sampled_variable]<<' '
        <<"value: "<<setw(15)<<partest(sampled_variable)<<' '
        <<"logAlpha: "<<setw(10)<<log_alpha_MH<<endl;

    if(gsl_rng_uniform(rng) < exp(log_alpha_MH)) {
        pout.G_tot  = G_tot_test;
        pout.gamma  = gamma_test;
        pout.DCaT   = DCaT_test;
        pout.Rf     = Rf_test;
        pout.gam_in = gam_in_test;
        pout.gam_out= gam_out_test;
        traj.C      = C_test;
    } else {
        pout.G_tot  = pin.G_tot;
        pout.gamma  = pin.gamma;
        pout.DCaT   = pin.DCaT;
        pout.Rf     = pin.Rf;
        pout.gam_in = pin.gam_in;
        pout.gam_out= pin.gam_out;
    }


}

void SMC::PF(const param &par){

    // define particle system
    cout<<"TIME = "<<TIME<<endl;
    cout<<"nparticles = "<<nparticles<<endl;

    // define weights
    double w[nparticles],logW[nparticles];

    // initialize all particles at time 0
    for(unsigned int i=0;i<nparticles;++i){
        rmu(particleSystem[0][i],data_y(0),par);
    }

    vector<double> g_noise(nparticles);
    vector<double> u_noise(nparticles);
    vector<int> a_noise(nparticles);

    for(unsigned int t=1;t<TIME;++t){
        // Multinomial resampling
        // // // collect weights
        for(unsigned int i=0;i<nparticles;i++){
            logW[i] = particleSystem[t-1][i].logWeight;
        }

        utils::w_from_logW(logW,w,nparticles);

        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
        
        // Pregenerate noise so we can have noise that is not dependent on the thread execution order
        double dt = 1.0/constants->sampling_frequency;
        double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));  
        for(unsigned int i=0;i<nparticles;i++){
            if(i != 0) {
                a_noise[i] = gsl_ran_discrete(rng,rdisc);
                u_noise[i] = gsl_rng_uniform(rng);
                g_noise[i] = gsl_ran_gaussian(rng,sigma_B_posterior);
            }
        }

        #pragma omp parallel for schedule(static)
        for(unsigned int i=0;i<nparticles;i++){
            
            //set ancestor of particle i
            
            particleSystem[t][i].ancestor = a_noise[i]; 
            particleSystem[t][i].logWeight = -log(nparticles);

            //move particle
            move_and_weight(particleSystem[t][i], particleSystem[t-1][a_noise[i]], data_y(t), par, g_noise[i], u_noise[i]);
        }

        gsl_ran_discrete_free(rdisc);

    }

    ofstream outfile_C("PF_output_C.dat");
    ofstream outfile_B("PF_output_B.dat");
    ofstream outfile_A("PF_output_ancestors.dat");
    ofstream outfile_test("PF_all_part.dat");
    ofstream outfile_S("PF_output_S.dat");


    for(unsigned int i=0;i<nparticles;i++){
        unsigned int a=i;
        unsigned int t=TIME;

        while(t>0){
            t--;
            outfile_A<<a<<' ';
            outfile_C<<particleSystem[t][a].C(0)<<' ';
            outfile_S<<particleSystem[t][a].S<<' ';
            outfile_B<<particleSystem[t][a].B<<' ';
            a=particleSystem[t][a].ancestor;
        }
        outfile_C<<endl;
        outfile_B<<endl;
        outfile_A<<endl;
        outfile_S<<endl;
    }

  
}

// A global to keep track whether Kokkos has been setup.
bool Kokkos_Initialized = false;

void init_kokkos()
{
	if (!Kokkos_Initialized) {
		Kokkos::initialize();
		Kokkos_Initialized = true;
	}
}

void SMC::PGAS(const param &par, const Trajectory &traj_in, Trajectory &traj_out){

    unsigned int t,a;
    int i;

    double dt=1.0/constants->sampling_frequency;

    // set model parameters
    model->setParams(par.G_tot,par.gamma,par.DCaT,par.Rf, par.gam_in, par.gam_out);

    // set particle 0 from input trajectory
    // Because trajectories do not store the full GCaMP state over time we have to re-evolve from the initial state.
    model->init();  // to initialize the GCaMP state to the steady state.
		
    for(t=0;t<TIME;t++){
        particleSystem[t][0].B=traj_in.B(t);
        particleSystem[t][0].burst=traj_in.burst(t);
        particleSystem[t][0].S=traj_in.S(t);

        model->evolve(dt,traj_in.S(t));
        particleSystem[t][0].C=model->state;
				
        particleSystem[t][0].ancestor=0;
				
    }

    particleSystem[0][0].ancestor = -1;


    // set spike count to all particles if training
    if(constants->KNOWN_SPIKES){
        for(t=0;t<TIME;t++){
            for(i=0;i<nparticles;i++){
                particleSystem[t][i].S=traj_in.S(t);
            }
        }
    }

    // define weights
    double w[nparticles],logW[nparticles];
    double ar_w[nparticles], ar_logW[nparticles];
		

    // initialize all particles at time 0 (excluded particle 0) and set all weights
    for(i=0;i<nparticles;++i){
        rmu(particleSystem[0][i],data_y(0),par,i==0); // i==0 allows to generate latent states only if i!=0
    }

    vector<double> g_noise(nparticles);
    vector<double> u_noise(nparticles);
        
    init_kokkos();
    
    // Copy the state of particles for all times to our structure of arrays on device memory
    ParticleArray particleArray(nparticles, TIME);
    for(t=0;t<TIME;t++)
        for(i=0;i<nparticles;i++)
            particleArray.set_particle(t, i, particleSystem[t][i]);
    particleArray.copy_to_device();

    // Create a GPU view of the data_y vector
    VectorType data_y_view = VectorType("data_y_view", data_y.n_elem);
    VectorType::HostMirror data_y_h = Kokkos::create_mirror_view(data_y_view);
    for (unsigned int i = 0; i < data_y.n_elem; i++)
        data_y_h(i) = data_y(i);
    Kokkos::deep_copy(data_y_view, data_y_h);

    // Create the noise views
    VectorType g_noise_view = VectorType("g_noise_view", nparticles);
    VectorType u_noise_view = VectorType("u_noise_view", nparticles);
    VectorType::HostMirror g_noise_h = Kokkos::create_mirror_view(g_noise_view);
    VectorType::HostMirror u_noise_h = Kokkos::create_mirror_view(u_noise_view);

    GCaMP_params params;
    model->read_params(params);

    bool die = false;
    for(t=1;t<TIME;++t){

        if (die)
            break;

        cout<<"    "<<t<<"      \r"<<flush;
        // Retrieve weights and calculate ancestor resampling weights for particle 0
        // #pragma omp parallel for schedule(static)
        // for(i=0;i<nparticles;i++){
        //     logW[i] = particleSystem[t-1][i].logWeight;
        //     ar_logW[i] = logW[i]+logf(particleSystem[t-1][i],particleSystem[t][0],par);
        // }

        particleArray.calc_ancestor_resampling(t, par, constants);
        for(i=0;i<nparticles;i++){
            logW[i] = particleArray.logW_h(i);;
            ar_logW[i] = particleArray.ar_logW_h(i);
        }

        // Check logW and ar_logW are equal on particleArray and particleSystem
        // for(i=0;i<nparticles;i++){
        //     double lw_cpu = logW[i];
        //     double lw_gpu = particleArray.logW_h(i);
        //     double ar_lw_cpu = ar_logW[i];
        //     double ar_lw_gpu = particleArray.ar_logW_h(i);
        //     if (fabs(lw_cpu - lw_gpu) > 1e-10 || fabs(ar_lw_cpu - ar_lw_gpu) > 1e-10) {
        //         cout << "Error: logW or ar_logW mismatch at particle " << i << " at time " << t << endl;
        //         cout << "CPU: logW = " << lw_cpu << ", ar_logW = " << ar_lw_cpu << endl;
        //         cout << "GPU: logW = " << lw_gpu << ", ar_logW = " << ar_lw_gpu << endl;
        //         die = true;
        //     }
        // }

        utils::w_from_logW(logW,w,nparticles);
        utils::w_from_logW(ar_logW,ar_w,nparticles);

        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
        gsl_ran_discrete_t *ar_rdisc = gsl_ran_discrete_preproc(nparticles, ar_w);

        // Ancestor resampling of particle 0
        a = gsl_ran_discrete(rng,ar_rdisc);
        particleSystem[t][0].ancestor = a;
        
        // Resampling particles 1:nparticles
        for(i=1;i<nparticles;i++){
            a = gsl_ran_discrete(rng,rdisc);
            particleSystem[t][i].ancestor = a;
        }

        // Copy ancestors to device memory
        for(i=0;i<nparticles;i++)
            particleArray.ancestor_h(i, t) = particleSystem[t][i].ancestor;
        Kokkos::deep_copy(particleArray.ancestor, particleArray.ancestor_h);

        // Pregenerate noise so we can have noise that is not dependent on the thread execution order
        double dt = 1.0/constants->sampling_frequency;
        double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));  
        for(i=0;i<nparticles;i++){
            if(i != 0) {
                u_noise[i] = gsl_rng_uniform(rng);
                g_noise[i] = gsl_ran_gaussian(rng,sigma_B_posterior);
            }
        }

        // Copy noise to device memory
        for(i=0;i<nparticles;i++){
            g_noise_h(i) = g_noise[i];
            u_noise_h(i) = u_noise[i];
        }
        Kokkos::deep_copy(g_noise_view, g_noise_h);
        Kokkos::deep_copy(u_noise_view, u_noise_h);
    
        // Move and weight particles
        // #pragma omp parallel for schedule(static)
        // for(i=0;i<nparticles;i++){
        //     a = particleSystem[t][i].ancestor;
        //     if(constants->KNOWN_SPIKES){
        //         move_and_weight_GTS(particleSystem[t][i], particleSystem[t-1][a], data_y(t), par, g_noise[i], u_noise[i], i==0);
        //     } else {
        //         move_and_weight(particleSystem[t][i], particleSystem[t-1][a], data_y(t), par, g_noise[i], u_noise[i], i==0);
        //     }
        // }

        // die = !particleArray.check_particle_system(t-1, particleSystem);
        
        // Copy back to host so we can theck things.
        particleArray.move_and_weight(t, data_y_view, par, constants, g_noise_view, u_noise, params);

        // die = !particleArray.check_particle_system(t, particleSystem);
        
        gsl_ran_discrete_free(rdisc);
        gsl_ran_discrete_free(ar_rdisc);
        
    }
		
    // Copy the partilce system back to the CPU
    particleArray.copy_to_host();
    for(t=1;t<TIME;++t)
        for(i=0;i<nparticles;i++)
            particleArray.get_particle(t, i, particleSystem[t][i]);

    // Now use the last particle set to resample the new trajectory
    for(i=0;i<nparticles;i++){
        logW[i] = particleSystem[TIME-1][i].logWeight;
    }

    utils::w_from_logW(logW,w,nparticles);

    gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
    i = gsl_ran_discrete(rng,rdisc);

    t=TIME;

    if(die)
        return;

    while(t>0){
        t--;
        // if(t>0) cout<<" -> "<<i<<" -> "<<particleSystem[t][i].ancestor;
        traj_out.B(t)=particleSystem[t][i].B;
        traj_out.burst(t)=particleSystem[t][i].burst;
        traj_out.C(t)=model->getDFF(particleSystem[t][i].C);
        traj_out.S(t)=particleSystem[t][i].S;
        traj_out.Y(t)=data_y(t);
        
        if(t>0){
            i=particleSystem[t][i].ancestor;
        }
    }
 
}
 
