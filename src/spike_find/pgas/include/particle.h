#ifndef PARTICLE_H
#define PARTICLE_H

#include<iostream>
#include"param.h"
#include"constants.h"
#include"mvn.h"
#include<armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include<string>
#include"include/GCaMP_model.h"

#include <Kokkos_Core.hpp>


using namespace std;

class Particle
{
    public:
        Particle& operator=(const Particle&);
        double B;
        int burst;
        arma::vec C;
        int S;
        double logWeight;
        int ancestor;
        void print();
        Particle();
        int index;
};

// typedef Kokkos::Serial   ExecSpace;
// typedef Kokkos::Serial   MemSpace;
typedef Kokkos::OpenMP   ExecSpace;
typedef Kokkos::OpenMP        MemSpace;
// typedef Kokkos::Cuda          ExecSpace;
// typedef Kokkos::CudaSpace     MemSpace;

typedef Kokkos::View<double*, MemSpace>   VectorType;
typedef Kokkos::View<double**, MemSpace>  MatrixType;
typedef Kokkos::View<int*, MemSpace>   IntVectorType;
typedef Kokkos::View<int**, MemSpace>  IntMatrixType;

// A class to store N particles over time. First dimension of all arrays is time, second is particle index
class ParticleArray
{
    public:
        const int maxspikes = 2;

        MatrixType B;
        IntMatrixType burst;
        Kokkos::View<double**[12], MemSpace> C;
        IntMatrixType S;
        MatrixType logWeight;
        IntMatrixType ancestor;
        IntMatrixType index;

        // Host mirrors
        MatrixType::HostMirror B_h;
        IntMatrixType::HostMirror burst_h;
        Kokkos::View<double**[12], MemSpace>::HostMirror C_h;
        IntMatrixType::HostMirror S_h;
        MatrixType::HostMirror logWeight_h;
        IntMatrixType::HostMirror ancestor_h;
        IntMatrixType::HostMirror index_h;

        ParticleArray(int N, int T);

        void set_particle(int t, int idx, const Particle &p);
        void get_particle(int t, int idx, Particle &p);

        void copy_to_device();
        void copy_to_host();

        void move_and_weight(int t, VectorType y, const param &par, constpar *constants, 
                             VectorType g_noise, std::vector<double> & u_noise, const GCaMP_params & params); 

        void calc_ancestor_resampling(int t, const param &par, constpar *constants);

        double logf(int part_idx_in, int t_in, int part_idx_out, int t_out, const param &par, constpar *constants);

        bool check_particle_system(int t, std::vector<Particle*> & particleSystem);

        // Number of particles
        int N;

        // Number of time points
        int T;

    // private:

        MatrixType log_probs;
        MatrixType::HostMirror log_probs_h;

        MatrixType probs;
        MatrixType::HostMirror probs_h;

        IntVectorType discrete;
        IntVectorType::HostMirror discrete_h;

        VectorType logW;
        VectorType::HostMirror logW_h;

        VectorType ar_logW;
        VectorType::HostMirror ar_logW_h;

};

class Trajectory
{
    public:
        Trajectory(unsigned int, string);
        unsigned int size;
        arma::vec B;
        arma::ivec burst;
        arma::vec C;  // this is only DFF
        arma::vec S;
        arma::vec Y;

        void simulate(gsl_rng*, const param&, const constpar*);
        void simulate_doubles(gsl_rng*, const param&, const constpar*);
        void simulate_doubles2(gsl_rng*, const param&, const constpar*,int);
        void write(ofstream &, unsigned int idx=0);
        double logp(const param*, const constpar*);
        Trajectory& operator=(const Trajectory&);

        string filename;

};

class SMC{
    public:

        SMC(string,int,constpar&,bool,int seed=0, unsigned int maxlen=0, string GParam_file=""); 
        SMC(arma::vec&,constpar&,bool verbose); 

        double logf(const Particle&,const Particle&, const param&);
        void move_and_weight(Particle&, const Particle&, double, const param &, double, double, bool); // move particles and weight
        void move_and_weight_GTS(Particle&, const Particle&, double, const param &, double, double, bool); // move particles and weight

        void rmu(Particle &, double, const param &, bool); // draw initial state
        
        void PGAS(const param &theta, const Trajectory &traj_in, Trajectory &traj_out);

        void PF(const param &theta); // to draw the initial trajectory

        void MCMC(int iterations);

        void sampleParameters(const param&, param &, Trajectory&);

        unsigned int nparticles;
        unsigned int TIME;

        arma::vec data_y;
        arma::vec data_time;

        vector<Particle*> particleSystem;

        GCaMP* model;

    private:
        gsl_rng *rng;
        arma::mat tracemat;
        constpar *constants;
        bool verbose;

};

#endif
