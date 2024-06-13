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

// typedef Kokkos::OpenMP   ExecSpace;
// typedef Kokkos::OpenMP        MemSpace;
typedef Kokkos::Cuda          ExecSpace;
typedef Kokkos::CudaSpace     MemSpace;

// typedef Kokkos::LayoutLeft   Layout;
typedef Kokkos::LayoutRight  Layout;

typedef Kokkos::View<double*, Layout, MemSpace>   VectorType;
typedef Kokkos::View<double**, Layout, MemSpace>  MatrixType;
typedef Kokkos::View<int*, Layout, MemSpace>   IntVectorType;
typedef Kokkos::View<int**, Layout, MemSpace>  IntMatrixType;

// A class to store N particles over time. First dimension of all arrays is time, second is particle index
class ParticleArray
{
    public:
        MatrixType B;
        IntMatrixType burst;
        Kokkos::View<double**[12]> C;
        IntMatrixType S;
        MatrixType logWeight;
        IntMatrixType ancestor;
        IntMatrixType index;

        // Host mirrors
        MatrixType::HostMirror B_h;
        IntMatrixType::HostMirror burst_h;
        Kokkos::View<double**[12]>::HostMirror C_h;
        IntMatrixType::HostMirror S_h;
        MatrixType::HostMirror logWeight_h;
        IntMatrixType::HostMirror ancestor_h;
        IntMatrixType::HostMirror index_h;

        ParticleArray(int N, int T);

        void set_particle(int t, int idx, const Particle &p);
        void get_particle(int t, int idx, Particle &p);

        void copy_to_device();
        void copy_to_host();

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
