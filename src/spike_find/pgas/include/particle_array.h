#ifndef PARTICLE_ARRAY_H
#define PARTICLE_ARRAY_H

#include "particle.h"
#include <Kokkos_Core.hpp>

// typedef Kokkos::Serial   ExecSpace;
// typedef Kokkos::Serial   MemSpace;
// typedef Kokkos::OpenMP   ExecSpace;
// typedef Kokkos::OpenMP        MemSpace;
typedef Kokkos::Cuda          ExecSpace;
typedef Kokkos::CudaSpace     MemSpace;

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

#endif