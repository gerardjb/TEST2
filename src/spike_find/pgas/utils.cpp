#include<algorithm>
#include<gsl/gsl_math.h>
#include<armadillo>
#include"include/utils.h"
#include<gsl/gsl_sf.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

namespace utils {

    void w_from_logW(const double *src, double* out, unsigned int n){
        double maxLogWeight = *max_element(src,src+n);
        for(unsigned int i=0;i<n;i++) out[i] = exp(src[i]-maxLogWeight);
    }

    double Z_from_logW(const double *src, unsigned int n){
        double maxLogWeight = *max_element(src,src+n);
        double Z=0;
        for(unsigned int i=0; i<n; i++){
            Z+=exp(src[i]-maxLogWeight);
        }
        return(Z*exp(maxLogWeight));
    }

    double Z_factor(double x,double xprop,double sigma,double cutoff){
        return(log(gsl_sf_erf_Q((cutoff-x)/sigma)/gsl_sf_erf_Q((cutoff-xprop)/sigma)));
    }

    // Function to read values from a file and store them in a dynamically allocated array
    double* read_vector_from_file(const string& filename, size_t& size) {
        ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return nullptr;
        }

        vector<double> values;
        double value;
        while (file >> value) {
            values.push_back(value);
        }
        file.close();

        size = values.size();
        double* read_in = static_cast<double*>(std::malloc(size * sizeof(double)));
        if (read_in == nullptr) {
            std::cerr << "Memory allocation failed." << std::endl;
            return nullptr;
        }

        for (size_t i = 0; i < size; ++i) {
            read_in[i] = values[i];
        }

        return read_in;
    }

    // Create an Armadillo matrix of size (rows, cols) from double*
    arma::mat double_to_arma_mat(double* data, int rows, int cols) {
        
        arma::mat arma_mat(rows, cols);
        
        // Copy data from double* to Armadillo matrix
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                arma_mat(i, j) = data[i * cols + j];
            }
        }
        
        return arma_mat;
    }

    // Function to convert arma::mat to double*
    double* arma_mat_to_double(const arma::mat& matrix, int rows, int cols) {
        // Allocate memory for double* array
        double* data = new double[rows * cols];
        
        // Copy data from Armadillo matrix to double* array
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i * cols + j] = matrix(i, j);
            }
        }
        
        return data;
    }

    //Function for multiplying 
    void unrolled_matrix_vector_multiplication(double* matrix, int rows, int cols, double* vector, double* result) {  
        // Check for compatible dimensions
        if (cols != vector[0]) {
            std::cerr << "Error: Matrix columns (" << cols << ") don't match vector size (" << vector[0] << ")" << std::endl;
            return;
        }

        // Loop through each row of the matrix
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;

            // Unroll the loop for better performance (adjust unrolling factor as needed)
            for (int j = 0; j < cols / 4; ++j) {
                // Access elements using pointer arithmetic
                sum += matrix[i * cols + 4 * j] * vector[4 * j];
                sum += matrix[i * cols + 4 * j + 1] * vector[4 * j + 1];
                sum += matrix[i * cols + 4 * j + 2] * vector[4 * j + 2];
                sum += matrix[i * cols + 4 * j + 3] * vector[4 * j + 3];
            }

            // Handle remaining elements (if cols is not divisible by 4)
            for (int j = cols / 4 * 4; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
            }

            result[i] = sum;
        }
    }
    // Utility function to perform Gaussian elimination and find null space
    double** null_space(double* mat_vec, int rows, int cols) {
        // Create an augmented matrix [A|I]
        double** augmented = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            augmented[i] = new double[cols + rows];
            for (int j = 0; j < cols; ++j) {
                augmented[i][j] = mat_vec[i * cols + j]; // Copy matrix A into augmented
            }
            for (int j = cols; j < cols + rows; ++j) {
                augmented[i][j] = (i == j - cols) ? 1.0 : 0.0; // Identity matrix I
            }
        }
        
        // Perform Gaussian elimination
        for (int i = 0; i < rows; ++i) {
            // Find the pivot row
            int pivot = i;
            for (int j = i + 1; j < rows; ++j) {
                if (std::fabs(augmented[j][i]) > std::fabs(augmented[pivot][i])) {
                    pivot = j;
                }
            }
            if (std::fabs(augmented[pivot][i]) < 1e-10) {
                continue; // Skip if the pivot is too small (i.e., the column is already null)
            }
            
            // Swap the current row with the pivot row
            std::swap(augmented[i], augmented[pivot]);
            
            // Normalize the pivot row
            double pivot_value = augmented[i][i];
            for (int j = 0; j < cols + rows; ++j) {
                augmented[i][j] /= pivot_value;
            }
            
            // Eliminate the current column in other rows
            for (int j = 0; j < rows; ++j) {
                if (j != i) {
                    double factor = augmented[j][i];
                    for (int k = 0; k < cols + rows; ++k) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
        
        // Extract the null space (right side of the augmented matrix)
        int nullspace_count = 0;
        for (int i = 0; i < rows; ++i) {
            bool is_zero_row = true;
            for (int j = 0; j < cols; ++j) {
                if (std::fabs(augmented[i][j]) >= 1e-10) {
                    is_zero_row = false;
                    break;
                }
            }
            if (is_zero_row) {
                nullspace_count++;
            }
        }
        
        // Allocate memory for null space vectors
        double** nullspace = new double*[nullspace_count];
        int nullspace_index = 0;
        for (int i = 0; i < rows; ++i) {
            bool is_zero_row = true;
            for (int j = 0; j < cols; ++j) {
                if (std::fabs(augmented[i][j]) >= 1e-10) {
                    is_zero_row = false;
                    break;
                }
            }
            if (is_zero_row) {
                nullspace[nullspace_index] = new double[rows];
                for (int j = 0; j < rows; ++j) {
                    nullspace[nullspace_index][j] = augmented[i][cols + j];
                }
                nullspace_index++;
            }
        }
        
        // Deallocate augmented matrix memory
        for (int i = 0; i < rows; ++i) {
            delete[] augmented[i];
        }
        delete[] augmented;
        
        return nullspace;
    }
    
}

namespace stats {
    double gsl_gamma_logpdf(const double x, const double shape, const double scale){
        return ( (shape-1)*log(x) - x/scale );
    }
}

