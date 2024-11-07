#ifndef ARMA_PYBIND11_H
#define ARMA_PYBIND11_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace py = pybind11;
namespace pybind11 {
namespace detail {

// Type caster for arma::mat
/* template <> struct type_caster<arma::mat> {
public:
    PYBIND11_TYPE_CASTER(arma::mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        if (!isinstance<array_t<double>>(src)) {
            return false;
        }
        auto buf = array_t<double>::ensure(src);
        if (!buf) {
            return false;
        }
        auto dims = buf.ndim();
        if (dims != 2) {
            return false;
        }

        auto rows = buf.shape()[0];
        auto cols = buf.shape()[1];
        value = arma::mat(rows, cols);
        std::memcpy(value.memptr(), buf.data(), rows * cols * sizeof(double));
        return true;
    }

    static handle cast(const arma::mat &src, return_value_policy, handle) {
        array_t<double> array({src.n_rows, src.n_cols}, src.memptr());
        return array.release();
    }
}; */

// Type caster for arma:mat
template <> struct type_caster<arma::mat> {
public:
    PYBIND11_TYPE_CASTER(arma::mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        if (!isinstance<array_t<double>>(src)) {
            return false;
        }
        py::array_t<double> buf = py::array_t<double>::ensure(src);
        if (!buf) {
            return false;
        }
        if (buf.ndim() != 2) {
            return false;
        }

        size_t rows = buf.shape(0);
        size_t cols = buf.shape(1);

        // Check if the array is Fortran-contiguous
        if (buf.strides(0) != sizeof(double) || buf.strides(1) != sizeof(double)*rows) {
            return false;
        }

        value = arma::mat(rows, cols);
        std::memcpy(value.memptr(), buf.data(), rows * cols * sizeof(double));
        return true;
    }

    // Conversion from C++ to Python
    static handle cast(const arma::mat &src, return_value_policy, handle) {
        // Define shape and strides with explicit casts to py::ssize_t
        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(src.n_rows),
            static_cast<py::ssize_t>(src.n_cols)
        };
        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(sizeof(double)),
            static_cast<py::ssize_t>(sizeof(double) * src.n_rows)
        };

        // Debugging: Print shape and strides
        std::cout << "Casting arma::mat to NumPy array with shape (" 
                    << shape[0] << ", " << shape[1] << ") and strides (" 
                    << strides[0] << ", " << strides[1] << ")" << std::endl;

        // Create a NumPy array without copying the data
        py::array_t<double> array(shape, strides, src.memptr(), py::none());

        // Make a copy to ensure ownership in Python
        py::array_t<double> copy = array.attr("copy")();

        // Return the copied array as a handle
        return copy.release();
    }
};

// Type caster for arma::vec
template <> struct type_caster<arma::vec> {
public:
    PYBIND11_TYPE_CASTER(arma::vec, _("numpy.ndarray"));

    bool load(handle src, bool) {
			//std::cout << "Type caster called for arma::vec with type: " << pybind11::str(src.get_type()) << std::endl;
        if (!isinstance<array_t<double>>(src)) {
					//std::cout << "Not a numpy array of doubles." << std::endl;
            return false;
        }
        auto buf = array_t<double>::ensure(src);
        if (!buf) {
					//std::cout << "Failed to ensure numpy array." << std::endl;
            return false;
        }
        auto dims = buf.ndim();
        if (dims != 1) {
					//std::cout << "Array is not 1-dimensional." << std::endl;
            return false;
        }

        auto size = buf.shape()[0];
        value = arma::vec(size);
        std::memcpy(value.memptr(), buf.data(), size * sizeof(double));
				  //std::cout << "Successfully casted to arma::vec with size: " << size << std::endl;
        return true;
    }

    static handle cast(const arma::vec &src, return_value_policy, handle) {
        array_t<double> array(src.n_rows, src.memptr());
        return array.release();
    }
};

template <typename T>
struct type_caster<std::vector<std::vector<T>>> {
public:
    PYBIND11_TYPE_CASTER(std::vector<std::vector<T>>, _("List[List[" PYBIND11_STRINGIFY(T) "]]"));

    bool load(handle src, bool) {
        // Not needed for output conversion
        return false;
    }

    static handle cast(const std::vector<std::vector<T>>& src, return_value_policy /* policy */, handle /* parent */) {
        // Define the alias within the method's scope
        namespace py = pybind11;

        if (src.empty()) {
            // Return an empty array
            return py::array(py::dtype::of<T>(), {0, 0}).release();
        }

        size_t rows = src.size();
        size_t cols = src[0].size();

        // Create a NumPy array with the appropriate shape
        py::array_t<T> result({rows, cols});
        auto buffer = result.request();
        T* ptr = static_cast<T*>(buffer.ptr);

        // Copy data into the NumPy array
        for (size_t i = 0; i < rows; ++i) {
            if (src[i].size() != cols) {
                throw std::runtime_error("All inner vectors must have the same size");
            }
            std::copy(src[i].begin(), src[i].end(), ptr + i * cols);
        }

        return result.release();
    }
};


}  // namespace detail
}  // namespace pybind11

#endif  // ARMA_PYBIND11_H
