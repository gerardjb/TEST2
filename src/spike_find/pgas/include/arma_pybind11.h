#ifndef ARMA_PYBIND11_H
#define ARMA_PYBIND11_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace pybind11 {
namespace detail {

// Type caster for arma::mat
template <> struct type_caster<arma::mat> {
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

}  // namespace detail
}  // namespace pybind11

#endif  // ARMA_PYBIND11_H
