//Required pybind libraries for expected conversions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//External libraries called by pgas
#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
//Type conversion for armadillo library in pybind11
#include "../include/arma_pybind11.h"
//Bound modules
#include "../include/Analyzer.h"
//In case any other functions or classes need to be exposed to python
#include "../include/GCaMP_model.h"
/*#include "../include/constants.h"
#include "../include/mvn.h"
#include "../include/param.h"
#include "../include/particle.h"
#include "../include/utils.h"*/


namespace py = pybind11;

//method to extract final array entries as numpy array
py::array_t<double> get_final_params(Analyzer& analyzer) {
  // Create a NumPy array from the std::vector
  py::array_t<double> result(analyzer.final_params.size());
  std::copy(analyzer.final_params.begin(), analyzer.final_params.end(), result.mutable_data());
  return result;
}

PYBIND11_MODULE(pgas_bound, m) {
	// bindings for Analyzer.cpp
		py::class_<Analyzer>(m, "Analyzer")
        .def(py::init<const std::string&, const std::string&, const std::string&, unsigned int, const std::string&, unsigned int,
                      const std::string&, bool, unsigned int, bool, const std::string&, bool, bool, unsigned int, const std::string&>(),
             py::arg("data_file"), py::arg("constants_file"), py::arg("output_folder"), py::arg("column"), py::arg("tag"),
             py::arg("niter") = 0, py::arg("trainedPriorFile") = "", py::arg("append") = false, py::arg("trim") = 1,
             py::arg("verbose") = true, py::arg("gtSpike_file") = "", py::arg("has_trained_priors") = false, py::arg("has_gtspikes") = false,
             py::arg("maxlen") = 0, py::arg("Gparam_file") = "")
        .def("run", &Analyzer::run)
				.def("get_final_params", &get_final_params, "Get final parameters as a NumPy array");
	
	//In case any other functions or classes need to be exposed to python during future iterations
	/*// bindings for constants.cpp
    py::class_<constpar>(m, "constpar")
        .def(py::init<const std::string &>())
        .def("print", &constpar::print)
        .def_readwrite("bm_sigma", &constpar::bm_sigma)
        .def_readwrite("alpha_sigma2", &constpar::alpha_sigma2)
        .def_readwrite("beta_sigma2", &constpar::beta_sigma2)
        .def_readwrite("alpha_rate_b0", &constpar::alpha_rate_b0)
        .def_readwrite("beta_rate_b0", &constpar::beta_rate_b0)
        .def_readwrite("alpha_rate_b1", &constpar::alpha_rate_b1)
        .def_readwrite("beta_rate_b1", &constpar::beta_rate_b1)
        .def_readwrite("alpha_w01", &constpar::alpha_w01)
        .def_readwrite("beta_w01", &constpar::beta_w01)
        .def_readwrite("alpha_w10", &constpar::alpha_w10)
        .def_readwrite("beta_w10", &constpar::beta_w10)
        .def_readwrite("G_tot_mean", &constpar::G_tot_mean)
        .def_readwrite("G_tot_sd", &constpar::G_tot_sd)
        .def_readwrite("gamma_mean", &constpar::gamma_mean)
        .def_readwrite("gamma_sd", &constpar::gamma_sd)
        .def_readwrite("DCaT_mean", &constpar::DCaT_mean)
        .def_readwrite("DCaT_sd", &constpar::DCaT_sd)
        .def_readwrite("Rf_mean", &constpar::Rf_mean)
        .def_readwrite("Rf_sd", &constpar::Rf_sd)
        .def_readwrite("gam_in_mean", &constpar::gam_in_mean)
        .def_readwrite("gam_in_sd", &constpar::gam_in_sd)
        .def_readwrite("gam_out_mean", &constpar::gam_out_mean)
        .def_readwrite("gam_out_sd", &constpar::gam_out_sd)
        .def_readwrite("G_tot_prop_sd", &constpar::G_tot_prop_sd)
        .def_readwrite("gamma_prop_sd", &constpar::gamma_prop_sd)
        .def_readwrite("DCaT_prop_sd", &constpar::DCaT_prop_sd)
        .def_readwrite("Rf_prop_sd", &constpar::Rf_prop_sd)
        .def_readwrite("gam_in_prop_sd", &constpar::gam_in_prop_sd)
        .def_readwrite("gam_out_prop_sd", &constpar::gam_out_prop_sd)
        .def_readwrite("sampling_frequency", &constpar::sampling_frequency)
        .def_readwrite("MOVE_SPIKES", &constpar::MOVE_SPIKES)
        .def_readwrite("SAMPLE_KINETICS", &constpar::SAMPLE_KINETICS)
        .def_readwrite("SAMPLE_SPIKES", &constpar::SAMPLE_SPIKES)
        .def_readwrite("SAMPLE_PARAMETERS", &constpar::SAMPLE_PARAMETERS)
        .def_readwrite("CROP_TRACE", &constpar::CROP_TRACE)
        .def_readwrite("KNOWN_SPIKES", &constpar::KNOWN_SPIKES)
        .def_readwrite("seed", &constpar::seed)
        .def_readwrite("niter", &constpar::niter)
        .def_readwrite("nparticles", &constpar::nparticles)
        .def_readwrite("t_min", &constpar::t_min)
        .def_readwrite("t_max", &constpar::t_max)
        .def_readwrite("baseline_frames", &constpar::baseline_frames)
        .def_readwrite("nospike_before", &constpar::nospike_before)
        .def_readwrite("normalization_index", &constpar::normalization_index)
        .def_readwrite("output_folder", &constpar::output_folder)
        .def_readwrite("tag", &constpar::tag)
        .def_readwrite("TSMode", &constpar::TSMode)
        .def("set_time_scales", &constpar::set_time_scales);*/
				
	/*// bindings for mvn.cpp
    py::class_<mvn>(m, "mvn")
        .def(py::init<>())
        .def(py::init<gsl_rng *>())
        .def(py::init<gsl_rng *, const arma::vec &, const arma::mat &>())
        .def("setRNG", &mvn::setRNG)
        .def("setParams", &mvn::setParams)
        .def("printL", &mvn::printL)
        .def("rmvn_mat", py::overload_cast<int>(&mvn::rmvn_mat))
        .def("rmvn_vec", py::overload_cast<>(&mvn::rmvn_vec))
        .def("rmvn_vec", py::overload_cast<const arma::vec &, const arma::mat &>(&mvn::rmvn_vec))
        .def("rmvn_mat", py::overload_cast<const arma::vec &, const arma::mat &, int>(&mvn::rmvn_mat))
        .def("dmvn", py::overload_cast<const arma::vec &>(&mvn::dmvn))
        .def("dmvn", py::overload_cast<const arma::vec &, const arma::vec &, const arma::mat>(&mvn::dmvn))
        .def("dmvn_log", py::overload_cast<const arma::vec &>(&mvn::dmvn_log))
        .def("dmvn_log", py::overload_cast<const arma::vec &, const arma::vec &, const arma::mat>(&mvn::dmvn_log))
        .def("getInvCov", &mvn::getInvCov)
        .def("rWishart", py::overload_cast<int>(&mvn::rWishart))
        .def("rWishart", py::overload_cast<int, const arma::mat &>(&mvn::rWishart))
        .def("rInvWishart", py::overload_cast<int>(&mvn::rInvWishart))
        .def("rInvWishart", py::overload_cast<int, const arma::mat &>(&mvn::rInvWishart));*/
				
	// bindings for GCaMP_model.cpp			
		py::class_<GCaMP>(m, "GCaMP")
        .def(py::init<double, double, double, double, double, double, std::string>())
				.def(py::init<std::string, std::string>(), py::arg("Gparam_file"), py::arg("Cparam_file"))
				.def(py::init<const arma::vec,const arma::vec>(), py::arg("Gparams_in"), py::arg("Cparams_in"))
        .def("setParams", &GCaMP::setParams)
        .def("setGmat", &GCaMP::setGmat)
        .def("flux", &GCaMP::flux)
        .def("steady_state", &GCaMP::steady_state)
        .def("init", &GCaMP::init)
        .def("setState", &GCaMP::setState)
        .def("setTimeStepMode", &GCaMP::setTimeStepMode)
        .def("evolve", py::overload_cast<double, int, const arma::vec&>(&GCaMP::evolve))
        .def("evolve", py::overload_cast<double, int>(&GCaMP::evolve))
        .def("fixedStep", &GCaMP::fixedStep)
        .def("fixedStep_LA", &GCaMP::fixedStep_LA)
        .def("getDFF", py::overload_cast<>(&GCaMP::getDFF))
        .def("getDFF", py::overload_cast<const arma::vec&>(&GCaMP::getDFF))
        .def("getAmplitude", &GCaMP::getAmplitude)
				.def("integrateOverTime", &GCaMP::integrateOverTime, py::arg("time"), py::arg("spike_times"))
				.def("integrateOverTime2", &GCaMP::integrateOverTime2, py::arg("time"), py::arg("spike_times"))
        .def("getDFFValues", &GCaMP::getDFFValues);
				// .def_readwrite("DFF", &GCaMP::DFF); // Removed to prevent crosstalk with integrateOverTime
				
	/*// bindings for param.cpp
		py::class_<param>(m, "param")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def("print", &param::print)
        .def("write", &param::write)
        .def("logPrior", &param::logPrior)
        .def("__copy__", [](const param &self) { return param(self); })
        .def("__deepcopy__", [](const param &self, py::dict) { return param(self); });*/
	
	/*//bindings for classes present in particle.cpp
		// Particle class bindings
    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def("print", &Particle::print)
        .def_readwrite("C", &Particle::C)
        .def_readwrite("B", &Particle::B)
        .def_readwrite("burst", &Particle::burst)
        .def_readwrite("S", &Particle::S)
        .def_readwrite("ancestor", &Particle::ancestor)
        .def_readwrite("logWeight", &Particle::logWeight)
        .def_readwrite("index", &Particle::index);
        //.def(py::self = py::self);

    // Trajectory class bindings
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<unsigned int, std::string>())
        .def("simulate", &Trajectory::simulate)
        .def("write", &Trajectory::write)
        .def("logp", &Trajectory::logp)
        .def_readwrite("filename", &Trajectory::filename)
        .def_readwrite("size", &Trajectory::size)
        .def_readwrite("B", &Trajectory::B)
        .def_readwrite("burst", &Trajectory::burst)
        .def_readwrite("C", &Trajectory::C)
        .def_readwrite("S", &Trajectory::S)
        .def_readwrite("Y", &Trajectory::Y);
        //.def(py::self = py::self);

    // SMC class bindings
    py::class_<SMC>(m, "SMC")
        .def(py::init<std::string, int, constpar&, bool, int, unsigned int, std::string>())
        .def(py::init<arma::vec&, constpar&, bool>())
        .def("rmu", &SMC::rmu)
        .def("logf", &SMC::logf)
        .def("move_and_weight_GTS", &SMC::move_and_weight_GTS)
        .def("move_and_weight", &SMC::move_and_weight)
        .def("sampleParameters", &SMC::sampleParameters)
        .def("PF", &SMC::PF)
				.def("PGAS", &SMC::PGAS)
        .def_readwrite("nparticles", &SMC::nparticles)
        .def_readwrite("TIME", &SMC::TIME)
        //.def_readwrite("constants", &SMC::constants)
        .def_readwrite("model", &SMC::model)
        .def_readwrite("particleSystem", &SMC::particleSystem)
        .def_readwrite("data_time", &SMC::data_time)
        .def_readwrite("data_y", &SMC::data_y);
        //.def_readwrite("verbose", &SMC::verbose);*/
	
	/*// bindings for functions contained in utils.cpp
		m.def("w_from_logW", [](const std::vector<double>& src) {
        std::vector<double> out(src.size());
        utils::w_from_logW(src.data(), out.data(), src.size());
        return out;
    }, "Convert log weights to weights");

    m.def("Z_from_logW", [](const std::vector<double>& src) {
        return utils::Z_from_logW(src.data(), src.size());
    }, "Compute normalization constant from log weights");

    m.def("Z_factor", &utils::Z_factor, "Compute the Z factor for given parameters");

    m.def("gsl_gamma_logpdf", &stats::gsl_gamma_logpdf, "Compute the log PDF of the gamma distribution");*/
}