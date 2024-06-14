#ifndef ANALYZER_H
#define ANALYZER_H

#include <string>

class Analyzer {
public:

    Analyzer(const arma::vec& time,const arma::vec& data, const std::string& constants_file, const std::string& output_folder,
                 unsigned int column, const std::string& tag, unsigned int niter = 0, const std::string& trainedPriorFile = "",
                 bool append = false, unsigned int trim = 1, bool verbose = true, const std::string& gtSpike_file = "",
                 bool has_trained_priors = false, bool has_gtspikes = false, unsigned int maxlen = 0, const std::string& Gparam_file = "");

    void run();
		std::vector<double> final_params;

private:
    arma::vec time;
    arma::vec data;
    std::string constants_file;
    std::string output_folder;
    unsigned int column;
    std::string gtSpike_file;
    std::string tag;

    std::string trainedPriorFile;
    bool has_trained_priors;
    bool has_gtspikes;
    std::string Gparam_file;
    bool append;
    unsigned int niter;
    unsigned int trim;
    bool verbose;
    unsigned int maxlen;
		
};

#endif // ANALYZER_H
