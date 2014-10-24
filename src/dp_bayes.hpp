#include "armadillo"

struct dp_result {
    arma::uvec J;
    double NDist;
    arma::vec q2LL;
};

dp_result dp_bayes(arma::vec q1, arma::vec q1L, arma::vec q2L, int times, int cut);

