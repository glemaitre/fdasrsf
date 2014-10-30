#include "armadillo"

using namespace arma;

struct simu_result {
    vec best_match;
    mat match_collect;
    vec dist_collect;
    vec kappafamily;
    vec log_posterior;
    float dist_min;
};

simu_result simuiter(int iter, int p, vec qt1_5, vec qt2_5, int L, float tau, int times, float kappa, float alpha, float beta, float powera, float dist, float dist_min, vec best_match, vec match, int thin, int cut);
 
