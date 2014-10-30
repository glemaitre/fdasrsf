import numpy as np
cimport numpy as np
cimport cbayes as cb

include "armapy.pxi"


cdef extern from "dp_bayes.hpp":
    cdef struct dp_result:
        uvec J
        double NDist
        vec q2LL

    dp_result dp_bayes(vec q1, vec q1L, vec q2L, int times, int cut)

cdef extern from "bayes_simuiter.hpp":
    cdef struct simu_result:
        vec best_match
        mat match_collect
        vec dist_collect
        vec kappafamily
        vec log_posterior
        float dist_min

    simu_result simuiter(int iter, int p, vec qt1_5, vec qt2_5, int L,
                         float tau, int times, float kappa, float alpha,
                         float beta, float powera, float dist, float dist_min,
                         vec best_match, vec match, int thin, int cut)


def DP(np.ndarray[double, ndim=1, mode="c"] q1,
       np.ndarray[double, ndim=1, mode="c"] q2,
       np.ndarray[double, ndim=1, mode="c"] q2L, int times, int cut):

    cdef vec q1a = dndtovec(q1)
    cdef vec q2a = dndtovec(q2)
    cdef vec q2La = dndtovec(q2L)

    cdef dp_result result
    result = cb.dp_bayes(q1a, q2a, q2La, times, cut)

    cdef np.ndarray[double, ndim=1] q2LL=dvectond(result.q2LL)
    cdef np.ndarray[int, ndim=1] J=duvectond(result.J)

    return(J, result.NDist, q2LL)


def simuiterb(int iters, int p, np.ndarray[double, ndim=1, mode="c"] q1,
              np.ndarray[double, ndim=1, mode="c"] q2, int L, float tau,
              int times, float kappa, float alpha, float beta, float powera,
              float dist, float dist_min,
              np.ndarray[double, ndim=1, mode="c"] best_match,
              np.ndarray[double, ndim=1, mode="c"] match, int thin, int cut):

    cdef vec qt1_5 = dndtovec(q1)
    cdef vec qt2_5 = dndtovec(q2)
    cdef vec best_match1 = dndtovec(best_match)
    cdef vec match1 = dndtovec(match)

    cdef simu_result result
    result = cb.simuiter(iters, p, qt1_5, qt2_5, L, tau, times, kappa, alpha,
                         beta, powera, dist, dist_min, best_match1, match1,
                         thin, cut)

    cdef np.ndarray[double, ndim=1] bm_out=dvectond(result.best_match)
    cdef np.ndarray[double, ndim=2] mc_out=dmattond(result.match_collect)
    cdef np.ndarray[double, ndim=1] dc_out=dvectond(result.dist_collect)
    cdef np.ndarray[double, ndim=1] kf_out=dvectond(result.kappafamily)
    cdef np.ndarray[double, ndim=1] lp_out=dvectond(result.log_posterior)

    out = dict(best_match=bm_out, match_collect=mc_out, dist_collect=dc_out,
               kappafamily=kf_out, log_posterior=lp_out,
               dist_min=result.dist_min)


