import numpy as np
cimport numpy as np

include "armapy.pxi"

cdef extern from "dp_bayes.hpp":
    cdef struct dp_result:
        uvec J
        double NDist
        vec q2LL

    dp_result dp_bayes(vec q1, vec q1L, vec q2L, int times, int cut)


def DP(np.ndarray[double, ndim=1, mode="c"] q1,np.ndarray[double, ndim=1, mode="c"] q2, np.ndarray[double, ndim=1, mode="c"] q2L, int times, int cut):
    cdef vec q1a = dndtovec(q1)
    cdef vec q2a = dndtovec(q2)
    cdef vec q2La = dndtovec(q2L)

    cdef dp_result result
    result = dp_bayes(q1a, q2a, q2La, times, cut)

    cdef np.ndarray[double, ndim=1] q2LL=dvectond(result.q2LL)
    cdef np.ndarray[int, ndim=1] J=duvectond(result.J)

    return(J, result.NDist, q2LL)
