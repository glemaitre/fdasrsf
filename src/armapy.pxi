from cpython.version cimport PY_MAJOR_VERSION
import numpy as n
cimport numpy as n


cdef extern from "string" namespace "std":
    cdef cppclass string:
        string()
        string(char*)
cdef extern from "complex" namespace "std":
    cdef cppclass complex[T]:
        complex(T,T)
        complex()
        T real()
        T imag()

ctypedef unsigned int us

cdef extern from "<armadillo>" namespace "arma":
    cdef cppclass vec:
        vec(int)
        vec(double*,int,bool,bool)
        void raw_print()
        vec()
        double operator()(int)
        void zeros()
        double* memptr()
        int size()
    cdef cppclass uvec:
        uvec(int)
        uvec(us*,int,bool,bool)
        void raw_print()
        uvec()
        double operator()(int)
        void zeros()
        us* memptr()
        int size()
    cdef cppclass cx_vec:
        cx_vec(int)
        cx_vec()
        void zeros()
        complex operator()(int)
        cx_vec(double*,int,bool,bool)
        void set_real(vec)
        void set_imag(vec)
        int size()

    vec real(cx_vec)
    vec imag(cx_vec)
ctypedef cx_vec vc
ctypedef vec vd
ctypedef complex[double] c
ctypedef double d

cdef cx_vec cndtovec(n.ndarray[n.complex128_t,ndim=1] ndarr):
    cdef n.ndarray[n.float64_t,ndim=1,mode='c'] arrR=n.ascontiguousarray(ndarr.real)
    cdef n.ndarray[n.float64_t,ndim=1,mode='c'] arrI=n.ascontiguousarray(ndarr.imag)
    cdef int size=ndarr.shape[0]
    cdef vd varrR=vd(&arrR[0],size,True,True)
    cdef vd varrI=vd(&arrI[0],size,True,True)
    cdef cx_vec varr=cx_vec(size)
    varr.set_real(varrR)
    varr.set_imag(varrI)
    return varr
cdef vd dndtovec(n.ndarray[n.float64_t,ndim=1] ndarr):
    cdef int size=ndarr.shape[0]
    cdef vd varr
    varr=vd(&ndarr[0],size,False,False)
    return varr

cdef n.ndarray cvectond(vc varr):
    cdef int size=varr.size()
    cdef vd varrR,varrI
    varrR=real(varr)
    varrI=imag(varr)
    cdef n.float_t* Rdata = varrR.memptr()
    cdef n.float_t* Idata = varrI.memptr()
    cdef n.float64_t[:] Rdata_view = <n.float64_t[:size]> Rdata
    cdef n.float64_t[:] Idata_view = <n.float64_t[:size]> Idata

    cdef n.ndarray[n.float64_t, ndim=1, mode='c'] Rresult
    cdef n.ndarray[n.float64_t, ndim=1, mode='c'] Iresult
    cdef n.ndarray[n.complex128_t, ndim=1, mode='c'] result
    Rresult = n.asarray(Rdata_view, dtype=n.float64, order='C')
    Iresult = n.asarray(Idata_view, dtype=n.float64, order='C')
    result=Rresult+1j*Iresult
#   print('array content in Cython: ' + repr(result))
    return result

cdef n.ndarray duvectond(uvec varr):
    cdef int size=varr.size()
    cdef us* data = varr.memptr()
    cdef us[:] data_view = <us[:size]> data
    cdef n.ndarray[n.uint64_t, ndim=1, mode='c'] result
    result = n.asarray(data_view, dtype=n.uint64, order='C')
    return result

cdef n.ndarray dvectond(vd varr):
    cdef int size=varr.size()
    cdef n.float_t* data = varr.memptr()
    cdef n.float64_t[:] data_view = <n.float64_t[:size]> data
    cdef n.ndarray[n.float64_t, ndim=1, mode='c'] result
    result = 1.0*n.asarray(data_view, dtype=n.float64, order='C')
#   print('array content in Cython: ' + repr(result))
    return result



cdef unicode _ustring(s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError(...)

cdef pyxstring(string):
        return _ustring(string).encode()
