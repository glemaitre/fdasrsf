from cpython.version cimport PY_MAJOR_VERSION
import numpy as n
cimport numpy as n

from libcpp cimport bool

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

    cdef cppclass mat:
        mat(double* aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict)
        mat(double* aux_mem, int n_rows, int n_cols)
        mat(int n_rows, int n_cols)
        mat() nogil
        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero
        # fuctions
        mat i()  #inverse
        mat t()  #transpose
        vec diag()
        vec diag(int)
        fill(double)
        void raw_print(char*)
        void raw_print()
        vec unsafe_col(int)
        vec col(int)
        #print(char)
        #management
        mat reshape(int, int)
        mat resize(int, int)
        double* memptr()
        # opperators
        double& operator[](int)
        double& operator[](int,int)
        double& at(int,int)
        double& at(int)
        mat operator*(mat)
        mat operator%(mat)
        vec operator*(vec)
        mat operator+(mat)
        mat operator*(double)
        mat operator-(double)
        mat operator+(double)
        mat operator/(double)

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

cdef np.ndarray[np.double_t, ndim=2] dmattond(mat X):
    cdef int n_rows = X.n_rows
    cdef int n_cols = X.n_cols
    cdef int size = n_rows*n_cols
    cdef n.float_t* data = X.memptr()
    cdef n.float64_t[:, :] data_view = <n.float64_t[:n_rows, :n_cols]> data
    cdef n.ndarray[n.float64_t, ndim=2, mode='c'] result
    result = 1.0*n.asarray(data_view, dtype=n.float64, order='C')
    return result

