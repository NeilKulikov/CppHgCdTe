import ctypes as ct
import numpy as np

lib = ct.cdll.LoadLibrary("./libinterop.dylib")

lib.make_model.restype = ct.c_void_p
lib.make_model.argtypes = [ct.c_ulong, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]

def make_model(zs : np.ndarray, xs : np.ndarray):
    "Return raw C pointer to model structure"
    assert len(zs) == len(xs), "Invalid length"
    zcp = zs.ctypes.data_as(ct.POINTER(ct.c_double))
    xcp = xs.ctypes.data_as(ct.POINTER(ct.c_double))
    return lib.make_model(len(xs), zcp, xcp)

lib.del_model.restype = ct.c_int
lib.del_model.argtypes = [ct.c_void_p]

def del_model(md : ct.c_void_p):
    lib.del_model(md)

lib.make_hcore.restype = ct.c_void_p
lib.make_hcore.argtypes = [ct.c_void_p, ct.c_ulong]

def make_hcore(md : ct.c_void_p, bs : int = 101):
    "Return raw C pointer to core of hamiltonian"
    assert bs > 1, "Invalid length"
    return (lib.make_hcore(md, bs), bs)

lib.del_hcore.restype = ct.c_int
lib.del_hcore.argtypes = [ct.c_void_p]

def del_hcore(hc : ct.c_void_p):
    lib.del_hcore(hc)

lib.make_hinst.restype = ct.c_void_p
lib.make_hinst.argtypes = [ct.c_void_p, ct.c_double, ct.c_double]

def make_hinst(hc : (ct.c_void_p, int), k : (float, float)  = (0., 0.)):
    assert len(k) == 2, "Invalid length"
    return (lib.make_hinst(hc[0], k[0], k[1]), hc[1])

lib.del_hinst.restype = ct.c_int
lib.del_hinst.argtypes = [ct.c_void_p]

def del_hinst(hi : ct.c_void_p):
    lib.del_hinst(hi)

lib.gen_eigen.restype = ct.POINTER(ct.c_double)
lib.gen_eigen.argtypes = [ct.c_void_p]

def gen_eigen(hi : (ct.c_void_p, int)):
    return np.ctypeslib.as_array(
        lib.gen_eigen(hi[0]), 
        shape = (8 * hi[1],))

def get_spectre(hc : (ct.c_void_p, int), k : (float, float)  = (0., 0.)):
    "Generates "
    hi = make_hinst(hc, k)
    rv = gen_eigen(hi)
    del_hinst(hi[0])
    return rv

class model:
    def __init__(self, zs : np.ndarray, xs : np.ndarray):
        self.body = make_model(zs, xs)
    def __del__(self): 
        del_model(self.body)

class hcore:
    def __init__(self, md : model, bs : int = 61):
        self.bsiz = bs
        self.body = make_hcore(md, bs)
    def __del__(self):
        del_hcore(self.body)
    def spectre(self, k : (float, float)  = (0., 0.)):
        return get_spectre(self.body, k)
    







