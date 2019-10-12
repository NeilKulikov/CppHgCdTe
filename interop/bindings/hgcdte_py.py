import ctypes as ct
import numpy as np
import os

lib = ct.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + "/libinterop.dylib")

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
lib.make_hcore.argtypes = [ct.c_void_p, ct.c_size_t]

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

lib.del_diag_ws.restype = ct.c_int
lib.del_diag_ws.argtypes = [ct.c_void_p]

def del_diag_ws(ws : ct.c_void_p):
    lib.del_diag_ws(ws)

lib.get_diag_ws.restype = ct.c_void_p
lib.get_diag_ws.argtypes = [ct.c_size_t]

def get_diag_ws(si : int):
    assert si > 2, "Invalid workspace size"
    return lib.get_diag_ws(si)

lib.gen_eigen.restype = ct.POINTER(ct.c_double)
lib.gen_eigen.argtypes = [ct.c_void_p]

def gen_eigenb(hi : (ct.c_void_p, int)):
    return np.ctypeslib.as_array(
        lib.gen_eigen(hi[0]), 
        shape = (8 * hi[1],))

lib.gen_eigena.restype = ct.POINTER(ct.c_double)
lib.gen_eigena.argtypes = [ct.c_void_p, ct.c_void_p]

def gen_eigena(hi : (ct.c_void_p, int), ws: ct.c_void_p):
    return np.ctypeslib.as_array(
        lib.gen_eigen(hi[0], ws), 
        shape = (8 * hi[1],))

def gen_eigen(hi : (ct.c_void_p, int), ws = None):
    if ws == None:
        return gen_eigenb(hi)
    assert type(ws) == ct.c_void_p, "Invalid type of workspace"
    return gen_eigena(hi, ws)

lib.get_matr.restype = ct.POINTER(ct.c_double)
lib.get_matr.argtypes = [ct.c_void_p]

def get_matr(hi : (ct.c_void_p, int)):
    rv = np.ctypeslib.as_array(
            lib.get_matr(hi[0]),
            shape = (2 * hi[1]**2,))
    real, imag = rv[::2], rv[1::2]
    comp = real + 1j * imag
    return comp.reshape((hi[1], hi[1]))


class diag_ws:
    def __init__(self, size: int = 488):
        assert size > 2, "Invalid basis size"
        self.body = get_diag_ws(size)
    def __del__(self):
        del_diag_ws(self.body)

def get_spectre(hc : (ct.c_void_p, int), 
                k : (float, float)  = (0., 0.), 
                ws: diag_ws = None):
    "Generates number of eigenvalues"
    hi = make_hinst(hc, k)
    rv = None
    if ws == None:
        rv = gen_eigen(hi)
    else:
        rv = gen_eigena(hi, ws.body)
    del_hinst(hi[0])
    return rv

class model:
    def __init__(self, zs : np.ndarray, xs : np.ndarray):
        self.body = make_model(zs, xs)
    def __del__(self): 
        del_model(self.body)

class hcore:
    def __init__(self, 
            md : model, bs : int = 61, 
            own_ws: bool = True):
        assert bs > 2, "Invalid basis size"
        self.bsiz = bs
        self.body = make_hcore(md.body, bs)
        self.ws = None if own_ws == False else diag_ws(bs * 8)
    def __del__(self):
        del_hcore(self.body)
    def spectre(self, k : (float, float)  = (0., 0.)):
        return get_spectre(self.body, k, self.ws)
    def hinst(self, k : (float, float)  = (0., 0.)):
        hi = make_hinst(self.body, k)
        rv = get_matr((hi[0], 8 * hi[1]))
        del_hinst(hi[0])
        return rv







