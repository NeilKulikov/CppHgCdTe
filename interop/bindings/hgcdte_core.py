import ctypes as ct
import numpy as np
import os

lib = ct.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + "/libinterop.dylib")

lib.make_model.restype = ct.c_void_p
lib.make_model.argtypes = [ct.c_size_t, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]

def make_model(zs : np.ndarray, xs : np.ndarray):
    "Return raw C pointer to model structure"
    assert len(zs) == len(xs), "Invalid length"
    zcp = zs.ctypes.data_as(ct.POINTER(ct.c_double))
    xcp = xs.ctypes.data_as(ct.POINTER(ct.c_double))
    return lib.make_model(len(xs), zcp, xcp)

lib.make_rotator.restype = ct.c_void_p
lib.make_rotator.argtypes = [ct.c_double, ct.c_double, ct.c_double]

def make_rotator(ags : np.ndarray):
    "Return raw C pointer to rotator"
    assert len(ags) == 3, "Should be only 3 angles"
    return lib.make_rotator(ags[0], ags[1], ags[2])

lib.make_strain_model.restype = ct.c_void_p
lib.make_strain_model.argtypes = [ct.c_size_t, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double]

def make_strain_model(zs : np.ndarray, xs : np.ndarray, bufx : float):
    "Return raw C pointer to model structure"
    assert len(zs) == len(xs), "Invalid length"
    assert 0. <= bufx <= 1., "Invalid buffer"
    zcp = zs.ctypes.data_as(ct.POINTER(ct.c_double))
    xcp = xs.ctypes.data_as(ct.POINTER(ct.c_double))
    return lib.make_strain_model(len(xs), zcp, xcp, bufx)

lib.del_model.restype = ct.c_int
lib.del_model.argtypes = [ct.c_void_p]

def del_model(md : ct.c_void_p):
    lib.del_model(md)

lib.del_rotator.restype = ct.c_int
lib.del_rotator.argtypes = [ct.c_void_p]

def del_rotator(rt : ct.c_void_p):
    lib.del_rotator(rt)

lib.del_strain_model.restype = ct.c_int
lib.del_strain_model.argtypes = [ct.c_void_p]

def del_strain_model(md : ct.c_void_p):
    lib.del_strain_model(md)

lib.make_hcore.restype = ct.c_void_p
lib.make_hcore.argtypes = [ct.c_void_p, ct.c_size_t]

def make_hcore(md : ct.c_void_p, bs : int = 101):
    "Return raw C pointer to core of hamiltonian"
    assert bs >= 1, "Invalid length"
    return (lib.make_hcore(md, bs), bs)

lib.make_strain_hcore.restype = ct.c_void_p
lib.make_strain_hcore.argtypes = [ct.c_void_p, ct.c_size_t]

def make_strain_hcore(md : ct.c_void_p, bs : int = 101):
    "Return raw C pointer to core of hamiltonian"
    assert bs >= 1, "Invalid length"
    return (lib.make_strain_hcore(md, bs), bs)

lib.del_hcore.restype = ct.c_int
lib.del_hcore.argtypes = [ct.c_void_p]

def del_hcore(hc : ct.c_void_p):
    lib.del_hcore(hc)

lib.del_strain_hcore.restype = ct.c_int
lib.del_strain_hcore.argtypes = [ct.c_void_p]

def del_strain_hcore(hc : ct.c_void_p):
    lib.del_strain_hcore(hc)

lib.make_hinst.restype = ct.c_void_p
lib.make_hinst.argtypes = [ct.c_void_p, ct.c_double, ct.c_double]

def make_hinst(hc : (ct.c_void_p, int), k : (float, float)  = (0., 0.)):
    assert len(k) == 2, "Invalid length"
    return (lib.make_hinst(hc[0], k[0], k[1]), hc[1])

lib.make_hinstr.restype = ct.c_void_p
lib.make_hinstr.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_void_p]

def make_hinstr(
        hc : (ct.c_void_p, int), 
        k : (float, float)  = (0., 0.),
        rot : ct.c_void_p = None):
    assert len(k) == 2, "Invalid length"
    assert rot != None, "Invalid rot"
    return (lib.make_hinstr(hc[0], k[0], k[1], rot), hc[1])

lib.make_strain_hinst.restype = ct.c_void_p
lib.make_strain_hinst.argtypes = [ct.c_void_p]

def make_strain_hinst(hc : (ct.c_void_p, int)):
    res = lib.make_strain_hinst(hc[0])
    return (res, hc[1])

def get_strain_hinst(md: (ct.c_void_p, int), bs: int = 101):
    hc = make_strain_hcore(md, bs)
    hi = make_strain_hinst(hc)
    del_strain_hcore(hc[0])
    return hi

lib.sum_hinsts.restype = ct.c_void_p
lib.sum_hinsts.argtypes = [ct.c_void_p, ct.c_void_p]

def sum_hinst(hi1: (ct.c_void_p, int), hi2: (ct.c_void_p, int)):
    assert hi1[1] == hi2[1], "Invalid length's in h instances"
    return (lib.sum_hinsts(hi1[0], hi2[0]), hi1[1])

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


