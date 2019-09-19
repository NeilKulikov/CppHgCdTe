import os
import numpy
import ctypes

lib_hgcdte_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "libinterop.dylib"))
 
lib_hgcdte = ctypes.CDLL(lib_hgcdte_name)

def gen_model(zs: numpy.ndarray, xs: numpy.ndarray):
    zr = zs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xr = zs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    rv = lib_hgcdte.make_model(zr, xr)
    return rv

