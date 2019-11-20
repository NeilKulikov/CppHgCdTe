import numpy as np

import hkl
import hgcdte_core as hct

class rotator:
    def init_by_ags(self, ags):
        assert len(ags) == 3, "3 angles should be provided"
        nags = np.array(ags)
        self.body = hct.make_rotator(nags)
    def init_by_hkl(self, idx):
        self.init_by_ags(hkl.angles(idx))
    def __init__(self, hkl = None, ags = [0., 0., 1.]):
        if hkl != None:
            self.init_by_hkl(hkl)
        else:
            self.init_by_ags(ags)
    def __del__(self):
        hct.del_rotator(self.body)

class strain_model:
    def __init__(self, zs, xs, bufx : float = 0.7):
        nzs, nxs = np.array(zs), np.array(xs)
        self.body = hct.make_strain_model(nzs, nxs, bufx)
    def __del__(self):
        hct.del_strain_model(self.body)

class model:
    def __init__(self, zs, xs):
        nzs, nxs = np.array(zs), np.array(xs)
        assert len(nzs) == len(nxs), "Xs and Zs should have equal size"
        self.body = hct.make_model(zs, xs)
    def __del__(self): 
        hct.del_model(self.body)

class diag_ws:
    def __init__(self, size: int = 488):
        assert size > 2, "Invalid basis size"
        self.body = hct.get_diag_ws(size)
    def __del__(self):
        hct.del_diag_ws(self.body)

class hcore:
    def __init__(self, md : model, bs : int = 61, own_ws: bool = True, 
            rot : rotator = None, stm : strain_model = None):
        assert bs > 0, "Basis size should be > 0"
        self.bsiz = bs
        self.body = hct.make_hcore(md.body, bs)
        self.ws = None if own_ws == False else diag_ws(bs * 8)
        self.strain = hct.get_strain_hinst(stm.body, bs) if stm != None else None
        self.rotator = rot
    def strain_hinst(self):
        if self.strain != None:
            rv = hct.get_matr((self.strain[0], 8 * self.strain[1]))
            return rv
        return None
    def hinst_matr(self, k : (float, float)  = (0., 0.)):
        hi = None
        if self.rotator == None:
            hi = hct.make_hinst(self.body, k)
        else:
            hi = hct.make_hinstr(self.body, k, self.rotator.body)
        return hi
    def hinst(self, k : (float, float)  = (0., 0.)):
        hi = self.hinst_matr(k)
        rv = hct.get_matr((hi[0], 8 * hi[1]))
        hct.del_hinst(hi[0])
        return rv  
    def spectre(self, k : (float, float)  = (0., 0.)):
        rv = None
        hi = self.hinst_matr(k)
        if self.strain != None:
            nh = hct.sum_hinst(hi, self.strain)
            hct.del_hinst(hi[0])
            hi = nh
        if self.ws == None:
            rv = hct.gen_eigen(hi)
        else:
            rv = hct.gen_eigena(hi, self.ws.body)
        hct.del_hinst(hi[0])
        return rv
    def __del__(self):
        hct.del_hcore(self.body)
        if self.strain != None:
            hct.del_hinst(self.strain[0])
