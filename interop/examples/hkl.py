import numpy as np

def normalize(vec):
    nvec = np.array(vec)
    return nvec / np.linalg.norm(nvec)

def rot_matrix(vec, ort = [0., 0., 1.]):
    nvec, nort = normalize(vec), normalize(ort)
    nper = np.cross(nvec, nort)
    return np.array([nvec, nper, nort])

def angle(sin, cos):
    if cos > 0.:
        return np.arctan(sin / cos)
    elif cos == 0.:
        return 0.5 * np.pi * np.sign(sin)
    else:
        return np.pi * np.sign(sin) + np.arctan(sin / cos)

def euler_ags(rmat, thr = 1.e-10):
    cosb = rmat[2, 2]
    sinb = np.sqrt(1. - rmat[2, 2]**2)
    if np.abs(sinb) < thr:
        return [angle(rmat[1, 0], rmat[0, 0]), angle(0., cosb), angle(0., 1.)]
    else:
        nrmt = np.array(rmat) / sinb
        return [angle(nrmt[2, 1], -nrmt[2, 0]), angle(sinb, cosb), angle(nrmt[1, 2], nrmt[0, 2])]

def angles(hkl):
    norm = normalize(hkl)
    rmat = rot_matrix(norm)
    eulr = euler_ags(rmat)
    return np.array(eulr)
