import numpy as np

def normalize(vec):
    nvec = np.array(vec)
    return nvec / np.linalg.norm(nvec)

def rot_matrix(vz, vx = [1, 0, 0]):
    nx, nz = normalize(vx), normalize(vz) 
    assert(abs(np.dot(nx, nz)) < 1.e-12)
    ny = np.cross(nz, nx)
    return np.array([nx, ny, nz])

def angle(sin, cos):
    if cos > 0.:
        return np.arctan(sin / cos)
    elif cos == 0.:
        return 0.5 * np.pi * np.sign(sin)
    else:
        return np.pi * np.sign(sin) + np.arctan(sin / cos)

def euler_ags(rmat, thr = 1.e-10):
    scb = rmat[2, 2], np.sqrt(1. - rmat[2, 2]**2)
    if scb[0] < thr:
        sca = rmat[1, 0], rmat[0, 0]
        scb = 0., 1.
        scc = 0., 1.
    else:
        norm = np.array(rmat) / scb[0]
        sca = norm[2, 1], - norm[2, 0]
        scc = norm[1, 2], norm[0, 2]
    return np.array([angle(*x) for x in [sca, scb, scc]])



def angles(hkl):
    norm = normalize(hkl)
    rmat = rot_matrix(norm)
    eulr = euler_ags(rmat)
    print(eulr)
    return np.array(eulr)
