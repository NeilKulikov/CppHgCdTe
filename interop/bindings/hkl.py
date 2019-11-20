import math as m
import numpy as np
from scipy.optimize import root

def rotator(angles):
    a, b, g = angles
    gr = np.array([[m.cos(g), m.sin(g), 0.], [-m.sin(g), m.cos(g), 0.], [0., 0., 1.]])
    br = np.array([[m.cos(b), 0., -m.sin(b)], [0., 1., 0.], [m.sin(b), 0., m.cos(b)]])
    ar = np.array([[m.cos(a), m.sin(a), 0.], [-m.sin(a), m.cos(a), 0.], [0., 0., 1.]])     
    return gr @ br @ ar

def rotate(xs, ags):
    return rotator(ags) @ np.array(xs)

def normalize(vec):
    nvec = np.array(vec)
    return nvec / np.linalg.norm(nvec)

def adopt(x):
    n = int(0.5 * x / m.pi)
    return x - 2. * m.pi * float(n)

adopt_v = np.vectorize(adopt)

def angles(hkl):
    res = normalize(hkl)
    ort = np.array([0., 0., 1.])
    fopt = lambda x: rotate(ort, x) - res
    result = {'success' : False}
    while result['success'] == False:
        result = root(fopt, np.random.rand(3))
    return adopt_v(result['x'])
