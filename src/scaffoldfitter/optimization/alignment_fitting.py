import scipy
from scipy.spatial import cKDTree
from scipy.optimize import leastsq

from .utils import transform_rigid_3d_about_com, transform_rigid_scale_3d_about_com


def _sample_data(data, ratio):
    """
    Sample evenly spaced points from data

    :param data:
    :param ratio:
    :return:
    """

    total_data = len(data)
    N = int(total_data * ratio)

    if N < 1:
        raise ValueError("src.scaffoldfitter.optimization.alignment_fitting._sample_data(): N must be > 1")
    elif N > len(data):
        return data
    else:
        i = scipy.linspace(0, len(data)-1, N).astype(int)
        return data[i, :]


def fit_rigid_scale(data, target, xtol=1e-8, maxfev=10000, t0=None, sample=None, output_errors=False, scale_threshold=None):
    """

    :param data:
    :param target:
    :param xtol:
    :param maxfev:
    :param t0:
    :param sample:
    :param output_errors:
    :param scale_threshold:
    :return:
    """

    T = scipy.array(target)

    if sample is not None:
        D = data
        T = _sample_data(T, sample)
    else:
        D = data
        T = T

    if t0 is None:
        t0 = scipy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    D = scipy.array(D)

    if scale_threshold is not None:
        def objective_function(t):
            DT, Tfinal = transform_rigid_3d_about_com(D, t)
            DTtree = cKDTree(DT)
            d = DTtree.query(T)[0]
            s = max(t[-1], 1.0/t[-1])
            if s > scale_threshold:
                sw = 1000.0 * s
            else:
                sw = 1.0
            return d*d + sw
    else:
        def objective_function(t):
            DT, Tfinal = transform_rigid_scale_3d_about_com(D, t)
            DTtree = cKDTree(DT)
            d = DTtree.query(T)[0]
            return d*d

    initial_rms = scipy.sqrt(objective_function(t0).mean())
    t0pt = leastsq(objective_function, t0, xtol=xtol, maxfev=maxfev)[0]
    fitted_data, Tfinal = transform_rigid_3d_about_com(D, t0pt)
    final_rms = scipy.sqrt(objective_function(t0pt).mean())

    if output_errors:
        return t0pt, fitted_data, (initial_rms, final_rms), Tfinal
    else:
        return t0pt, fitted_data, Tfinal