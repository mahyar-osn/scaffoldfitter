import scipy
from scipy.spatial import cKDTree


def fit_rigid_scale(data, target, xtol=1e-5, maxfev=0, t0=None, sample=None, output_errors=0, scale_threshold=None):
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

