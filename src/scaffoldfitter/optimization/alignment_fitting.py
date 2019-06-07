import scipy
from scipy.spatial import cKDTree
from scipy.optimize import leastsq

from .utils import transformRigid3DAboutCom, transformRigidScale3DAboutCom


def _sampleData(data, ratio):
    """
    Sample evenly spaced points from data

    :param data:
    :param ratio:
    :return:
    """

    totalData = len(data)
    N = int(totalData * ratio)

    if N < 1:
        raise ValueError("src.scaffoldfitter.optimization.alignment_fitting._sampleData(): N must be > 1")
    elif N > len(data):
        return data
    else:
        i = scipy.linspace(0, len(data)-1, N).astype(int)
        return data[i, :]


def fitRigidScale(data, target, xtol=1e-8, maxfev=10000, t0=None, sample=None, outputErrors=False, scaleThreshold=None):
    """

    :param data:
    :param target:
    :param xtol:
    :param maxfev:
    :param t0:
    :param sample:
    :param outputErrors:
    :param scaleThreshold:
    :return:
    """

    T = scipy.array(target)

    if sample is not None:
        D = data
        T = _sampleData(T, sample)
    else:
        D = data
        T = T

    if t0 is None:
        t0 = scipy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    D = scipy.array(D)

    if scaleThreshold is not None:
        def objectiveFunction(t):
            DT, Tfinal = transformRigid3DAboutCom(D, t)
            DTtree = cKDTree(DT)
            d = DTtree.query(T)[0]
            s = max(t[-1], 1.0/t[-1])
            if s > scaleThreshold:
                sw = 1000.0 * s
            else:
                sw = 1.0
            return d*d + sw
    else:
        def objectiveFunction(t):
            DT, Tfinal = transformRigidScale3DAboutCom(D, t)
            DTtree = cKDTree(DT)
            d = DTtree.query(T)[0]
            return d*d

    initialRMS = scipy.sqrt(objectiveFunction(t0).mean())
    t0pt = leastsq(objectiveFunction, t0, xtol=xtol, maxfev=maxfev)[0]
    fittedData, Tfinal = transformRigid3DAboutCom(D, t0pt)
    finalRMS = scipy.sqrt(objectiveFunction(t0pt).mean())

    if outputErrors:
        return t0pt, fittedData, (initialRMS, finalRMS), Tfinal
    else:
        return t0pt, fittedData, Tfinal