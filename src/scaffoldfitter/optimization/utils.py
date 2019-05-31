import scipy


def transform_scale_3d(x, S):
    """
    Applies scaling to a list of points.
    :param x:
    :param S:
    :return:
    """
    return scipy.multiply(x, S)


def transform_rigid_3d(x, t):
    """
    Performs a rigid transformation to a list of points.

    :param x:
    :param t:
    :return:
    """

    X = scipy.vstack((x.T, scipy.ones(x.shape[0])))
    T = scipy.array([[1.0, 0.0, 0.0, t[0]],
                     [0.0, 1.0, 0.0, t[1]],
                     [0.0, 0.0, 1.0, t[2]],
                     [0.0, 0.0, 0.0, 1.0]])

    Rx = scipy.array([[1.0, 0.0, 0.0],
                      [0.0, scipy.cos(t[3]), -scipy.sin(t[3])],
                      [0.0, scipy.sin(t[3]), scipy.cos(t[3])]])

    Ry = scipy.array([[scipy.cos(t[4]), 0.0, scipy.sin(t[4])],
                      [0.0, 1.0, 0.0],
                      [-scipy.sin(t[4]), 0.0, scipy.cos(t[4])]])

    Rz = scipy.array([[scipy.cos(t[5]), -scipy.sin(t[5]), 0.0],
                      [scipy.sin(t[5]), scipy.cos(t[5]), 0.0],
                      [0.0, 0.0, 1.0]])

    T[:3, :3] = scipy.dot(scipy.dot(Rx, Ry), Rz)
    return scipy.dot(T, X)[:3, :].T, T


def transform_rigid_3d_about_com(x, t):
    """
    Performs a rigid transformation to a list of points.
    Rotation is about the centre of mass. Rotates first then translates.

    :param x:
    :param t:
    :return:
    """
    centre_of_mass = x.mean(0)
    x0 = x - centre_of_mass
    x0T, Tfinal = transform_rigid_3d(x0, t)
    return x0T + centre_of_mass, Tfinal


def transform_rigid_scale_3d_about_com(x, t):
    """
    Performs a rigid + scale transformation to a list of points.
    Rotation is about the centre of mass. Rotates first then translates.


    :param x:
    :param t:
    :return:
    """
    centre_of_mass = x.mean(0)
    x0 = x - centre_of_mass
    x0S = transform_scale_3d(x0, t[6:])
    x0T, Tfinal = transform_rigid_3d(x0S, t[:6])
    return x0T, Tfinal
