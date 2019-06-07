from math import sqrt, cos, sin, fabs, atan2


def magnitude(v):
    return sqrt(sum((v[i] * v[i]) for i in range(len(v))))


def add(u, v):
    return [(u[i] + v[i]) for i in range(len(u))]


def sub(u, v):
    return [(u[i] - v[i]) for i in range(len(u))]


def dot(u, v):
    return sum((u[i] * v[i]) for i in range(len(u)))


def eldiv(u, v):
    return [(u[i] / v[i]) for i in range(len(u))]


def elmult(u, v):
    return [(u[i] * v[i]) for i in range(len(u))]


def normalize(v):
    vmag = magnitude(v)
    return [(val / vmag) for val in v]


def cross(u, v):
    c = [u[1] * v[2] - u[2] * v[1],
         u[2] * v[0] - u[0] * v[2],
         u[0] * v[1] - u[1] * v[0]]

    return c


def mult(u, c):
    return [(u[i] * c) for i in range(len(u))]


def div(u, c):
    return [(u[i] / c) for i in range(len(u))]


def rotmx(quaternion):
    """
    This method takes a quaternion representing a rotation
    and turns it into a rotation matrix.
    """
    mag_q = magnitude(quaternion)
    norm_q = div(quaternion, mag_q)
    qw, qx, qy, qz = norm_q
    mx = [[qw * qw + qx * qx - qy * qy - qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
          [2 * qx * qy + 2 * qw * qz, qw * qw - qx * qx + qy * qy - qz * qz, 2 * qy * qz - 2 * qw * qx],
          [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, qw * qw - qx * qx - qy * qy + qz * qz]]
    return mx


def mxmult(mx, u):
    return []


def matrixconstantmult(m, c):
    """
    Multiply components of matrix m by constant c
    """
    return [mult(row_m, c) for row_m in m]


def matrixvectormult(m, v):
    """
    Post multiply matrix m by vector v
    """
    return [dot(row_m, v) for row_m in m]


def vectormatrixmult(v, m):
    """
    Premultiply matrix m by vector v
    """
    rows = len(m)
    if len(v) != rows:
        raise ValueError('vectormatmult mismatched rows')
    columns = len(m[0])
    result = []
    for c in range(0, columns):
        result.append(sum(v[r] * m[r][c] for r in range(rows)))
    return result


def matrixmult(a, b):
    """
    Multiply 2 matrices: first index is down row, second is across column.
    Assumes sizes are compatible (
    """
    return [vectormatrixmult(row_a, b) for row_a in a]


def eulerToRotationMatrix3(euler_angles):
    """
    From OpenCMISS-Zinc graphics_library.cpp
    """
    cos_azimuth = cos(euler_angles[0])
    sin_azimuth = sin(euler_angles[0])
    cos_elevation = cos(euler_angles[1])
    sin_elevation = sin(euler_angles[1])
    cos_roll = cos(euler_angles[2])
    sin_roll = sin(euler_angles[2])
    mat3x3 = [
        [cos_azimuth * cos_elevation, sin_azimuth * cos_elevation, -sin_elevation],
        [cos_azimuth * sin_elevation * sin_roll - sin_azimuth * cos_roll,
         sin_azimuth * sin_elevation * sin_roll + cos_azimuth * cos_roll, cos_elevation * sin_roll],
        [cos_azimuth * sin_elevation * cos_roll + sin_azimuth * sin_roll,
         sin_azimuth * sin_elevation * cos_roll - cos_azimuth * sin_roll, cos_elevation * cos_roll]]
    return mat3x3


def rotationMatrix3ToEuler(matrix):
    """
    From OpenCMISS-Zinc graphics_library.cpp
    """
    MATRIX_TO_EULER_TOLERANCE = 1.0E-12
    euler_angles = [0.0, 0.0, 0.0]
    if (fabs(matrix[0][0]) > MATRIX_TO_EULER_TOLERANCE) and (fabs(matrix[0][1]) > MATRIX_TO_EULER_TOLERANCE):
        euler_angles[0] = atan2(matrix[0][1], matrix[0][0])
        euler_angles[2] = atan2(matrix[1][2], matrix[2][2])
        euler_angles[1] = atan2(-matrix[0][2], matrix[0][0] / cos(euler_angles[0]))
    elif fabs(matrix[0][0]) > MATRIX_TO_EULER_TOLERANCE:
        euler_angles[0] = atan2(matrix[0][1], matrix[0][0])
        euler_angles[2] = atan2(matrix[1][2], matrix[2][2])
        euler_angles[1] = atan2(-matrix[0][2], matrix[0][0] / cos(euler_angles[0]))
    elif fabs(matrix[0][1]) > MATRIX_TO_EULER_TOLERANCE:
        euler_angles[0] = atan2(matrix[0][1], matrix[0][0])
        euler_angles[2] = atan2(matrix[1][2], matrix[2][2])
        euler_angles[1] = atan2(-matrix[0][2], matrix[0][1] / sin(euler_angles[0]))
    else:
        euler_angles[1] = atan2(-matrix[0][2], 0)  # get +/-1
        euler_angles[0] = 0
        euler_angles[2] = atan2(-matrix[2][1], -matrix[2][0] * matrix[0][2])
    return euler_angles


def axisAngleToQuaternion(axis, angle):
    return [cos(angle / 2), axis[0] * sin(angle / 2), axis[1] * sin(angle / 2), axis[2] * sin(angle / 2)]


def axisAngleToRotationMatrix(axis, angle):
    pass
