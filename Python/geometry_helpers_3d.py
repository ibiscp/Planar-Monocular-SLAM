import numpy as np
import math

# Rotation matrix around x axis
def Rx(rot_x):
    c = math.cos(rot_x)
    s = math.sin(rot_x)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R

# Rotation matrix around y axis
def Ry(rot_y):
    c = math.cos(rot_y)
    s = math.sin(rot_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R

# Rotation matrix around z axis
def Rz(rot_z):
    c = math.cos(rot_z)
    s = math.sin(rot_z)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

# From 6d vector to homogeneous matrix
def v2t(v):
    # print(v)
    T = np.eye(4)
    T[0:3,0:3] = Rx(v[3]) @ Ry(v[4]) @ Rz(v[5])
    T[0:3,[3]] = v[0:3].reshape((3, 1))
    return T

# Converts a rotation matrix into a unit normalized quaternion
def mat2quat(R):
    qw4 = 2 * math.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    q  = np.array([(R[2,1]-R[1,2])/qw4, (R[0,2]-R[2,0])/qw4, (R[1,0]-R[0,1])/qw4])
    return q

# From homogeneous matrix to 6d vector
def t2v(X):
    x = np.zeros(6)
    x[0:3] = X[0:3,3]
    x[3:6] = mat2quat(X[0:3,0:3])
    return x

# Calculate skew matrix
def skew(v):
    S = np.array([[0, -v.item(2), v.item(1)], [v.item(2), 0, -v.item(0)], [-v.item(1), v.item(0), 0]])
    return S

def flattenIsometryByColumns(T):
    v = np.zeros([12, 1])
    v[0:9] = np.reshape(T[0:3,0:3].transpose(), [9, 1])
    v[9:12] = T[0:3,[3]]
    return v

# Derivative of rotation matrix w.r.t rotation around x, in 0
Rx0 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

# Derivative of rotation matrix w.r.t rotation around y, in 0
Ry0 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])

# Derivative of rotation matrix w.r.t rotation around z, in 0
Rz0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
