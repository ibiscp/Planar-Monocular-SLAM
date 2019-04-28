from helper import *
import numpy as np
import math
from numpy.linalg import inv

# Error and jacobian of a measured pose, all poses are in world frame
# Input:
#   Xi: the observing robot pose (4x4 homogeneous matrix)
#   Xj: the observed robot pose (4x4 homogeneous matrix)
#   Z:   the relative transform measured between Xr1 and Xr2
# Output:
#   e: 12x1 is the difference between prediction, and measurement, vectorized
#   Ji : 12x6 derivative w.r.t a the error and a perturbation of the
#       first pose
#   Jj : 12x6 derivative w.r.t a the error and a perturbation of the
#       second pose
def poseErrorAndJacobian(Xi,Xj,Z):
    Ri = Xi[0:3, 0:3]
    Rj = Xj[0:3, 0:3]
    tj = Xj[0:3, 3]
    Ri_transpose = Ri.transpose()

    # Partial derivatives - dh/deltaa_i
    dR_dax = Ri_transpose @ Rx0 @ Rj
    dR_day = Ri_transpose @ Ry0 @ Rj
    dR_daz = Ri_transpose @ Rz0 @ Rj

    # Chordal Jacobian - dh(Xj, Xi + dxi)/deltax_i
    Jj = np.zeros([12,6])
    Jj[0:9, 3] = np.reshape(dR_dax, 9, 1)
    Jj[0:9, 4] = np.reshape(dR_day, 9, 1)
    Jj[0:9, 5] = np.reshape(dR_daz, 9, 1)
    Jj[9:12, 0:3] = Ri_transpose
    Jj[9:12, 3:6] =- Ri_transpose @ skew(tj)

    # Chordal Jacobian - dh(Xj + dxj, Xi)/deltax_j
    Ji=-Jj

    # Pose error
    Z_hat = inv(Xi) @ Xj
    pose_error = flattenIsometryByColumns(Z_hat - Z)

    return pose_error, Ji, Jj

# Linearizes the robot-robot measurements
# Inputs:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   ZR: the robot_robot measuremenrs (4x4xnum_measurements: array of homogeneous matrices)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[i_idx, j_idx]' means the kth measurement
#                 refers to an observation made from pose i_idx, that
#                 observed the pose j_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   kernel_threshod: robust kernel threshold
# Outputs:
#   H: the H matrix, filled
#   b: the b vector, filled
#   chi_tot: the total chi2 of the current round
#   num_inliers: number of measurements whose error is below kernel_threshold
def linearizePoses(XR, Zr, associations, num_poses, num_landmarks, kernel_threshold):
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    H = np.zeros([system_size, system_size])
    b = np.zeros([system_size, 1])

    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zr.shape[2]):
        Omega = np.eye(12)
        #Omega[0:9, 0:9] *= 1e3 # we need to pimp the rotation  part a little
        pose_i_index = associations[0, measurement_num]
        pose_j_index = associations[1, measurement_num]
        Z = Zr[:, :, measurement_num]
        Xi = XR[:, :, pose_i_index]
        Xj = XR[:, :, pose_j_index]

        # Calculate Error and Jacobian
        e, Ji, Jj = poseErrorAndJacobian(Xi, Xj, Z)

        chi = e.transpose() @ Omega @ e
        is_inlier = True
        if chi > kernel_threshold:
            Omega = Omega * math.sqrt(kernel_threshold/chi)
            chi = kernel_threshold
            is_inlier = False
        else:
            num_inliers += 1
        chi_tot += chi

        if not is_inlier:
            continue

        # Indices
        pose_i_matrix_index = poseMatrixIndex(pose_i_index, num_poses, num_landmarks)
        pose_j_matrix_index = poseMatrixIndex(pose_j_index, num_poses, num_landmarks)

        # Fill H and b
        H[pose_i_matrix_index:pose_i_matrix_index+pose_dim,
          pose_i_matrix_index:pose_i_matrix_index+pose_dim] += Ji.transpose() @ Omega @ Ji

        H[pose_i_matrix_index:pose_i_matrix_index+pose_dim,
          pose_j_matrix_index:pose_j_matrix_index+pose_dim] += Ji.transpose() @ Omega @ Jj

        H[pose_j_matrix_index:pose_j_matrix_index+pose_dim,
          pose_i_matrix_index:pose_i_matrix_index+pose_dim] += Jj.transpose() @ Omega @ Ji

        H[pose_j_matrix_index:pose_j_matrix_index+pose_dim,
          pose_j_matrix_index:pose_j_matrix_index+pose_dim] += Jj.transpose() @ Omega @ Jj

        b[pose_i_matrix_index:pose_i_matrix_index+pose_dim] += Ji.transpose() @ Omega @ e
        b[pose_j_matrix_index:pose_j_matrix_index+pose_dim] += Jj.transpose() @ Omega @ e


    return H, b, chi_tot, num_inliers
