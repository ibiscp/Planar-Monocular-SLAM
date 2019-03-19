import numpy as np
from total_least_squares_landmarks import linearizeLandmarks
from total_least_squares_poses import linearizePoses
from total_least_squares_projections import linearizeProjections
from total_least_squares_indices import *
from geometry_helpers_3d import *

# Applies a perturbation to a set of landmarks and robot poses
# Input:
#   XR: the robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the landmark pose (3xnum_landmarks matrix of landmarks)
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   dx: the perturbation vector of appropriate dimensions
#       the poses come first, then the landmarks
# Output:
#   XR: the robot poses obtained by applying the perturbation
#   XL: the landmarks obtained by applying the perturbation

def boxPlus(XR, XL, num_poses, num_landmarks, dx):
    for pose_index in range(num_poses):
        pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
        dxr = dx[pose_matrix_index:pose_matrix_index + pose_dim]
        XR[:, :, pose_index] = v2t(dxr) @ XR[:, :, pose_index]

    for landmark_index in range(num_landmarks):
        landmark_matrix_index = landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)
        dxl = dx[landmark_matrix_index:landmark_matrix_index + landmark_dim, :]
        XL[:, [landmark_index]] += dxl

    return XR, XL

# Implementation of the optimization loop with robust kernel
# applies a perturbation to a set of landmarks and robot poses
# Input:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (3xnum_measurements)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   num_iterations: the number of iterations of least squares
#   damping:      damping factor (in case system not spd)
#   kernel_threshod: robust kernel threshold
# Output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats_{l,p,r}: array 1:num_iterations, containing evolution of chi2 for landmarks, projections and poses
#   num_inliers{l,p,r}: array 1:num_iterations, containing evolution of inliers landmarks, projections and poses

def doTotalLS(XR, XL, Zl, landmark_associations, Zp, projection_associations,
	     Zr, pose_associations, num_poses, num_landmarks, num_iterations,
	     damping, kernel_threshold):

    # print(np.sum(XR))
    # print(np.sum(XL))
    # print(np.sum(Zl))
    # print(np.sum(landmark_associations))
    # print(np.sum(Zp))
    # print(np.sum(projection_associations))
    # print(np.sum(Zr))
    # print(np.sum(pose_associations))
    # print(np.sum(num_poses))
    # print(np.sum(num_landmarks))
    # print(np.sum(num_iterations))
    # print(np.sum(damping))
    # print(np.sum(kernel_threshold))

    chi_stats_l = np.zeros(num_iterations)
    num_inliers_l = np.zeros(num_iterations)
    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)

    # Size of the linear system
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    for iteration in range(num_iterations):
        print('Iteration: ' + str(iteration))
        H = np.zeros([system_size, system_size])
        b = np.zeros([system_size, 1])

        if (num_landmarks):
            H_landmarks, b_landmarks, chi_, num_inliers_ = linearizeLandmarks(XR, XL, Zl, landmark_associations,num_poses, num_landmarks, kernel_threshold)
            chi_stats_l[iteration] = chi_
            num_inliers_l[iteration] = num_inliers_
            # print('total_least_squares')
            # print(np.sum(H_landmarks))
            # print(np.sum(b_landmarks))
            # print(np.sum(chi_))
            # print(np.sum(num_inliers_))

            H_projections, b_projections, chi_, num_inliers_ = linearizeProjections(XR, XL, Zp, projection_associations,num_poses, num_landmarks, kernel_threshold)
            chi_stats_p[iteration] = chi_stats_p[iteration] + chi_
            num_inliers_p[iteration] = num_inliers_

            # print('total_least_squares')
            # print(np.sum(H_projections))
            # print(np.sum(b_projections))
            # print(np.sum(chi_))
            # print(np.sum(num_inliers_))

        H_poses, b_poses, chi_, num_inliers_ = linearizePoses(XR, XL, Zr, pose_associations, num_poses, num_landmarks, kernel_threshold)
        chi_stats_r[iteration] += chi_
        num_inliers_r[iteration] = num_inliers_

        H = H_poses
        b = b_poses
        if (num_landmarks):
            H += H_landmarks + H_projections
            b += b_landmarks + b_projections


        H += np.eye(system_size) * damping
        dx = np.zeros([system_size, 1])

        # we solve the linear system, blocking the first pose
        # this corresponds to "remove" from H and b the locks
        # of the 1st pose, while solving the system

        dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:,0]).reshape([-1,1])
        # print(np.sum(dx))
        XR, XL = boxPlus(XR, XL, num_poses, num_landmarks, dx)

    return XR, XL, chi_stats_l, num_inliers_l, chi_stats_p, num_inliers_p,chi_stats_r, num_inliers_r, H, b
