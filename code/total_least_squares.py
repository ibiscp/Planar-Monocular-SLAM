from poses import linearizePoses
from projections import linearizeProjections
from helper import *

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
def doTotalLS(XR, XL, Zp, projection_associations,
	     Zr, pose_associations, num_poses, num_landmarks, num_iterations,
	     damping, kernel_threshold_proj, kernel_threshold_pose):

    # Chi and Inliers
    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)

    # Size of the linear system
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    iteration = 0
    error = 1e6
    while iteration < num_iterations and error > 1e-6:
        print('Iteration: ' + str(iteration))

        # Projections
        H_projections, b_projections, chi_, num_inliers_ = linearizeProjections(XR, XL, Zp, projection_associations,
                                                                                num_poses, num_landmarks,
                                                                                kernel_threshold_proj)
        chi_stats_p[iteration] = chi_stats_p[iteration] + chi_
        num_inliers_p[iteration] = num_inliers_

        # Poses
        H_poses, b_poses, chi_, num_inliers_ = linearizePoses(XR, Zr, pose_associations, num_poses, num_landmarks,
                                                              kernel_threshold_pose)
        chi_stats_r[iteration] += chi_
        num_inliers_r[iteration] = num_inliers_

        # Construct H and b
        H = H_poses + H_projections
        b = b_poses + b_projections

        # Add damping
        H += np.eye(system_size) * damping

        # Solve linear system (block first pose)
        dx = np.zeros([system_size, 1])
        dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:,0]).reshape([-1,1])

        # Box plus
        XR, XL = boxPlus(XR, XL, num_poses, num_landmarks, dx)

        # Print iteration
        iteration += 1
        error = np.sum(np.absolute(dx))
        print("Error: " + str(error))

    return XR, XL, chi_stats_p, num_inliers_p,chi_stats_r, num_inliers_r, H, b, iteration
