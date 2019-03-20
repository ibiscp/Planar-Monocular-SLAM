import numpy as np
import geometry_helpers_3d
from total_least_squares_indices import *
from geometry_helpers_3d import skew
import math

# Define landmark id given appearance
# Input:
#   appearance: appearance of given landmark
#   landmarks: list of landmarks appearances
#   num_landmarks: total number of landmarks
# Output:
#   index: index of the landmark
def landmarkAssociation(appearance, land_appearances):

    observ_appearance = np.asarray(appearance).reshape([-1, 1])

    diff = land_appearances - observ_appearance
    error = np.linalg.norm(diff, ord=2, axis=0)
    index = np.argmin(error)
    return index

# Error and jacobian of a measured landmark
# Input:
#   Xr: the robot pose in world frame (4x4 homogeneous matrix)
#   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
#   z: measured position of landmark
# Output:
#   e: 3x1 is the difference between prediction and measurement
#   Jr: 3x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 3x3 derivative w.r.t a the error and a perturbation on the
#       landmark
def landmarkErrorAndJacobian(Xr,Xl,z):
    # inverse transform
    iR = Xr[0:3,0:3].transpose()
    it = -iR @ Xr[0:3,[3]]

    # print('landmarkErrorAndJacobian')
    # print(iR)
    # print(skew(Xl))
    # print(iR @ skew(Xl))

    # prediction
    z_hat = iR @ Xl + it
    e = (z_hat - z).reshape([-1, 1])
    Jr = np.zeros([3,6])
    Jr[0:3, 0:3] = -iR
    Jr[0:3, 3:6] = iR @ skew(Xl)
    Jl = iR

    return e, Jr, Jl

# Linearizes the robot-landmark measurements
# Iutput:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (3xnum_measurements)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   kernel_threshod: robust kernel threshold
# Output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers
def linearizeLandmarks(XR, XL, Zl, associations, num_poses, num_landmarks, kernel_threshold):

    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    H = np.zeros([system_size, system_size])
    b = np.zeros([system_size, 1])
    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zl.shape[1]):
        pose_index = associations[0, measurement_num]
        landmark_index = associations[1, measurement_num]
        z = Zl[:, [measurement_num]]
        Xr = XR[:, :, pose_index]
        Xl = XL[:, [landmark_index]]
        e, Jr, Jl = landmarkErrorAndJacobian(Xr, Xl, z)
        # print('total_least_squares_landmarks')
        # print(np.sum(e))
        # print(np.sum(Jr))
        # print(np.sum(Jl))


        chi = e.transpose() @ e
        if chi > kernel_threshold:
            e = e * math.sqrt(kernel_threshold/chi)
            chi = kernel_threshold
        else:
            num_inliers += 1
        chi_tot += chi

        pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
        landmark_matrix_index=landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ Jr

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jr.transpose() @ Jl

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ Jl

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += Jl.transpose() @ Jr

        b[pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ e
        b[landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ e

    return H, b, chi_tot, num_inliers
