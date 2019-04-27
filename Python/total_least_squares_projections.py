# Assembly of the projection problem
import numpy as np
from total_least_squares_indices import *
from geometry_helpers_3d import skew
import math
from total_least_squares_landmarks import landmarkAssociation
from numpy.linalg import inv

# Camera matrix
K = np.array([[180, 0, 320], [0, 180, 240], [0, 0, 1]])
cam_transform = np.array([[0, 0, 1, 0.2], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# Image_size
image_rows=480
image_cols=640
z_near = 0
z_far = 5

# Dimension of projection
projection_dim=2

def triangulate(num_landmarks, num_poses, observations, land_apperances, XR):
    XL_guess = np.zeros([3, num_landmarks])
    D = np.zeros([2 * num_poses, 4*num_landmarks])
    index_vec = np.zeros([1, num_landmarks], dtype=int)
    cam = K @ np.eye(3, 4) @ inv(cam_transform)

    for pose_num in range(num_poses):
        for landmark_observ in range(len(observations[pose_num])):

            land_id = landmarkAssociation(observations[pose_num][landmark_observ][3], land_apperances)
            x = observations[pose_num][landmark_observ][2]

            if abs(x[0]) > 50 and abs(x[1]) > 50:
                P = cam @ inv(XR[:, :,pose_num])

                index = index_vec[0, land_id]

                D[2 * index, 4*land_id:4*land_id+4] = x[0] * P[2,:] - P[0,:]
                D[2 * index + 1, 4*land_id:4*land_id+4] = x[1] * P[2,:] - P[1,:]

                index_vec[0, land_id] += 1

    iter = 0
    ids = {}#np.zeros([1, num_landmarks], dtype=int)
    for landmark_num in range(num_landmarks):
        index = index_vec[0, landmark_num]

        A = D[0:2*index, 4*landmark_num:4*landmark_num+4]

        _, b, vh = np.linalg.svd(A)

        if len(b) == 4:
            point = np.array([vh[3,0]/vh[3,3], vh[3,1]/vh[3,3], vh[3,2]/vh[3,3]])
            XL_guess[:, iter] = point
            ids[landmark_num] = iter
            iter += 1
        else:
            print(landmark_num)
        # if (abs(point[0]) > 12 and abs(point[1]) > 12 and abs(point[2]) > 3):
        #     XL_guess[:, iter] = np.array([0,0,0])
        # else:


    XL_guess = XL_guess[:,0:iter]

    return ids, XL_guess

# Error and jacobian of a measured landmark
# Input:
#   Xr: the robot pose in world frame (4x4 homogeneous matrix)
#   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
#   z:  projection of the landmark on the image plane
# Output:
#   e: 2x1 is the difference between prediction and measurement
#   Jr: 2x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 2x3 derivative w.r.t a the error and a perturbation on the
#       landmark
#   is_valid: true if projection ok
def projectionErrorAndJacobian(Xr,Xl,z):
    is_valid = False
    e = np.zeros([2, 1])
    Jr = np.zeros([2,6])
    Jl = np.zeros([2,3])

    # inverse transform
    w2c = Xr @ cam_transform
    iR = w2c[0:3,0:3].transpose()
    it = -iR @ w2c[0:3,3]

    pw = iR @ Xl + it #point prediction, in world scale
    if (pw[2]<z_near or pw[2]>z_far):
        return is_valid, e, Jr, Jl

    Jwr = np.zeros([3,6])
    Jwr[0:3, 0:3] = -iR
    Jwr[0:3, 3:6] = iR @ skew(Xl)
    Jwl = iR

    p_cam = K @ pw
    iz = 1. / p_cam[2]
    z_hat = p_cam[0:2] * iz
    if (z_hat[0]<0 or z_hat[0]>image_cols or z_hat[1]<0 or z_hat[1]>image_rows):
        return is_valid, e, Jr, Jl

    iz2 = iz * iz
    Jp = np.array([[iz,  0, -p_cam[0] * iz2],
                   [0,  iz, -p_cam[1] * iz2]])

    e = (z_hat - z).reshape([-1, 1])
    Jr = Jp @ K @ Jwr
    Jl = Jp @ K @ Jwl
    is_valid = True
    return is_valid, e, Jr, Jl


# Linearizes the robot-landmark measurements
# Input:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (2xnum_measurements)
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
def linearizeProjections(XR, XL, Zl, associations, num_poses, num_landmarks, kernel_threshold):
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    H = np.zeros([system_size, system_size])
    b = np.zeros([system_size,1])

    chi_tot = 0
    num_inliers = 0
    for measurement_num in range(Zl.shape[1]):
        pose_index = associations[0, measurement_num]
        landmark_index = associations[1, measurement_num]
        z = Zl[:, measurement_num]
        Xr = XR[:, :, pose_index]
        Xl = XL[:, landmark_index]
        is_valid, e, Jr, Jl = projectionErrorAndJacobian(Xr, Xl, z)
        if not is_valid:
           continue
        chi = e.transpose() @ e
        if chi>kernel_threshold:
            e *= math.sqrt(kernel_threshold / chi)
            chi = kernel_threshold
        else:
            num_inliers += 1
        chi_tot += chi

        # Indices
        pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
        landmark_matrix_index = landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)

        # Fill H and b
        omega_proj = 0.01 * np.identity(2)

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ omega_proj @ Jr

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jr.transpose() @ omega_proj @ Jl

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ omega_proj @ Jl

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += Jl.transpose() @ omega_proj @ Jr

        b[pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ omega_proj @ e
        b[landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ omega_proj @ e

    return H, b, chi_tot, num_inliers
