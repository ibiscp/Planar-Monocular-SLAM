# Assembly of the projection problem
import numpy as np
from total_least_squares_indices import *
from geometry_helpers_3d import skew
import math

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

# Projects a point
# function p_img=projectPoint(Xr,Xl)
#   global image_cols;
#   global image_rows;
#   global K;
#   iXr=inv(Xr);
#   p_img=[-1;-1];
#   pw=iXr(1:3,1:3)*Xl+iXr(1:3,4);
#   if (pw(3)<0)
#      return;
#   endif;
#   p_cam=K*pw;
#   iz=1./p_cam(3);
#   p_cam*=iz;
#   if (p_cam(1)<0 ||
#       p_cam(1)>image_cols ||
#       p_cam(2)<0 ||
#       p_cam(2)>image_rows)
#     return;
#   endif;
#   p_img=p_cam(1:2);
#   return p_img

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
    if (pw[1]<z_near or pw[1]>z_far):
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
    Jp = np.array([[iz, 0, -p_cam[0] * iz2], [0, iz, -p_cam[1] * iz2]])

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

        pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
        landmark_matrix_index = landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)

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
