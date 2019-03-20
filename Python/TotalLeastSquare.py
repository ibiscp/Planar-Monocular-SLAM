## import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from geometry_helpers_3d import v2t
from total_least_squares import *

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

projection_dim = 2

damping = 0
kernel_threshold = 1e3
num_iterations = 10

################################## TRAJECTORY ##################################
datFile = '../dataset/trajectory.dat'

trajectory = []

with open(datFile, 'r') as file:
    for line in file:
        word = line.split()

        id = int(word[1])
        odometry = [float(word[2]), float(word[3]), float(word[4])]
        ground_truth = [float(word[5]), float(word[6]), float(word[7])]

        trajectory.append([id, odometry, ground_truth])

# Number of poses
num_poses = len(trajectory)

# Generate poses homogeneous matrices
XR_true = np.zeros([4, 4, num_poses])
XR_guess = np.zeros([4, 4, num_poses])
traj_true = np.zeros([3, num_poses])
traj_guess = np.zeros([3, num_poses])
traj_estimated = np.zeros([3, num_poses])
for i in range(num_poses):
    # True
    pose = trajectory[i][2]
    u = np.array([pose[0], pose[1], 0, 0, 0, pose[2]])
    XR_true[:,:,i] = v2t(u)
    traj_true[:,i] = u[0:3]

    # Guess
    pose = trajectory[i][1]
    ug = np.array([pose[0], pose[1], 0, 0, 0, pose[2]])
    XR_guess[:,:,i] = v2t(ug)
    traj_guess[:, i] = ug[0:3]

################################## LANDMARKS ###################################
datFile = '../dataset/world.dat'

landmarks = []

with open(datFile, 'r') as file:    #file is the variable
    for line in file:
        word = line.split()

        landmark_id = int(word[0])
        position = [float(word[1]), float(word[2]), float(word[3])]

        # Apearance
        apearance = []
        for i in range(4,14):
            apearance.append(float(word[i]))

        landmarks.append([landmark_id, position, apearance])

num_landmarks = len(landmarks)

XL_true = np.zeros([3, num_landmarks])
for i in range(num_landmarks):
    landmark = landmarks[i][1]
    XL_true[:,i] = landmark

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(pos_x, pos_y, pos_z)

################################# OBSERVATIONS #################################
files = glob.glob("../dataset/meas-*.dat")
files.sort()

observations = []
num_landmark_measurements = 0

for f in files:
    with open(f, 'r') as file:
        lines = [l for l in (line.strip() for line in file) if l]
        lines = lines[3:]
        landmark_observation = []
        for line in lines:
            num_landmark_measurements += 1
            word = line.split()

            id = int(word[1])
            landmark_id = int(word[2])
            image_position = [float(word[3]), float(word[4])]

            # Apearance
            apearance = []
            for i in range(5,15):
                apearance.append(float(word[i]))

            landmark_observation.append([id, landmark_id,
                                         image_position, apearance])
    observations.append(landmark_observation)

############################ LANDMARK MEASUREMENTS #############################
num_landmark_measurements=num_poses*num_landmarks
Zl = np.zeros([landmark_dim, num_landmark_measurements])
landmark_associations = np.zeros([2, num_landmark_measurements]).astype(int)

measurement_num = 0
for pose_num in range(num_poses):
    Xr = np.linalg.inv(XR_true[:,:,pose_num])
    for landmark_num in range(num_landmarks):#len(observations[pose_num])):
        # landmark_id = observations[pose_num][landmark_observ][1]
        Xl=XL_true[:,landmark_num]
        landmark_associations[:,measurement_num] = [pose_num, landmark_num]
        Zl[:,measurement_num] = Xr[0:3,0:3] @ Xl + Xr[0:3,3]
        measurement_num += 1

########################### PROJECTION MEASUREMENTS ############################
Zp = np.zeros([projection_dim, num_landmark_measurements])
projection_associations = np.zeros([2, num_landmark_measurements]).astype(int)

measurement_num = 0
for pose_num in range(num_poses):
    Xr = XR_true[:,:,pose_num]
    for landmark_observ in range(len(observations[pose_num])):
        landmark_id = observations[pose_num][landmark_observ][1]
        landmark_img = observations[pose_num][landmark_observ][2]

        projection_associations[:,measurement_num] = [pose_num, landmark_id]
        Zp[:, measurement_num] = landmark_img
        measurement_num += 1

############################## POSE MEASUREMENTS ###############################
# Generate an odometry trajectory for the robot
num_pose_measurements=num_poses - 1
Zr = np.zeros([4,4,num_pose_measurements])
pose_associations = np.zeros([2, num_pose_measurements]).astype(int)

measurement_num = 0
for pose_num in range(num_poses-1):
    Xi=XR_true[:, :, pose_num]
    Xj=XR_true[:, :, pose_num+1]
    pose_associations[:, measurement_num] = [pose_num, pose_num+1]
    Zr[:, :, measurement_num] = np.linalg.inv(Xi) @ Xj
    measurement_num += 1

################################ GENERATION OF (WRONG) INITIAL GUESS ##################################
### apply a perturbation to each ideal pose (construct the estimation problem)
pert_deviation=1;
XL_guess = np.copy(XL_true);

# apply a perturbation to each landmark
dXl=(np.random.rand(landmark_dim, num_landmarks)-0.5)*pert_deviation;
XL_guess+=dXl;

################################# CALL SOLVER  #################################

# Uncomment the following to suppress pose-landmark measurements
# Zl = np.zeros[3,0]

# Uncomment the following to suppress pose-landmark-projection measurements
# num_landmarks = 0
# Zp = np.zeros[3,0]

# Uncomment the following to suppress pose-pose measurements
# Zr = np.zeros[4,4,0]

# print(np.sum(XR_true))
# print(np.sum(XR_guess))
# print(np.sum(XL_true))
# print(np.sum(XL_guess))
# print(np.sum(Zl))
# print(np.sum(Zp))
# print(np.sum(Zr))

XR, XL, chi_stats_l, num_inliers_l, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b = doTotalLS(np.copy(XR_guess), np.copy(XL_true),
											      Zl, landmark_associations,
											      Zp, projection_associations,
											      Zr, pose_associations,
											      num_poses,
											      num_landmarks,
											      num_iterations,
											      damping,
											      kernel_threshold)

fig = plt.figure(1)
fig.suptitle("Landmark and poses", fontsize=16)

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(XL_true[0,:],XL_true[1,:],XL_true[2,:], 'o', mfc='none', color='b')
ax1.plot(XL_guess[0,:],XL_guess[1,:],XL_guess[2,:], 'x', color='r')
ax1.set_title("Landmark true and guess values", fontsize=10)

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(XL_true[0,:],XL_true[1,:],XL_true[2,:], 'o', mfc='none', color='b')
ax2.plot(XL[0,:],XL[1,:],XL[2,:], 'x', color='r')
ax2.set_title("Landmark true and estimated values", fontsize=10)

# Estimated trajectory
for i in range(num_poses):
    traj_estimated[:,i] = t2v(XR[:,:,i])[0:3]

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(traj_true[0,:],traj_true[1,:],traj_true[2,:], 'o', mfc='none', color='b')
ax3.plot(traj_guess[0,:],traj_guess[1,:],traj_guess[2,:], 'x', color='r')
ax3.set_zlim([-0.04,0.04])
ax3.set_title("Robot true and odometry values", fontsize=10)

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(traj_true[0,:],traj_true[1,:],traj_true[2,:], 'o', mfc='none', color='b')
ax4.plot(traj_estimated[0,:],traj_estimated[1,:],traj_estimated[2,:], 'x', color='r')
ax4.set_zlim([-0.04,0.04])
ax4.set_title("Robot true and estimated values", fontsize=10)

fig = plt.figure(2)
fig.suptitle("Error and inliners", fontsize=16)

ax1 = fig.add_subplot(321)
ax1.plot(chi_stats_l)
ax1.set_title("Chi Landmarks", fontsize=10)
ax2 = fig.add_subplot(322)
ax2.plot(num_inliers_l)
ax2.set_title("Inliers Landmarks", fontsize=10)

ax3 = fig.add_subplot(323)
ax3.plot(chi_stats_r)
ax3.set_title("Chi Poses", fontsize=10)
ax4 = fig.add_subplot(324)
ax4.plot(num_inliers_r)
ax4.set_title("Inliers Poses", fontsize=10)

ax5 = fig.add_subplot(325)
ax5.plot(chi_stats_p)
ax5.set_title("Chi Projections", fontsize=10)
ax6 = fig.add_subplot(326)
ax6.plot(num_inliers_p)
ax6.set_title("Inliers Projections", fontsize=10)

plt.show()