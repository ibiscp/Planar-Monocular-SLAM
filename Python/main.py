import glob
from total_least_squares import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *
from projections import triangulate

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
XR_odom = np.zeros([4, 4, num_poses])
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
    XR_odom[:, :, i] = v2t(ug)
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
land_apperances = np.zeros([10, num_landmarks])
for i in range(num_landmarks):
    landmark = landmarks[i][1]
    XL_true[:,i] = landmark
    land_apperances[:, i] = landmarks[i][2]


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

########################### LANDMARKS TRIANGULATION ############################
landmark_ids, XL_triang, num_landmarks = triangulate(num_landmarks, num_poses, observations, land_apperances, XR_odom)

########################### PROJECTION MEASUREMENTS ############################
Zp = np.zeros([projection_dim, num_landmark_measurements])
projection_associations = np.zeros([2, num_landmark_measurements]).astype(int)

measurement_num = 0
for pose_num in range(num_poses):
    for landmark_observ in range(len(observations[pose_num])):
        landmark_id = landmarkAssociation(observations[pose_num][landmark_observ][3], land_apperances)
        landmark_img = observations[pose_num][landmark_observ][2]

        try:
            id = landmark_ids[landmark_id]
            projection_associations[:,measurement_num] = [pose_num, id]
            Zp[:, measurement_num] = landmark_img
            measurement_num += 1
        except:
            continue

projection_associations = projection_associations[:, 0:measurement_num]
Zp = Zp[:, 0:measurement_num]

############################## POSE MEASUREMENTS ###############################
# Generate an odometry trajectory for the robot
num_pose_measurements=num_poses - 1
Zr = np.zeros([4,4,num_pose_measurements])
pose_associations = np.zeros([2, num_pose_measurements]).astype(int)

measurement_num = 0
for pose_num in range(num_poses-1):
    Xi= XR_odom[:, :, pose_num]
    Xj= XR_odom[:, :, pose_num + 1]
    pose_associations[:, measurement_num] = [pose_num, pose_num+1]
    Zr[:, :, measurement_num] = np.linalg.inv(Xi) @ Xj
    measurement_num += 1

################################# CALL SOLVER  #################################
# Parameters
damping = 1e-4
kernel_threshold_proj = 5000
kernel_threshold_pose = 0.01
max_iterations = 30

XR, XL, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b, iteration =\
    doTotalLS(XR_odom, np.copy(XL_triang),
              Zp, projection_associations,
              Zr, pose_associations,
              num_poses,
              num_landmarks,
              max_iterations,
              damping,
              kernel_threshold_proj,
              kernel_threshold_pose)

################################# CREATE GRAPHS  #################################

# Landmark and poses
fig1 = plt.figure(1)
fig1.set_size_inches(16, 12)
fig1.suptitle("Landmark and Poses", fontsize=16)

ax1 = fig1.add_subplot(221, projection='3d')
ax1.plot(XL_true[0,:],XL_true[1,:],XL_true[2,:], 'o', mfc='none', color='b', markersize=3)
ax1.plot(XL_triang[0, :], XL_triang[1, :], XL_triang[2, :], 'x', color='r', markersize=3)
ax1.axis([-15,15,-15,15])
ax1.set_zlim([-3,3])
ax1.set_title("Landmark true and guess values", fontsize=10)

ax2 = fig1.add_subplot(222, projection='3d')
ax2.plot(XL_true[0,:],XL_true[1,:],XL_true[2,:], 'o', mfc='none', color='b', markersize=3)
ax2.plot(XL[0,:],XL[1,:],XL[2,:], 'x', color='r', markersize=3)
ax2.axis([-15,15,-15,15])
ax2.set_zlim([-3,3])
ax2.set_title("Landmark true and estimated values", fontsize=10)

# Estimated trajectory
for i in range(num_poses):
    traj_estimated[:,i] = t2v(XR[:,:,i])[0:3]

ax3 = fig1.add_subplot(223)
ax3.plot(traj_true[0,:],traj_true[1,:], 'o', mfc='none', color='b', markersize=3)
ax3.plot(traj_guess[0,:],traj_guess[1,:], 'x', color='r', markersize=3)
ax3.axis([-10,10,-10,10])
ax3.set_title("Robot true and odometry values", fontsize=10)

ax4 = fig1.add_subplot(224)
ax4.plot(traj_true[0,:],traj_true[1,:], 'o', mfc='none', color='b', markersize=3)
ax4.plot(traj_estimated[0,:],traj_estimated[1,:], 'x', color='r', markersize=3)
ax4.axis([-10,10,-10,10])
ax4.set_title("Robot true and estimated values", fontsize=10)

# Chi and inliers
fig2 = plt.figure(2)
fig2.set_size_inches(16, 12)
fig2.suptitle("Chi and Inliers", fontsize=16)

ax3 = fig2.add_subplot(221)
ax3.plot(chi_stats_r[0:iteration])
ax3.set_title("Chi Poses", fontsize=10)
ax4 = fig2.add_subplot(222)
ax4.plot(num_inliers_r[0:iteration])
ax4.set_title("Inliers Poses", fontsize=10)

ax5 = fig2.add_subplot(223)
ax5.plot(chi_stats_p[0:iteration])
ax5.set_title("Chi Projections", fontsize=10)
ax6 = fig2.add_subplot(224)
ax6.plot(num_inliers_p[0:iteration])
ax6.set_title("Inliers Projections", fontsize=10)

# Save figures
fig1.savefig("../images/landmark_and_pose.png", dpi=1000)
fig2.savefig("../images/chi_and_inliers.png", dpi=1000)

plt.show()