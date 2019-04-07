import numpy as np
import glob
from total_least_squares import *
from total_least_squares_landmarks import landmarkAssociation
from numpy.linalg import inv


def midpoint(p1, p2):
    mid = (p1 + p2) / 2

    return mid

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

###################################################################################

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

###################################################################################

XL_guess = np.zeros([3, num_landmarks])
first_obs = np.zeros([3, num_landmarks]) # pose_id, u, v

def triangulate(num_landmarks, num_poses, observations, land_apperances, XR):
    for pose_num in range(num_poses):

        for landmark_observ in range(len(observations[pose_num])):

            landmark_id = landmarkAssociation(observations[pose_num][landmark_observ][3], land_apperances)

            # if first time seeing landmark
            if np.sum(first_obs[:, landmark_id]) == 0:
                landmark_img = observations[pose_num][landmark_observ][2]
                first_obs[:, landmark_id] = [pose_num, landmark_img[0], landmark_img[1]]

            else:
                # execute triangulation
                previous_id = int(first_obs[0, landmark_id])
                x = first_obs[1:, landmark_id]
                y = observations[pose_num][landmark_observ][2]

                P1 = K @ np.eye(3, 4) @ inv(cam_transform) @ inv(XR[:,:,previous_id])
                P2 = K @ np.eye(3, 4) @ inv(cam_transform) @ inv(XR[:, :,pose_num])

                D = np.array([x[0] * P1[2,:] - P1[0,:],
                              x[1] * P1[2,:] - P1[1,:],
                              y[0] * P2[2,:] - P2[0,:],
                              y[1] * P2[2,:] - P2[1,:]])

                _, _, vh = np.linalg.svd(D)

                point = np.array([vh[3, 0] / vh[3, 3], vh[3, 1] / vh[3, 3], vh[3, 2] / vh[3, 3]])

                first_obs[:, landmark_id] = [pose_num, y[0], y[1]]

                if np.sum(XL_guess[:, landmark_id]) == 0:
                    XL_guess[:, landmark_id] = point
                else:
                    XL_guess[:, landmark_id] = midpoint(XL_guess[:, landmark_id], point)

    return XL_guess

stop = [31, 32, 371, 443, 983]

def triangulate2(num_landmarks, num_poses, observations, land_apperances, XR):
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
    ids = {}
    for landmark_num in range(num_landmarks):
        index = index_vec[0, landmark_num]

        A = D[0:2*index, 4*landmark_num:4*landmark_num+4]

        a, b, vh = np.linalg.svd(A)

        if len(b) == 4:

            point = np.array([vh[3,0]/vh[3,3], vh[3,1]/vh[3,3], vh[3,2]/vh[3,3]])

            # if landmark_num in stop:
            #     print(b)
            #     print(point)
            #     print(XL_true[:,landmark_num])

            if abs(point[0]) <=10 and abs(point[1]) <=10 and abs(point[2]) <=2:
                XL_guess[:, landmark_num] = point
                ids[landmark_num] = iter
                iter += 1

    XL_guess = XL_guess[:, 0:iter]
    # ids = ids[:, 0:iter]

    return ids, XL_guess

ids, XL_guess = triangulate2(num_landmarks, num_poses, observations, land_apperances, XR_guess)

count = 0
num_landmarks = XL_guess.shape[1]
for i in range(num_landmarks):
    id = ids[0,i]
    error = abs(XL_true[:, id] - XL_guess[:,i])
    if np.sum(error) > 20:
        count += 1
        print("Landmark ", i, count)
        print(XL_guess[:,i])
        print(XL_true[:,id])