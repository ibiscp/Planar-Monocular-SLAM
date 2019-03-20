import glob
import numpy as np

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

land_apperances = np.zeros([10, num_landmarks])
for l in range(num_landmarks):
    land_apperances[:,l] = landmarks[l][2]

correct = 0
wrong = 0
for i in range(len(observations)):
    observ_pose = observations[i]

    for j in observ_pose:
        observ_appearance = np.asarray(j[3]).reshape([-1,1])

        diff = land_apperances - observ_appearance
        error = np.linalg.norm(diff, ord=2, axis=0)

        index = np.argmin(error)

        if index == j[1]:
            correct += 1
            # print("Pose " + str(i) + " - Observation " + str(j[0]) + " Correct")
        else:
            wrong += 1
            # print("Pose " + str(i) + " - Observation " + str(j[0]) + " Wrong")

print("Correct: " + str(correct))
print("Wrong: " + str(wrong))