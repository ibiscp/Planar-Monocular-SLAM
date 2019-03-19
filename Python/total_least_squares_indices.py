pose_dim = 6
landmark_dim = 3

# Retrieves the index in the perturbation vector, that corresponds to
# a certain pose
# Input:
#   pose_index:     the index of the pose for which we want to compute the
#                   index
#   num_poses:      number of pose variables in the state
#   num_landmarks:  number of pose variables in the state
# Output:
#   v_idx: the index of the sub-vector corrsponding to
#          pose_index, in the array of perturbations  (-1 if error)

def poseMatrixIndex(pose_index, num_poses, num_landmarks):
    if (pose_index>num_poses):
        return -1
    v_idx = pose_index * pose_dim
    return v_idx

# Retrieves the index in the perturbation vector, that corresponds to
# a certain landmark
# Input:
#   landmark_index: the index of the landmark for which we want to compute the
#                   index
#   num_poses:      number of pose variables in the state
#   num_landmarks:  number of pose variables in the state
# Output:
#   v_idx: the index of the perturnation corrsponding to the
#           landmark_index, in the array of perturbations

def landmarkMatrixIndex(landmark_index, num_poses, num_landmarks):
    if (landmark_index>num_landmarks):
        return -1
    v_idx = num_poses * pose_dim + landmark_index * landmark_dim
    return v_idx
