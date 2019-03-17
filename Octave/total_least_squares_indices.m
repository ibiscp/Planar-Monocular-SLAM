# bookeeping of the indices
# dimensions

%(minimal) size of pose and landmarks
global pose_dim=6;
global landmark_dim=3;


# retrieves the index in the perturbation vector, that corresponds to
# a certain pose
# input:
#   pose_index:     the index of the pose for which we want to compute the
#                   index
#   num_poses:      number of pose variables in the state
#   num_landmarks:  number of pose variables in the state
# output:
#   v_idx: the index of the sub-vector corrsponding to 
#          pose_index, in the array of perturbations  (-1 if error)


function v_idx=poseMatrixIndex(pose_index, num_poses, num_landmarks)
  global pose_dim;
  global landmark_dim;

  if (pose_index>num_poses)
    v_idx=-1;
    return;
  endif;
  v_idx=1+(pose_index-1)*pose_dim;
endfunction;

# retrieves the index in the perturbation vector, that corresponds to
# a certain landmark
# input:
#   landmark_index:     the index of the landmark for which we want to compute the
#                   index
#   num_poses:      number of pose variables in the state
#   num_landmarks:  number of pose variables in the state
# output:
#   v_idx: the index of the perturnation corrsponding to the
#           landmark_index, in the array of perturbations

function v_idx=landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)
  global pose_dim;
  global landmark_dim;
  if (landmark_index>num_landmarks)
    v_idx=-1;
    return;
  endif;
  v_idx=1 + (num_poses)*pose_dim + (landmark_index-1) * landmark_dim;
endfunction;
