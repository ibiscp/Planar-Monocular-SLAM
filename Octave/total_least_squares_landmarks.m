# Assembly of the landmark-based problem
source "./geometry_helpers_3d.m"
source "./total_least_squares_indices.m"
%(minimal) size of pose and landmarks


# error and jacobian of a measured landmark
# input:
#   Xr: the robot pose in world frame (4x4 homogeneous matrix)
#   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
#   z:  measured position of landmark
# output:
#   e: 3x1 is the difference between prediction and measurement
#   Jr: 3x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 3x3 derivative w.r.t a the error and a perturbation on the
#       landmark

function [e,Jr,Jl]=landmarkErrorAndJacobian(Xr,Xl,z)
  # inverse transform
  iR=Xr(1:3,1:3)';
  it=-iR*Xr(1:3,4);
  
  ##  disp("landmarkErrorAndJacobian")
  ##  iR
  ##  skew(Xl)
  ##  iR*skew(Xl)
  ##  
  ##  pause()
  
  #prediction
  z_hat=iR*Xl+it;
  e=z_hat-z;
  Jr=zeros(3,6);
  Jr(1:3,1:3)=-iR;
  Jr(1:3,4:6)=iR*skew(Xl);
  Jl=iR;
endfunction;


#linearizes the robot-landmark measurements
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
# output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers

function [H,b, chi_tot, num_inliers]=linearizeLandmarks(XR, XL, Zl, associations,num_poses, num_landmarks, kernel_threshold)
  global pose_dim;
  global landmark_dim;
  system_size=pose_dim*num_poses+landmark_dim*num_landmarks; 
  H=zeros(system_size, system_size);
  b=zeros(system_size,1);
  chi_tot=0;
  num_inliers=0;
  for (measurement_num=1:size(Zl,2))
    pose_index=associations(1,measurement_num);
    landmark_index=associations(2,measurement_num);
    z=Zl(:,measurement_num);
    Xr=XR(:,:,pose_index);
    Xl=XL(:,landmark_index);
    [e,Jr,Jl] = landmarkErrorAndJacobian(Xr, Xl, z);
    ##    disp("total_least_squares_landmarks")
    ##    sum(e(:))
    ##    sum(Jr(:))
    ##    sum(Jl(:))
    ##    
    ##    pause()
    ##  
    chi=e'*e;
    if (chi>kernel_threshold)
      e*=sqrt(kernel_threshold/chi);
      chi=kernel_threshold;
    else
      num_inliers++;
    endif;
    chi_tot+=chi;

    pose_matrix_index=poseMatrixIndex(pose_index, num_poses, num_landmarks);
    landmark_matrix_index=landmarkMatrixIndex(landmark_index, num_poses, num_landmarks);

    H(pose_matrix_index:pose_matrix_index+pose_dim-1,
      pose_matrix_index:pose_matrix_index+pose_dim-1)+=Jr'*Jr;

    H(pose_matrix_index:pose_matrix_index+pose_dim-1,
      landmark_matrix_index:landmark_matrix_index+landmark_dim-1)+=Jr'*Jl;

    H(landmark_matrix_index:landmark_matrix_index+landmark_dim-1,
      landmark_matrix_index:landmark_matrix_index+landmark_dim-1)+=Jl'*Jl;

    H(landmark_matrix_index:landmark_matrix_index+landmark_dim-1,
      pose_matrix_index:pose_matrix_index+pose_dim-1)+=Jl'*Jr;

    b(pose_matrix_index:pose_matrix_index+pose_dim-1)+=Jr'*e;
    b(landmark_matrix_index:landmark_matrix_index+landmark_dim-1)+=Jl'*e;
  endfor
endfunction
