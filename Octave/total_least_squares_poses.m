source "./geometry_helpers_3d.m"
source "./total_least_squares_indices.m"

# error and jacobian of a measured pose, all poses are in world frame
# input:
#   Xi: the observing robot pose (4x4 homogeneous matrix)
#   Xj: the observed robot pose (4x4 homogeneous matrix)
#   Z:   the relative transform measured between Xr1 and Xr2
#   e: 12x1 is the difference between prediction, and measurement, vectorized
#   Ji : 12x6 derivative w.r.t a the error and a perturbation of the
#       first pose
#   Jj : 12x6 derivative w.r.t a the error and a perturbation of the
#       second pose

function [e,Ji,Jj]=poseErrorAndJacobian(Xi,Xj,Z)
  global Rx0;
  global Ry0;
  global Rz0;
  Ri=Xi(1:3,1:3);
  Rj=Xj(1:3,1:3);
  ti=Xi(1:3,4);
  tj=Xj(1:3,4);
  tij=tj-ti;
  Ri_transpose=Ri';
  Ji=zeros(12,6);
  Jj=zeros(12,6);
  
  dR_dax=Ri_transpose*Rx0*Rj;
  dR_day=Ri_transpose*Ry0*Rj;
  dR_daz=Ri_transpose*Rz0*Rj;
  
  Jj(1:9,4)=reshape(dR_dax, 9, 1);
  Jj(1:9,5)=reshape(dR_day, 9, 1);
  Jj(1:9,6)=reshape(dR_daz, 9, 1);
  Jj(10:12,1:3)=Ri_transpose;
  
  Jj(10:12,4:6)=-Ri_transpose*skew(tj);
  Ji=-Jj;

  Z_hat=eye(4);
  Z_hat(1:3,1:3)=Ri_transpose*Rj;
  Z_hat(1:3,4)=Ri_transpose*tij;
  e=flattenIsometryByColumns(Z_hat-Z);
  
  ##    disp("total_least_squares_poses (poseErrorAndJacobian)")
  ##    Z_hat(1:3,1:3)
  ##    Z_hat(1:3,4)
  ##    Z
  ##    Z_hat-Z
  ##    e
  ##    pause()
  
 endfunction;

#linearizes the robot-robot measurements
# inputs:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   ZR: the robot_robot measuremenrs (4x4xnum_measurements: array of homogeneous matrices)
#   associations: 2xnum_measurements. 
#                 associations(:,k)=[i_idx, j_idx]' means the kth measurement
#                 refers to an observation made from pose i_idx, that
#                 observed the pose j_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   kernel_threshod: robust kernel threshold
# outputs:
#   H: the H matrix, filled
#   b: the b vector, filled
#   chi_tot: the total chi2 of the current round
#   num_inliers: number of measurements whose error is below kernel_threshold

function [H,b, chi_tot, num_inliers]=linearizePoses(XR, XL, Zr, associations,num_poses, num_landmarks, kernel_threshold)
  global pose_dim;
  global landmark_dim;
  system_size=pose_dim*num_poses+landmark_dim*num_landmarks; 
  H=zeros(system_size, system_size);
  b=zeros(system_size,1);
  chi_tot=0;
  num_inliers=0;
  
  for (measurement_num=1:size(Zr,3))
    Omega=eye(12);
    Omega(1:9,1:9)*=1e3; # we need to pimp the rotation  part a little
    pose_i_index=associations(1,measurement_num);
    pose_j_index=associations(2,measurement_num);
    Z=Zr(:,:,measurement_num);
    Xi=XR(:,:,pose_i_index);
    Xj=XR(:,:,pose_j_index);
    [e,Ji,Jj] = poseErrorAndJacobian(Xi, Xj, Z);
    ##    disp("total_least_squares_poses")
    ##    sum(e(:))
    ##    sum(Ji(:))
    ##    sum(Jj(:))
    ##    pause()
  
    chi=e'*Omega*e;
    if (chi>kernel_threshold)
      Omega*=sqrt(kernel_threshold/chi);
      chi=kernel_threshold;
    else
      num_inliers ++;
    endif;
    chi_tot+=chi;

    pose_i_matrix_index=poseMatrixIndex(pose_i_index, num_poses, num_landmarks);
    pose_j_matrix_index=poseMatrixIndex(pose_j_index, num_poses, num_landmarks);
    
    H(pose_i_matrix_index:pose_i_matrix_index+pose_dim-1,
      pose_i_matrix_index:pose_i_matrix_index+pose_dim-1)+=Ji'*Omega*Ji;
##    Ji'*Omega*Ji
##    pause()

    H(pose_i_matrix_index:pose_i_matrix_index+pose_dim-1,
      pose_j_matrix_index:pose_j_matrix_index+pose_dim-1)+=Ji'*Omega*Jj;
##    Ji'*Omega*Jj
##    pause()

    H(pose_j_matrix_index:pose_j_matrix_index+pose_dim-1,
      pose_i_matrix_index:pose_i_matrix_index+pose_dim-1)+=Jj'*Omega*Ji;
##    Jj'*Omega*Ji
##    pause()

    H(pose_j_matrix_index:pose_j_matrix_index+pose_dim-1,
      pose_j_matrix_index:pose_j_matrix_index+pose_dim-1)+=Jj'*Omega*Jj;
##    Jj'*Omega*Jj
##    pause()

    b(pose_i_matrix_index:pose_i_matrix_index+pose_dim-1)+=Ji'*Omega*e;
##    Ji'*Omega*e
##    pause()
    b(pose_j_matrix_index:pose_j_matrix_index+pose_dim-1)+=Jj'*Omega*e;
##    Jj'*Omega*e
##    pause()
    
##    sum(H(:))
##    sum(b(:))
##    sum(chi_tot(:))
##    sum(num_inliers(:))
##    pause()
    
    
  endfor
endfunction
