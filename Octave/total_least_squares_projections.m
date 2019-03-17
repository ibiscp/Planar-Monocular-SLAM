# Assembly of the projection problem
source "./geometry_helpers_3d.m"
source "./total_least_squares_indices.m"


# camera matrix
global K=[180,   0, 320;
            0, 180, 240;
            0,   0,   1];

global cam_transform = [0,  0, 1, 0.2;
                       -1,  0, 0, 0;
                        0, -1, 0, 0;
                        0,  0, 0, 1];
    
# image_size
global image_rows=480;
global image_cols=640;
global z_near = 0;
global z_far = 5;

# dimension of projection
global projection_dim=2;

# projects a point
function p_img=projectPoint(Xr,Xl)
  global image_cols;
  global image_rows;
  global K;
  global cam_transform;
  global z_near;
  global z_far;

  p_img=[-1;-1];
  
  iXr = inv(cam_transform) * inv(Xr);
  pw = iXr(1:3,1:3) * Xl + iXr(1:3,4);
  if (pw(3) < z_near || pw(3) > z_far)
     return;
  endif;
  p_cam=K*pw;
  iz=1./p_cam(3);
  p_cam*=iz;
  if (p_cam(1)<0 || p_cam(1)>image_cols || p_cam(2)<0 || p_cam(2)>image_rows)
    return;
  endif;
  p_img=p_cam(1:2);
endfunction

# error and jacobian of a measured landmark
# input:
#   Xr: the robot pose in world frame (4x4 homogeneous matrix)
#   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
#   z:  projection of the landmark on the image plane
# output:
#   e: 2x1 is the difference between prediction and measurement
#   Jr: 2x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 2x3 derivative w.r.t a the error and a perturbation on the
#       landmark
#   is_valid: true if projection ok

function [is_valid, e,Jr,Jl]=projectionErrorAndJacobian(Xr,Xl,z)
  global K;
  global cam_transform;
  global image_rows;
  global image_cols;
  global z_near;
  global z_far;
  is_valid=false;
  e=[0;0];
  Jr=zeros(2,6);
  Jl=zeros(2,3);
  
  # inverse transform
  w2c = Xr * cam_transform;
  iR=w2c(1:3,1:3)';
  it=-iR*w2c(1:3,4);

  pw=iR*Xl+it; #point prediction, in world scale
  if (pw(2) < z_near || pw(2) > z_far)
     return;
  endif

  Jwr=zeros(3,6);
  Jwr(1:3,1:3)=-iR;
  Jwr(1:3,4:6)=iR*skew(Xl);
  Jwl=iR;
  
  p_cam=K*pw;
  iz=1./p_cam(3);
  z_hat=p_cam(1:2)*iz;
  if (z_hat(1)<0 || 
      z_hat(1)>image_cols ||
      z_hat(2)<0 || 
      z_hat(2)>image_rows)
    return;
  endif;

  iz2=iz*iz;
  Jp=[iz, 0, -p_cam(1)*iz2;
      0, iz, -p_cam(2)*iz2];
  
  e=z_hat-z;
  Jr=Jp*K*Jwr;
  Jl=Jp*K*Jwl;
  is_valid=true;
endfunction;


#linearizes the robot-landmark measurements
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
# output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers

function [H,b, chi_tot, num_inliers]=linearizeProjections(XR, XL, Zl, associations,num_poses, num_landmarks, kernel_threshold)
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
    [is_valid, e,Jr,Jl] = projectionErrorAndJacobian(Xr, Xl, z);
    if (! is_valid)
       continue;
    endif;
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
