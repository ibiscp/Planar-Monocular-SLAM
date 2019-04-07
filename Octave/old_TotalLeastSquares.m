close all
clear
clc

# This is an integrated example that comprises
# 
source "./total_least_squares.m"

# synthesis of the virtual world
num_landmarks=100;
num_poses=10;
world_size=10;

# landmarks in a matrix, one per column
P_world=(rand(landmark_dim, num_landmarks)-0.5)*world_size;

global K; % camera matrix
global image_rows;
global image_cols;


# poses in an array of 4x4 homogeneous transform matrices
XR_true=zeros(4,4,num_poses);
XL_true=P_world;

# initialize 1st pose
XR_true(:,:,1)=eye(4);

# scaling coefficient for uniform random pose generation
# adjusts the translation to cover world_size
# adjusts the rotation to span the three angles;
rand_scale=eye(6);
rand_scale(1:3,1:3)*=(0.5*world_size);
rand_scale(4:6,4:6)*=pi;

for (pose_num=2:num_poses)
    xr=rand(6,1)-0.5;
    Xr=v2t(rand_scale*xr);
    XR_true(:,:,pose_num)=Xr;
endfor;

############################## GENERATION OF (WRONG) INITIAL GUESS ################################## 

# apply a perturbation to each ideal pose (construct the estimation problem)
pert_deviation=0.5;
pert_scale=eye(6)*pert_deviation;
XR_guess=XR_true;
XL_guess=XL_true;

for (pose_num=2:num_poses)
    xr=rand(6,1)-0.5;
    dXr=v2t(pert_scale*xr);
    XR_guess(:,:,pose_num)=dXr*XR_guess(:,:,pose_num);
endfor;

#apply a perturbation to each landmark
dXl=(rand(landmark_dim, num_landmarks)-0.5)*pert_deviation;
XL_guess+=dXl;

######################################## LANDMARK MEASUREMENTS ######################################## 
# generate an ideal number of landmark measurements
# each pose observes each landmark
num_landmark_measurements=num_poses*num_landmarks;
Zl=zeros(landmark_dim,num_landmark_measurements);
landmark_associations=zeros(2,num_landmark_measurements);

measurement_num=1;
for (pose_num=1:num_poses)
    Xr=inv(XR_true(:,:,pose_num));
    for (landmark_num=1:num_landmarks)
      Xl=XL_true(:,landmark_num);
      landmark_associations(:,measurement_num)=[pose_num,landmark_num]';
      Zl(:,measurement_num)=Xr(1:3,1:3)*Xl+Xr(1:3,4);
      measurement_num++;
    endfor;
endfor

######################################## PROJECTION MEASUREMENTS ######################################## 
# generate an ideal number of projection measurements
# each pose observes each landmark
num_projection_measurements=num_poses*num_landmarks;
Zp=zeros(projection_dim,num_projection_measurements);
projection_associations=zeros(2,num_projection_measurements);

measurement_num=1;
for (pose_num=1:num_poses)
    Xr=XR_guess(:,:,pose_num);
    for (landmark_num=1:num_landmarks)
      Xl=XL_guess(:,landmark_num);
      z_img=projectPoint(Xr,Xl);
      if (z_img(1)>0)
        projection_associations(:,measurement_num)=[pose_num, landmark_num]';
        Zp(:,measurement_num)=z_img;
        measurement_num++;
      endif;
    endfor;
endfor
# crop the projection associations to something meaningful
num_projection_measurements=measurement_num-1;
projection_associations=projection_associations(:,1:num_projection_measurements);
Zp=Zp(:,1:num_projection_measurements);


######################################## POSE MEASUREMENTS ######################################## 

# generate an odometry trajectory for the robot
num_pose_measurements=num_poses-1;
Zr=zeros(4,4,num_pose_measurements);
pose_associations=zeros(2,num_pose_measurements);

measurement_num=1;
for (pose_num=1:num_poses-1)

    Xi=XR_guess(:,:,pose_num);
    Xj=XR_guess(:,:,pose_num+1);
    pose_associations(:,measurement_num)=[pose_num, pose_num+1]';
    Zr(:,:,measurement_num)=inv(Xi)*Xj;

    measurement_num++;
endfor



############################## CALL SOLVER  ################################## 

# uncomment the following to suppress pose-landmark measurements
#Zl=zeros(3,0);

# uncomment the following to suppress pose-landmark-projection measurements
#num_landmarks=0;
# Zp=zeros(3,0);

# uncomment the following to suppress pose-pose measurements
# Zr=zeros(4,4,0);

damping=0.5;
kernel_threshold=100;
num_iterations=50;
[XR, XL,chi_stats_l, num_inliers_l, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b]=doTotalLS(XR_guess, XL_guess, 
												      Zl, landmark_associations, 
												      Zp, projection_associations, 
												      Zr, pose_associations, 
												      num_poses, 
												      num_landmarks, 
												      num_iterations, 
												      damping, 
												      kernel_threshold);
												      
												      
												      
# Plot State
figure(1);
hold on;
grid;

subplot(2,2,1);
title("Landmark Initial Guess");
plot3(XL_true(1,:),XL_true(2,:),XL_true(3,:),'b*',"linewidth",2);
hold on;
plot3(XL_guess(1,:),XL_guess(2,:),XL_guess(3,:),'ro',"linewidth",2);
legend("Landmark True", "Guess");grid;


subplot(2,2,2);
title("Landmark After Optimization");
plot3(XL_true(1,:),XL_true(2,:),XL_true(3,:),'b*',"linewidth",2);
hold on;
plot3(XL(1,:),XL(2,:),XL(3,:),'ro',"linewidth",2);
legend("Landmark True", "Guess");grid;


subplot(2,2,3);
title("Poses Initial Guess");
plot3(XR_true(1,:),XR_true(2,:),XR_true(3,:),'b*',"linewidth",2);
hold on;
plot3(XR_guess(1,:),XR_guess(2,:),XR_guess(3,:),'ro',"linewidth",2);
legend("Poses True", "Guess");grid;


subplot(2,2,4);
title("Poses After Optimization");
plot3(XR_true(1,:),XR_true(2,:),XR_true(3,:),'b*',"linewidth",2);
hold on;
plot3(XR(1,:),XR(2,:),XR(3,:),'ro',"linewidth",2);
legend("Poses True", "Guess"); grid;


figure(2);
hold on;
grid;
title("chi evolution");

subplot(3,2,1);
plot(chi_stats_r, 'r-', "linewidth", 2);
legend("Chi Poses"); grid; xlabel("iterations");
subplot(3,2,2);
plot(num_inliers_r, 'b-', "linewidth", 2);
legend("#inliers"); grid; xlabel("iterations");

subplot(3,2,3);
plot(chi_stats_l, 'r-', "linewidth", 2);
legend("Chi Landmark"); grid; xlabel("iterations");
subplot(3,2,4);
plot(num_inliers_l, 'b-', "linewidth", 2);
legend("#inliers"); grid; xlabel("iterations");

subplot(3,2,5);
plot(chi_stats_p, 'r-', "linewidth", 2);
legend("Chi Proj"); grid; xlabel("iterations");
subplot(3,2,6);
plot(num_inliers_p, 'b-', "linewidth", 2);
legend("#inliers");grid; xlabel("iterations");

figure(3);
title("H matrix");
H_ =  H./H;                      # NaN and 1 element
H_(isnan(H_))=0;                 # Nan to Zero
H_ = abs(ones(size(H_)) - H_);   # switch zero and one
H_ = flipud(H_);                 # switch rows
colormap(gray(64));
hold on;
image([0.5, size(H_,2)-0.5], [0.5, size(H_,1)-0.5], H_*64);
hold off;
