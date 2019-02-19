Project 10 - Planar Monocular SLAM

We provided a small simulator of a vacuum cleaner-ish robot that navigates in an environment
collecting data from a monocular camera. The simulator can be used to generate other datasets.

We provided, though, a ready to use dataset. It is composed by the following files:

- world.dat
  It contains information about the map.
  Every row contains:
   LANDMARK_ID POSITION APPEARANCE

- camera.dat
  It contains information about the camera used to gather data:
  - camera matrix
  - cam_transform: pose of the camera w.r.t. robot
  - z_near/z_far how close/far the camera can perceive stuff
  - width/height of images

- trajectory.dat
  pose: POSE_ID ODOMETRY_POSE GROUND_TRUTH_POSE
  the ODOMETRY_POSE is obtained by adding Gaussian Noise (0; 0.001) to the actual robot commands

- meas-XXXX.dat
  Every measurement contains a sequence number, ground truth (of the robot) and odometry pose and measurement information:
  - point POINT_ID_CURRENT_MESUREMENT ACTUAL_POINT_ID IMAGE_POINT APPEARANCE

  The Image_Point represents the pair [col;row] where the landmark is observed in the image
  The Appearance is represented by a Vector of 10 float number. Can be used to perform data association (see HBST)

Expected Output:
 - 1. Visual Aided Odometry:
   - instantiate the landmarks using the odometry, and perform SLAM in bearing-only fashion
 - 2. Bundle Adjustment:
   - embed the obtained measures in a full bundle adjustment pipeline (see total_least_squares)
