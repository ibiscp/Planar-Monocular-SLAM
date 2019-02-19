1. connect a joystick
2. $> ./runba_sim | gnuplot
3. move the joystick

you can occasionally look at the trajectory by plotting
columns 3 and 4 of the file trajectory.dat
gnuplot> plot "trajectory.dat" using 3:4 w l

Notes:
1. sample gaussian noise is used to obtain a odometry
1. rand noise is used to mess up the appearance of the landmarks
