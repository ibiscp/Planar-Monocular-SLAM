#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <iostream>
#include <fstream>
#include <termios.h>
#include <fcntl.h>
#include <linux/joystick.h>

#include <random>

using namespace std;

// joy stuff
struct JoyArgs{
  int fd;
  double max_tv=0.001;
  double max_rv=0.001;
  int tv_axis=1;
  int rv_axis=3;
  int boost_button=4;
  int halt_button=5;
  const char* joy_device="/dev/input/js0";
} ;

volatile bool run=true;
volatile float des_tv=0, des_rv=0;

/** this stuff is to read the joystick**/
void* joyThread(void* args_){
  JoyArgs* args=(JoyArgs*)args_;
  args->fd = open (args->joy_device, O_RDONLY|O_NONBLOCK);
  if (args->fd<0) {
    cerr<<"no joy found on [" << args->joy_device << "]\n";
    return 0;
  }

  float tv = 0;
  float rv = 0;
  float tvscale = args->max_tv/32767.0;
  float rvscale = args->max_rv/32767.0;
  float gain = 1;

  struct js_event e;
  while (run) {
    if (read (args->fd, &e, sizeof(e)) > 0 ){
      fflush(stdout);
      int axis = e.number;
      int value = e.value;
      int type = e.type;
      if (axis == args->tv_axis && type == 2) {
        tv = -value * tvscale;
      }
      if (axis == args->rv_axis && type == 2) {
        rv = -value *rvscale;
      }
      if (axis == args->halt_button && type==1){
        tv = 0;
        rv = 0;
      } else if (axis == args->boost_button && type == 1) {
        if(value)
          gain = 2.0;
        else
          gain = 1.0;
      } 
      des_rv = rv*gain;
      des_tv = tv*gain;
    }
    usleep(10000); // 10 ms
  }
  close(args->fd);
  return 0;
};

// always contemporary, never out of fashion
Eigen::Vector3f t2v(const Eigen::Isometry2f&  iso) {
  Eigen::Vector3f v;
  v.head<2>()=iso.translation();
  v.z()=Eigen::Rotation2Df(iso.linear()).angle();
  return v;
}

Eigen::Isometry2f v2t(const Eigen::Vector3f& pose) {
  Eigen::Isometry2f _transform;
  _transform.setIdentity();
  _transform.translation()=pose.head<2>();
  _transform.linear() = Eigen::Rotation2Df(pose.z()).toRotationMatrix();
  return _transform;
}

Eigen::Isometry3f t2t3d(const Eigen::Isometry2f& iso) {
  Eigen::Isometry3f _transform;
  _transform.setIdentity();
  _transform.linear().block<2,2>(0,0)=iso.linear();
  _transform.translation().block<2,1>(0,0)=iso.translation();
  return _transform;
}


// 3D andmark with an own appearance (a 10 float vector)
struct WorldLandmark {
  Eigen::Vector3f position;
  Eigen::VectorXf appearance=Eigen::VectorXf(10);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

// a vector of the things above
using WorldLandmarkVector= std::vector<WorldLandmark, Eigen::aligned_allocator<WorldLandmark> >;

// world is a set of landmarks, with an extent.
struct World{
  World(int num_samples_, float x_range_, float y_range_, float z_range_):
    landmarks(num_samples_),
    x_range(x_range_),
    y_range(y_range_),
    z_range(z_range_){
    
    for (auto& wl: landmarks) {
      wl.position=Eigen::Vector3f::Random();
      wl.position.x()=wl.position.x()*x_range;
      wl.position.y()=wl.position.y()*y_range;
      wl.position.z()=fabs(wl.position.z()*z_range);
      wl.appearance=Eigen::VectorXf::Random(wl.appearance.size());
      
    }
  }
  WorldLandmarkVector landmarks;
  float x_range, y_range, z_range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

// a measurement of a landmark
struct PointMeasurement{
  int gt_idx; // the index of the landmark observed (for debug)
  Eigen::Vector2f image_point; // the 2D projection of the landmark in cam
  Eigen::VectorXf appearance;  // the appearance of the landmark
};

// a vector of the above
using PointMeasurementVector=std::vector<PointMeasurement, Eigen::aligned_allocator<PointMeasurement> >;


// a full fledge image+odom measurement
// with odometry (synchronous with cam)
struct ImageMeasurement{
  int seq; // progressive numba of the measurement
  Eigen::Isometry2f gt_pose; // the true pose
  Eigen::Isometry2f odom_pose; // the pose as seen by the odometry
  // the points seen
  PointMeasurementVector point_measurements;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


// prints an image+odom measurement
ostream& operator << (ostream & os, const ImageMeasurement& meas) {
  os << "seq: " << meas.seq << endl;
  os << "gt_pose: " << t2v(meas.gt_pose).transpose() <<endl;
  os << "odom_pose: " << t2v(meas.odom_pose).transpose() <<endl;
  for (size_t i=0; i<meas.point_measurements.size(); ++i) {
    const PointMeasurement&  pm=meas.point_measurements[i];
    os << "point " << i << " " << pm.gt_idx << " " << pm.image_point.transpose() << " "
       << pm.appearance.transpose() << endl;
  }
  return os;
}

// adds random noise to the appearance (btw [-0.1;0.1])
inline Eigen::VectorXf getNoisyAppearance(const Eigen::VectorXf& appearance) {
  Eigen::VectorXf noisy_appearance = appearance;
  const size_t size = noisy_appearance.size();
  for(size_t i = 0; i < size; ++i)
    noisy_appearance(i) += (-0.05 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.1))));

  return noisy_appearance;
}


// our lousy camera
// has intrinsics and extrinsics
struct Camera  {
  Eigen::Matrix3f K;  // camera matrix
  Eigen::Isometry3f inv_cam_pose; // pose of robot w.r.t camera
  int z_far;         // far clipping planr
  int z_near;        // near clipping plane
  float width;       // image width (pixels)
  float height;      // image height (pixels)

  
  //generates a measurement from a word and a robot position
  inline void projectPoints(PointMeasurementVector& point_measurements,
                            const WorldLandmarkVector& world_landmarks,
                            const Eigen::Isometry2f& robot_pose) const {
    //world to robot, bloated to 3D 
    Eigen::Isometry3f inv_robot_pose=t2t3d(robot_pose.inverse());
    
    Eigen::Isometry3f w2c=inv_cam_pose*inv_robot_pose;
    Eigen::Matrix3f KR=K*w2c.linear();
    Eigen::Vector3f Kt=K*w2c.translation();
    for (size_t i=0; i<world_landmarks.size(); ++i) {
      const WorldLandmark& wp=world_landmarks[i];
      Eigen::Vector3f cp=KR*wp.position+Kt;
      if (cp.z()<z_near)
        continue;
      if (cp.z()>z_far)
        continue;
      Eigen::Vector2f ip=cp.head<2>()/cp.z();
      if (ip.x()<0 || ip.y()<0)
        continue;
      if (ip.x()>width)
        continue;
      if (ip.y()>height)
        continue;
      // ok we accept the measurement
      PointMeasurement pm;
      pm.gt_idx=i;
      pm.image_point=ip;
      // passing a noisy appearance
      pm.appearance=getNoisyAppearance(wp.appearance);
      point_measurements.push_back(pm);
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

int main(int argc, char** argv) {

  //1. generate the world
  // 1st arg: nuber of landmarks
  // 2nd and 3rd: range on x, and y where to put landmarks
  // 3rd range on z
  World world(1000, 10, 10, 2);

  
  //2. print the world, all landmarks and their "true" appearance
  ofstream os("world.dat");
  int k=0;
  for (auto p:world.landmarks) {
    os << k << " "
       << p.position.transpose()  << " "
       << p.appearance.transpose() << endl;
    ++k;
  }
  os.close();

  //3. configure a camera looking forward w.r.t yhe robot base
  Camera cam;
  Eigen::Isometry3f cam_pose;
  cam_pose.setIdentity();
  cam_pose.linear() <<
    0,0,1,
    -1,0, 0,
    0, -1, 0; 
  cam_pose.translation() << 0.2, 0, 0;

  
  cam.inv_cam_pose=cam_pose.inverse();
  // camera matrix
  cam.K << 180, 0,  320,
    0,  180, 240,
    0, 0, 1;
  // image size
  cam.width=640;
  cam.height=480;
  
  // min/max distance
  cam.z_near=0.1;
  cam.z_far=5;

  //save parameters
  os.open("camera.dat");
  os << "camera matrix:" << endl;
  os << cam.K << endl;
  os << "cam_transform:" << endl;
  os << cam_pose.matrix() << endl;
  os << "z_near: " << cam.z_near << endl;
  os << "z_far:  " << cam.z_far  << endl;
  os << "width:  " << cam.width << endl;
  os << "height: " << cam.height << endl;
  os.close();

  
  // 4. initialize joy thrtead to control robot
  // it writes on two volatile variables
  // des_rv and des_tv
  JoyArgs joy_args;
  pthread_t joy_thread;
  pthread_create(&joy_thread, 0, joyThread,  &joy_args);


  //  5. initialize variables used in simulation (robot pose, counters etc)
  
  // true robot pose
  Eigen::Isometry2f robot_pose=Eigen::Isometry2f::Identity();

  // odometry estimate
  Eigen::Isometry2f odom_pose=Eigen::Isometry2f::Identity();

  // last pose for which we generated a measurement
  Eigen::Isometry2f last_robot_pose=Eigen::Isometry2f::Identity();

  // translational displacement when to benerate measurement
  float min_tdisp=0.2;
  
  // rotational displacement
  float min_rdisp=0.2; 

  // save the trajectory (true and odom) here
  os.open("trajectoy.dat");
  
  bool first=true;
  // progressive number of measurement
  int measurement_count=0;

  //configure gnuplot
  cout << "set size ratio -1" << endl;

  // 6. configure noise figures
  const float mean = 0.f;
  const float std = 0.001;
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(mean,std);
  
  // 7. start simulation
  while (1) {
    // if joy says we move, we move, otherwise we stay
    // exceptiom on 1st cycle, when we generate the measurement anyway
    if (fabs(des_rv)>=1e-5 || fabs(des_tv)>=1e-5 || first) {
      // compute a motion control from oystic command
      float rv=des_rv;
      float tv=des_tv;
      Eigen::Vector3f motion_vector(tv, 0, rv);
      Eigen::Isometry2f motion=v2t(motion_vector);

      // get corresponding noise control
      float noise_rv = distribution(generator);
      float noise_tv = distribution(generator);
      Eigen::Vector3f noisy_motion_vector(tv+noise_tv, 0, rv+noise_rv);
      Eigen::Isometry2f noisy_motion=v2t(noisy_motion_vector);
      
      // apply to true pose
      robot_pose = robot_pose * motion;

      // apply to odom (the noisy motion)
      odom_pose  = odom_pose * noisy_motion;

      // generate an image measurement
      ImageMeasurement image_measurement;
      image_measurement.seq=measurement_count;
      image_measurement.odom_pose=odom_pose;
      image_measurement.gt_pose=robot_pose;
      
      // project the ponts based on the robot pose (and camera pose)
      PointMeasurementVector& point_measurements(image_measurement.point_measurements);
      cam.projectPoints(point_measurements,
                        world.landmarks,
                        robot_pose);
      
      // use gnuplot to show output (sucks big times)
      cout << "set yrange [0:250]" << endl;
      cout << "set xrange [0:600]" << endl;
      cout << "plot '-' w p" << endl;
      for (auto& pm:point_measurements) {
        cout << pm.image_point.transpose() << endl;
      }
      cout << "e" << endl;

      // if we don't move enough or the measurement is not the 1st
      // we dont generate any measurement
      
      Eigen::Vector3f motion_since_last_frame=t2v(last_robot_pose.inverse()*robot_pose);
      if (motion_since_last_frame.head<2>().norm()<min_tdisp
          && fabs(motion_since_last_frame.z())<min_rdisp
          && !first)
        continue;

      // generate measurement
      last_robot_pose=robot_pose;

      os << "pose: " << measurement_count << " "
         << t2v(odom_pose).transpose() << " "
         << t2v(robot_pose).transpose() << endl;

      char filename[1024];
      sprintf(filename, "meas-%05d.dat", measurement_count);
      cerr << "measurement: " << measurement_count << " "
           << "pose: " << t2v(robot_pose).transpose() << " "
           << "points: " << point_measurements.size() << endl;
      
      ofstream meas_stream(filename);
      meas_stream << image_measurement << endl;
      meas_stream.close();
      
      ++measurement_count;
      first=false;
      
    }
  }  
}


