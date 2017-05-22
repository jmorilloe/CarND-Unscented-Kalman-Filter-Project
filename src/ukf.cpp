#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
 
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools tools;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
 
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
 
  // initial state vector
  x_ = VectorXd(5);
 
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
 
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;
 
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;
 
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
 
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
 
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
 
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
 
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
 
  /**
  TODO:
 
  Complete the initialization. See ukf.h for other member properties.
 
  Hint: one or more values initialized above might be wildly off...
  */
 
  is_initialized_ = false;
 
  // time when the state is true, in us
  time_us_ = 0.0;
 
  //state dimension
  n_x_ = 5;
 
  //augmented state dimension
  n_aug_ = 7;
 
  //spreading parameter
  lambda_ = 3 - n_x_;
 
  //set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  
  for (int i = 1; i<2 * n_aug_ + 1; i++) {
    weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }
}
 
UKF::~UKF() {}
 
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
 
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
 
    // Initialize the state x_
    cout << "UKF: " << endl;
    x_ << 1, 1, 1, 1, 1;
 
    // initialize prediction covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
 
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      double vx = rho_dot * cos(phi);//0.0;
      double vy = rho_dot * sin(phi);//0.0;
 
      double phi_dot = atan2(vy, vx);
 
      if (px == 0 && py == 0) {
        px = 0.001;
        py = 0.001;
      }
 
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Set state ekf_.x_ to the first measurement.
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
 
      //if initial values are zero
      if (px == 0 && py == 0) {
          px = 0.001;
          py = 0.001;
      }
 
      x_ << px, py, 0, 0, 0;
    }
 
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
 
  //compute the time elapsed between the current and previous measurements
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;    //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
 
  // Prediction
  Prediction(delta_t);
 
  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
}
 
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:
 
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
 
  // Augmented means vector
  VectorXd x_aug = VectorXd(7);
  // Augmented covariance matrix
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.setZero();
  // Sigma points matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  //Q matrix
  MatrixXd Q(2, 2);
  Q << std_a_ * std_a_, 0,
       0, std_yawdd_ * std_yawdd_;
 
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
 
  P_aug.topLeftCorner<5, 5>() = P_;
  P_aug.bottomRightCorner<2, 2>() = Q;
 
  MatrixXd L = P_aug.llt().matrixL(); 
 
  // Set sigma points
  Xsig_aug.col(0) = x_aug;
 
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
 
  // Prediction
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
 
 
  for (int i = 0; i< 2 * n_aug_ + 1; i++) {
 
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
 
    double px_p, py_p;
 
    // Control division by zero
 
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
 
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
 
    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
 
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
 
 
  // Predict means
  x_.fill(0.0);
 
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
 
 
  // Predict covariance matrix
  P_.fill(0.0);
 
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools.normalizeAngle(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
 
  }
}
 
/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
 
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
 
  You'll also need to calculate the lidar NIS.
  */
 
  // Sensor measurement dimension
  int n_z = 2;
 
  // Sigma points matrix
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
 
  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
 
  }
 
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
 
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
 
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    z_diff(1) = tools.normalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
 
  }
 
  // Add measurement noise covariance matrix
 
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S = S + R;
 
  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
 
  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
 
    // Angle normalization
    z_diff(1) = tools.normalizeAngle(z_diff(1));
 
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
 
    // Angle normalization
    x_diff(3) = tools.normalizeAngle(x_diff(3));
 
    // Cross correlation matrix
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
 
  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();
 
  // Actual measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
 
  // Residual
  VectorXd z_diff = z - z_pred;
 
  // Angle normalization
  z_diff(1) = tools.normalizeAngle(z_diff(1));
 
  // Calculate the lidar NIS.
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
 
  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
 
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
 
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
 
  You'll also need to calculate the radar NIS.
  */
 
  // Sensor state dimension
  int n_z = 3;
 
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
 
  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
 
  }
 
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
 
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
 
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    z_diff(1) = tools.normalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
 
  }
 
  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S = S + R;
 
 
  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
 
  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
 
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
 
    // Angle normalization
    z_diff(1) = tools.normalizeAngle(z_diff(1));
 
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
 
    // Angle normalization
    x_diff(3) = tools.normalizeAngle(x_diff(3));
 
    // Cross correlation matrix
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
 
  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();
 
  // Actual measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
 
  // Residual
  VectorXd z_diff = z - z_pred;
 
  // Angle normalization
  z_diff(1) = tools.normalizeAngle(z_diff(1));
 
  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
 
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}