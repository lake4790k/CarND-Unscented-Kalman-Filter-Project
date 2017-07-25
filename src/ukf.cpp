#include "ukf.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    //set state dimension
    n_x_ = 5;
    //set augmented dimension
    n_aug_ = 7;

    lambda_ = 3 - n_aug_;

    nis_radar_count_ = 0.;
    nis_radar_ok_count_ = 0.;
    nis_lidar_count_ = 0.;
    nis_lidar_ok_count_ = 0.;
    nis_threshold_ = 7.8;

    // initial state vector
    x_ = VectorXd(n_x_);

    Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

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

    weights_ = VectorXd(2 * n_aug_ + 1);
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights_(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights_(i) = weight;
    }

    P_ << .1, 0, 0, 0, 0,
            0, .1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

    R_lidar_ = MatrixXd(2, 2);
    R_lidar_ << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;

    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    cout << "UKF created" << endl;
}

UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage &meas_package) {
    bool isRadar = meas_package.sensor_type_ == MeasurementPackage::RADAR;
    if ((isRadar && !use_radar_) || (!isRadar && !use_laser_)) {
        return;
    }

    const VectorXd &z = meas_package.raw_measurements_;
    if (!is_initialized_) {
        if (isRadar) {
            double rho = z(0);
            double phi = z(1);
            double px = rho * cos(phi);
            double py = rho * sin(phi);
            x_ << px, py, 0, 0, 0;
        } else {
            const double px = z(0);
            const double py = z(1);
            x_ << px, py, 0, 0, 0;
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    if (isRadar) {
        UpdateRadar(z);
    } else {
        UpdateLidar(z);
    }
//    cout << x_ << endl << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */
    AugmentedSigmaPoints();
    SigmaPointPrediction(delta_t);
    PredictMeanAndCovariance();
}

void UKF::AugmentedSigmaPoints() {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug_.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }
}

void UKF::SigmaPointPrediction(double delta_t) {
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_aug_(0, i);
        double p_y = Xsig_aug_(1, i);
        double v = Xsig_aug_(2, i);
        double yaw = Xsig_aug_(3, i);
        double yawd = Xsig_aug_(4, i);
        double nu_a = Xsig_aug_(5, i);
        double nu_yawdd = Xsig_aug_(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
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
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
}


void UKF::PredictMeanAndCovariance() {
    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        normalizeAngles(x_diff, 3);

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const VectorXd &z) {
    PredictLidarMeasurement(z);
    UpdateStateLidar(z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const VectorXd &z) {
    PredictRadarMeasurement(z);
    UpdateStateRadar(z);
}

void UKF::PredictRadarMeasurement(const VectorXd &z) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    Zsig_ = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig_(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
        Zsig_(1, i) = atan2(p_y, p_x);                                 //phi
        Zsig_(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //r_dot
    }

    //mean predicted measurement
    z_pred_ = VectorXd(n_z);
    z_pred_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
    }

    //measurement covariance matrix S
    S_ = MatrixXd(n_z, n_z);
    S_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred_;

        normalizeAngles(z_diff, 1);

        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S_ = S_ + R_radar_;
}


void UKF::PredictLidarMeasurement(const VectorXd &z) {
    //set measurement dimension, lidar can measure x,y
    int n_z = 2;

    Zsig_ = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // measurement model
        Zsig_(0, i) = Xsig_pred_(0, i);
        Zsig_(1, i) = Xsig_pred_(1, i);
    }

    //mean predicted measurement
    z_pred_ = VectorXd(n_z);
    z_pred_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
    }

    //measurement covariance matrix S
    S_ = MatrixXd(n_z, n_z);
    S_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred_;

        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S_ = S_ + R_lidar_;
}

void UKF::UpdateStateRadar(const VectorXd &z) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred_;
        normalizeAngles(z_diff, 1);

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normalizeAngles(x_diff, 3);

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_.inverse();

    //residual
    VectorXd z_diff = z - z_pred_;

    normalizeAngles(z_diff, 1);

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S_ * K.transpose();

    double nis = z_diff.transpose() * S_.inverse() * z_diff;
    nis_radar_count_++;
    if (nis < nis_threshold_) {
        nis_radar_ok_count_++;
    }
    consistency_radar_ = nis_radar_ok_count_ / nis_radar_count_;
//    cout << "radar nis = " << nis << "consistency = "<< consistency_radar_ << endl;
}

void UKF::UpdateStateLidar(const VectorXd &z) {
    //set measurement dimension, radar can measure x,y
    int n_z = 2;

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred_;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normalizeAngles(x_diff, 3);

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_.inverse();

    //residual
    VectorXd z_diff = z - z_pred_;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S_ * K.transpose();

    double nis = z_diff.transpose() * S_.inverse() * z_diff;
    nis_lidar_count_++;
    if (nis < nis_threshold_) {
        nis_lidar_ok_count_++;
    }
    consistency_lidar_ = nis_lidar_ok_count_ / nis_lidar_count_;
//    cout << "lidar nis = " << nis << "consistency = " << consistency_lidar_ << endl;
}

void UKF::normalizeAngles(VectorXd &v, int i) {
    while (v(i) > M_PI) v(i) -= 2. * M_PI;
    while (v(i) < -M_PI) v(i) += 2. * M_PI;
}