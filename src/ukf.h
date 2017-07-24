#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    UKF();

    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage &meas_package);

private:
    ///* state covariance matrix
    MatrixXd P_;

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* sigma point matrix
    MatrixXd Xsig_aug_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;
    VectorXd z_pred_;
    MatrixXd Zsig_;
    MatrixXd S_;
    MatrixXd R_radar_;
    MatrixXd R_lidar_;

    double nis_radar_count_;
    double nis_radar_ok_count_;
    double nis_lidar_count_;
    double nis_lidar_ok_count_;
    double nis_threshold_;
    double consistency_radar_;
    double consistency_lidar_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    ///* Sigma point spreading parameter
    double lambda_;

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(const VectorXd &z);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(const VectorXd &z);

    void AugmentedSigmaPoints();
    void SigmaPointPrediction(double delta_t);
    void PredictMeanAndCovariance();

    void PredictRadarMeasurement(const VectorXd &z);
    void UpdateStateRadar(const VectorXd &z);

    void PredictLidarMeasurement(const VectorXd &z);
    void UpdateStateLidar(const VectorXd &z);

};

#endif /* UKF_H */
