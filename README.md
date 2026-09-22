# CM2024_Group10
KTH technology and health project course. IMUs measuring knee angle to validate method for correlating to ACL injury risk in a more accessible way.

Files:

imu_processor - main, runs data analysis

/src folder/
- IMU_filters - has all classes and functions related to the extended kalman filter (EKF) and calculating knee angles
- load_movesense_data - has all functions related to loading raw data from Movesense IMUs

/data folder/ - has raw data that can be used in imu_processor, user must point processor at raw exercise data it wants to analyze

What code does:
[1] The raw data from each sensor (Movesense) needed to be re-packaged, merging the accelerometer and gyroscope data files to align to one common time grid.

[2] The data across all of the devices needed to be synchronized in time to create a cohesive data set, interpolating when needed. The output of this phase was one csv file with all of the IMU data from a specific exercise trial.

[3] We used an Extended Kalman Filter (EKF) to estimate the orientation of each IMU. This filter takes the gyroscope data sets, creates a model-based prediction, and combines that with the measurements from the accelerometer, weighing each by their uncertainty. The initial prediction phase uses the gyroscope’s angular velocity to know what orientation the next data point will be at. The calibration data set, from our subject standing still, helps estimate the gyroscope’s bias and noise, which tells the filter how its uncertainty will increase during prediction. Then there is an update phase which is built from accelerometer data. When the sensor is not accelerating in space, the accelerometer is only measuring gravity and tells the filter which way is down. This phase compares the measured “down” direction with the prediction from the phase before and adjusts the orientation by a weighted amount, known as the Kalman gain. This helps correct pitch and roll. During initial analysis, we noticed that during exercises with more movement (such as a jump compared to a lunge), the plots were drifting in a strange way. We realized the filter assumed that the measured acceleration direction was always just gravity, which was not the case during the dynamic motions. This caused the orientation estimate to be adjusted toward an incorrect reference. To combat this, we added a threshold for the accelerometer data so when it is measuring further from the value of gravity (using g=9.8m/s2), indicating large linear acceleration during dynamic movement, the filter will trust the accelerometer adjustment less. The EKF was applied to each IMU to get that segment’s orientation in space. 

[4] Next we translated each IMU's orientation from the EKF to 3 knee angles: flexion/extension, abduction/adduction, and rotation. This was done by utilizing equations in Fan et al. 2021 (linked below). This paper provided us with the mathamatical model to get from IMU orientation to joint angles.

[5] Plotted knee angles vs time for each exercise recorded.


Resources: 
Fan, B., Xia, H., Xu, J., Li, Q., & Shull, P. B. (2021). IMU-based knee flexion, abduction and internal rotation estimation during drop landing and cutting tasks. Journal of Biomechanics, 124, 110549. https://pubmed.ncbi.nlm.nih.gov/34167019/ 

