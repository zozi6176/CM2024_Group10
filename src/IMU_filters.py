import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

'''
Orientation Extended Kalman Filter (EKF) for a single IMU.
 
STATE DESIGN: this uses an "error-state" (indirect) EKF, which is the
standard approach in inertial navigation, rather than putting the
quaternion directly in a linear state vector.
- The TRUE orientation is tracked as a quaternion via scipy's Rotation
  class, updated nonlinearly each step (never linearized directly -- this
  avoids hand-deriving quaternion kinematics matrices, a common source of
  silent sign-convention bugs).
- The KALMAN FILTER's 6-element state is just the small ERROR relative to
  that tracked orientation: 3 attitude error components (rad) + 3 gyro
  bias components (rad/s). Errors are small by construction, so the
  small-angle linear approximation used in the Jacobians is valid.
'''
class OrientationEKF:
    def __init__(self, initial_quat_xyzw, gyro_bias_init=None,
                 gyro_noise_std=0.01, bias_noise_std=0.0001, accel_noise_std=0.5,
                 accel_gating_strength=50.0):
        """
        initial_quat_xyzw: starting orientation as [x, y, z, w] (scipy's
            quaternion convention).
        gyro_noise_std: gyro measurement noise, rad/s. Adjustable.
        bias_noise_std: expected gyro bias drift rate, rad/s per sqrt(s). Adjustable.
        accel_noise_std: accelerometer measurement noise, m/s^2. Adjustable.
        accel_gating_strength: threshold for accelerometer gating. Adjustable.
        """
        self.q = Rotation.from_quat(initial_quat_xyzw)
        self.bias = gyro_bias_init if gyro_bias_init is not None else np.zeros(3)
 
        self.P = np.eye(6) * 0.1  #how wrong do I think current state is (covariance of error state)
        self.Q = np.diag([gyro_noise_std**2] * 3 + [bias_noise_std**2] * 3)  #how much I expect the state to change between steps (process noise)
        self.R = np.eye(3) * accel_noise_std**2  #how much I expect the accelerometer to be noisy (measurement noise)
        self.accel_gating_strength = accel_gating_strength
        self.gravity = 9.81
 
    def predict(self, gyro_dps, dt):
        """Propagate orientation forward using a gyro reading.
        gyro_dps: [x, y, z] angular velocity in deg/s (matches raw data units).
        dt: time step in seconds.
        """
        gyro_rad = np.deg2rad(gyro_dps)  #convert to rad/s for internal calculations
        gyro_corrected = gyro_rad - self.bias  #remove estimated error from raw gyro data
 
        # Nonlinear propagation: compose current orientation with the incremental rotation this gyro reading implies over dt
        # Bascially the orientation update from our calculated error
        delta_rotation = Rotation.from_rotvec(gyro_corrected * dt)
        self.q = self.q * delta_rotation
 
        # Linearized error-state transition (standard small-angle result)
        # Bascially the error state update from calculated error
        wx, wy, wz = gyro_corrected
        skew = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0],
        ])
        F = np.eye(6)
        F[0:3, 0:3] -= skew * dt
        F[0:3, 3:6] = -np.eye(3) * dt
 
        self.P = F @ self.P @ F.T + self.Q
 
    def update(self, accel_ms2):
        """Correct orientation using the accelerometer's gravity direction.
        accel_ms2: [x, y, z] accelerometer reading, m/s^2.
        Only trustworthy when the sensor is near-static.
        """
        # turn accel reading into unit vector, only care about gravity direction
        accel_measured = np.asarray(accel_ms2, dtype=float)
        accel_norm = np.linalg.norm(accel_measured)
        accel_unit = accel_measured / accel_norm if accel_norm > 0 else np.array([0.0, 0.0, 1.0])

        g_world = np.array([0.0, 0.0, 1.0])  # unit "up"
        g_pred_body = self.q.inv().apply(g_world)  # rotates sensor to world/gravity frame then back to sensor frame
        # difference between what accel measured and expected measurement
        residual = accel_unit - g_pred_body
 
        gx, gy, gz = g_pred_body
        # H gives relationship between error state and expected measurement, linearized around current orientation
        H = np.zeros((3, 6))
        H[:, 0:3] = np.array([
            [0, -gz, gy],
            [gz, 0, -gx],
            [-gy, gx, 0],
        ])

        # adaptive measurement noise - if accel magnitude is far from 9.81, trust it less
        deviation = abs(accel_norm - self.gravity) / self.gravity
        R_effective = self.R*(1+self.accel_gating_strength*deviation**2)

        # Kalman gain calculation
        S = H @ self.P @ H.T + R_effective
        K = self.P @ H.T @ np.linalg.inv(S)
        correction = K @ residual
        # Orientation correction
        attitude_correction = Rotation.from_rotvec(correction[0:3])
        self.q = self.q * attitude_correction
        self.bias = self.bias + correction[3:6]
 
        self.P = (np.eye(6) - K @ H) @ self.P
 
    def get_euler_deg(self):
        """Current orientation as XYZ Euler angles, degrees - for
        VISUALIZATION/SANITY-CHECKING ONLY. DO not use in downstream calculations."""
        return self.q.as_euler("xyz", degrees=True)
 
 
def calibrate_from_static_trial(calib_df, label):
    """Compute gyro bias and sensor-to-segment alignment from a static
    standing-still calibration trial (Upright-INIT method).
 
    Returns (gyro_bias_dps, q_calib):
      - gyro_bias_dps: mean raw gyro reading over the window, in dps.
        Since the person was standing genuinely still, TRUE angular
        velocity was 0 so whatever the gyro reports IS bias, computed
        directly from raw data
      - q_calib: the sensor's orientation during calibration (a scipy
        Rotation), representing the fixed sensor-to-segment alignment
        offset (paper's BS_q, Upright-INIT: true segment orientation is
        assumed zero roll/pitch, forward-facing yaw). Apply during a real
        trial via: q_segment(t) = q_sensor(t) * q_calib.inv()
 
    2026/09/15 CAVEAT (found by testing against real data, not theoretical): yaw
    specifically showed meaningful residual drift (12-40+ degrees across
    devices) even within this static window, since it's unobservable by
    the accelerometer and only partially fixed by mean-bias removal.
    Roll/pitch converge reliably; treat any yaw-sensitive downstream
    result (e.g. abduction/adduction) with extra caution until validated
    against mocap.
    """
    gyro_bias_dps = calib_df[[f"{label}_gyro_x", f"{label}_gyro_y", f"{label}_gyro_z"]].mean().values
    gyro_bias_rad = np.deg2rad(gyro_bias_dps)
 
    initial_accel = calib_df.iloc[0][[f"{label}_acc_x", f"{label}_acc_y", f"{label}_acc_z"]].values.astype(float)
    initial_accel_unit = initial_accel / np.linalg.norm(initial_accel)
    r_align, _ = Rotation.align_vectors([[0, 0, 1]], [initial_accel_unit])
 
    ekf = OrientationEKF(r_align.as_quat(), gyro_bias_init=gyro_bias_rad.copy())
 
    t = calib_df["rel_time_ms"].values
    for i in range(1, len(calib_df)):
        dt = (t[i] - t[i - 1]) / 1000.0
        gyro_raw = calib_df.iloc[i][[f"{label}_gyro_x", f"{label}_gyro_y", f"{label}_gyro_z"]].values.astype(float)
        gyro_debiased = gyro_raw - gyro_bias_dps
        accel = calib_df.iloc[i][[f"{label}_acc_x", f"{label}_acc_y", f"{label}_acc_z"]].values.astype(float)
        ekf.predict(gyro_debiased, dt)
        ekf.update(accel)
 
    return gyro_bias_dps, ekf.q


def run_ekf_on_device(full_trial_df, label, gyro_bias_dps, q_calib, show_plot=True):
    """Run the calibrated orientation kalman filter on one device's data from synchronized trial DataFrame.
    Args (full_trial_df, label, gyro_bias_dps, q_calib):
    - full_trial_df: trial dataframe with all 3 IMU data synced to common time grid, output of process_trial_folder()
    - label: one of the DEVICE_LABELS values, e.g. "under_k", "above_k", "ankle"
    - gyro_bias_dps: bias computed from static calibration trial, in deg/s
    - q_calib: sensor-to-segment alignment quaternion, output of calibrate_from_static_trial()
    Returns (segment_quaternions, segment_eulers_deg, t_seconds):
    - segment_quaternions: list of scipy Rotation objects, one per timestep
    - segment_eulers_deg: (N, 3) array of roll/pitch/yaw, for plotting/inspection
    - t_seconds: time axis matching both arrays above (starts at 0)
    """
    gyro_bias_rad = np.deg2rad(gyro_bias_dps)
 
    initial_accel = full_trial_df.iloc[0][[f"{label}_acc_x", f"{label}_acc_y", f"{label}_acc_z"]].values.astype(float)
    r_align, _ = Rotation.align_vectors([[0, 0, 1]], [initial_accel / np.linalg.norm(initial_accel)])
    ekf = OrientationEKF(r_align.as_quat(), gyro_bias_init=gyro_bias_rad.copy())
 
    t = full_trial_df["rel_time_ms"].values
    segment_quaternions = []
    segment_eulers = []
    for i in range(1, len(full_trial_df)):
        dt = (t[i] - t[i-1]) / 1000.0
        gyro = full_trial_df.iloc[i][[f"{label}_gyro_x", f"{label}_gyro_y", f"{label}_gyro_z"]].values.astype(float)
        accel = full_trial_df.iloc[i][[f"{label}_acc_x", f"{label}_acc_y", f"{label}_acc_z"]].values.astype(float)
        ekf.predict(gyro, dt)
        ekf.update(accel)
        q_segment = ekf.q * q_calib.inv()  # apply calibration
        segment_quaternions.append(q_segment)
        segment_eulers.append(q_segment.as_euler("xyz", degrees=True))
 
    segment_eulers = np.array(segment_eulers)
    tt = (t[1:] - t[0]) / 1000
 
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tt, segment_eulers[:, 0], label="roll")
        ax.plot(tt, segment_eulers[:, 1], label="pitch")
        ax.plot(tt, segment_eulers[:, 2], label="yaw")
        ax.legend()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("deg")
        ax.set_title(f"{label} calibrated segment orientation")
        plt.show()
 
    return segment_quaternions, segment_eulers, tt


def relative_knee_angle(q_thigh, q_shank):
    """Compute the knee angle (shank relative to thigh) from two calibrated
    segment quaternions, following the paper's Eq 4 and Eq 5 (see references)
 
    q_thigh, q_shank: scipy Rotation objects (calibrated segment
    orientation, i.e. already through run_ekf_on_device) AT THE SAME
    TIMESTEP - guaranteed automatically if both come from the same synchronized trial_df.
 
    Returns [phi_IE, theta_AA, psi_FE] in degrees:
      IE = internal/external rotation
      AA = abduction/adduction
      FE = flexion/extension
    """
    q_TS = q_thigh.inv() * q_shank  # Eq 4: B_qTS = GB_qT* (x) GB_qS
    xyzw = q_TS.as_quat()  # scipy order: [x, y, z, w]
    q0, q1, q2, q3 = xyzw[3], xyzw[0], xyzw[1], xyzw[2]  # paper order: scalar first
 
    phi_IE = np.arctan2(-2*q1*q2 + 2*q0*q3, q1**2 + q0**2 - q3**2 - q2**2)
    theta_AA = np.arcsin(np.clip(2*q1*q3 + 2*q0*q2, -1, 1))
    psi_FE = np.arctan2(-2*q2*q3 + 2*q0*q1, q3**2 - q2**2 - q1**2 + q0**2)
 
    return np.degrees([phi_IE, theta_AA, psi_FE])