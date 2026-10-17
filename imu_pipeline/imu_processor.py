import json
import sys
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# Add src/ to Python's search path, regardless of where this script is run from
sys.path.append(str(Path(__file__).parent / "src"))
from load_movesense_data import load_movesense_json, merge_acc_gyro, synchronize_devices, plot_qc, process_trial_folder
from IMU_filters import OrientationEKF, calibrate_from_static_trial, run_ekf_on_device, relative_knee_angle, estimate_flexion_axis
from mag_calibration import calibrate_folder, load_mag_json, load_calibration, check_static_mag
from scipy.spatial.transform import Rotation

DEVICE_LABELS = {
    "233830000601": "under_k", # IMU on lower leg, under knee
    "233830000617": "above_k", # IMU on upper leg, above knee
    "233830000625": "ankle", # IMU on ankle or foot
}

# Session settings: change per session
SESSION = "wmag_20261002"
CALIB_FOLDER = f"data/raw/stand_{SESSION}"
AXIS_REFERENCE_FOLDER = f"data/raw/lunge_{SESSION}"   # hinge axis estimated from this trial
TRIAL_FOLDER = f"data/raw/lunge_{SESSION}"   # the exercise to process
PLOT_TITLE = "Lunge Trial" # for plotting

# Exercise trial to process
full_trial_df = process_trial_folder(TRIAL_FOLDER, DEVICE_LABELS, show_qc_plots=False)
# Calibrate from the static standing trial
calib_df = process_trial_folder(CALIB_FOLDER, DEVICE_LABELS)
# Axis reference to estimate hinge axis
axis_ref_df = process_trial_folder(AXIS_REFERENCE_FOLDER, DEVICE_LABELS)


# under knee calibration
under_k_axis = estimate_flexion_axis(axis_ref_df, "under_k")
under_k_bias, under_k_q_calib = calibrate_from_static_trial(calib_df, "under_k", lateral_axis_sensor=under_k_axis)
under_k_quats, under_k_eulers, tt = run_ekf_on_device(full_trial_df, "under_k", under_k_bias, under_k_q_calib, show_plot=True)

# above knee calibration
above_k_axis = estimate_flexion_axis(axis_ref_df, "above_k")
above_k_bias, above_k_q_calib = calibrate_from_static_trial(calib_df, "above_k", lateral_axis_sensor=above_k_axis)
above_k_quats, above_k_eulers, _ = run_ekf_on_device(full_trial_df, "above_k", above_k_bias, above_k_q_calib)


# Get calibration-period knee angles (needed for Eq 6 zeroing)
# Run the same EKF on the STATIC calibration trial itself, not just the exercise trial
calib_under_k_quats, _, _ = run_ekf_on_device(calib_df, "under_k", under_k_bias, under_k_q_calib, show_plot=False)
calib_above_k_quats, _, _ = run_ekf_on_device(calib_df, "above_k", above_k_bias, above_k_q_calib, show_plot=False)

calib_knee_angles = np.array([
    relative_knee_angle(qt, qs)
    for qt, qs in zip(calib_above_k_quats, calib_under_k_quats)
])
knee_angle_offset = calib_knee_angles.mean(axis=0)

# Compute knee angle for the actual trial (thigh = above_k, shank = under_k)
knee_angles = np.array([
    relative_knee_angle(qt, qs)
    for qt, qs in zip(above_k_quats, under_k_quats)
])
knee_angles_zeroed = knee_angles - knee_angle_offset  # Eq 6
knee_angles_zeroed[:, 0] = np.unwrap(knee_angles_zeroed[:, 0], period=360)
knee_angles_zeroed[:, 1] = np.unwrap(knee_angles_zeroed[:, 1], period=360)
knee_angles_zeroed[:, 2] = np.unwrap(knee_angles_zeroed[:, 2], period=360)


# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tt, knee_angles_zeroed[:, 0], label="IE (int/ext rotation)")
ax.plot(tt, knee_angles_zeroed[:, 1], label="AA (abduction/adduction)")
ax.plot(tt, knee_angles_zeroed[:, 2], label="FE (flexion/extension)")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.legend()
ax.set_xlabel("time (s)")
ax.set_ylabel("deg")
ax.set_title(f"Knee angle - {PLOT_TITLE}")
plt.show()


# ------- Below is code for project components that we ran out of time for but could be continued later -------

# ankle calibration - add ankle stability/pronation to analysis of injury risk
# not needed for knee angle analysis so commented out for now
'''
ankle_bias, ankle_q_calib = calibrate_from_static_trial(calib_df, "ankle")
# ankle_quats, ankle_eulers, _ = run_ekf_on_device(full_trial_df, "ankle", ankle_bias, ankle_q_calib)
'''
# Magnetometer calibration - may give more precise orientation estimates from IMUs
# due to how much noise is in the magnetometer data, we didn't have time to implement this in the main analysis
'''
calibs = calibrate_folder("data/raw/mag_cal", "data/processed/mag_calibration")
static_files = {
    "233830000601": r"C:\Dev\imu_project\data\raw\stand_wmag_20261002\20261002T081416Z_233830000601_magn_stream.json",
    "233830000617": r"C:\Dev\imu_project\data\raw\stand_wmag_20261002\20261002T081416Z_233830000617_magn_stream.json",
    "233830000625": r"C:\Dev\imu_project\data\raw\stand_wmag_20261002\20261002T081416Z_233830000625_magn_stream.json",
}
for dev_id, path in static_files.items():
    print(f"--- static check: {dev_id} ---")
    print(check_static_mag(load_mag_json(path), calibs[dev_id]))
'''
