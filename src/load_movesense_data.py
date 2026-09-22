import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt


"""
Movesense JSON loader.
Movesense exports data in PACKETS, not individual samples: each entry in
data["data"] has one Timestamp (packet start, ms) and an array of several
samples (e.g. ArrayAcc) collected in that packet. This script reconstructs
a per-sample timestamp by evenly spacing samples between one packet's
timestamp and the next packet's timestamp.
ASSUMPTIONS:
- Timestamp units are milliseconds
- Acc values are already in m/s^2 (y-axis values around 9.8 = gravity).
"""
def plot_qc(df, sensor_label):
    """Stacked x/y/z subplots sharing a time axis - for eyeballing sync
    taps, sensor range headroom, and overall signal sanity."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for ax, axis_name in zip(axes, ["x", "y", "z"]):
        ax.plot(df["timestamp_ms"] / 1000, df[axis_name])
        ax.set_ylabel(f"{sensor_label}_{axis_name}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{sensor_label} QC plot")
    fig.tight_layout()
    plt.show()


def load_movesense_json(filepath):
    """Load a Movesense stream JSON file into DataFrame.
    Auto-detects the sensor key (e.g. 'acc', 'gyro') and its array key
    (e.g. 'ArrayAcc', 'ArrayGyro') from the first packet, so function works
    for either an acc_stream.json or a gyro_stream.json file.
    """
    with open(filepath) as f: # open json file
        raw = json.load(f)

    packets = raw["data"]
    sensor_key = list(packets[0].keys())[0] # identify accel or gyro data file
    array_key = [k for k in packets[0][sensor_key].keys() if k.startswith("Array")][0] # identify accel or gyro data values

    packet_timestamps = [p[sensor_key]["Timestamp"] for p in packets] # find all timestamps for time correction

    rows = []
    for i, packet in enumerate(packets): # go through all packets of data
        samples = packet[sensor_key][array_key] 
        t0 = packet[sensor_key]["Timestamp"]
        n = len(samples)

        if i < len(packets) - 1:
            dt = (packet_timestamps[i + 1] - t0) / n # rebuild timestamps evenly spaced btw packets
        else:
            # last packet has no "next" timestamp to interpolate with, reuse previous packet spacing
            dt = (packet_timestamps[i] - packet_timestamps[i - 1]) / n if i > 0 else 0

        for j, sample in enumerate(samples): # add rows for added timestamps and x/y/z values
            rows.append((t0 + j * dt, sample["x"], sample["y"], sample["z"]))

    df = pd.DataFrame(rows, columns=["timestamp_ms", "x", "y", "z"]) # build dataframe
    return df, sensor_key


def merge_acc_gyro(acc_df, gyro_df, label):
    """Merge one device's acc + gyro DataFrames into one table, prefixed
    with segment label (e.g. 'below_knee', 'above_knee', 'ankle'). Verifies the
    timestamps match exactly before merging - if they don't, something is
    wrong with the pairing and this will raise an error rather than merging misaligned data."""
    timestamps_match = (
        len(acc_df) == len(gyro_df)
        and (acc_df["timestamp_ms"].values == gyro_df["timestamp_ms"].values).all()
    )

    if timestamps_match:
        return pd.DataFrame({
            "timestamp_ms": acc_df["timestamp_ms"],
            f"{label}_acc_x": acc_df["x"], f"{label}_acc_y": acc_df["y"], f"{label}_acc_z": acc_df["z"],
            f"{label}_gyro_x": gyro_df["x"], f"{label}_gyro_y": gyro_df["y"], f"{label}_gyro_z": gyro_df["z"],
        })

    # else if timestamps don't match...
    print(f"[{label}] acc ({len(acc_df)} samples) and gyro ({len(gyro_df)} samples) "
          f"don't match 1:1 -- interpolating onto a shared grid instead.")
    # Interpolate onto whichever sensor has the smaller sample spacing
    acc_dt = acc_df["timestamp_ms"].diff().median() # find timestamp spacing for accel
    gyro_dt = gyro_df["timestamp_ms"].diff().median() # find timestamp spacing for gyro
    step_ms = min(acc_dt, gyro_dt) # select smaller timestamp spacing
 
    t_start = max(acc_df["timestamp_ms"].iloc[0], gyro_df["timestamp_ms"].iloc[0])
    t_end = min(acc_df["timestamp_ms"].iloc[-1], gyro_df["timestamp_ms"].iloc[-1])
    grid = np.arange(t_start, t_end, step_ms) # create common time grid using overlapping time
 
    return pd.DataFrame({
        "timestamp_ms": grid, # interpolate accel and gyro data onto common time grid
        f"{label}_acc_x": np.interp(grid, acc_df["timestamp_ms"], acc_df["x"]),
        f"{label}_acc_y": np.interp(grid, acc_df["timestamp_ms"], acc_df["y"]),
        f"{label}_acc_z": np.interp(grid, acc_df["timestamp_ms"], acc_df["z"]),
        f"{label}_gyro_x": np.interp(grid, gyro_df["timestamp_ms"], gyro_df["x"]),
        f"{label}_gyro_y": np.interp(grid, gyro_df["timestamp_ms"], gyro_df["y"]),
        f"{label}_gyro_z": np.interp(grid, gyro_df["timestamp_ms"], gyro_df["z"]),
    })



# !!!! NEED TO UPDATE ONCE WE HAVE ZEROING EVENT IN RAW DATA - TEMP APPROX FOR NOW
def synchronize_devices(device_dfs, grid_step_ms=10):
    """Align multiple devices (each with independent, non-comparable clocks)
    onto one shared relative-time grid.
 
    device_dfs: dict of {label: merged_df} (output of merge_acc_gyro), each
    still carrying its own device's raw timestamp_ms column.
    grid_step_ms: time step (default 10 ms), used for common time grid 
    """
    # Convert each device to its own relative time (starts at 0)
    rel_dfs = {}
    for label, df in device_dfs.items():
        df = df.copy() # copies dataframe for timestamp modifications
        df["rel_time_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0] # reset time, start at 0
        rel_dfs[label] = df
 
    # find where all IMUs overlap in time
    common_duration = min(df["rel_time_ms"].iloc[-1] for df in rel_dfs.values()) # find shortest time of the IMUs
    grid = np.arange(0, common_duration, grid_step_ms) # create common time grid for all IMUs
 
    merged = pd.DataFrame({"rel_time_ms": grid}) # create datafram with common time
    for label, df in rel_dfs.items(): # go through each device's dataframe
        value_cols = [c for c in df.columns if c not in ("timestamp_ms", "rel_time_ms")] # ignore the timestamps from individual devices
        for col in value_cols: 
            merged[col] = np.interp(grid, df["rel_time_ms"], df[col]) 
            # using linear interpolation to fill in values from each device to fit in common time grid
    return merged


def process_trial_folder(folder_path, device_labels, show_qc_plots=False):
    """Process one full trial folder containing acc + gyro JSON files for
    multiple devices, and return one synchronized DataFrame for the trial.
 
    folder_path: path to a folder, expected to contain files named
    '{session}_{device_id}_{acc|gyro}_stream.json' for each device.
 
    device_labels: dict mapping device_id -> segment label, e.g.
    {"233830000601": "under_k", "233830000617": "above_k", "233830000625": "ankle"}
 
    show_qc_plots: if True, pops up a stacked x/y/z QC plot for each
    device's acc and gyro signals before merging. Off by default -- turn
    on when you want to actually eyeball a trial (new motion type, first
    look at a session, checking for clipping), off for routine reprocessing
    once you've already confirmed a trial looks clean.
 
    Returns the same kind of shared-time-grid DataFrame that
    synchronize_devices() produces.
    """
    folder_path = Path(folder_path)
    pattern = re.compile(r"(\d{8}T\d{6}Z)_(\d+)_(acc|gyro)_stream\.json")
 
    # Group files by device_id: {device_id: {"acc": path, "gyro": path}}
    files_by_device = {}
    for f in folder_path.glob("*.json"):
        match = pattern.match(f.name)
        if not match:
            print(f"Skipping unrecognized filename: {f.name}")
            continue
        _, device_id, sensor_type = match.groups()
        files_by_device.setdefault(device_id, {})[sensor_type] = f
 
    device_dfs = {}
    for device_id, files in files_by_device.items():
        if "acc" not in files or "gyro" not in files:
            print(f"Skipping device {device_id}: missing acc or gyro file.")
            continue
        if device_id not in device_labels:
            print(f"Skipping device {device_id}: no label provided in device_labels.")
            continue
 
        label = device_labels[device_id]
        acc_df, _ = load_movesense_json(files["acc"])
        gyro_df, _ = load_movesense_json(files["gyro"])
 
        if show_qc_plots:
            plot_qc(acc_df, f"{label}_acc")
            plot_qc(gyro_df, f"{label}_gyro")
 
        device_dfs[label] = merge_acc_gyro(acc_df, gyro_df, label)
 
    return synchronize_devices(device_dfs)


