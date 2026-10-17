"""
Magnetometer calibration for Movesense IMUs.
The idea:
    A perfect magnetometer, rotated through every orientation in a clean field,
    traces a sphere centred on the origin with radius = local field strength.
    Real sensors trace a shifted, squashed ellipsoid instead:
      - hard-iron  = magnetised stuff that moves WITH the sensor (battery, PCB,
                     clip) -> adds a constant vector -> shifts the centre.
      - soft-iron  = nearby material that bends the field -> stretches/squashes
                     the sphere into an ellipsoid.
    Calibration = fit that ellipsoid, then undo it:
        m_calibrated = W @ (m_raw - offset)
    offset (3,)  -> hard-iron correction (the ellipsoid centre)
    W      (3,3) -> soft-iron correction (squashes the ellipsoid back to a sphere)

Workflow:
    1. calibrate_folder()        -> fits each device's figure-8/tumble file,
                                    saves one JSON per device
    2. check_static_mag()        -> sanity check on a standing-still recording
    3. apply_mag_calibration()   -> used on every trial before the EKF
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

MAG_COLS = ["mag_x", "mag_y", "mag_z"]
DEVICE_ID_PATTERN = re.compile(r"_(\d{12})_")  # e.g. ..._233830000617_magn_stream.json


# Loading file
def load_mag_json(path):
    """Load a Movesense magn_stream JSON into a DataFrame (time_s, mag_x/y/z in uT).
    Same as load_movesense_json() but for magnetometer data
    """
    with open(path) as f:
        packets = json.load(f)["data"]

    packet_ts = np.array([p["magn"]["Timestamp"] for p in packets], dtype=float)
    n_per_packet = len(packets[0]["magn"]["ArrayMagn"])
    dt_ms = np.median(np.diff(packet_ts)) / n_per_packet

    rows = []
    for ts, p in zip(packet_ts, packets):
        for i, s in enumerate(p["magn"]["ArrayMagn"]):
            rows.append((ts + i * dt_ms, s["x"], s["y"], s["z"]))

    df = pd.DataFrame(rows, columns=["time_ms"] + MAG_COLS)
    df["time_s"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0
    return df[["time_s"] + MAG_COLS]


# Fitting 
def fit_sphere(m):
    """Hard-iron only. Fits |m - offset| = radius by linear least squares.
    """
    A = np.column_stack([2 * m, np.ones(len(m))])
    b = np.sum(m**2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    offset = sol[:3]
    radius = np.sqrt(sol[3] + offset @ offset)
    return offset, np.eye(3), radius


def fit_ellipsoid(m):
    """Hard- and soft-iron. Fits a general ellipsoid  m'Mm + 2n'm = 1.
    """
    x, y, z = m.T
    D = np.column_stack([x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z])
    coef, *_ = np.linalg.lstsq(D, np.ones(len(m)), rcond=None)
    a, b, c, f, g, h, p, q, r = coef

    M = np.array([[a, h, g],
                  [h, b, f],
                  [g, f, c]])
    n = np.array([p, q, r])

    offset = -np.linalg.solve(M, n) # center of the ellipsoid
    k = 1 + offset @ M @ offset # rescales so (m-offset)'(M/k)(m-offset) = 1
    M_unit = M / k

    eig = np.linalg.eigvalsh(M_unit)
    if np.any(eig <= 0):
        raise ValueError("Fit is not an ellipsoid - orientation coverage is too poor.")

    semi_axes = 1 / np.sqrt(eig)
    radius = np.prod(semi_axes) ** (1 / 3)
    W = radius * np.real(sqrtm(M_unit)) # maps the ellipsoid onto a sphere of radius
    return offset, W, radius


def octant_coverage(m_cal, min_frac=0.02):
    """How many of the 8 direction octants hold at least `min_frac` of the samples.
    A good tumble covers all 8. A flat figure-8 in one plane covers ~4.
    """
    u = m_cal / np.linalg.norm(m_cal, axis=1, keepdims=True)
    octant = (u[:, 0] > 0) * 1 + (u[:, 1] > 0) * 2 + (u[:, 2] > 0) * 4
    frac = np.bincount(octant, minlength=8) / len(u)
    return int(np.sum(frac >= min_frac))


def fit_mag_calibration(df, method="auto", min_octants=6):
    """Fit a calibration to one device's tumble recording.
    method: "sphere"    -> hard-iron only (robust, needs less coverage)
            "ellipsoid" -> hard + soft iron (needs good coverage)
            "auto"      -> ellipsoid if coverage >= min_octants, else sphere
    """
    m = df[MAG_COLS].to_numpy()

    # Sphere fit first: always works, and lets us measure coverage
    offset, W, radius = fit_sphere(m)
    used = "sphere"
    coverage = octant_coverage(m - offset)
    warnings = []

    if method == "ellipsoid" or (method == "auto" and coverage >= min_octants):
        try:
            offset, W, radius = fit_ellipsoid(m)
            used = "ellipsoid"
        except ValueError as e:
            warnings.append(f"Ellipsoid fit failed ({e}); kept sphere fit.")
    elif method == "auto":
        warnings.append(f"Only {coverage}/8 octants covered - used sphere (hard-iron only).")

    m_cal = (m - offset) @ W.T
    norm = np.linalg.norm(m_cal, axis=1)
    norm_std_pct = 100 * norm.std() / norm.mean()

    if coverage < min_octants:
        warnings.append("Low orientation coverage - re-record with a fuller tumble.")
    if norm_std_pct > 3:
        warnings.append(f"Calibrated magnitude varies {norm_std_pct:.1f}% - expected < ~3%.")
    if not 25 < radius < 75:
        warnings.append(f"Fitted field {radius:.1f} uT is far from Earth's (~50 uT) - "
                        "check for nearby metal or a bad fit.")

    return {
        "method": used,
        "offset_uT": offset.tolist(),
        "W": W.tolist(),
        "radius_uT": float(radius),
        "n_samples": int(len(m)),
        "octants_covered": coverage,
        "norm_std_pct": float(norm_std_pct),
        "warnings": warnings,
    }


# Applying calibration
def apply_mag_calibration(df, calib, cols=MAG_COLS):
    """Return a copy of df with calibrated magnetometer columns.
    Works on any DataFrame that has the mag columns - raw, merged or synced.
    """
    out = df.copy()
    offset = np.array(calib["offset_uT"])
    W = np.array(calib["W"])
    out[cols] = (out[cols].to_numpy() - offset) @ W.T
    return out


def save_calibration(calib, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


# Checks and plots
def check_static_mag(static_df, calib):
    """Sanity-check a calibration with a standing-still recording.

    - noise_std_uT: sensor noise floor per axis (sensor isn't moving).
    - field_error_pct: how far the calibrated |m| at the standing spot is from
      the calibration radius. A big number means the field where you stand
      is disturbed (force plates, steel) or the calibration is off.
    """
    cal = apply_mag_calibration(static_df, calib)
    norm = np.linalg.norm(cal[MAG_COLS].to_numpy(), axis=1)
    return {
        "noise_std_uT": static_df[MAG_COLS].std().round(3).to_dict(),
        "calibrated_field_uT": float(norm.mean()),
        "field_error_pct": float(100 * (norm.mean() - calib["radius_uT"]) / calib["radius_uT"]),
    }


def plot_mag_calibration(df, calib, title=""):
    """Raw (grey) vs calibrated (color) in three 2D projections, plus |m| over time.
    Calibrated points should form circles centered on 0 with equal radii,
    and the |m| line should be flat.
    """
    raw = df[MAG_COLS].to_numpy()
    cal = apply_mag_calibration(df, calib)[MAG_COLS].to_numpy()
    r = calib["radius_uT"]

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax[k].scatter(raw[:, i], raw[:, j], s=2, c="grey", label="raw")
        ax[k].scatter(cal[:, i], cal[:, j], s=2, c="tab:blue", label="calibrated")
        ax[k].add_patch(plt.Circle((0, 0), r, fill=False, ls="--", color="tab:red"))
        ax[k].set_xlabel(MAG_COLS[i]); ax[k].set_ylabel(MAG_COLS[j])
        ax[k].set_aspect("equal"); ax[k].axhline(0, lw=0.5); ax[k].axvline(0, lw=0.5)
    ax[0].legend(markerscale=4)

    ax[3].plot(df["time_s"], np.linalg.norm(raw, axis=1), c="grey", label="raw |m|")
    ax[3].plot(df["time_s"], np.linalg.norm(cal, axis=1), c="tab:blue", label="calibrated |m|")
    ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("uT"); ax[3].legend()

    fig.suptitle(f"{title}  [{calib['method']}, {calib['octants_covered']}/8 octants, "
                 f"|m| std {calib['norm_std_pct']:.1f}%]")
    fig.tight_layout()
    plt.show()


# Folder-level driver
def calibrate_folder(folder, out_dir, method="auto", show_plots=True):
    """Fit and save a calibration for every device with a magn file in `folder`.
    Saves out_dir/mag_calib_<deviceID>.json. If a device has several magn
    files, they are stacked (more orientations = better fit).
    """
    folder, out_dir = Path(folder), Path(out_dir)
    by_device = {}
    for path in sorted(folder.glob("*magn*.json")):
        match = DEVICE_ID_PATTERN.search(path.name)
        if match:
            by_device.setdefault(match.group(1), []).append(path)

    if not by_device:
        raise FileNotFoundError(f"No *magn*.json files with a device ID found in {folder}")

    results = {}
    for device_id, paths in by_device.items():
        df = pd.concat([load_mag_json(p) for p in paths], ignore_index=True)
        calib = fit_mag_calibration(df, method=method)
        calib["device_id"] = device_id
        calib["source_files"] = [p.name for p in paths]
        save_calibration(calib, out_dir / f"mag_calib_{device_id}.json")
        results[device_id] = calib

        print(f"{device_id}: {calib['method']}, radius {calib['radius_uT']:.1f} uT, "
              f"{calib['octants_covered']}/8 octants, |m| std {calib['norm_std_pct']:.1f}%")
        for w in calib["warnings"]:
            print(f"   WARNING: {w}")
        if show_plots:
            plot_mag_calibration(df, calib, title=device_id)

    return results
