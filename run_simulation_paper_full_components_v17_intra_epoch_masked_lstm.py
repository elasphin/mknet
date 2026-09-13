"""Single-file GNSS/LEO/INS + Masked KalmanNet reproduction.

Scientific policy
-----------------
The implementation follows Yan et al. (2026) and its cited references wherever
the papers publish enough information to reproduce a step. Two intentional
project substitutions are preserved exactly as requested:
1) pseudorange-only measurements (no pseudorange-rate/Doppler);
2) TLE/SGP4 LEO orbits instead of STK/HPOP.

Published components retained/restored in this revision include:
- 15-state ECEF INS error propagation and Eq. (8) IMU-compensation interface;
- full 15-row classical and learned measurement corrections in the Eq. (7)
  order [delta-p, delta-v, delta-theta, b_a, b_g]; the six bias-gain rows are
  retained and zero-initialized, but no artificial direct bias target is invented;
- Yan Eqs. (1),(2),(5) LEO pseudorange geometry/atmosphere/light-time path;
- the LEO stochastic variance is intentionally kept identical to v6 at the user's
  request: ionosphere + troposphere + Ref. [35] Eq. (18) elevation-only MP/NLOS,
  with URA/receiver-noise terms not added to this project branch;
- Eqs. (10)-(21) fixed offline X/M/Y data, masked CNN/LSTM/attention
  Eqs. (22)-(29), Table III hyperparameters, Eq. (30)/(32) supervised training,
  and Ref. [15]-style alternating optimization;
- Fig. 8 KG state-update path and the eta_k output/IMU-feedback interface;
- online raw-innovation FDE before the learned KG, Ref. [33] DIA
  detection/identification and Yan Eq. (34) adaptation bookkeeping. A single
  FDE_DIA_ON boolean block permits a controlled ON/OFF ablation of this entire
  module while leaving the rest of the online pipeline unchanged;
- hard fault elimination before Masked CLA without the former unpublished
  post-exclusion rejection loop;
- covariance-based HPL/VPL outputs for the PL/Stanford branch using the explicit
  Ref. [35] HPL/VPL equations;
- closed-loop spectral-radius logging for the stability analysis.

This revision keeps the non-invasive divergence diagnostics and uses a compact
training completion guided by Ref. [15], which Yan et al. explicitly cite for
alternating optimization:
- eta_k is not trained with an extra direct sensor-error MSE absent from Yan
  Eq. (32); its unpublished timing is completed through the next-interval
  Eq. (8)+INS state consequence and the same Eq. (30)-type state loss;
- a Data01-only affine input standardization is applied as numerical conditioning;
  this changes neither the published feature definitions nor any navigation/FDE/
  state-update equation;
- Yan's published fixed offline X/M/Y construction (Eqs. 18-20) remains the sole
  training dataset. Each padded X_bar[k] is processed as one masked sequence at
  fusion epoch k: CNN -> pooling/flatten -> masked LSTM -> attention -> FC. The
  LSTM recurrence therefore runs inside the padded sequence and is reset for the
  next fusion epoch; no unpublished hidden/cell carry is introduced between
  navigation epochs. The filter and representation blocks are optimized
  alternately following the general training principle of Ref. [15]. No
  full-trajectory BPTT or network-regenerated training features are used.

This CNN/LSTM revision retains the Pooling and Flatten stages drawn explicitly in
Yan Fig. 8. The mask controls valid positions inside each padded X_bar[k]. Because
valid entries form a contiguous prefix, packed-sequence LSTM evaluation is used:
valid positions update the recurrent state and padded suffix positions are skipped,
which is the simple implementation of the behavior described around Eqs. (24)-(25).
The exact pooling operator remains unpublished; the implemented same-length masked
max-pool is labelled as a Fig.-8-guided completion.

The separate post-training recursive Data01 rollout remains diagnostic only and
never updates a network parameter or feeds information into Data02.

The paper does not publish enough information to uniquely reconstruct the exact
pooling operator in Fig. 8, the eta_k training target/timing, feature scaling,
the random LEO masking-angle distribution, the exact FC tensorization, the LEO
clock-error generator, or the FDE false-alarm probability. Those details are not
silently attributed to Yan et al.; compact completions are labelled explicitly.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import csv
import os
import json
import math
import random
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as SpatialRotation
from scipy.stats import chi2
from sgp4.api import SGP4_ERRORS, Satrec, WGS72
from sgp4.io import verify_checksum


# =============================================================================
# CONFIGURATION, CONSTANTS, AND COORDINATE/TIME UTILITIES
# =============================================================================
# -----------------------------------------------------------------------------
# Physical constants (SI)
# -----------------------------------------------------------------------------
EARTH_SEMI_MAJOR_AXIS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SEMI_MINOR_AXIS_M = EARTH_SEMI_MAJOR_AXIS_M * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_ROTATION_RATE_RADPS = 7.292115e-5
EARTH_GRAVITATIONAL_PARAMETER_M3PS2 = 3.986004418e14
SPEED_OF_LIGHT_MPS = 299792458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0

# -----------------------------------------------------------------------------
# Current experiment paths -- exact Kaggle layout
# -----------------------------------------------------------------------------
IN_KAGGLE = (
    Path("/kaggle/input").is_dir()
    and Path("/kaggle/working").is_dir()
)

KAGGLE_PROJECT_ROOT = Path(
    os.environ.get(
        "MKNET_PROJECT_ROOT",
        "/kaggle/input/datasets/elasphin/mknet-project",
    )
)

# In this Kaggle dataset Data01/Data02 live directly under the dataset root.
DATASET_ROOT = KAGGLE_PROJECT_ROOT
SMARTPNT_ROOT = KAGGLE_PROJECT_ROOT

TRAIN_DATASET_DIR = (
    DATASET_ROOT / "Data01_20230102_ISA-100C_Vehicle_Complex"
)

TEST_DATASET_DIR: Path | None = (
    DATASET_ROOT / "Data02_20220309_ISA-100C_Vehicle_Complex"
)

DATASET_DIR = TRAIN_DATASET_DIR
README_XML_PATH = TRAIN_DATASET_DIR / "README.xml"

# IMUErrorModel.txt is stored at the Kaggle dataset root.
IMU_ERROR_MODEL_PATH = KAGGLE_PROJECT_ROOT / "IMUErrorModel.txt"

ROVE_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ROVE_GroundTruth.txt"
IMU_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ISA-100C_GroundTruth.txt"
RINEX_OBS_PATH = TRAIN_DATASET_DIR / "ROVE.23O"
IMR_PATH = TRAIN_DATASET_DIR / "ISA-100C.imr"

# Data01 precise products are stored at the dataset root.
SP3_PATH = KAGGLE_PROJECT_ROOT / "WUM0MGXFIN_20230020000_01D_05M_ORB.SP3"
CLK_PATH = KAGGLE_PROJECT_ROOT / "WUM0MGXFIN_20230020000_01D_30S_CLK.CLK"

NAV_PATH = TRAIN_DATASET_DIR / "brdm0020.23p"

# TLE directory confirmed in the Kaggle dataset.
LEO_TLE_DIR = KAGGLE_PROJECT_ROOT / "LEO_TLE"


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "all"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer or 'none'")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _default_output_dir() -> Path:
    override = os.environ.get("MKNET_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    if IN_KAGGLE:
        return Path("/kaggle/working/direct_run")
    return Path("direct_run")


# Current project assumptions / settings.
MAX_TRUTH_INTERPOLATION_GAP_S = 2.0
MIN_GNSS_ELEVATION_DEG = 5.0
USE_IONOSPHERE = True
USE_TROPOSPHERE = True
LEO_MIN_ELEVATION_DEG = 10.0
LEO_SEED = 0
TLE_MAX_AGE_DAYS = 1.0
TLE_ALLOW_DEGRADED_EOP = False
TLE_ALLOW_NON_TLE_FILES = True
LEO_TX_EPSILON_POSITION_M = 1e-3
LEO_TX_MAX_ITERATIONS = 20

# =============================================================================
# FDE / DIA EXPERIMENT BLOCK
# =============================================================================
# Change ONLY this line for the controlled comparison:
#   True  -> paper online FDE/DIA block is present (Yan Sec. II-D, Eqs. 33-34)
#   False -> the whole FDE/DIA block is bypassed; the same raw measurements go
#            directly to Masked-CLA/KalmanNet.
#
# Keep True for the paper-reproduction run. Use False only as an ablation run
# to measure the effect of adding/removing the FDE/DIA block.
FDE_DIA_ON = True

# =============================================================================
# DIVERGENCE-DIAGNOSTIC CONTROLS
# =============================================================================
# Keep the published Fig. 2 / Eq. (8) eta_k feedback ON in the reproduction path.
# The preceding eta-OFF ablation showed that disabling this feedback did not remove
# the divergence, so the paper-path behavior is restored here.
ETA_FEEDBACK_ON = True

# Diagnostic only: after training, replay Data01 recursively with the trained
# Masked-CLA using the SAME online closed-loop logic used for Data02.  This does
# not train on the rollout, modify any model weight, replace any paper equation,
# or feed Data01 diagnostic results into Data02.  It only distinguishes
# fixed-offline-training instability from cross-dataset generalization failure.
RUN_RECURSIVE_DATA01_DIAGNOSTIC = True

# The classical Data02 baseline was already verified stable (~7.84 m 3D RMSE in
# the preceding run). Repeating it would add runtime but no new diagnostic value.
RUN_CLASSICAL_TEST_BASELINE = False

# Numerical conditioning only. Yan et al. publish the physical feature vector,
# zero-padding, and masking (Eqs. 15-21) but do not state a feature-scaling rule.
# The prior closed-loop diagnostics showed that raw recursive features leave the
# Data01 offline range by many orders of magnitude. Standardization therefore
# applies one fixed affine reparameterization fitted ONLY on Data01; the same
# stored statistics are reused unchanged for recursive Data01 and independent
# Data02. Targets, innovations used by the state update, FDE, INS, and all
# navigation equations remain in physical units. Set False only for a controlled
# numerical ablation; True is the corrected numerically conditioned run.
FEATURE_STANDARDIZATION_ON = True

# Runtime/accuracy trade-off controls.
# The final scientific LEO mask remains exactly LEO_MIN_ELEVATION_DEG.  This guard is
# used only to avoid expensive transmit-time iteration for satellites clearly below it.
LEO_PREFILTER_GUARD_DEG = 0.5

# For ||A*dt||_1 <= 0.2, the Taylor-10 remainder bound is O(1e-15) in matrix norm.
# Larger steps/states fall back to scipy.linalg.expm exactly.
VAN_LOAN_TAYLOR_ORDER = 10
VAN_LOAN_TAYLOR_MAX_NORM_1 = 0.20
VAN_LOAN_VALIDATE_CALLS = 8

# Yan et al. Eq. (2): the text places the ionosphere approximately between
# 100 and 1000 km. The paper denotes these limits h_L and h_H but does not
# list separate numerical values; using those stated bounds is a paper-guided
# completion rather than an additional external model.
LEO_IONOSPHERE_LOWER_HEIGHT_M = 100_000.0
LEO_IONOSPHERE_UPPER_HEIGHT_M = 1_000_000.0

# Ref. [35] error-model constants used by Yan Eq. (3).
# Ref. [35] states typical URA values of 1-2 m; the midpoint is used because
# Yan et al. do not publish the exact simulated LEO URA index.
LEO_URA_SIGMA_M = 1.5
LEO_CODE_LOOP_BANDWIDTH_HZ = 2.0
LEO_CORRELATOR_SPACING_CHIPS = 0.1
LEO_CORRELATOR_ACCUMULATION_S = 0.02
LEO_CODE_CHIPPING_RATE_HZ = 1.023e6

# Yan Eq. (4) cites goGPS [37]. These are the sample parameters published by
# goGPS for its elevation/CN0 weighting law.
GOGPS_A = 30.0
GOGPS_S0_DBHZ = 10.0
GOGPS_S1_DBHZ = 50.0
GOGPS_A_DB = 20.0

# Ref. [35] Eqs. (57)-(58), used for the PL branch shown in Yan Fig. 2/Fig. 20.
PL_VERTICAL_MULTIPLIER = 5.33
PL_HORIZONTAL_MULTIPLIER = 6.0


@dataclass(frozen=True)
class GPSTime:
    week: int
    tow_s: float


def calendar_to_gpst_seconds(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    time_system: str = "GPS",
) -> float:
    """Calendar epoch -> continuous GPST seconds."""
    sec_int = int(math.floor(second))
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    base = (dt - GPS_EPOCH).total_seconds() + sec_int + (second - sec_int)
    system = time_system.upper()
    if system in {"GPS", "GPST", "GAL", "GST", "QZS", "QZSST", "IRN"}:
        return float(base)
    if system in {"BDT", "BDS"}:
        return float(base + 14.0)
    if system in {"UTC", "GLO"}:
        return float(base + 18.0)
    raise ValueError(f"Unsupported time system: {time_system}")


def gpst_seconds_to_week_tow(time_gpst_s: float) -> GPSTime:
    week = int(math.floor(float(time_gpst_s) / GPS_WEEK_S))
    return GPSTime(week, float(time_gpst_s) - week * GPS_WEEK_S)


def anchor_imr_tow_to_gpst_seconds(
    tow_s: np.ndarray,
    anchor_time_gpst_s: float,
) -> np.ndarray:
    """Attach GPS week(s) to IMR TOW and validate strict time ordering.

    Runtime optimization: the original sample-by-sample week-unwrapping loop is
    expressed with vectorized adjacent-TOW jumps and an integer cumulative sum.
    The rollover rule, GPST values, and validation are unchanged.
    """
    tow_s = np.asarray(tow_s, dtype=float).reshape(-1)
    if tow_s.size == 0:
        return tow_s.copy()
    if not np.all(np.isfinite(tow_s)):
        raise ValueError("IMR TOW contains non-finite values")
    if np.any((tow_s < 0.0) | (tow_s >= GPS_WEEK_S)):
        raise ValueError("IMR TOW must lie in [0, 604800) seconds")
    if not math.isfinite(float(anchor_time_gpst_s)):
        raise ValueError("RINEX anchor time must be finite")

    anchor_week = int(math.floor(float(anchor_time_gpst_s) / GPS_WEEK_S))
    candidates = np.array([
        (anchor_week + offset) * GPS_WEEK_S + tow_s[0]
        for offset in (-1, 0, 1)
    ])
    first = float(
        candidates[np.argmin(np.abs(candidates - float(anchor_time_gpst_s)))]
    )
    first_week = int(math.floor(first / GPS_WEEK_S))

    if tow_s.size == 1:
        return np.asarray([first], dtype=float)

    delta_tow = np.diff(tow_s)
    week_step = np.zeros(delta_tow.size, dtype=np.int64)
    week_step[delta_tow < -0.5 * GPS_WEEK_S] = 1
    week_step[delta_tow > 0.5 * GPS_WEEK_S] = -1
    week_offset = np.empty(tow_s.size, dtype=np.int64)
    week_offset[0] = 0
    np.cumsum(week_step, out=week_offset[1:])
    week_number = first_week + week_offset

    time_gpst_s = week_number.astype(float) * GPS_WEEK_S + tow_s
    time_gpst_s[0] = first
    if np.any(np.diff(time_gpst_s) <= 0.0):
        raise ValueError("anchored IMR GPST tags must be strictly increasing")
    return time_gpst_s


def validate_strict_time_axis(name: str, time_gpst_s: np.ndarray) -> np.ndarray:
    """Return a finite, strictly increasing one-dimensional GPST time axis."""
    time_gpst_s = np.asarray(time_gpst_s, dtype=float).reshape(-1)
    if time_gpst_s.size == 0:
        raise ValueError(f"{name} time axis is empty")
    if not np.all(np.isfinite(time_gpst_s)):
        raise ValueError(f"{name} time axis contains non-finite values")
    if time_gpst_s.size > 1 and np.any(np.diff(time_gpst_s) <= 0.0):
        raise ValueError(f"{name} time axis must be strictly increasing")
    return time_gpst_s


@dataclass(frozen=True)
class PropagationSegment:
    start_time_gpst_s: float
    end_time_gpst_s: float
    imu_index: int


@dataclass(frozen=True)
class FusionMarker:
    time_gpst_s: float
    fusion_index: int


def build_exact_fusion_timeline(
    imu_time_gpst_s: np.ndarray,
    fusion_time_gpst_s: np.ndarray,
    *,
    through_last_fusion: bool = True,
):
    """Yield exact propagation/fusion events without materializing the timeline.

    This is numerically identical to the previous scheduler, but it avoids creating
    and retaining one Python object for every IMU propagation interval.
    """
    imu_time = validate_strict_time_axis("IMU", imu_time_gpst_s)
    fusion_time = np.asarray(fusion_time_gpst_s, dtype=float).reshape(-1)

    if fusion_time.size == 0:
        return
    validate_strict_time_axis("fusion", fusion_time)
    if fusion_time[0] < imu_time[0] or fusion_time[-1] > imu_time[-1]:
        raise ValueError("fusion epochs must lie inside the IMU time span")

    if through_last_fusion:
        stop = int(np.searchsorted(imu_time, fusion_time[-1], side="left")) + 1
        imu_time = imu_time[:min(max(stop, 2), len(imu_time))]

    fusion_index = 0
    segment_start = float(imu_time[0])

    for imu_index in range(1, len(imu_time)):
        interval_end = float(imu_time[imu_index])

        while (
            fusion_index < len(fusion_time)
            and fusion_time[fusion_index] <= interval_end + 1e-12
        ):
            t = float(fusion_time[fusion_index])
            if t < segment_start - 1e-12:
                raise RuntimeError(
                    "fusion scheduler encountered a fusion epoch before "
                    "the current propagation segment"
                )
            if t > segment_start + 1e-12:
                yield PropagationSegment(segment_start, t, imu_index)
            yield FusionMarker(t, fusion_index)
            segment_start = t
            fusion_index += 1

        if interval_end > segment_start + 1e-12:
            yield PropagationSegment(segment_start, interval_end, imu_index)
        segment_start = interval_end

    if fusion_index != len(fusion_time):
        raise RuntimeError("not all requested fusion epochs were scheduled")


def ecef_to_llh(position_ecef_m: np.ndarray) -> tuple[float, float, float]:
    """ECEF [m] -> WGS-84 latitude [rad], longitude [rad], height [m]."""
    x, y, z = np.asarray(position_ecef_m, dtype=float).reshape(3)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    if p < 1e-8:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        return float(lat), lon, float(abs(z) - EARTH_SEMI_MINOR_AXIS_M)

    lat = float(np.arctan2(z, p * (1.0 - EARTH_ECCENTRICITY_SQUARED)))
    for _ in range(15):
        sin_lat = np.sin(lat)
        N = EARTH_SEMI_MAJOR_AXIS_M / np.sqrt(
            1.0 - EARTH_ECCENTRICITY_SQUARED * sin_lat * sin_lat
        )
        h = p / np.cos(lat) - N
        new_lat = float(
            np.arctan2(
                z,
                p * (1.0 - EARTH_ECCENTRICITY_SQUARED * N / (N + h)),
            )
        )
        if abs(new_lat - lat) < 1e-13:
            lat = new_lat
            break
        lat = new_lat

    sin_lat = np.sin(lat)
    N = EARTH_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - EARTH_ECCENTRICITY_SQUARED * sin_lat * sin_lat
    )
    h = p / np.cos(lat) - N
    return float(lat), lon, float(h)


def c_ecef_to_ned(lat_rad: float, lon_rad: float) -> np.ndarray:
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    return np.array([
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ])


def c_vehicle_to_body_zxy(x_rot_deg: float, y_rot_deg: float, z_rot_deg: float) -> np.ndarray:
    """SmartPNT passive vehicle->body Z-X-Y mounting rotation."""
    gamma, beta, alpha = np.deg2rad([x_rot_deg, y_rot_deg, z_rot_deg])
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Ry = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    Rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def transform_lever_arm_vehicle_to_body(
    lever_vehicle_m: np.ndarray,
    x_rot_deg: float,
    y_rot_deg: float,
    z_rot_deg: float,
) -> np.ndarray:
    return c_vehicle_to_body_zxy(x_rot_deg, y_rot_deg, z_rot_deg) @ np.asarray(
        lever_vehicle_m, dtype=float
    ).reshape(3)


def saastamoinen_delay_m(height_m: float, elevation_rad: float) -> float:
    """Standard Saastamoinen tropospheric delay used by the current GNSS path."""
    if elevation_rad <= 0.0:
        return float("inf")
    h = max(-100.0, min(float(height_m), 10000.0))
    temperature_k = 15.0 - 0.0065 * h + 273.15
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-5 * h) ** 5.2568
    water_vapor_hpa = 6.108 * 0.7 * math.exp(
        (17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45)
    )
    z = math.pi / 2.0 - elevation_rad
    return 0.002277 / math.cos(z) * (
        pressure_hpa + (1255.0 / temperature_k + 0.05) * water_vapor_hpa
        - 1.16 * math.tan(z) ** 2
    )


def klobuchar_delay_m(
    time_gps_tow_s: float,
    latitude_rad: float,
    longitude_rad: float,
    elevation_rad: float,
    azimuth_rad: float,
    alpha_s,
    beta_s,
) -> float:
    """GPS ICD Klobuchar model; returns delay in metres."""
    alpha = np.asarray(alpha_s, dtype=float).reshape(4)
    beta = np.asarray(beta_s, dtype=float).reshape(4)
    lat_sc = latitude_rad / math.pi
    lon_sc = longitude_rad / math.pi
    elev_sc = elevation_rad / math.pi
    psi = 0.0137 / (elev_sc + 0.11) - 0.022
    phi_i = np.clip(lat_sc + psi * math.cos(azimuth_rad), -0.416, 0.416)
    lam_i = lon_sc + psi * math.sin(azimuth_rad) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    local_time_s = (43200.0 * lam_i + time_gps_tow_s) % 86400.0
    basis = np.array([1.0, phi_m, phi_m**2, phi_m**3])
    amplitude_s = max(0.0, float(alpha @ basis))
    period_s = max(72000.0, float(beta @ basis))
    phase = 2.0 * math.pi * (local_time_s - 50400.0) / period_s
    F = 1.0 + 16.0 * (0.53 - elev_sc) ** 3
    if abs(phase) < 1.57:
        delay_s = F * (5e-9 + amplitude_s * (1.0 - phase**2 / 2.0 + phase**4 / 24.0))
    else:
        delay_s = F * 5e-9
    return SPEED_OF_LIGHT_MPS * delay_s


def geometric_range(
    receiver_position_ecef_m: np.ndarray,
    satellite_position_tx_ecef_m: np.ndarray,
    transit_s: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Earth-rotation corrected one-way range and satellite->receiver LOS."""
    angle = EARTH_ROTATION_RATE_RADPS * float(transit_s)
    c, s = np.cos(angle), np.sin(angle)
    C_rx_tx = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sat_rx = C_rx_tx @ np.asarray(satellite_position_tx_ecef_m, dtype=float).reshape(3)
    los = np.asarray(receiver_position_ecef_m, dtype=float).reshape(3) - sat_rx
    rho = float(np.linalg.norm(los))
    return rho, los / rho, sat_rx


def elevation_azimuth_from_ned_matrix(
    c_ecef_ned: np.ndarray,
    los_satellite_to_receiver_ecef: np.ndarray,
) -> tuple[float, float]:
    """Elevation/azimuth using a receiver NED matrix already computed for the epoch."""
    receiver_to_satellite_ned = np.asarray(c_ecef_ned, dtype=float) @ -np.asarray(
        los_satellite_to_receiver_ecef, dtype=float
    )
    elevation = float(np.arcsin(np.clip(-receiver_to_satellite_ned[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(receiver_to_satellite_ned[1], receiver_to_satellite_ned[0]) % (2.0 * np.pi))
    return elevation, azimuth


def elevation_azimuth_from_ecef_los(
    receiver_position_ecef_m: np.ndarray,
    los_satellite_to_receiver_ecef: np.ndarray,
) -> tuple[float, float]:
    lat, lon, _ = ecef_to_llh(receiver_position_ecef_m)
    return elevation_azimuth_from_ned_matrix(
        c_ecef_to_ned(lat, lon),
        los_satellite_to_receiver_ecef,
    )


# =============================================================================
# DATASET READERS
# =============================================================================
# =============================================================================
# IMU noise model
# =============================================================================
@dataclass(frozen=True)
class IMUNoiseModel:
    imu_type: str
    isdv_pos_m: np.ndarray
    isdv_vel_mps: np.ndarray
    isdv_att_deg: np.ndarray
    isdv_accel_bias_mps2: np.ndarray
    isdv_gyro_bias_deg_s: np.ndarray
    pnsd_pos_m_sqrt_s: np.ndarray
    pnsd_vel_mps_sqrt_s: np.ndarray
    pnsd_att_deg_sqrt_s: np.ndarray
    pnsd_accel_bias_mps2_sqrt_s: np.ndarray
    pnsd_gyro_bias_deg_s_sqrt_s: np.ndarray


@dataclass(frozen=True)
class IMUNoiseModelSI:
    imu_type: str
    isdv_pos_m: np.ndarray
    isdv_vel_mps: np.ndarray
    isdv_att_rad: np.ndarray
    isdv_accel_bias_mps2: np.ndarray
    isdv_gyro_bias_rad_s: np.ndarray
    pnsd_pos_m_sqrt_s: np.ndarray
    pnsd_vel_mps_sqrt_s: np.ndarray
    pnsd_att_rad_sqrt_s: np.ndarray
    pnsd_accel_bias_mps2_sqrt_s: np.ndarray
    pnsd_gyro_bias_rad_s_sqrt_s: np.ndarray


def imu_model_to_si(model: IMUNoiseModel) -> IMUNoiseModelSI:
    d2r = np.pi / 180.0
    return IMUNoiseModelSI(
        model.imu_type,
        model.isdv_pos_m.copy(),
        model.isdv_vel_mps.copy(),
        model.isdv_att_deg * d2r,
        model.isdv_accel_bias_mps2.copy(),
        model.isdv_gyro_bias_deg_s * d2r,
        model.pnsd_pos_m_sqrt_s.copy(),
        model.pnsd_vel_mps_sqrt_s.copy(),
        model.pnsd_att_deg_sqrt_s * d2r,
        model.pnsd_accel_bias_mps2_sqrt_s.copy(),
        model.pnsd_gyro_bias_deg_s_sqrt_s * d2r,
    )


def read_imu_error_models(path: str | Path) -> dict[str, IMUNoiseModel]:
    """Read SmartPNT IMUErrorModel.txt and return models by IMU type."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    keys = (
        "ISDV_Pos", "ISDV_Vel", "ISDV_Att", "ISDV_AccelBias", "ISDV_GyrosBias",
        "PNSD_Pos", "PNSD_Vel", "PNSD_Att", "PNSD_AccelBias", "PNSD_GyrosBias",
    )
    models = {}
    for match in re.finditer(r"IMU\s*\{(.*?)\}", text, flags=re.S):
        block = match.group(1)
        type_match = re.search(r'IMU_Type\s*=\s*"([^"]+)"', block)
        if not type_match:
            continue
        values = {}
        for key in keys:
            value_match = re.search(rf"{key}\s*=\s*([^\r\n]+)", block)
            if value_match:
                values[key] = np.fromstring(value_match.group(1), sep=" ", dtype=float)
        if len(values) != len(keys):
            continue
        imu_type = type_match.group(1)
        models[imu_type] = IMUNoiseModel(
            imu_type,
            values["ISDV_Pos"], values["ISDV_Vel"], values["ISDV_Att"],
            values["ISDV_AccelBias"], values["ISDV_GyrosBias"],
            values["PNSD_Pos"], values["PNSD_Vel"], values["PNSD_Att"],
            values["PNSD_AccelBias"], values["PNSD_GyrosBias"],
        )
    return models


# =============================================================================
# RINEX observation: one pseudorange signal per GPS/BDS satellite
# =============================================================================
_FREQUENCY_HZ = {
    ("G", "1"): 1575.42e6,
    ("G", "2"): 1227.60e6,
    ("G", "5"): 1176.45e6,
    ("C", "1"): 1575.42e6,
    ("C", "2"): 1561.098e6,
    ("C", "5"): 1176.45e6,
    ("C", "7"): 1207.140e6,
    ("C", "8"): 1191.795e6,
    ("C", "6"): 1268.52e6,
}
_SIGNAL_PREFS = {
    "G": ("1C", "1W", "1P", "2W", "2L", "2X", "5Q", "5X", "5I"),
    "C": ("2I", "1I", "2X", "1X", "1P", "1D", "5X", "5P", "5D", "7I", "7X", "6I", "6X"),
}


@dataclass(frozen=True)
class SelectedSignal:
    suffix: str
    code_type: str
    frequency_hz: float


@dataclass(frozen=True)
class SatelliteMeasurement:
    sat_id: str
    constellation: str
    signal: SelectedSignal
    pseudorange_m: float
    cn0_dbhz: float | None = None


@dataclass(frozen=True)
class ObservationEpoch:
    time_gpst_s: float
    gps_week: int
    tow_s: float
    measurements: tuple[SatelliteMeasurement, ...]


class RINEXObservationFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.version = 0.0
        self.time_scale = "GPS"
        self.observation_types: dict[str, tuple[str, ...]] = {}
        self._index_by_type: dict[str, dict[str, int]] = {}
        self._read_header()

    @classmethod
    def open(cls, path: str | Path) -> "RINEXObservationFile":
        return cls(path)

    def _read_header(self) -> None:
        types: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        current_system = None
        with self.path.open("r", encoding="ascii", errors="replace") as f:
            first = f.readline()
            self.version = float(first[:9])
            for line in f:
                label = line[60:80].strip() if len(line) >= 60 else ""
                if label == "END OF HEADER":
                    break
                if label == "TIME OF FIRST OBS":
                    self.time_scale = line[48:51].strip() or "GPS"
                elif label == "SYS / # / OBS TYPES":
                    if line[0:1].strip():
                        current_system = line[0]
                        counts[current_system] = int(line[3:6])
                        types[current_system] = []
                    if current_system:
                        types[current_system].extend(line[7:60].split())
                        types[current_system] = types[current_system][:counts[current_system]]
        self.observation_types = {system: tuple(values) for system, values in types.items()}
        self._index_by_type = {
            system: {obs_type: i for i, obs_type in enumerate(values)}
            for system, values in self.observation_types.items()
        }

    def _select_signal_and_pseudorange(
        self,
        constellation: str,
        obs_types: tuple[str, ...],
        fields: list[str],
    ) -> tuple[SelectedSignal, float] | None:
        """Choose the first preferred pseudorange that is valid for this satellite/epoch.

        RINEX observation types are declared per constellation in the header, but an
        individual satellite can have a blank/invalid value for the most-preferred
        code while another supported pseudorange is present. Selection therefore
        must be performed on the actual fields of each satellite record, not once
        for the entire constellation.
        """
        index_by_type = self._index_by_type.get(constellation, {})
        for suffix in _SIGNAL_PREFS.get(constellation, ()):
            code_type = "C" + suffix
            index = index_by_type.get(code_type)
            if index is None or index >= len(fields):
                continue

            raw = fields[index].ljust(16)[:14].strip()
            if not raw:
                continue
            try:
                pseudorange = float(raw.replace("D", "E"))
            except ValueError:
                continue
            if not math.isfinite(pseudorange) or pseudorange <= 0.0:
                continue

            if constellation == "C" and self.version < 3.04 and suffix in {"1I", "1Q", "1X"}:
                frequency = 1561.098e6
            else:
                frequency = _FREQUENCY_HZ.get((constellation, suffix[0]))
            if frequency is None:
                continue

            return SelectedSignal(suffix, code_type, frequency), float(pseudorange)
        return None

    @staticmethod
    def _fields(first_payload: str, stream, count: int) -> list[str]:
        payload = first_payload.rstrip("\n")
        fields = [payload[i:i + 16] for i in range(0, len(payload), 16)]
        while len(fields) < count:
            line = stream.readline()
            if not line:
                break
            payload = line[3:].rstrip("\n")
            fields.extend(payload[i:i + 16] for i in range(0, len(payload), 16))
        return fields[:count]

    def iter_epochs(
        self,
        allowed_constellations: set[str] | None = None,
        *,
        start_time_gpst_s: float | None = None,
        end_time_gpst_s: float | None = None,
        max_epochs: int | None = None,
        require_measurements: bool = False,
    ):
        """Iterate RINEX epochs with optional exact time/counter pruning.

        The parser still reads each selected epoch identically.  The optional bounds
        only stop work that the main pipeline would later discard anyway.
        """
        yielded = 0
        with self.path.open("r", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if len(line) >= 60 and line[60:80].strip() == "END OF HEADER":
                    break

            for line in stream:
                if not line.startswith(">"):
                    continue
                parts = line[1:].split()
                if len(parts) < 8:
                    continue
                year, month, day, hour, minute = map(int, parts[:5])
                second = float(parts[5])
                epoch_flag = int(parts[6])
                satellite_count = int(parts[7])
                if epoch_flag not in (0, 1):
                    for _ in range(satellite_count):
                        stream.readline()
                    continue

                time_gpst_s = calendar_to_gpst_seconds(
                    year, month, day, hour, minute, second, self.time_scale
                )
                # RINEX epochs are time ordered.  Once the upper bound is crossed,
                # no later epoch can be used by this run.
                if end_time_gpst_s is not None and time_gpst_s > float(end_time_gpst_s):
                    break

                gps_time = gpst_seconds_to_week_tow(time_gpst_s)
                measurements = []

                for _ in range(satellite_count):
                    sat_line = stream.readline()
                    if not sat_line:
                        break
                    sat_id = sat_line[:3].strip()
                    if not sat_id:
                        continue
                    constellation = sat_id[0]
                    obs_types = self.observation_types.get(constellation, ())
                    fields = self._fields(sat_line[3:], stream, len(obs_types))
                    if allowed_constellations and constellation not in allowed_constellations:
                        continue
                    selected = self._select_signal_and_pseudorange(
                        constellation, obs_types, fields
                    )
                    if selected is None:
                        continue
                    signal, pseudorange = selected

                    snr_type = "S" + signal.suffix
                    cn0 = None
                    snr_index = self._index_by_type.get(constellation, {}).get(snr_type)
                    if snr_index is not None and snr_index < len(fields):
                        raw_snr = fields[snr_index].ljust(16)[:14].strip()
                        if raw_snr:
                            value = float(raw_snr.replace("D", "E"))
                            cn0 = value if math.isfinite(value) else None
                    measurements.append(
                        SatelliteMeasurement(sat_id, constellation, signal, pseudorange, cn0)
                    )

                if start_time_gpst_s is not None and time_gpst_s < float(start_time_gpst_s):
                    continue
                if require_measurements and not measurements:
                    continue

                yield ObservationEpoch(
                    time_gpst_s, gps_time.week, gps_time.tow_s, tuple(measurements)
                )
                yielded += 1
                if max_epochs is not None and yielded >= int(max_epochs):
                    break


# =============================================================================
# RINEX precise clock
# =============================================================================
class RINEXClock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.time_scale = "GPS"
        self._series = self._parse()
        self._satellites = tuple(sorted(self._series))
        self._satellite_set = frozenset(self._series)

    def _parse(self):
        records = defaultdict(lambda: [[], [], []])
        with self.path.open("r", encoding="ascii", errors="replace") as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ""
                if label == "END OF HEADER":
                    break
                if label == "TIME SYSTEM ID":
                    fields = line[:10].split()
                    if fields:
                        self.time_scale = fields[0]

            for line in stream:
                if line[:2] != "AS":
                    continue
                fields = line.split()
                if len(fields) < 10:
                    continue
                sat_id = fields[1]
                year, month, day, hour, minute = map(int, fields[2:7])
                second = float(fields[7])
                value_count = int(fields[8])
                values = [float(x.replace("D", "E")) for x in fields[9:]]
                while len(values) < value_count:
                    values.extend(float(x.replace("D", "E")) for x in stream.readline().split())
                t = calendar_to_gpst_seconds(year, month, day, hour, minute, second, self.time_scale)
                records[sat_id][0].append(t)
                records[sat_id][1].append(values[0])
                records[sat_id][2].append(values[2] if value_count >= 3 else np.nan)

        out = {}
        for sat_id, (times, bias, drift) in records.items():
            order = np.argsort(times)
            out[sat_id] = (
                np.asarray(times)[order],
                np.asarray(bias)[order],
                np.asarray(drift)[order],
            )
        return out

    @property
    def satellites(self) -> tuple[str, ...]:
        return self._satellites

    def has_satellite(self, sat_id: str) -> bool:
        return sat_id in self._satellite_set

    def bias(self, sat_id: str, time_gpst_s: float) -> float:
        times, bias, _ = self._series[sat_id]
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f"CLK time outside product span for {sat_id}")
        return float(np.interp(time_gpst_s, times, bias))



# =============================================================================
# RINEX navigation header: Klobuchar coefficients only
# =============================================================================
def read_rinex_navigation_header(path: str | Path) -> dict[str, tuple[float, ...]]:
    """Return only the IONOSPHERIC CORR fields used by Klobuchar."""
    corrections = {}
    with Path(path).open("r", encoding="ascii", errors="replace") as stream:
        stream.readline()
        for line in stream:
            label = line[60:80].strip() if len(line) >= 60 else ""
            if label == "END OF HEADER":
                break
            if label == "IONOSPHERIC CORR":
                fields = line[:60].split()
                if len(fields) >= 5:
                    corrections[fields[0]] = tuple(float(x.replace("D", "E")) for x in fields[1:5])
    return corrections



# =============================================================================
# SP3 precise orbit
# =============================================================================
class SP3Orbit:
    def __init__(self, path: str | Path, interpolation_points: int = 9):
        self.path = Path(path)
        self.interpolation_points = interpolation_points
        self.series = self._parse()

    def _parse(self):
        temporary = defaultdict(lambda: [[], [], []])
        velocities = {}
        drifts = {}
        current_time = None
        time_scale = "GPS"

        with self.path.open("r", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("%c") and len(line) >= 12:
                    candidate = line[9:12].strip()
                    # The first SP3 ``%c`` record carries the time system. The
                    # following record commonly contains the literal placeholder
                    # ``ccc`` in the same columns and must not overwrite it.
                    if candidate.upper() in {
                        "GPS", "GPST", "GAL", "GST", "QZS", "IRN",
                        "BDT", "BDS", "UTC", "GLO",
                    }:
                        time_scale = candidate
                elif line.startswith("*"):
                    f = line[1:].split()
                    current_time = calendar_to_gpst_seconds(
                        int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[5]), time_scale
                    )
                elif current_time is not None and line.startswith("P"):
                    sat_id = line[1:4].strip()
                    v = line[4:].split()
                    if len(v) < 4:
                        continue
                    pos_km = np.asarray(v[:3], dtype=float)
                    if np.any(np.abs(pos_km) >= 999999.0) or np.allclose(pos_km, 0.0):
                        continue
                    clock_us = float(v[3])
                    temporary[sat_id][0].append(current_time)
                    temporary[sat_id][1].append(pos_km * 1000.0)
                    temporary[sat_id][2].append(np.nan if abs(clock_us) >= 999999.0 else clock_us * 1e-6)
                elif current_time is not None and line.startswith("V"):
                    sat_id = line[1:4].strip()
                    v = line[4:].split()
                    if len(v) < 4:
                        continue
                    velocities[(sat_id, current_time)] = np.asarray(v[:3], dtype=float) * 0.1
                    rate = float(v[3])
                    drifts[(sat_id, current_time)] = np.nan if abs(rate) >= 999999.0 else rate * 1e-10

        series = {}
        for sat_id, (times, positions, clocks) in temporary.items():
            times = np.asarray(times, dtype=float)
            order = np.argsort(times)
            times = times[order]
            clock = np.asarray(clocks, dtype=float)[order]
            velocity = np.vstack([
                velocities.get((sat_id, float(t)), [np.nan] * 3) for t in times
            ])
            drift = np.asarray([
                drifts.get((sat_id, float(t)), np.nan) for t in times
            ])
            finite_clock = np.isfinite(clock)
            finite_drift = np.isfinite(drift)
            series[sat_id] = {
                "time": times,
                "position": np.asarray(positions, dtype=float)[order],
                "clock": clock,
                "velocity": velocity,
                "velocity_finite": np.all(np.isfinite(velocity), axis=1),
                "drift": drift,
                "clock_time": times[finite_clock],
                "clock_value": clock[finite_clock],
                "drift_time": times[finite_drift],
                "drift_value": drift[finite_drift],
            }
        return series

    @staticmethod
    def _interpolate(times: np.ndarray, values: np.ndarray, query: float):
        scale = max(float(np.max(np.abs(times - query))), 1.0)
        x = (times - query) / scale
        y, dy = [], []
        for column in range(values.shape[1]):
            p = BarycentricInterpolator(x, values[:, column], rng=0)
            y.append(float(p(0.0)))
            dy.append(float(p.derivative(0.0)) / scale)
        return np.asarray(y), np.asarray(dy)

    @staticmethod
    def _interpolate_values_only(times: np.ndarray, values: np.ndarray, query: float):
        """Same barycentric position interpolation without unused derivatives."""
        scale = max(float(np.max(np.abs(times - query))), 1.0)
        x = (times - query) / scale
        y = []
        for column in range(values.shape[1]):
            p = BarycentricInterpolator(x, values[:, column], rng=0)
            y.append(float(p(0.0)))
        return np.asarray(y)

    def position(self, sat_id: str, time_gpst_s: float) -> np.ndarray:
        """SP3 position using the exact same interpolation window as state()."""
        s = self.series[sat_id]
        times = s["time"]
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f"SP3 time outside product span for {sat_id}")
        count = min(self.interpolation_points, len(times))
        i = int(np.searchsorted(times, time_gpst_s))
        start = max(0, min(len(times) - count, i - count // 2))
        w = slice(start, start + count)
        return self._interpolate_values_only(
            times[w], s["position"][w], time_gpst_s
        )

    def clock_bias(self, sat_id: str, time_gpst_s: float) -> float | None:
        """SP3 clock bias without computing position/velocity."""
        s = self.series[sat_id]
        times = s["time"]
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f"SP3 time outside product span for {sat_id}")
        if s["clock_time"].size == 0:
            return None
        return float(np.interp(time_gpst_s, s["clock_time"], s["clock_value"]))

    def state(self, sat_id: str, time_gpst_s: float):
        s = self.series[sat_id]
        times = s["time"]
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f"SP3 time outside product span for {sat_id}")
        count = min(self.interpolation_points, len(times))
        i = int(np.searchsorted(times, time_gpst_s))
        start = max(0, min(len(times) - count, i - count // 2))
        w = slice(start, start + count)
        position, position_derivative = self._interpolate(times[w], s["position"][w], time_gpst_s)
        native_velocity = s["velocity"][w]
        finite_v = s["velocity_finite"][w]
        if finite_v.sum() >= 2:
            velocity, _ = self._interpolate(times[w][finite_v], native_velocity[finite_v], time_gpst_s)
        else:
            velocity = position_derivative
        clock = (
            None
            if s["clock_time"].size == 0
            else float(np.interp(time_gpst_s, s["clock_time"], s["clock_value"]))
        )
        drift = (
            None
            if s["drift_time"].size == 0
            else float(np.interp(time_gpst_s, s["drift_time"], s["drift_value"]))
        )
        return position, velocity, clock, drift


# =============================================================================
# SmartPNT IMR
# =============================================================================
IMR_HEADER_FORMAT_BODY = "8scdiidddiid32s?BBB32s6h?iii354s"
IMR_RECORD_FORMAT_BODY = "d6i"


@dataclass(frozen=True)
class IMRHeader:
    endian: str
    delta_theta: int
    delta_velocity: int
    data_rate_hz: float
    gyro_scale: float
    accel_scale: float
    utc_or_gps_time: int
    receiver_or_corrected_time: int
    time_tag_bias_ms: float


@dataclass
class IMRData:
    header: IMRHeader
    tow_s: np.ndarray
    angular_rate_body_radps: np.ndarray
    acceleration_body_mps2: np.ndarray
    raw_gyro_counts: np.ndarray
    raw_accel_counts: np.ndarray


def _read_imr_layout(path: str | Path) -> tuple[Path, IMRHeader, np.dtype, int, int]:
    """Read only the fixed IMR header and binary-record layout metadata."""
    path = Path(path)
    with path.open("rb") as stream:
        buffer = stream.read(512)
    endian = "<" if buffer[8] == 0 else ">"
    values = struct.unpack(endian + IMR_HEADER_FORMAT_BODY, buffer)
    header = IMRHeader(
        endian=endian,
        delta_theta=int(values[3]),
        delta_velocity=int(values[4]),
        data_rate_hz=float(values[5]),
        gyro_scale=float(values[6]),
        accel_scale=float(values[7]),
        utc_or_gps_time=int(values[8]),
        receiver_or_corrected_time=int(values[9]),
        time_tag_bias_ms=float(values[10]),
    )

    record_bytes = struct.calcsize(endian + IMR_RECORD_FORMAT_BODY)
    payload_bytes = max(0, path.stat().st_size - 512)
    record_count = payload_bytes // record_bytes
    record_dtype = np.dtype([
        ("tow", endian + "f8"),
        ("counts", endian + "i4", (6,)),
    ], align=False)
    if record_dtype.itemsize != record_bytes:
        raise RuntimeError("NumPy IMR dtype does not match the documented record size")
    return path, header, record_dtype, record_count, record_bytes


def _adjust_imr_tow(tow: np.ndarray, header: IMRHeader) -> np.ndarray:
    tow = np.asarray(tow, dtype=np.float64)
    tow = np.where(tow > 604800.0, tow - 604800.0, tow)
    tow -= header.time_tag_bias_ms * 1e-3
    return tow


def read_imr_tow_only(path: str | Path) -> tuple[IMRHeader, np.ndarray, int]:
    """Read only IMR time tags.

    A memory map exposes the strided TOW field without converting the six raw-count
    channels.  This lets partial/debug runs determine the exact required IMU window
    before allocating and scaling the sensor arrays.
    """
    path, header, record_dtype, record_count, _ = _read_imr_layout(path)
    records = np.memmap(
        path, dtype=record_dtype, mode="r", offset=512, shape=(record_count,)
    )
    tow = np.asarray(records["tow"], dtype=np.float64).copy()
    del records
    return header, _adjust_imr_tow(tow, header), record_count


def read_imr(
    path: str | Path,
    scaling_mode: str = "cpp_exact",
    *,
    start_record: int = 0,
    stop_record: int | None = None,
) -> IMRData:
    """Read an exact contiguous SmartPNT IMR record window.

    The binary layout, scaling equations, float precision, units, and sample order are
    unchanged.  For partial runs only records that can influence the requested fusion
    epochs are converted to floating-point sensor values.
    """
    path, header, record_dtype, record_count, record_bytes = _read_imr_layout(path)
    start_record = int(start_record)
    if stop_record is None:
        stop_record = record_count
    stop_record = int(stop_record)
    if not (0 <= start_record <= stop_record <= record_count):
        raise ValueError(
            f"invalid IMR record window [{start_record}, {stop_record}) for {record_count} records"
        )

    count = stop_record - start_record
    records = np.fromfile(
        path,
        dtype=record_dtype,
        count=count,
        offset=512 + start_record * record_bytes,
    )
    tow = _adjust_imr_tow(records["tow"].astype(np.float64, copy=True), header)
    counts = records["counts"]
    gyro_counts = counts[:, :3]
    accel_counts = counts[:, 3:6]

    gyro = gyro_counts.astype(float) * header.gyro_scale
    accel = accel_counts.astype(float) * header.accel_scale
    if scaling_mode == "cpp_exact":
        gyro *= header.data_rate_hz
        accel *= header.data_rate_hz
    else:
        if header.delta_theta:
            gyro *= header.data_rate_hz
        if header.delta_velocity:
            accel *= header.data_rate_hz

    return IMRData(
        header,
        tow,
        np.deg2rad(gyro),
        accel,
        gyro_counts,
        accel_counts,
    )


# =============================================================================
# SmartPNT README.xml
# =============================================================================
@dataclass(frozen=True)
class RoverMetadata:
    imu_type: str
    lever_arm_vehicle_m: np.ndarray
    mounting_xyz_deg: np.ndarray


def load_smartpnt_metadata(path: str | Path, rover_id: str = "01") -> RoverMetadata:
    root = ET.fromstring(Path(path).read_text(encoding="utf-8", errors="replace"))
    for rover in root.findall("ROVE"):
        if (rover.findtext("ID") or "").strip() == str(rover_id):
            return RoverMetadata(
                (rover.findtext("SINS_IMUType") or "").strip(),
                np.fromstring(rover.findtext("SINS_LeverArm_GNSS") or "", sep=" "),
                np.fromstring(rover.findtext("SINS_RotAngle_IMU") or "", sep=" "),
            )
    raise KeyError(f"Rover {rover_id} not found")


@dataclass(frozen=True)
class DatasetFiles:
    root: Path
    readme_xml: Path
    rover_ground_truth: Path
    imu_ground_truth: Path
    rinex_obs: Path
    imr: Path
    sp3: Path
    clk: Path
    nav: Path


def _unique_file(directory: Path, patterns: tuple[str, ...], label: str) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {label} file in {directory}, found {len(matches)}: {names}"
        )
    return matches[0]


def resolve_dataset_files(dataset_dir: str | Path, rover_id: str = "01") -> tuple[DatasetFiles, RoverMetadata]:
    """Resolve one SmartPNT-Pos dataset without assuming date-specific product filenames."""
    root = Path(dataset_dir)
    readme = root / "README.xml"
    if not readme.exists():
        raise FileNotFoundError(f"Missing README.xml in dataset: {root}")

    rover = load_smartpnt_metadata(readme, rover_id)
    imu_truth = root / f"{rover.imu_type}_GroundTruth.txt"
    imr = root / f"{rover.imu_type}.imr"
    rover_truth = _unique_file(
        root,
        (
            "ROVE_GroundTruth.txt",
            f"ROVE_{rover_id}_GroundTruth.txt",
            f"Rove_{rover_id}_GroundTruth.txt",
        ),
        f"rover {rover_id} ground truth",
    )
    rover_observation = _unique_file(
        root,
        (
            "ROVE.*O",
            "ROVE.*o",
            f"ROVE_{rover_id}.*O",
            f"ROVE_{rover_id}.*o",
        ),
        f"rover {rover_id} RINEX observation",
    )

    files = DatasetFiles(
        root=root,
        readme_xml=readme,
        rover_ground_truth=rover_truth,
        imu_ground_truth=imu_truth,
        rinex_obs=rover_observation,
        imr=imr,
        sp3=_unique_file(root, ("*.SP3", "*.sp3"), "SP3 precise-orbit"),
        clk=_unique_file(root, ("*.CLK", "*.clk"), "RINEX clock"),
        nav=_unique_file(root, ("brdm*.*p", "brdm*.*P", "brdm*.rnx", "BRDM*.RNX"), "broadcast navigation"),
    )
    required = [
        files.readme_xml, files.rover_ground_truth, files.imu_ground_truth, files.rinex_obs,
        files.imr, files.sp3, files.clk, files.nav,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))
    return files, rover


def resolve_test_dataset_dir() -> Path:
    """Select the explicitly configured test dataset or the single non-training dataset."""
    if TEST_DATASET_DIR is not None:
        return Path(TEST_DATASET_DIR)

    candidates = sorted(
        path for path in DATASET_ROOT.iterdir()
        if path.is_dir() and path.resolve() != TRAIN_DATASET_DIR.resolve()
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(
            "TEST_DATASET_DIR is not set and automatic selection is ambiguous. "
            f"Found {len(candidates)} non-training dataset directories: {names}. "
            "Set TEST_DATASET_DIR to the exact second SmartPNT-Pos dataset."
        )
    return candidates[0]


# =============================================================================
# Inertial Explorer ground truth: only fields used by this project
# =============================================================================
@dataclass
class GroundTruth:
    week: np.ndarray
    tow_s: np.ndarray
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray
    heading_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray


def load_ie_ground_truth(path: str | Path) -> GroundTruth:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    started = False
    for line in lines:
        fields = line.split()
        if len(fields) < 24:
            if started:
                break
            continue
        try:
            row = (
                int(fields[0]), float(fields[1]),
                float(fields[9]), float(fields[10]), float(fields[11]),
                float(fields[15]), float(fields[16]), float(fields[17]),
                float(fields[21]), float(fields[22]), float(fields[23]),
            )
        except (ValueError, IndexError):
            if started:
                break
            continue
        started = True
        rows.append(row)

    a = np.asarray(rows, dtype=float)
    return GroundTruth(
        a[:, 0].astype(int),
        a[:, 1],
        a[:, 2:5],
        a[:, 5:8],
        a[:, 8],
        a[:, 9],
        a[:, 10],
    )


def interpolate_ground_truth(
    truth: GroundTruth,
    query_time_gpst_s: np.ndarray,
    max_gap_s: float,
):
    """Interpolate IE position/velocity/HPR at requested GPST epochs."""
    truth_time = truth.week.astype(float) * GPS_WEEK_S + truth.tow_s
    query = np.asarray(query_time_gpst_s, dtype=float)
    upper = np.searchsorted(truth_time, query, side="left")
    upper = np.clip(upper, 0, len(truth_time) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = truth_time[upper] == query
    lower[exact] = upper[exact]
    gap = truth_time[upper] - truth_time[lower]
    if np.any(gap > float(max_gap_s)):
        raise ValueError("Ground-truth interpolation gap is too large")
    if np.any(query < truth_time[0]) or np.any(query > truth_time[-1]):
        raise ValueError("Requested epoch is outside ground-truth time span")

    weight = np.zeros_like(query)
    nz = gap > 0.0
    weight[nz] = (query[nz] - truth_time[lower[nz]]) / gap[nz]
    w0 = 1.0 - weight

    position = w0[:, None] * truth.position_ecef_m[lower] + weight[:, None] * truth.position_ecef_m[upper]
    velocity = w0[:, None] * truth.velocity_ecef_mps[lower] + weight[:, None] * truth.velocity_ecef_mps[upper]
    heading_unwrapped = np.unwrap(np.deg2rad(truth.heading_deg))
    heading = np.rad2deg(w0 * heading_unwrapped[lower] + weight * heading_unwrapped[upper]) % 360.0
    pitch = w0 * truth.pitch_deg[lower] + weight * truth.pitch_deg[upper]
    roll = w0 * truth.roll_deg[lower] + weight * truth.roll_deg[upper]
    return position, velocity, heading, pitch, roll


def body_to_ecef_from_ie_hpr(
    position_ecef_m: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    mounting_xyz_deg: np.ndarray,
) -> np.ndarray:
    """Build body->ECEF DCM with the same IE/SmartPNT convention used at initialization."""
    lat, lon, _ = ecef_to_llh(position_ecef_m)
    C_e_n = c_ecef_to_ned(lat, lon).T
    heading, pitch, roll = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    C_n_f = np.array([
        [cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh],
        [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch],
        [-sp, sr * cp, cr * cp],
    ])
    C_f_v = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    C_b_v = c_vehicle_to_body_zxy(*mounting_xyz_deg)
    return _rotation(C_e_n @ (C_n_f @ C_f_v) @ C_b_v.T)


def attitude_error_state_target(prior_body_to_ecef: np.ndarray, truth_body_to_ecef: np.ndarray) -> np.ndarray:
    """Error-state attitude correction consistent with inject_error_state()."""
    relative = truth_body_to_ecef @ prior_body_to_ecef.T
    rotvec = SpatialRotation.from_matrix(relative).as_rotvec()
    return rotvec / ATTITUDE_FEEDBACK_SIGN


# =============================================================================
# TLE files
# =============================================================================
def read_tle_directory(path: str | Path, allow_non_tle_files: bool = True):
    """Return validated (line1, line2) pairs from every file in the TLE folder."""
    pairs = []
    for source in sorted(Path(path).iterdir()):
        if not source.is_file():
            continue
        lines = source.read_text(encoding="ascii", errors="replace").splitlines()
        found = False
        for i in range(len(lines) - 1):
            line1, line2 = lines[i].strip(), lines[i + 1].strip()
            if line1.startswith("1 ") and line2.startswith("2 "):
                try:
                    verify_checksum(line1, line2)
                except ValueError:
                    if allow_non_tle_files:
                        continue
                    raise
                pairs.append((line1, line2))
                found = True
        if not found and not allow_non_tle_files:
            raise ValueError(f"No valid TLE in {source}")
    return pairs


# =============================================================================
# ECEF INS, TC MEASUREMENT MODEL, AND TLE/SGP4 LEO SIMULATION
# =============================================================================
Array = np.ndarray
INS_STATE_DIM = 15
# Yan Eq. (7): x = [delta-p, delta-v, delta-theta, b_a, b_g]^T.  All 15
# components belong to the estimated TC error state.  The supplied Data01/Data02
# truth directly labels only the first nine navigation-error components; that
# dataset limitation must restrict only the direct loss, not the KF/KalmanNet
# gain dimension or the bias feedback in Eq. (8).
DIRECT_STATE_LABEL_DIM = 9
ACCELEROMETER_BIAS_STATE_SLICE = slice(9, 12)
GYROSCOPE_BIAS_STATE_SLICE = slice(12, 15)
ATTITUDE_FEEDBACK_SIGN = -1.0
J2_UNITLESS = 1.08262668e-3
OMEGA_IE_E = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RADPS])


def _skew(v: Array) -> Array:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _so3_exponential(rotation_vector_rad: Array) -> Array:
    """Closed-form exp(skew(rotvec)) via Rodrigues, stable for tiny angles.

    This is the exact SO(3) matrix exponential in closed form, not a reduced-order
    navigation approximation.  It replaces the generic 3x3 scipy.linalg.expm calls.
    """
    v = np.asarray(rotation_vector_rad, dtype=float).reshape(3)
    theta2 = float(v @ v)
    K = _skew(v)
    K2 = K @ K
    if theta2 < 1e-12:
        # Stable series for sin(theta)/theta and (1-cos(theta))/theta^2.
        theta4 = theta2 * theta2
        a = 1.0 - theta2 / 6.0 + theta4 / 120.0
        b = 0.5 - theta2 / 24.0 + theta4 / 720.0
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta2
    return np.eye(3) + a * K + b * K2


OMEGA_IE_SKEW = _skew(OMEGA_IE_E)


def _rotation(matrix: Array) -> Array:
    """Project a numerically drifted DCM back to SO(3)."""
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float).reshape(3, 3))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


@dataclass
class NavigationState:
    position_ecef_m: Array
    velocity_ecef_mps: Array
    body_to_ecef_dcm: Array
    accelerometer_bias_body_mps2: Array = field(default_factory=lambda: np.zeros(3))
    gyroscope_bias_body_radps: Array = field(default_factory=lambda: np.zeros(3))

    def copy(self):
        return NavigationState(
            self.position_ecef_m.copy(),
            self.velocity_ecef_mps.copy(),
            self.body_to_ecef_dcm.copy(),
            self.accelerometer_bias_body_mps2.copy(),
            self.gyroscope_bias_body_radps.copy(),
        )


# Paper Eq. (8).  Current run uses zero scale/misalignment matrices and no
# sample-wise stochastic-noise realization, but the complete compensation form
# is retained through optional matrices.
def compensate_imu(
    measured_angular_rate_body_radps: Array,
    measured_specific_force_body_mps2: Array,
    gyroscope_bias_body_radps: Array,
    accelerometer_bias_body_mps2: Array,
    S_g: Array | None = None,
    M_g: Array | None = None,
    S_a: Array | None = None,
    M_a: Array | None = None,
    gyroscope_measurement_error_body_radps: Array | None = None,
    accelerometer_measurement_error_body_mps2: Array | None = None,
) -> tuple[Array, Array]:
    """Paper Eq. (8) IMU error compensation.

    The final two optional arguments implement the residual measurement-error
    terms epsilon_g and epsilon_a in Eq. (8).  In the Masked-CLA path they are
    supplied by eta_k = [epsilon_g(3), epsilon_a(3)].  Passing None preserves the
    ordinary bias-only compensation used by the reference/feature path.
    """
    epsilon_g = (
        np.zeros(3)
        if gyroscope_measurement_error_body_radps is None
        else np.asarray(gyroscope_measurement_error_body_radps, dtype=float).reshape(3)
    )
    epsilon_a = (
        np.zeros(3)
        if accelerometer_measurement_error_body_mps2 is None
        else np.asarray(accelerometer_measurement_error_body_mps2, dtype=float).reshape(3)
    )
    gyro_rhs = (
        np.asarray(measured_angular_rate_body_radps, dtype=float)
        - np.asarray(gyroscope_bias_body_radps, dtype=float)
        - epsilon_g
    )
    accel_rhs = (
        np.asarray(measured_specific_force_body_mps2, dtype=float)
        - np.asarray(accelerometer_bias_body_mps2, dtype=float)
        - epsilon_a
    )

    # Current project configuration has zero scale/misalignment matrices. Solving
    # I*x=b millions of times is exactly equivalent to returning b directly.
    if S_g is None and M_g is None:
        gyro = gyro_rhs
    else:
        gyro = np.linalg.solve(
            np.eye(3)
            + (S_g if S_g is not None else 0.0)
            + (M_g if M_g is not None else 0.0),
            gyro_rhs,
        )

    if S_a is None and M_a is None:
        accel = accel_rhs
    else:
        accel = np.linalg.solve(
            np.eye(3)
            + (S_a if S_a is not None else 0.0)
            + (M_a if M_a is not None else 0.0),
            accel_rhs,
        )
    return gyro, accel


def _gravitation_j2_ecef(position_ecef_m: Array) -> Array:
    x, y, z = np.asarray(position_ecef_m, dtype=float).reshape(3)
    r = float(np.linalg.norm([x, y, z]))
    z2_r2 = z * z / (r * r)
    j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / r) ** 2
    xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / r**3
    return scale * np.array([x * xy_factor, y * xy_factor, z * z_factor])


def effective_gravity_ecef(position_ecef_m: Array) -> Array:
    r = np.asarray(position_ecef_m, dtype=float).reshape(3)
    return _gravitation_j2_ecef(r) - np.cross(OMEGA_IE_E, np.cross(OMEGA_IE_E, r))



@lru_cache(maxsize=128)
def _earth_rotation_transition(dt_s: float) -> Array:
    """Exact closed-form Earth-rotation transition for repeated IMU step sizes."""
    matrix = _so3_exponential(-OMEGA_IE_E * float(dt_s))
    matrix.setflags(write=False)
    return matrix

def mechanize_ecef(
    nav: NavigationState,
    angular_rate_body_radps: Array,
    specific_force_body_mps2: Array,
    dt_s: float,
) -> NavigationState:
    """Standard ECEF strapdown INS completion used by the project."""
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    C1 = _rotation(
        _earth_rotation_transition(dt)
        @ C0
        @ _so3_exponential(np.asarray(angular_rate_body_radps, dtype=float) * dt)
    )
    Cmid = 0.5 * (C0 + C1)
    acceleration = (
        Cmid @ np.asarray(specific_force_body_mps2)
        + effective_gravity_ecef(nav.position_ecef_m)
        - 2.0 * np.cross(OMEGA_IE_E, nav.velocity_ecef_mps)
    )
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavigationState(
        position,
        velocity,
        C1,
        nav.accelerometer_bias_body_mps2.copy(),
        nav.gyroscope_bias_body_radps.copy(),
    )





def _navigation_delta_9(
    reference_nav: NavigationState,
    candidate_nav: NavigationState,
) -> Array:
    """9-state correction that moves ``reference_nav`` to ``candidate_nav``."""
    return np.concatenate([
        candidate_nav.position_ecef_m - reference_nav.position_ecef_m,
        candidate_nav.velocity_ecef_mps - reference_nav.velocity_ecef_mps,
        attitude_error_state_target(
            reference_nav.body_to_ecef_dcm,
            candidate_nav.body_to_ecef_dcm,
        ),
    ])


def _replay_imu_interval_with_eta(
    start_nav: NavigationState,
    interval_segments,
    imr: IMRData,
    eta_6: Array,
) -> NavigationState:
    """Propagate one IMU interval with constant eta_k through paper Eq. (8).

    Online, eta_k is produced at a fusion epoch and held over the following
    IMU propagation interval until the next usable network output. Yan et al.
    do not publish eta_k's inter-fusion update rate; this zero-order hold is a
    declared timing completion, not a paper-claimed detail.
    """
    eta = np.asarray(eta_6, dtype=float).reshape(IMU_ERROR_DIM)
    nav = start_nav.copy()
    for imu_index, dt in interval_segments:
        gyro, accel = compensate_imu(
            imr.angular_rate_body_radps[int(imu_index)],
            imr.acceleration_body_mps2[int(imu_index)],
            nav.gyroscope_bias_body_radps,
            nav.accelerometer_bias_body_mps2,
            gyroscope_measurement_error_body_radps=eta[ETA_GYRO_SLICE],
            accelerometer_measurement_error_body_mps2=eta[ETA_ACCEL_SLICE],
        )
        nav = mechanize_ecef(nav, gyro, accel, float(dt))
    return nav


def eta_interval_state_sensitivity_9(
    start_nav: NavigationState,
    interval_segments,
    imr: IMRData,
    reference_end_nav: NavigationState | None = None,
) -> Array:
    """Teacher-forced d([dp,dv,dtheta])/d(eta_k) through Eq. (8)+INS.

    This Jacobian exists only in offline training to express the unpublished
    eta_k supervision through the *published state-estimation loss* rather than
    by inventing a direct eta label/loss. Online inference applies eta_k directly
    in ``compensate_imu`` and never uses this Jacobian.
    """
    if not interval_segments:
        return np.zeros((SUPERVISED_STATE_DIM, IMU_ERROR_DIM), dtype=float)

    zero_eta = np.zeros(IMU_ERROR_DIM)
    base_end = _replay_imu_interval_with_eta(
        start_nav, interval_segments, imr, zero_eta
    )

    if reference_end_nav is not None:
        replay_mismatch = _navigation_delta_9(reference_end_nav, base_end)
        if (
            np.linalg.norm(replay_mismatch[:3]) > 1e-5
            or np.linalg.norm(replay_mismatch[3:6]) > 1e-8
            or np.linalg.norm(replay_mismatch[6:9]) > 1e-10
        ):
            raise RuntimeError(
                "zero-eta IMU replay does not reproduce its reference end state; "
                f"delta9={replay_mismatch}"
            )

    steps = np.array(
        [ETA_FD_GYRO_STEP_RADPS] * 3
        + [ETA_FD_ACCEL_STEP_MPS2] * 3,
        dtype=float,
    )
    jacobian = np.zeros((SUPERVISED_STATE_DIM, IMU_ERROR_DIM), dtype=float)

    for axis in range(IMU_ERROR_DIM):
        eta = np.zeros(IMU_ERROR_DIM)
        eta[axis] = steps[axis]
        perturbed_end = _replay_imu_interval_with_eta(
            start_nav, interval_segments, imr, eta
        )
        jacobian[:, axis] = (
            _navigation_delta_9(base_end, perturbed_end) / steps[axis]
        )

    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError(
            "eta interval state sensitivity contains non-finite values"
        )
    return jacobian


def gnss_antenna_position(nav: NavigationState, lever_arm_b_m: Array) -> Array:
    """Paper Eq. (9): IMU reference position -> GNSS antenna position."""
    return nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)


# =============================================================================
# Pseudorange preparation
# =============================================================================
@dataclass(frozen=True)
class PseudorangeMeasurement:
    sat_id: str
    constellation: str
    pseudorange_m: float
    satellite_position_reception_ecef_m: Array
    satellite_clock_bias_s: float
    ionosphere_delay_m: float
    troposphere_delay_m: float
    sigma_code_m: float
    elevation_rad: float = float("nan")
    cn0_dbhz: float | None = None


def _iterate_transmit_time(
    position_at,
    sat_id: str,
    reception_time_gpst_s: float,
    receiver_position_ecef_m: Array,
    initial_position_ecef_m: Array,
    initial_transit_s: float,
    epsilon_position_m: float,
    max_iterations: int,
):
    previous_position = initial_position_ecef_m
    transit_s = initial_transit_s
    for _ in range(max_iterations):
        transmit_time = reception_time_gpst_s - transit_s
        position = position_at(sat_id, transmit_time)
        rho, _, _ = geometric_range(receiver_position_ecef_m, position, transit_s)
        if np.linalg.norm(position - previous_position) < epsilon_position_m:
            return transmit_time, transit_s, position
        previous_position = position
        transit_s = rho / SPEED_OF_LIGHT_MPS
    raise RuntimeError(f"Transmit-time iteration did not converge for {sat_id}")


class GNSSPreprocessor:
    def __init__(
        self,
        orbit: SP3Orbit,
        clock: RINEXClock,
        min_elevation_deg: float = 5.0,
        use_troposphere: bool = True,
        use_ionosphere: bool = True,
        broadcast_ionosphere_coefficients: dict[str, tuple[Array, Array]] | None = None,
    ):
        self.orbit = orbit
        self.clock = clock
        self.clock_satellites = frozenset(clock.satellites)
        self.min_elevation_rad = np.deg2rad(min_elevation_deg)
        self.use_troposphere = use_troposphere
        self.use_ionosphere = use_ionosphere
        self.iono = broadcast_ionosphere_coefficients or {}

    def _position(self, sat_id: str, time_gpst_s: float) -> np.ndarray:
        return self.orbit.position(sat_id, time_gpst_s)

    def _state(self, sat_id: str, time_gpst_s: float):
        position_ecef_m = self._position(sat_id, time_gpst_s)
        return position_ecef_m, self._clock_bias(sat_id, time_gpst_s)

    def _clock_bias(self, sat_id: str, time_gpst_s: float) -> float:
        """Return the same clock source without computing unused SP3 velocity."""
        if sat_id in self.clock_satellites:
            return float(self.clock.bias(sat_id, time_gpst_s))
        clock_bias_s = self.orbit.clock_bias(sat_id, time_gpst_s)
        if clock_bias_s is None:
            raise KeyError(sat_id)
        return float(clock_bias_s)

    def prepare_measurement(
        self,
        reception_time_gpst_s: float,
        raw: SatelliteMeasurement,
        receiver_position_ecef_m: Array,
        receiver_llh: tuple[float, float, float] | None = None,
        c_ecef_ned: Array | None = None,
    ) -> PseudorangeMeasurement | None:
        try:
            initial_position = self._position(raw.sat_id, reception_time_gpst_s)
        except (KeyError, ValueError):
            return None

        initial_range, _, _ = geometric_range(receiver_position_ecef_m, initial_position, 0.0)
        try:
            transmit_time, transit_s, state_tx_position = _iterate_transmit_time(
                self._position,
                raw.sat_id,
                reception_time_gpst_s,
                receiver_position_ecef_m,
                initial_position,
                initial_range / SPEED_OF_LIGHT_MPS,
                1e-4,
                10,
            )
        except (KeyError, ValueError):
            return None

        satellite_clock_bias_s = self._clock_bias(raw.sat_id, transmit_time)
        rho, los, satellite_position_rx = geometric_range(
            receiver_position_ecef_m, state_tx_position, transit_s
        )
        if receiver_llh is None:
            receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        if c_ecef_ned is None:
            c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
        if elevation < self.min_elevation_rad:
            return None

        ionosphere = 0.0
        height_m = None
        if self.use_ionosphere:
            alpha, beta = self.iono[raw.constellation]
            tow = gpst_seconds_to_week_tow(reception_time_gpst_s).tow_s
            if raw.constellation == "G":
                reference_frequency_hz = 1575.42e6
            else:  # Current allowed constellation is BDS.
                # BDS broadcast coefficients used here are B1I coefficients.
                if raw.signal.suffix not in {"2I", "2X", "1I", "1X"}:
                    return None
                tow = (tow - 14.0) % 604800.0
                reference_frequency_hz = 1561.098e6
            lat, lon, height_m = receiver_llh
            ionosphere = klobuchar_delay_m(
                tow, lat, lon, elevation, azimuth, alpha, beta
            ) * (reference_frequency_hz / raw.signal.frequency_hz) ** 2

        troposphere = 0.0
        if self.use_troposphere:
            if height_m is None:
                height_m = receiver_llh[2]
            troposphere = saastamoinen_delay_m(height_m, elevation)

        # Use the same physically motivated pseudorange precision components
        # referenced by Yan rather than a fixed 5 m covariance. GNSS observations
        # are real; this sigma is used only for weighting/FDE covariance.
        cn0_dbhz = float(raw.cn0_dbhz) if raw.cn0_dbhz is not None else 45.0
        iono_reference_frequency_hz = (
            1575.42e6 if raw.constellation == "G" else 1561.098e6
        )
        iono_sigma = (
            ref35_ionosphere_sigma_m(elevation, receiver_llh[0])
            * (iono_reference_frequency_hz / raw.signal.frequency_hz) ** 2
            if self.use_ionosphere else 0.0
        )
        tropo_sigma = ref35_troposphere_sigma_m(elevation) if self.use_troposphere else 0.0
        mp_sigma = yan_goGPS_mp_nlos_sigma_m(elevation, cn0_dbhz)
        receiver_sigma = ref35_receiver_noise_sigma_m(cn0_dbhz)
        sigma_code_m = math.sqrt(
            LEO_URA_SIGMA_M**2 + iono_sigma**2 + tropo_sigma**2
            + mp_sigma**2 + receiver_sigma**2
        )
        return PseudorangeMeasurement(
            raw.sat_id, raw.constellation, float(raw.pseudorange_m),
            satellite_position_rx, satellite_clock_bias_s,
            float(ionosphere), float(troposphere), float(sigma_code_m),
            float(elevation), raw.cn0_dbhz,
        )

    def prepare_epoch(
        self,
        epoch: ObservationEpoch,
        nav: NavigationState,
        lever_arm_b_m: Array,
    ) -> tuple[PseudorangeMeasurement, ...]:
        antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
        receiver_llh = ecef_to_llh(antenna_position)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch.measurements:
            measurement = self.prepare_measurement(
                epoch.time_gpst_s,
                raw,
                antenna_position,
                receiver_llh=receiver_llh,
                c_ecef_ned=c_ecef_ned,
            )
            if measurement is not None:
                out.append(measurement)
        return tuple(out)


# =============================================================================
# 15-state error dynamics / KF
# =============================================================================
def build_error_state_dynamics(nav: NavigationState, specific_force_body_mps2: Array) -> Array:
    """Paper Eq. (6)/(7), with signs matched to this file's feedback convention.

    ``inject_error_state`` applies attitude feedback with
    ``ATTITUDE_FEEDBACK_SIGN = -1`` and adds the estimated accelerometer bias to
    the nominal bias state. Under that convention, the velocity-error coupling
    from attitude and accelerometer-bias errors has the signs used below.
    """
    F = np.zeros((INS_STATE_DIM, INS_STATE_DIM))
    r_e = nav.position_ecef_m
    radius = float(np.linalg.norm(r_e))
    gravity = _gravitation_j2_ecef(r_e)
    radial = r_e / radius
    C = nav.body_to_ecef_dcm
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = -(2.0 / radius) * np.outer(gravity, radial)
    F[3:6, 3:6] = -2.0 * OMEGA_IE_SKEW
    F[3:6, 6:9] = _skew(C @ np.asarray(specific_force_body_mps2))
    F[3:6, ACCELEROMETER_BIAS_STATE_SLICE] = -C
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    F[6:9, GYROSCOPE_BIAS_STATE_SLICE] = C
    return F


def initial_covariance_from_imu_model(model: IMUNoiseModelSI) -> Array:
    sigma = np.concatenate([
        model.isdv_pos_m,
        model.isdv_vel_mps,
        model.isdv_att_rad,
        model.isdv_accel_bias_mps2,
        model.isdv_gyro_bias_rad_s,
    ])
    return np.diag(sigma**2)


def continuous_process_covariance_from_imu_model(model: IMUNoiseModelSI) -> Array:
    density = np.concatenate([
        model.pnsd_pos_m_sqrt_s,
        model.pnsd_vel_mps_sqrt_s,
        model.pnsd_att_rad_sqrt_s,
        model.pnsd_accel_bias_mps2_sqrt_s,
        model.pnsd_gyro_bias_rad_s_sqrt_s,
    ])
    return np.diag(density**2)


_VAN_LOAN_STATS = {
    "taylor_calls": 0,
    "exact_fallback_calls": 0,
    "validation_calls": 0,
    "max_norm_1": 0.0,
    "max_taylor_remainder_bound": 0.0,
    "max_validation_phi_abs": 0.0,
    "max_validation_qd_abs": 0.0,
}


def _matrix_exponential_taylor(matrix: Array, order: int) -> Array:
    """Evaluate exp(matrix) by a fixed-order Taylor series in float64."""
    B = np.asarray(matrix, dtype=float)
    E = np.eye(B.shape[0], dtype=float)
    term = np.eye(B.shape[0], dtype=float)
    for k in range(1, int(order) + 1):
        term = (term @ B) / float(k)
        E += term
    return E


def discretize_process_noise_van_loan(F: Array, Qc: Array, dt_s: float) -> tuple[Array, Array]:
    """Van Loan discretization with guarded fast exponential evaluation.

    For the normal high-rate IMU regime, ||A*dt||_1 is small and a 10th-order
    Taylor series is much cheaper than a general-purpose 30x30 matrix exponential.
    The fast path is used only below VAN_LOAN_TAYLOR_MAX_NORM_1.  Otherwise the
    original scipy.linalg.expm path is retained exactly.
    """
    A = np.zeros((30, 30), dtype=float)
    A[:15, :15] = F
    A[:15, 15:] = Qc
    A[15:, 15:] = -F.T
    B = A * float(dt_s)

    norm_1 = float(np.linalg.norm(B, 1))
    _VAN_LOAN_STATS["max_norm_1"] = max(_VAN_LOAN_STATS["max_norm_1"], norm_1)

    if norm_1 <= VAN_LOAN_TAYLOR_MAX_NORM_1:
        E = _matrix_exponential_taylor(B, VAN_LOAN_TAYLOR_ORDER)
        _VAN_LOAN_STATS["taylor_calls"] += 1

        # Conservative submultiplicative-norm remainder estimate for the matrix
        # exponential Taylor tail.  This is diagnostic only and does not alter E.
        remainder_bound = (
            math.exp(norm_1)
            * norm_1 ** (VAN_LOAN_TAYLOR_ORDER + 1)
            / math.factorial(VAN_LOAN_TAYLOR_ORDER + 1)
        )
        _VAN_LOAN_STATS["max_taylor_remainder_bound"] = max(
            _VAN_LOAN_STATS["max_taylor_remainder_bound"],
            float(remainder_bound),
        )

        # A handful of exact comparisons provide an in-run numerical regression
        # check while adding negligible cost compared with tens of thousands of calls.
        if _VAN_LOAN_STATS["validation_calls"] < VAN_LOAN_VALIDATE_CALLS:
            E_ref = expm(B)
            Phi_fast = E[:15, :15]
            Qd_fast = E[:15, 15:] @ Phi_fast.T
            Phi_ref = E_ref[:15, :15]
            Qd_ref = E_ref[:15, 15:] @ Phi_ref.T
            _VAN_LOAN_STATS["max_validation_phi_abs"] = max(
                _VAN_LOAN_STATS["max_validation_phi_abs"],
                float(np.max(np.abs(Phi_fast - Phi_ref))),
            )
            _VAN_LOAN_STATS["max_validation_qd_abs"] = max(
                _VAN_LOAN_STATS["max_validation_qd_abs"],
                float(np.max(np.abs(Qd_fast - Qd_ref))),
            )
            _VAN_LOAN_STATS["validation_calls"] += 1
    else:
        E = expm(B)
        _VAN_LOAN_STATS["exact_fallback_calls"] += 1

    Phi = E[:15, :15]
    Qd = E[:15, 15:] @ Phi.T
    return Phi, 0.5 * (Qd + Qd.T)


@dataclass(frozen=True)
class TCMeasurementModel:
    y: Array
    y_pred: Array
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]
    receiver_clock_bias_m_by_system: dict[str, float]
    clock_projector: Array


def predict_pseudorange(
    antenna_position_ecef_m: Array,
    measurement: PseudorangeMeasurement,
    receiver_clock_bias_m_by_system: dict[str, float],
) -> tuple[float, Array]:
    range_vector = np.asarray(antenna_position_ecef_m) - measurement.satellite_position_reception_ecef_m
    geometric_range = float(np.linalg.norm(range_vector))
    los = range_vector / geometric_range

    # Current project: LEO receiver clock is ideal zero. GPS/BDS clocks are
    # epoch-wise nuisance parameters eliminated from the measurement equations.
    receiver_clock_m = (
        0.0
        if measurement.constellation == "L"
        else float(receiver_clock_bias_m_by_system.get(measurement.constellation, 0.0))
    )

    predicted = (
        geometric_range
        + receiver_clock_m
        - SPEED_OF_LIGHT_MPS * measurement.satellite_clock_bias_s
        + measurement.ionosphere_delay_m
        + measurement.troposphere_delay_m
    )
    return float(predicted), los


def retain_clock_observable_measurements(measurements) -> tuple[PseudorangeMeasurement, ...]:
    """Drop a lone GPS/BDS pseudorange whose unknown receiver clock absorbs it fully.

    One unknown receiver-clock nuisance is eliminated independently for GPS and
    BDS at every epoch. A constellation represented by only one pseudorange has
    zero position-information degrees of freedom after that elimination. LEO uses
    the project's ideal-zero receiver clock and is therefore unaffected.
    """
    measurements = tuple(measurements)
    counts = Counter(
        m.constellation
        for m in measurements
        if m.constellation in {"G", "C"}
    )
    return tuple(
        m
        for m in measurements
        if m.constellation not in {"G", "C"} or counts[m.constellation] >= 2
    )


def estimate_receiver_clock_biases_wls_m(
    nav: NavigationState,
    measurements,
    lever_arm_b_m: Array,
) -> dict[str, float]:
    """Estimate one epoch-wise WLS receiver-clock nuisance for GPS and for BDS."""
    antenna = gnss_antenna_position(nav, lever_arm_b_m)
    grouped = {"G": [], "C": []}
    zero_clock = {"G": 0.0, "C": 0.0}
    for m in measurements:
        if m.constellation not in grouped:
            continue
        predicted, _ = predict_pseudorange(antenna, m, zero_clock)
        variance = float(m.sigma_code_m) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("GNSS pseudorange variance must be finite and positive")
        grouped[m.constellation].append(
            (m.pseudorange_m - predicted, 1.0 / variance)
        )

    out: dict[str, float] = {}
    for system, samples in grouped.items():
        if not samples:
            continue
        residual, weight = np.asarray(samples, dtype=float).T
        out[system] = float(np.sum(weight * residual) / np.sum(weight))
    return out


def _clock_projector_and_biases(
    measurements,
    variances: Array,
    raw_innovation: Array,
) -> tuple[Array, dict[str, float]]:
    n = len(measurements)
    projector = np.eye(n)
    receiver_clock: dict[str, float] = {}
    for system in ("G", "C"):
        index = np.asarray(
            [i for i, m in enumerate(measurements) if m.constellation == system],
            dtype=int,
        )
        if index.size == 0:
            continue
        if index.size < 2:
            raise RuntimeError(
                f"{system} receiver-clock elimination requires at least two measurements"
            )
        weights = 1.0 / variances[index]
        weight_sum = np.sum(weights)
        normalized_weight = weights / weight_sum
        receiver_clock[system] = float(
            np.sum(weights * raw_innovation[index]) / weight_sum
        )
        block = (
            np.eye(index.size)
            - np.ones((index.size, 1)) @ normalized_weight[None, :]
        )
        projector[np.ix_(index, index)] = block
    return projector, receiver_clock


def build_measurement_model(
    nav: NavigationState,
    measurements,
    lever_arm_b_m: Array,
) -> TCMeasurementModel:
    """Build the 15-state pseudorange model with clock-nuisance projection.

    Runtime optimization: zero-clock pseudorange predictions are evaluated once.
    The WLS receiver-clock estimate is then obtained from those same raw
    innovations and variances instead of repeating the prediction loop.
    """
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return TCMeasurementModel(
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.zeros((0, INS_STATE_DIM)),
            np.zeros((0, 0)),
            (),
            {},
            np.zeros((0, 0)),
        )

    y = np.empty(n)
    y_pred_zero_clock = np.empty(n)
    H_raw = np.zeros((n, INS_STATE_DIM))
    variances = np.empty(n)
    lever_e = nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    antenna_position = nav.position_ecef_m + lever_e
    lever_skew = _skew(lever_e)
    sat_ids: list[str] = []

    zero_clock = {"G": 0.0, "C": 0.0}
    for i, m in enumerate(measurements):
        y_pred_zero_clock[i], los = predict_pseudorange(
            antenna_position, m, zero_clock
        )
        y[i] = m.pseudorange_m
        H_raw[i, 0:3] = los
        H_raw[i, 6:9] = -ATTITUDE_FEEDBACK_SIGN * (los @ lever_skew)
        variance = float(m.sigma_code_m) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("pseudorange variance must be finite and positive")
        variances[i] = variance
        sat_ids.append(m.sat_id)

    raw_innovation = y - y_pred_zero_clock
    projector, receiver_clock = _clock_projector_and_biases(
        measurements, variances, raw_innovation
    )

    innovation = projector @ raw_innovation
    H = projector @ H_raw
    R_raw = np.diag(variances)
    R = projector @ R_raw @ projector.T
    R = 0.5 * (R + R.T)

    y_pred = y.copy() - innovation
    return TCMeasurementModel(
        y,
        y_pred,
        innovation,
        H,
        R,
        tuple(sat_ids),
        receiver_clock,
        projector,
    )


def build_innovation_only(
    nav: NavigationState,
    measurements,
    lever_arm_b_m: Array,
) -> Array:
    """Recompute posterior innovation without constructing unused H/R matrices."""
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return np.empty(0)

    antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
    zero_clock = {"G": 0.0, "C": 0.0}
    raw_innovation = np.empty(n)
    variances = np.empty(n)
    for i, m in enumerate(measurements):
        predicted, _ = predict_pseudorange(antenna_position, m, zero_clock)
        raw_innovation[i] = m.pseudorange_m - predicted
        variance = float(m.sigma_code_m) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("pseudorange variance must be finite and positive")
        variances[i] = variance

    projector, _ = _clock_projector_and_biases(
        measurements, variances, raw_innovation
    )
    return projector @ raw_innovation



@dataclass(frozen=True)
class Ref33FDEResult:
    """Raw-innovation DIA decision used before the Masked-CLA update."""

    tested_measurements: tuple[PseudorangeMeasurement, ...]
    tested_measurement_model: TCMeasurementModel
    measurements: tuple[PseudorangeMeasurement, ...]
    measurement_model: TCMeasurementModel
    detected: bool
    identified_sat_ids: tuple[str, ...]
    identified_constellations: tuple[str, ...]
    ambiguous_sat_ids: tuple[str, ...]
    excluded_sat_ids: tuple[str, ...]
    statistic: float
    threshold: float
    dof: int
    unresolved: bool
    local_identification_score: float
    estimated_fault_m: float
    Q_nu_nu: Array
    # Yan Eq. (34), evaluated in the classical DIA model before the learned KG.
    dia_state_correction: Array
    dia_covariance: Array
    # Kept for backward-compatible logs; v7 does not apply an unpublished
    # post-exclusion rejection loop.
    post_exclusion_statistic: float = 0.0
    post_exclusion_threshold: float = float("inf")
    post_exclusion_dof: int = 0
    post_exclusion_consistent: bool = True


def _ref33_psd_pinv_and_rank(matrix: Array) -> tuple[Array, int]:
    """Moore-Penrose inverse/rank in the effective residual subspace."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Q_nu_nu must be square")
    if matrix.size == 0:
        return np.zeros_like(matrix), 0
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("Q_nu_nu contains non-finite values")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 100.0 * np.finfo(float).eps * max(matrix.shape) * scale
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise FloatingPointError("Q_nu_nu is materially indefinite in Ref. [33] FDE")

    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    if rank == 0:
        return np.zeros_like(matrix), 0

    basis = eigenvectors[:, positive]
    pinv = (basis / eigenvalues[positive]) @ basis.T
    return 0.5 * (pinv + pinv.T), rank


def _ref33_detection_terms(
    prior_covariance: Array,
    measurement_model: TCMeasurementModel,
    alpha: float,
) -> tuple[float, float, int, Array, Array]:
    """Ref. [33] global detection statistic on Yan's raw INS innovation."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("FDE significance alpha must lie strictly between 0 and 1")

    P = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(measurement_model.H, dtype=float)
    R = np.asarray(measurement_model.R, dtype=float)
    nu = np.asarray(measurement_model.innovation, dtype=float)
    Q_nu_nu = 0.5 * ((H @ P @ H.T + R) + (H @ P @ H.T + R).T)
    Q_pinv, dof = _ref33_psd_pinv_and_rank(Q_nu_nu)
    if dof == 0:
        return 0.0, float("inf"), 0, Q_nu_nu, Q_pinv

    statistic = float(nu @ Q_pinv @ nu)
    threshold = float(chi2.ppf(1.0 - float(alpha), df=dof))
    if not math.isfinite(statistic) or not math.isfinite(threshold):
        raise FloatingPointError("non-finite Ref. [33] FDE statistic/threshold")
    return statistic, threshold, dof, Q_nu_nu, Q_pinv


def _ref33_identify_single_pseudorange_fault(
    measurement_model: TCMeasurementModel,
    Q_pinv: Array,
):
    """Ref. [33] Eq. (7), q_i=1, one measurement-fault hypothesis per code."""
    nu = np.asarray(measurement_model.innovation, dtype=float)
    projector = np.asarray(measurement_model.clock_projector, dtype=float)
    qpinv_norm_2 = float(np.linalg.norm(Q_pinv, ord=2))
    eps = np.finfo(float).eps
    candidates = []

    for j in range(len(nu)):
        C_i = projector[:, j].astype(float, copy=False)
        denominator = float(C_i @ Q_pinv @ C_i)
        tolerance = 100.0 * eps * max(
            qpinv_norm_2 * float(C_i @ C_i), np.finfo(float).tiny
        )
        if not math.isfinite(denominator) or denominator <= tolerance:
            continue
        numerator = float(C_i @ Q_pinv @ nu)
        local_statistic = float(numerator * numerator / denominator)
        candidates.append({
            "index": int(j),
            "C_i": C_i.copy(),
            "denominator": denominator,
            "numerator": numerator,
            "local_statistic": local_statistic,
            "local_score": float(chi2.cdf(local_statistic, df=1)),
            "estimated_fault_m": float(numerator / denominator),
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item["local_statistic"])
    best_stat = float(best["local_statistic"])
    best_C = np.asarray(best["C_i"], dtype=float)
    best_den = float(best["denominator"])
    subspace_tol = 1000.0 * eps * max(1, len(nu))
    statistic_tol = 1000.0 * eps * max(1.0, abs(best_stat))
    ambiguous = []
    for candidate in candidates:
        C_j = np.asarray(candidate["C_i"], dtype=float)
        den_j = float(candidate["denominator"])
        cosine = abs(float(best_C @ Q_pinv @ C_j)) / math.sqrt(
            max(best_den * den_j, np.finfo(float).tiny)
        )
        same_subspace = 1.0 - min(max(cosine, 0.0), 1.0) <= subspace_tol
        tied = abs(float(candidate["local_statistic"]) - best_stat) <= statistic_tol
        if same_subspace or tied:
            ambiguous.append(int(candidate["index"]))

    result = dict(best)
    result["ambiguous_indices"] = tuple(sorted(set(ambiguous)))
    result["unique"] = len(result["ambiguous_indices"]) == 1
    return result


def _yan_eq34_dia_adaptation(
    prior_covariance: Array,
    model: TCMeasurementModel,
    Q_nu_nu: Array,
    Q_pinv: Array,
    fault_direction: Array | None,
) -> tuple[Array, Array]:
    """Yan Eq. (34) with Ref. [33] Appendix Eq. (39).

    This is evaluated inside the classical DIA inference branch, where the
    orthogonality assumptions used by Ref. [33] hold. The resulting adapted
    state/covariance are retained for integrity/PL bookkeeping; the proposed
    Masked-CLA navigation update still uses the post-exclusion learned KG.
    """
    Pm = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(model.H, dtype=float)
    R = np.asarray(model.R, dtype=float)
    nu = np.asarray(model.innovation, dtype=float)
    K0 = Pm @ H.T @ Q_pinv
    x0_plus = K0 @ nu
    I_KH = np.eye(INS_STATE_DIM) - K0 @ H
    P0_plus = I_KH @ Pm @ I_KH.T + K0 @ R @ K0.T
    P0_plus = 0.5 * (P0_plus + P0_plus.T)

    if fault_direction is None:
        return x0_plus, P0_plus

    C = np.asarray(fault_direction, dtype=float).reshape(-1, 1)
    denom = (C.T @ Q_pinv @ C).item()
    if denom <= np.finfo(float).tiny:
        return x0_plus, P0_plus
    C_plus = (C.T @ Q_pinv) / denom
    L_i = K0 @ C @ C_plus
    x_i = x0_plus - L_i @ nu
    P_i = P0_plus + L_i @ Q_nu_nu @ L_i.T
    return np.asarray(x_i, dtype=float), 0.5 * (P_i + P_i.T)


def ref33_fde_dia_decision(
    nav: NavigationState,
    prior_covariance: Array,
    measurements,
    lever_arm_b_m: Array,
    alpha: float,
) -> Ref33FDEResult:
    """Yan Sec. II-D / Ref. [33] DIA on raw innovation before the learned KG.

    A selected fault is eliminated and the Masked-CLA update proceeds on the
    remaining measurements. The v6 post-exclusion global re-test that converted
    many epochs into INS-only updates is removed because Yan does not publish
    such a second rejection stage. If clock projection creates an exact tie, all
    indistinguishable candidates are conservatively removed together.
    """
    current = tuple(retain_clock_observable_measurements(measurements))
    tested_model = build_measurement_model(nav, current, lever_arm_b_m)
    empty_model = build_measurement_model(nav, (), lever_arm_b_m)
    zero_dx = np.zeros(INS_STATE_DIM)
    P0 = np.asarray(prior_covariance, dtype=float).copy()

    if not current:
        return Ref33FDEResult(
            (), tested_model, (), empty_model, False, (), (), (), (),
            0.0, float("inf"), 0, False, 0.0, 0.0, np.zeros((0, 0)),
            zero_dx, P0,
        )

    statistic, threshold, dof, Q_nu_nu, Q_pinv = _ref33_detection_terms(
        prior_covariance, tested_model, alpha
    )
    nominal_dx, nominal_P = _yan_eq34_dia_adaptation(
        prior_covariance, tested_model, Q_nu_nu, Q_pinv, None
    )
    if dof == 0 or statistic <= threshold:
        return Ref33FDEResult(
            current, tested_model, current, tested_model, False, (), (), (), (),
            statistic, threshold, dof, False, 0.0, 0.0, Q_nu_nu,
            nominal_dx, nominal_P,
        )

    identification = _ref33_identify_single_pseudorange_fault(tested_model, Q_pinv)
    if identification is None:
        return Ref33FDEResult(
            current, tested_model, (), empty_model, True, (), (),
            tuple(m.sat_id for m in current), (), statistic, threshold, dof, True,
            0.0, 0.0, Q_nu_nu, nominal_dx, nominal_P,
        )

    ambiguous_indices = tuple(identification["ambiguous_indices"])
    if bool(identification["unique"]):
        excluded_indices = (int(identification["index"]),)
        ambiguous_sat_ids = ()
    else:
        excluded_indices = ambiguous_indices
        ambiguous_sat_ids = tuple(current[j].sat_id for j in ambiguous_indices)

    excluded_set = set(excluded_indices)
    selected = current[int(identification["index"])]
    remaining = tuple(m for j, m in enumerate(current) if j not in excluded_set)
    usable = tuple(retain_clock_observable_measurements(remaining))
    filtered_model = build_measurement_model(nav, usable, lever_arm_b_m)
    excluded_sat_ids = tuple(current[j].sat_id for j in excluded_indices)
    excluded_constellations = tuple(current[j].constellation for j in excluded_indices)

    dia_dx, dia_P = _yan_eq34_dia_adaptation(
        prior_covariance,
        tested_model,
        Q_nu_nu,
        Q_pinv,
        np.asarray(identification["C_i"], dtype=float),
    )
    return Ref33FDEResult(
        current, tested_model, usable, filtered_model, True,
        (selected.sat_id,), (selected.constellation,),
        ambiguous_sat_ids, excluded_sat_ids,
        statistic, threshold, dof, not bool(usable),
        float(identification["local_score"]),
        float(identification["estimated_fault_m"]),
        Q_nu_nu, dia_dx, dia_P,
        0.0, float("inf"), 0, True,
    )


def _new_fde_stats() -> dict:
    return {
        "epochs_checked": 0,
        "epochs_detected": 0,
        "identified_fault_modes": 0,
        "hard_exclusion_epochs": 0,
        "excluded_measurements": 0,
        "unresolved_epochs": 0,
        "post_exclusion_failed_epochs": 0,
        "max_statistic_to_threshold_ratio": 0.0,
        "max_abs_estimated_fault_m": 0.0,
        "max_local_identification_score": 0.0,
        "identified_by_constellation": {"G": 0, "C": 0, "L": 0},
    }


def _fde_export_fields(result: Ref33FDEResult | None) -> dict:
    """Diagnostic fields only; None means FDE/DIA was not executed."""
    if result is None:
        return {
            "fde_detected": False, "fde_identified_sat_ids": (),
            "fde_excluded_sat_ids": (), "fde_ambiguous_sat_ids": (),
            "fde_statistic": float("nan"), "fde_threshold": float("nan"),
            "fde_dof": 0, "fde_post_exclusion_statistic": float("nan"),
            "fde_post_exclusion_threshold": float("nan"),
            "fde_post_exclusion_dof": 0, "fde_post_exclusion_consistent": True,
            "fde_local_identification_score": float("nan"),
            "fde_estimated_fault_m": float("nan"),
            "fde_hard_exclusion_applied": False,
            "fde_state_correction_norm": 0.0, "fde_unresolved": False,
        }
    return {
        "fde_detected": bool(result.detected),
        "fde_identified_sat_ids": tuple(result.identified_sat_ids),
        "fde_excluded_sat_ids": tuple(result.excluded_sat_ids),
        "fde_ambiguous_sat_ids": tuple(result.ambiguous_sat_ids),
        "fde_statistic": float(result.statistic),
        "fde_threshold": float(result.threshold), "fde_dof": int(result.dof),
        "fde_post_exclusion_statistic": float(result.post_exclusion_statistic),
        "fde_post_exclusion_threshold": float(result.post_exclusion_threshold),
        "fde_post_exclusion_dof": int(result.post_exclusion_dof),
        "fde_post_exclusion_consistent": bool(result.post_exclusion_consistent),
        "fde_local_identification_score": float(result.local_identification_score),
        "fde_estimated_fault_m": float(result.estimated_fault_m),
        "fde_hard_exclusion_applied": bool(result.excluded_sat_ids),
        "fde_state_correction_norm": float(np.linalg.norm(result.dia_state_correction)),
        "fde_unresolved": bool(result.unresolved),
    }


def _accumulate_fde_stats(stats: dict, result: Ref33FDEResult) -> None:
    stats["epochs_checked"] += 1
    stats["epochs_detected"] += int(result.detected)
    stats["identified_fault_modes"] += len(result.identified_sat_ids)
    stats["hard_exclusion_epochs"] += int(bool(result.excluded_sat_ids))
    stats["excluded_measurements"] += len(result.excluded_sat_ids)
    stats["unresolved_epochs"] += int(result.unresolved)
    stats["post_exclusion_failed_epochs"] += int(
        bool(result.excluded_sat_ids)
        and not result.post_exclusion_consistent
    )

    if math.isfinite(result.threshold) and result.threshold > 0.0:
        stats["max_statistic_to_threshold_ratio"] = max(
            stats["max_statistic_to_threshold_ratio"],
            float(result.statistic / result.threshold),
        )

    stats["max_abs_estimated_fault_m"] = max(
        stats["max_abs_estimated_fault_m"],
        abs(float(result.estimated_fault_m)),
    )
    stats["max_local_identification_score"] = max(
        stats["max_local_identification_score"],
        float(result.local_identification_score),
    )
    for constellation in result.identified_constellations:
        stats["identified_by_constellation"].setdefault(constellation, 0)
        stats["identified_by_constellation"][constellation] += 1


def _validate_ref33_statistical_core(
    alpha: float,
    *,
    sample_count: int = 100_000,
    seed: int = 24681357,
) -> dict:
    """Cheap synthetic regression checks for the project-specific FDE core.

    This does not validate real-data fault probabilities. It verifies two specific
    implementation properties that can be checked without labeled navigation
    faults: (1) the pseudoinverse/rank global statistic has the requested
    chi-square false-alarm rate for a singular projected Gaussian covariance, and
    (2) the exactly-two-measurement clock-projection case is reported as
    ambiguous instead of silently selecting the first hypothesis.
    """
    if sample_count <= 0:
        raise ValueError("FDE self-check sample_count must be positive")

    # Three equal-variance observations with one fitted common clock direction.
    projector = np.eye(3) - np.ones((3, 3)) / 3.0
    Q = projector @ np.diag([4.0, 4.0, 4.0]) @ projector.T
    Q = 0.5 * (Q + Q.T)
    Q_pinv, rank = _ref33_psd_pinv_and_rank(Q)
    if rank != 2:
        raise RuntimeError(f"FDE self-check expected rank 2, got {rank}")

    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    positive = eigenvalues > (
        100.0 * np.finfo(float).eps * max(Q.shape) * max(np.max(np.abs(eigenvalues)), 1.0)
    )
    basis = eigenvectors[:, positive]
    sqrt_values = np.sqrt(eigenvalues[positive])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sample_count, rank))
    nu = (z * sqrt_values) @ basis.T
    statistics = np.einsum("bi,ij,bj->b", nu, Q_pinv, nu)
    threshold = float(chi2.ppf(1.0 - float(alpha), df=rank))
    empirical_false_alarm = float(np.mean(statistics > threshold))
    standard_error = math.sqrt(
        float(alpha) * (1.0 - float(alpha)) / float(sample_count)
    )
    z_score = (empirical_false_alarm - float(alpha)) / max(standard_error, 1e-15)
    if abs(z_score) > 8.0:
        raise RuntimeError(
            "projected Ref. [33] chi-square self-check failed: "
            f"alpha={alpha}, empirical={empirical_false_alarm}, z={z_score}"
        )

    # Strong single-pseudorange fault injection in the 3-observation projected
    # model. This supplies a deterministic regression check for Missed Detection,
    # Correct Identification, and Wrong Identification bookkeeping without
    # claiming to reproduce real-data fault probabilities.
    fault_direction = projector[:, 0]
    injected_fault_m = 20.0
    fault_nu = nu + injected_fault_m * fault_direction[None, :]
    fault_statistics = np.einsum(
        "bi,ij,bj->b", fault_nu, Q_pinv, fault_nu
    )
    detected_fault = fault_statistics > threshold
    candidate_directions = projector
    weighted_directions = Q_pinv @ candidate_directions
    denominators = np.einsum(
        "ij,ij->j", candidate_directions, weighted_directions
    )
    numerators = fault_nu @ weighted_directions
    local_statistics = numerators**2 / denominators[None, :]
    identified_index = np.argmax(local_statistics, axis=1)
    missed_detection_rate = float(np.mean(~detected_fault))
    correct_identification_rate = float(
        np.mean(detected_fault & (identified_index == 0))
    )
    wrong_identification_rate = float(
        np.mean(detected_fault & (identified_index != 0))
    )
    if correct_identification_rate < 0.99:
        raise RuntimeError(
            "FDE strong-fault identification self-check failed: "
            f"CI={correct_identification_rate}, MD={missed_detection_rate}, "
            f"WI={wrong_identification_rate}"
        )

    # Exact two-observation clock projection: C1 and C2 span the same 1-D space.
    projector2 = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=float)
    Q2 = projector2 @ np.eye(2) @ projector2.T
    Q2_pinv, rank2 = _ref33_psd_pinv_and_rank(Q2)
    dummy_model = TCMeasurementModel(
        y=np.zeros(2),
        y_pred=np.zeros(2),
        innovation=np.array([5.0, -5.0]),
        H=np.zeros((2, INS_STATE_DIM)),
        R=Q2,
        sat_ids=("G01", "G02"),
        receiver_clock_bias_m_by_system={"G": 0.0},
        clock_projector=projector2,
    )
    ambiguity = _ref33_identify_single_pseudorange_fault(dummy_model, Q2_pinv)
    if (
        rank2 != 1
        or ambiguity is None
        or bool(ambiguity["unique"])
        or tuple(ambiguity["ambiguous_indices"]) != (0, 1)
    ):
        raise RuntimeError(
            "FDE ambiguity self-check failed for two projected pseudoranges"
        )

    return {
        "sample_count": int(sample_count),
        "requested_alpha": float(alpha),
        "empirical_false_alarm": empirical_false_alarm,
        "false_alarm_z_score": float(z_score),
        "projected_test_rank": int(rank),
        "strong_fault_injected_m": injected_fault_m,
        "strong_fault_missed_detection_rate": missed_detection_rate,
        "strong_fault_correct_identification_rate": correct_identification_rate,
        "strong_fault_wrong_identification_rate": wrong_identification_rate,
        "two_measurement_ambiguity_detected": True,
        "scope": "synthetic_core_only_not_real_data_integrity_validation",
    }


def kalman_measurement_update(P: Array, innovation: Array, H: Array, R: Array):
    """Conventional full 15-state TC measurement update.

    Yan et al. define innovation/residual/state-history features but do not publish
    the estimator used to generate those offline quantities. This conventional TC
    pass is therefore the feature-history completion and the independent classical
    test baseline; it is not the proposed online Masked-CLA estimator. The GPS/BDS
    clock projection makes S singular in removed clock directions, so a
    Moore-Penrose inverse is used.
    """
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)

    # Yan Eq. (7) is a 15-state TC model and Eq. (8) explicitly consumes the
    # estimated accelerometer/gyroscope biases.  Preserve all 15 rows of the
    # conventional Kalman gain; zeroing the bias rows would break the propagated
    # cross-covariance path that makes those biases observable over time.
    if not np.all(np.isfinite(K)):
        raise FloatingPointError("non-finite 15-state classical Kalman gain")

    dx = K @ innovation
    I_KH = np.eye(INS_STATE_DIM) - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_reset = reset_error_state_covariance(P_post, dx)
    return dx, P_reset, K


def learned_gain_covariance_update(
    prior_covariance: Array,
    learned_gain: Array,
    measurement_jacobian: Array,
    measurement_covariance: Array,
    injected_error_state: Array,
) -> Array:
    """Joseph covariance bookkeeping for a KalmanNet-produced gain.

    IMPORTANT:
    - The Masked CLA does NOT use ``R`` to compute its Kalman gain.
    - The navigation correction remains ``dx = K_net @ innovation``.
    - ``R`` appears here only to propagate a covariance estimate for diagnostics
      and future FDE/integrity work. Yan et al. do not publish an explicit
      learned-gain covariance-update equation, so this is a declared
      STANDARD-COMPLETION rather than a paper-exact step.
    - After the nonlinear attitude feedback, the covariance is mapped into the
      reset error-state coordinates used by the next propagation epoch.
    """
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    learned_gain = np.asarray(learned_gain, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    injected_error_state = np.asarray(
        injected_error_state, dtype=float
    ).reshape(INS_STATE_DIM)

    measurement_count = measurement_jacobian.shape[0]
    if (
        prior_covariance.shape != (INS_STATE_DIM, INS_STATE_DIM)
        or measurement_jacobian.ndim != 2
        or measurement_jacobian.shape[1] != INS_STATE_DIM
        or learned_gain.shape != (INS_STATE_DIM, measurement_count)
        or measurement_covariance.shape != (measurement_count, measurement_count)
    ):
        raise ValueError("inconsistent P/K/H/R dimensions in learned covariance update")

    if (
        not np.all(np.isfinite(prior_covariance))
        or not np.all(np.isfinite(learned_gain))
        or not np.all(np.isfinite(measurement_jacobian))
        or not np.all(np.isfinite(measurement_covariance))
        or not np.all(np.isfinite(injected_error_state))
    ):
        raise ValueError("non-finite P/K/H/R in learned covariance update")

    update_matrix = np.eye(INS_STATE_DIM) - learned_gain @ measurement_jacobian
    posterior_covariance = (
        update_matrix @ prior_covariance @ update_matrix.T
        + learned_gain @ measurement_covariance @ learned_gain.T
    )
    return reset_error_state_covariance(
        posterior_covariance,
        injected_error_state,
    )



def protection_levels_from_covariance(
    covariance_ecef: Array,
    position_ecef_m: Array,
) -> tuple[float, float]:
    """HPL/VPL from Ref. [35] Eqs. (57)-(58), used by Yan's PL branch."""
    P = np.asarray(covariance_ecef, dtype=float)
    lat, lon, _ = ecef_to_llh(position_ecef_m)
    C = c_ecef_to_ned(lat, lon)
    P_ned = C @ P[:3, :3] @ C.T
    P_ned = 0.5 * (P_ned + P_ned.T)
    pnn, pee, pdd = float(P_ned[0, 0]), float(P_ned[1, 1]), float(P_ned[2, 2])
    pne = float(P_ned[0, 1])
    horizontal_major_variance = (
        0.5 * (pnn + pee)
        + math.sqrt(max(0.25 * (pnn - pee) ** 2 + pne * pne, 0.0))
    )
    hpl = PL_HORIZONTAL_MULTIPLIER * math.sqrt(max(horizontal_major_variance, 0.0))
    vpl = PL_VERTICAL_MULTIPLIER * math.sqrt(max(pdd, 0.0))
    return float(hpl), float(vpl)


def closed_loop_spectral_radius(gain: Array, H: Array) -> float:
    """Spectral radius of the square correction operator I-KH.

    Yan Sec. II-E reports a closed-loop spectral-radius stability diagnostic but
    does not print the exact matrix expression. I-KH is the dimensionally valid
    KF correction operator associated with the learned gain and is logged here
    as the explicit diagnostic completion.
    """
    K = np.asarray(gain, dtype=float)
    Hm = np.asarray(H, dtype=float)
    if K.ndim != 2 or Hm.ndim != 2 or K.shape[1] != Hm.shape[0]:
        return float("nan")
    Acl = np.eye(K.shape[0]) - K @ Hm
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


def _so3_left_jacobian(rotation_vector_rad: Array) -> Array:
    """Exact SO(3) left Jacobian for the attitude error reset.

    If ``phi`` is the injected left-multiplicative rotation, a small pre-reset
    attitude error perturbation is expressed after feedback through
    ``J_l(phi)``.  The series branch keeps the zero/small-angle case stable.
    """
    phi = np.asarray(rotation_vector_rad, dtype=float).reshape(3)
    theta_squared = float(phi @ phi)
    K = _skew(phi)
    K2 = K @ K
    if theta_squared < 1e-12:
        # J_l(phi) = I + (1/2-theta^2/24)K
        #              + (1/6-theta^2/120)K^2 + O(theta^6)
        a = 0.5 - theta_squared / 24.0 + theta_squared**2 / 720.0
        b = 1.0 / 6.0 - theta_squared / 120.0 + theta_squared**2 / 5040.0
    else:
        theta = math.sqrt(theta_squared)
        a = (1.0 - math.cos(theta)) / theta_squared
        b = (theta - math.sin(theta)) / (theta_squared * theta)
    return np.eye(3) + a * K + b * K2


def reset_error_state_covariance(
    posterior_covariance: Array,
    injected_error_state: Array,
) -> Array:
    """Map posterior covariance into the post-feedback error coordinates.

    ``inject_error_state`` applies

        C_new = Exp(ATTITUDE_FEEDBACK_SIGN * dtheta) C_old.

    Position, velocity, and bias feedback are additive, so their reset Jacobian
    is identity.  The attitude block is the exact SO(3) left Jacobian evaluated
    at the injected rotation.  This is a standard error-state completion needed
    to keep the covariance used by the next FDE/propagation epoch consistent
    with the reset nominal state; Yan et al. do not print this bookkeeping step.
    """
    P = np.asarray(posterior_covariance, dtype=float)
    dx = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    if P.shape != (INS_STATE_DIM, INS_STATE_DIM):
        raise ValueError("posterior covariance has wrong shape for error-state reset")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(dx)):
        raise ValueError("non-finite covariance/error-state in feedback reset")

    reset_jacobian = np.eye(INS_STATE_DIM)
    reset_jacobian[6:9, 6:9] = _so3_left_jacobian(
        ATTITUDE_FEEDBACK_SIGN * dx[6:9]
    )
    reset_covariance = reset_jacobian @ P @ reset_jacobian.T
    reset_covariance = 0.5 * (reset_covariance + reset_covariance.T)
    if not np.all(np.isfinite(reset_covariance)):
        raise FloatingPointError("non-finite covariance after error-state reset")
    return reset_covariance


def inject_error_state(nav: NavigationState, dx: Array) -> NavigationState:
    dx = np.asarray(dx, dtype=float).reshape(15)
    out = nav.copy()
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _rotation(
        _so3_exponential(ATTITUDE_FEEDBACK_SIGN * dx[6:9])
        @ out.body_to_ecef_dcm
    )
    out.accelerometer_bias_body_mps2 += dx[ACCELEROMETER_BIAS_STATE_SLICE]
    out.gyroscope_bias_body_radps += dx[GYROSCOPE_BIAS_STATE_SLICE]
    return out


# =============================================================================
# TLE/SGP4 LEO orbit and pseudorange-only simulation
# =============================================================================
@dataclass(frozen=True)
class _TLEElement:
    epoch_gpst_s: float
    satrec: Satrec


@dataclass(frozen=True)
class LEOSatelliteState:
    position_ecef_m: Array
    velocity_ecef_mps: Array



@lru_cache(maxsize=8192)
def _gpst_to_utc_time_cached(time_gpst_s: float) -> Time:
    """Cache repeated receive-epoch GPST->UTC conversions used by all LEOs."""
    return Time(float(time_gpst_s), format="gps").utc

class TLESGP4Provider:
    """Approximate LEO orbit source: TLE + SGP4/WGS-72 + TEME->ITRS/ECEF."""
    def __init__(
        self,
        path: str | Path,
        max_tle_age_days: float,
        allow_degraded_eop: bool = False,
        allow_non_tle_files: bool = True,
    ):
        self.max_tle_age_s = float(max_tle_age_days) * 86400.0
        self.allow_degraded_eop = allow_degraded_eop
        by_id = {}
        for line1, line2 in read_tle_directory(path, allow_non_tle_files):
            satrec = Satrec.twoline2rv(line1, line2, WGS72)
            norad = str(satrec.satnum_str).strip()
            epoch = Time(satrec.jdsatepoch, satrec.jdsatepochF, format="jd", scale="utc")
            by_id.setdefault(f"NORAD-{norad}", []).append(
                _TLEElement(float(epoch.gps), satrec)
            )
        self._elements = {
            sat_id: tuple(sorted(elements, key=lambda e: e.epoch_gpst_s))
            for sat_id, elements in by_id.items()
        }
        self._element_times = {
            sat_id: tuple(element.epoch_gpst_s for element in elements)
            for sat_id, elements in self._elements.items()
        }
        self.satellite_ids = tuple(
            sorted(self._elements, key=lambda s: int(s.split("-")[1]))
        )

    def state_at(self, time_gpst_s: float, sat_id: str) -> LEOSatelliteState:
        elements = self._elements[sat_id]
        index = bisect_right(self._element_times[sat_id], time_gpst_s) - 1
        if index < 0:
            raise ValueError(f"No prior TLE for {sat_id}")
        element = elements[index]
        if time_gpst_s - element.epoch_gpst_s > self.max_tle_age_s:
            raise ValueError(f"TLE too old for {sat_id}")

        utc = _gpst_to_utc_time_cached(float(time_gpst_s))
        error, position_km, velocity_km_s = element.satrec.sgp4(float(utc.jd1), float(utc.jd2))
        if error:
            raise ValueError(SGP4_ERRORS.get(error, f"SGP4 error {error}"))

        position = CartesianRepresentation(np.asarray(position_km) * u.km)
        velocity = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        teme = TEME(position.with_differentials(velocity), obstime=utc)
        degraded = "warn" if self.allow_degraded_eop else "error"
        with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("iers_degraded_accuracy", degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        return LEOSatelliteState(
            np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3),
            np.asarray(itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s)).reshape(3),
        )


def ref35_ionosphere_sigma_m(elevation_rad: float, receiver_latitude_rad: float) -> float:
    """Ref. [35] Eqs. (15)-(16): 1-sigma ionospheric residual error [m]."""
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else (4.5 if latitude_deg <= 55.0 else 6.0)
    R = 6_378_140.0
    h_i = 350_000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))


def ref35_troposphere_sigma_m(elevation_rad: float) -> float:
    """Ref. [35] Eq. (17): Black-Eisner mapped 1-sigma troposphere error [m]."""
    return float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation_rad)) ** 2))


def ref35_multipath_sigma_m(elevation_rad: float) -> float:
    """Previous-code LEO MP/NLOS 1-sigma model: Ref. [35] Eq. (18).

    sigma_mp = 0.13 + 0.53 exp(-elevation_deg / 10).

    Ref. [35] models multipath rather than NLOS separately. Per the explicit
    project choice for this reproduction, this unchanged v6 term is used as the
    combined LEO MP/NLOS variance model. This is intentionally different from
    Yan Eq. (4), whose exact LEO C/N0-vs-elevation realization is not published.
    """
    elevation_deg = float(np.rad2deg(elevation_rad))
    if not math.isfinite(elevation_deg):
        raise ValueError("finite elevation is required")
    if elevation_deg < 0.0:
        raise ValueError("multipath model requires a nonnegative elevation angle")
    return float(0.13 + 0.53 * math.exp(-elevation_deg / 10.0))


def yan_goGPS_mp_nlos_sigma_m(elevation_rad: float, cn0_dbhz: float) -> float:
    """Yan Eq. (4), using the sample goGPS [37] parameters A,s0,s1,a.

    Yan prints Eq. (4) as an MP/NLOS *variance*. goGPS supplies the same law as
    a weighting function and publishes A=30, s0=10, s1=50, a=20 as sample
    values. We therefore follow Yan's own interpretation and return sqrt(Eq.4).
    """
    e = max(float(elevation_rad), 1e-6)
    cn0 = float(cn0_dbhz)
    if cn0 >= GOGPS_S1_DBHZ:
        variance = 1.0
    else:
        ratio = (cn0 - GOGPS_S1_DBHZ) / (GOGPS_S0_DBHZ - GOGPS_S1_DBHZ)
        term = (
            10.0 ** (-(cn0 - GOGPS_S1_DBHZ) / GOGPS_A_DB)
            * (
                (GOGPS_A / 10.0 ** (-(GOGPS_S0_DBHZ - GOGPS_S1_DBHZ) / GOGPS_A_DB) - 1.0)
                * ratio
                + 1.0
            )
        )
        variance = (1.0 / max(math.sin(e) ** 2, 1e-6)) * term
    return float(math.sqrt(max(variance, 1e-12)))


def ref35_receiver_noise_sigma_m(cn0_dbhz: float) -> float:
    """Ref. [35] Eqs. (19)-(20): early-minus-late code tracking noise [m]."""
    cn0_linear = 10.0 ** (float(cn0_dbhz) / 10.0)  # Hz
    d = LEO_CORRELATOR_SPACING_CHIPS
    bl = LEO_CODE_LOOP_BANDWIDTH_HZ
    tau = LEO_CORRELATOR_ACCUMULATION_S
    sigma_chips = math.sqrt(
        bl * d / (2.0 * cn0_linear)
        * (1.0 + 2.0 / ((2.0 - d) * cn0_linear * tau))
    )
    return float(SPEED_OF_LIGHT_MPS / LEO_CODE_CHIPPING_RATE_HZ * sigma_chips)


def leo_cn0_design_dbhz(elevation_rad: float) -> float:
    """Compact completion of the unpublished LEO elevation->C/N0 curve.

    Yan et al. state that this relationship is designed according to Ref. [35]
    but publish no equation or fitted coefficients. Ref. [35] shows real C/N0
    values rather than a deterministic mapping. A bounded monotone link-quality
    curve is therefore used only to make Yan Eq. (4) and Ref. [35] Eqs. (19)-(20)
    executable; it is not attributed to the paper.
    """
    s = max(0.0, math.sin(float(elevation_rad)))
    return float(np.clip(25.0 + 25.0 * math.sqrt(s), 20.0, 50.0))


def _unit_variance_student_t(rng: np.random.Generator, df: float = 3.0) -> float:
    """Heavy-tailed unit-variance draw for the paper's non-Gaussian LEO errors."""
    return float(rng.standard_t(df) * math.sqrt((df - 2.0) / df))


class LEODownlinkSimulator:
    """Yan Eqs. (1),(2),(5) with TLE/SGP4 and the project's unchanged v6 variance.

    The deterministic LEO geometry, light-time iteration, ionosphere-height
    weighting, and troposphere model remain aligned with the paper. Per the
    explicit project choice, the stochastic LEO variance is kept exactly as in
    v6: ionosphere + troposphere + Ref.[35] Eq.(18) elevation-only MP/NLOS.
    URA/receiver-noise variance and Yan Eq.(4)'s C/N0-dependent MP/NLOS term are
    not substituted into this branch because doing so would change the requested
    LEO variance.
    """
    def __init__(
        self,
        provider: TLESGP4Provider,
        klobuchar_coefficients: tuple[Array, Array] | None,
        seed: int = 0,
        tx_epsilon_position_m: float = 1e-3,
        tx_max_iterations: int = 20,
        minimum_elevation_deg: float = 10.0,
        prefilter_guard_deg: float = LEO_PREFILTER_GUARD_DEG,
        use_ionosphere: bool = True,
        use_troposphere: bool = True,
    ):
        self.provider = provider
        self.klobuchar_coefficients = klobuchar_coefficients
        self.tx_epsilon_position_m = float(tx_epsilon_position_m)
        self.tx_max_iterations = int(tx_max_iterations)
        self.minimum_elevation_rad = np.deg2rad(minimum_elevation_deg)
        self.prefilter_guard_rad = np.deg2rad(float(prefilter_guard_deg))
        if self.prefilter_guard_rad < 0.0:
            raise ValueError("LEO prefilter guard must be nonnegative")
        self.use_ionosphere = bool(use_ionosphere)
        self.use_troposphere = bool(use_troposphere)
        self.prefilter_checked = 0
        self.prefilter_rejected = 0
        self.prefilter_passed = 0
        # Reproducible independent residual draws for the Ref. [35] sigma models.
        self.rng = np.random.default_rng(seed)

    def _paper_ionosphere_delay_m(
        self,
        receive_time_gpst_s: float,
        receiver_position_ecef_m: Array,
        satellite_position_reception_ecef_m: Array,
        elevation_rad: float,
        azimuth_rad: float,
        receiver_llh: tuple[float, float, float] | None = None,
    ) -> tuple[float, float]:
        if not self.use_ionosphere:
            return 0.0, 0.0
        if self.klobuchar_coefficients is None:
            raise ValueError("Klobuchar coefficients are required for Yan et al. Eq. (2)")
        alpha, beta = self.klobuchar_coefficients
        if receiver_llh is None:
            receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        lat, lon, _ = receiver_llh
        tow = gpst_seconds_to_week_tow(receive_time_gpst_s).tow_s
        iklo_m = klobuchar_delay_m(tow, lat, lon, elevation_rad, azimuth_rad, alpha, beta)
        _, _, satellite_height_m = ecef_to_llh(satellite_position_reception_ecef_m)
        if satellite_height_m >= LEO_IONOSPHERE_UPPER_HEIGHT_M:
            scale = 1.0
        elif satellite_height_m <= LEO_IONOSPHERE_LOWER_HEIGHT_M:
            scale = 0.0
        else:
            scale = (
                (satellite_height_m - LEO_IONOSPHERE_LOWER_HEIGHT_M)
                / (LEO_IONOSPHERE_UPPER_HEIGHT_M - LEO_IONOSPHERE_LOWER_HEIGHT_M)
            )
        return float(scale * iklo_m), float(scale)

    def simulate_one(
        self,
        receive_time_gpst_s: float,
        receiver_position_ecef_m: Array,
        sat_id: str,
        receiver_llh: tuple[float, float, float] | None = None,
        c_ecef_ned: Array | None = None,
    ):
        if receiver_llh is None:
            receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        if c_ecef_ned is None:
            c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])

        initial_state = self.provider.state_at(receive_time_gpst_s, sat_id)
        initial_range, initial_los, _ = geometric_range(
            receiver_position_ecef_m, initial_state.position_ecef_m, 0.0
        )
        initial_elevation, _ = elevation_azimuth_from_ned_matrix(
            c_ecef_ned, initial_los
        )
        self.prefilter_checked += 1
        if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
            self.prefilter_rejected += 1
            return None
        self.prefilter_passed += 1

        transmit_time, transit_s, state_tx_position = _iterate_transmit_time(
            lambda sid, t: self.provider.state_at(t, sid).position_ecef_m,
            sat_id,
            receive_time_gpst_s,
            receiver_position_ecef_m,
            initial_state.position_ecef_m,
            initial_range / SPEED_OF_LIGHT_MPS,
            self.tx_epsilon_position_m,
            self.tx_max_iterations,
        )
        rho, los, sat_rx = geometric_range(
            receiver_position_ecef_m, state_tx_position, transit_s
        )
        elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
        if elevation < self.minimum_elevation_rad:
            return None

        receiver_lat, _, receiver_height_m = receiver_llh

        # Yan et al. Eq. (2): Klobuchar above the ionosphere, linearly weighted
        # by satellite height when the LEO is inside the ionosphere.
        ionosphere_m, ionosphere_path_scale = self._paper_ionosphere_delay_m(
            receive_time_gpst_s,
            receiver_position_ecef_m,
            sat_rx,
            elevation,
            azimuth,
            receiver_llh=receiver_llh,
        )

        # Yan et al.: troposphere can be mitigated/modelled as in traditional
        # GNSS. The project already uses a standard Saastamoinen delay model.
        troposphere_m = (
            saastamoinen_delay_m(receiver_height_m, elevation)
            if self.use_troposphere
            else 0.0
        )

        # Keep the LEO stochastic variance exactly as in the previous v6 code.
        # Ionosphere/troposphere residual sigmas come from Ref. [35], while the
        # combined MP/NLOS term uses the previous elevation-only Ref. [35] Eq. (18)
        # model requested for this project. URA and receiver-noise variance are
        # deliberately NOT added here, so the LEO covariance is not changed.
        iono_sigma_m = (
            ionosphere_path_scale
            * ref35_ionosphere_sigma_m(elevation, receiver_lat)
            if self.use_ionosphere
            else 0.0
        )
        tropo_sigma_m = (
            ref35_troposphere_sigma_m(elevation)
            if self.use_troposphere
            else 0.0
        )
        mp_sigma_m = ref35_multipath_sigma_m(elevation)

        # Same residual realization as v6: independent zero-mean Gaussian terms
        # with component variances matching the covariance used in R.
        ionosphere_residual_m = (
            float(self.rng.normal(0.0, iono_sigma_m))
            if iono_sigma_m > 0.0
            else 0.0
        )
        troposphere_residual_m = (
            float(self.rng.normal(0.0, tropo_sigma_m))
            if tropo_sigma_m > 0.0
            else 0.0
        )
        mp_nlos_error_m = (
            float(self.rng.normal(0.0, mp_sigma_m))
            if mp_sigma_m > 0.0
            else 0.0
        )

        total_variance_m2 = (
            iono_sigma_m**2
            + tropo_sigma_m**2
            + mp_sigma_m**2
        )
        sigma_code_m = math.sqrt(max(total_variance_m2, 1e-12))

        pseudorange_m = (
            rho
            + ionosphere_m
            + troposphere_m
            + ionosphere_residual_m
            + troposphere_residual_m
            + mp_nlos_error_m
        )
        return PseudorangeMeasurement(
            sat_id,
            "L",
            float(pseudorange_m),
            sat_rx,
            0.0,
            float(ionosphere_m),
            float(troposphere_m),
            float(sigma_code_m),
            float(elevation),
            None,
        )

    def simulate_epoch(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array):
        receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        measurements = []
        for sat_id in self.provider.satellite_ids:
            try:
                measurement = self.simulate_one(
                    receive_time_gpst_s,
                    receiver_position_ecef_m,
                    sat_id,
                    receiver_llh=receiver_llh,
                    c_ecef_ned=c_ecef_ned,
                )
            except ValueError:
                continue
            except RuntimeError as exc:
                # TLE/SGP4 is an intentional project replacement for the paper's
                # STK/HPOP orbit.  A single light-time iteration failure must not
                # destroy a many-hour training run; that satellite is treated as
                # unavailable for this epoch only.  Other RuntimeErrors propagate.
                if "Transmit-time iteration did not converge" in str(exc):
                    continue
                raise
            if measurement is not None:
                measurements.append(measurement)
        return tuple(measurements)


# =============================================================================
# MASKED CLA NETWORK
# =============================================================================
# Paper-supported architecture:
#   Eqs. (22)-(23): masked convolution + mask propagation
#   Eqs. (24)-(25): masked LSTM state update
#   Eqs. (26)-(29): masked additive attention
#   Fig. 8: zero padding/mask -> Masked CNN -> Pooling -> Flatten
#           -> Masked LSTM-Attention -> Masked FC
#   Table III: 24 Conv1D filters, stride 1, ReLU, 64 LSTM units, 5 layers,
#              dropout 0.2.
#
# Conservative paper-aligned sequence interpretation used here:
#   - k indexes a fusion epoch/sample;
#   - t indexes the zero-padded observation-sequence position within that epoch;
#   - CNN/Pooling/Flatten are applied to each epoch independently;
#   - k is the LSTM's recurrent time direction (fusion epochs); the Nmax slot
#     states are processed in parallel with shared weights;
#   - Eq. (25) uses M[k,t] to update or retain each slot's h/c from epoch k-1;
#   - following the original KalmanNet sequence contract, state is reset at each
#     trajectory boundary and is never shared between Data01 and Data02.
#
# Explicit project completions that remain because Yan et al. do not publish them:
#   - one pseudorange-only satellite slot is represented by
#       [previous residual, current innovation]
#   - the fixed 36 IMU/state features are broadcast to each valid satellite slot
#   - Fig. 8 shows Pooling but does not give its type/kernel/stride. A same-length
#     masked max-pool (kernel 3, stride 1) is used so Eq. (23) can preserve
#     M_out == M_in; this is a declared figure-guided completion
#   - Fig. 8's Flatten is implemented as per-slot flattening of the 24 pooled CNN
#     feature maps into a 24-element vector before the LSTM
#   - Fig. 8's KG_k -> State Update path is explicit in training; Eq. (30) is
#     evaluated after composing that state update with the postprocessed truth
#   - Yan Eq. (7) requires a full 15-row KG_k.  The supplied truth directly labels
#     only position/velocity/attitude, so Eq. (30) is evaluated on those available
#     nine coordinates; no artificial bias labels are introduced
#   - the six bias-gain rows are conservatively zero-initialized; the supplied
#     dataset does not provide direct b_a/b_g truth, so no artificial bias target
#     is introduced
#   - Fig. 8 also outputs eta_k for inertial-navigation measurement-error
#     estimation. Its target/timing are unpublished. Instead of inventing a
#     direct eta label, eta_k is supervised through its next-epoch navigation
#     state effect via Eq. (8)+INS and the same Eq. (30) state criterion.
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 36
OBSERVATION_FEATURE_DIM = 2                 # [previous residual, current innovation]
IMU_ERROR_DIM = 6
ETA_GYRO_SLICE = slice(0, 3)
ETA_ACCEL_SLICE = slice(3, 6)
# Numerical differentiation steps used only by the offline teacher-forced
# eta_k supervision completion. They are not physical priors or output bounds.
ETA_FD_GYRO_STEP_RADPS = 1e-6
ETA_FD_ACCEL_STEP_MPS2 = 1e-4
SUPERVISED_STATE_DIM = DIRECT_STATE_LABEL_DIM  # directly labeled [p, v, attitude]
MASK_NORMALIZATION_EPS = 1e-6
FIG8_POOL_KERNEL_SIZE = 3


@dataclass(frozen=True)
class MaskedCLAOutput:
    """One Masked-CLA forward pass.

    Shapes
    ------
    kalman_gain : [B, 15, Nmax]
    imu_error   : [B, 6] -- eta_k = [epsilon_g(3) rad/s, epsilon_a(3) m/s^2]
    attention   : [B, Nmax]
    recurrent_state : None (no cross-fusion hidden-state carry)
    """

    kalman_gain: torch.Tensor
    imu_error: torch.Tensor
    attention: torch.Tensor
    recurrent_state: None = None


class MaskedConv1d(nn.Module):
    """Yan Eqs. (22)-(23) plus Pooling/Flatten drawn explicitly in Fig. 8.

    The convolution uses the Table-III values already adopted by this
    reproduction: 24 output filters, kernel width 3, stride 1 and ReLU. The
    valid-window normalization is mask-aware, so padded zeros do not bias edge
    features.

    Fig. 8 includes Pooling followed by Flatten, but Yan et al. do not publish
    the pooling operator or its hyperparameters. The minimal figure-guided
    completion used here is a shape-preserving masked max-pool with kernel=3,
    stride=1. It preserves the padded sequence length required by Eq. (23).
    "Flatten" is the final [B,C,N] -> [B,N,C] conversion: each sequence position
    receives its 24 pooled CNN feature maps as one LSTM input vector.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 24,
        kernel_size: int = 3,
        pool_kernel_size: int = FIG8_POOL_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if pool_kernel_size <= 0 or pool_kernel_size % 2 == 0:
            raise ValueError("pool_kernel_size must be a positive odd integer")

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.pool_kernel_size = int(pool_kernel_size)
        self.pool_padding = self.pool_kernel_size // 2
        self.register_buffer(
            "_mask_kernel",
            torch.ones(1, 1, self.kernel_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError("MaskedConv1d expects x=[B,N,C] and mask=[B,N]")
        if x.shape[:2] != mask.shape:
            raise ValueError("MaskedConv1d x/mask sequence shapes do not match")

        m = mask.to(dtype=x.dtype).unsqueeze(1)  # [B,1,N]

        x_masked = x.transpose(1, 2) * m         # [B,Cin,N]
        z = F.conv1d(
            x_masked,
            self.weight,
            bias=None,
            stride=1,
            padding=self.padding,
        )

        mask_kernel = self._mask_kernel.to(dtype=x.dtype)
        local_count = F.conv1d(
            m,
            mask_kernel,
            stride=1,
            padding=self.padding,
        )
        valid_window = (local_count > 0).to(dtype=x.dtype)
        denom = local_count.clamp_min(MASK_NORMALIZATION_EPS)

        feature_maps = F.relu(
            z / denom
            + self.bias.view(1, -1, 1) * valid_window
        )
        feature_maps = feature_maps * m

        # Fig. 8 Pooling. Exact type/kernel/stride are unpublished.
        if self.pool_kernel_size > 1:
            very_negative = torch.finfo(feature_maps.dtype).min
            pool_input = torch.where(
                m.bool(),
                feature_maps,
                torch.full_like(feature_maps, very_negative),
            )
            pooled = F.max_pool1d(
                pool_input,
                kernel_size=self.pool_kernel_size,
                stride=1,
                padding=self.pool_padding,
            )
            feature_maps = torch.where(
                m.bool(),
                pooled,
                torch.zeros_like(pooled),
            )

        # Fig. 8 Flatten: preserve sequence position t and flatten only the CNN
        # feature-map dimension into one 24-element vector per position.
        return feature_maps.transpose(1, 2)      # [B,N,24]


class MaskedStackedLSTM(nn.Module):
    """Yan Eqs. (24)-(25): masked LSTM over one padded fusion-epoch sequence.

    Each call receives one or more padded X_bar[k] sequences with shape [B,N,C].
    The valid mask is a contiguous prefix [1,...,1,0,...,0], as constructed by
    Yan Eqs. (16)-(21). A standard stacked LSTM is therefore evaluated only over
    the valid prefix using ``pack_padded_sequence``. The padded suffix never
    updates h/c and its returned features are forced to zero. Hidden/cell state
    is local to this call and is not carried to the next navigation fusion epoch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError(
                "input_size, hidden_size and num_layers must be positive"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None]:
        if recurrent_state is not None:
            raise ValueError(
                "cross-fusion recurrent_state is intentionally disabled; "
                "each fusion epoch is one masked LSTM sequence"
            )
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "MaskedStackedLSTM expects x=[B,N,C] and mask=[B,N]"
            )
        if x.shape[:2] != mask.shape:
            raise ValueError(
                "MaskedStackedLSTM x/mask sequence shapes do not match"
            )

        batch_size, slot_count = x.shape[:2]
        if slot_count <= 0:
            raise ValueError("MaskedStackedLSTM requires a non-empty sequence")

        valid = mask.bool()
        lengths = valid.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("each MaskedStackedLSTM sample needs at least one valid slot")

        # Yan's mask is a valid prefix followed by padding. Reject accidental holes.
        positions = torch.arange(slot_count, device=mask.device).unsqueeze(0)
        expected_prefix = positions < lengths.unsqueeze(1)
        if not torch.equal(valid, expected_prefix):
            raise ValueError(
                "masked LSTM expects a contiguous valid prefix as in Yan Eq. (21)"
            )

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=slot_count,
        )
        output = output * valid.unsqueeze(-1).to(dtype=output.dtype)
        return output, None


class MaskedAttention(nn.Module):
    """Paper Eqs. (26)-(29): masked additive attention."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "MaskedAttention expects h=[B,N,H] and mask=[B,N]"
            )
        if h.shape[:2] != mask.shape:
            raise ValueError(
                "MaskedAttention h/mask sequence shapes do not match"
            )

        score = self.v(
            torch.tanh(self.proj(h))
        ).squeeze(-1)

        valid = mask.bool()
        score = score.masked_fill(~valid, -torch.inf)

        all_masked = ~valid.any(dim=1)
        safe_score = score.masked_fill(
            all_masked.unsqueeze(1), 0.0
        )

        alpha = torch.softmax(safe_score, dim=1)
        alpha = alpha * valid.to(dtype=alpha.dtype)
        denom = alpha.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        alpha = torch.where(
            all_masked.unsqueeze(1),
            torch.zeros_like(alpha),
            alpha / denom,
        )

        context = torch.sum(
            alpha.unsqueeze(-1) * h,
            dim=1,
        )
        return context, alpha


class MaskedCLA(nn.Module):
    """Masked CNN-LSTM-attention Kalman-gain estimator.

    Paper/figure-supported feature path:

        zero-padding + M[k,t]
            -> Masked Conv1D
            -> Pooling
            -> Flatten
            -> 5-layer Masked LSTM
            -> Masked Attention
            -> FC heads -> KG_k and eta_k.

    Each X_bar[k] is one padded observation sequence. The masked CNN processes
    its valid slots, and the masked LSTM runs along those slots inside the same
    fusion epoch. Hidden/cell state is reset for the next X_bar[k+1]; Yan et al.
    do not publish a cross-fusion h/c carry, so none is introduced here.

    Yan's fixed offline labeled X/M/Y dataset of Eqs. (18)-(20) is the training
    set, while alternating optimization follows the general Ref. [15] principle.
    Exact pooling hyperparameters, FC-head tensorization, eta_k supervision/timing
    and behavior when N_test>Nmax_train remain explicitly documented completions.
    """

    def __init__(
        self,
        nmax: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if nmax <= 0:
            raise ValueError("nmax must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")

        self.nmax = int(nmax)
        self.dropout = float(dropout)

        self.conv = MaskedConv1d(
            in_channels=FIXED_FEATURE_DIM + OBSERVATION_FEATURE_DIM,
            out_channels=24,
            kernel_size=3,
            pool_kernel_size=FIG8_POOL_KERNEL_SIZE,
        )
        self.lstm = MaskedStackedLSTM(
            input_size=24,
            hidden_size=64,
            num_layers=5,
            dropout=self.dropout,
        )
        self.attention = MaskedAttention(64)

        # STANDARD-COMPLETION for variable measurement counts:
        # the paper does not publish the exact FC-head tensorization.  A shared
        # per-satellite head is used so learned parameters do not depend on the
        # training-set Nmax.  This lets the same trained weights process a test
        # epoch with N > training Nmax without truncating observations or using
        # test data to resize/retrain the network.  Each gain column depends on
        # the local masked-LSTM token and the global attention context.
        self.gain_head = nn.Linear(
            2 * 64,
            INS_STATE_DIM,
        )
        # The paper's Eq. (7)/Fig. 8 path requires all 15 gain rows.  Data01 has
        # no direct b_a/b_g labels, so initialize only those six rows at zero to
        # prevent arbitrary feedback. Yan et al. do not publish a separate bias
        # target, so this reproduction does not fabricate one. The full 15-row
        # output is retained for Eq. (7) compatibility. This initialization is a
        # declared completion.
        with torch.no_grad():
            self.gain_head.weight[SUPERVISED_STATE_DIM:, :].zero_()
            self.gain_head.bias[SUPERVISED_STATE_DIM:].zero_()
        # Fig. 8 explicitly produces eta_k in addition to KG_k. Eq. (8) identifies
        # eta physically as residual gyro/accelerometer measurement errors. Yan
        # does not publish its target/timing; training below therefore supervises
        # its navigation-state consequence rather than a fabricated direct label.
        self.imu_error_head = nn.Linear(
            64,
            IMU_ERROR_DIM,
        )
        # Zero-output initialization prevents arbitrary IMU feedback before
        # training; the branch remains trainable through the next-state Eq. (30)
        # objective. Initialization is an explicit reproducibility completion.
        nn.init.zeros_(self.imu_error_head.weight)
        nn.init.zeros_(self.imu_error_head.bias)

    def forward(
        self,
        fixed: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        channel_mask: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> MaskedCLAOutput:
        if (
            fixed.ndim != 2
            or observations.ndim != 3
            or mask.ndim != 2
            or channel_mask.ndim != 3
        ):
            raise ValueError(
                "fixed/observations/mask/channel_mask have invalid ranks"
            )

        batch_size = fixed.shape[0]
        sequence_length = observations.shape[1]
        if sequence_length <= 0:
            raise ValueError("observations must contain at least one satellite slot")
        expected_observations = (
            batch_size,
            sequence_length,
            OBSERVATION_FEATURE_DIM,
        )

        if tuple(fixed.shape) != (
            batch_size,
            FIXED_FEATURE_DIM,
        ):
            raise ValueError(
                f"fixed must have shape {(batch_size, FIXED_FEATURE_DIM)}, "
                f"got {tuple(fixed.shape)}"
            )
        if tuple(observations.shape) != expected_observations:
            raise ValueError(
                f"observations must have shape {expected_observations}, "
                f"got {tuple(observations.shape)}"
            )
        if tuple(mask.shape) != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                f"mask must have shape {(batch_size, sequence_length)}, "
                f"got {tuple(mask.shape)}"
            )
        if tuple(channel_mask.shape) != expected_observations:
            raise ValueError(
                f"channel_mask must have shape {expected_observations}, "
                f"got {tuple(channel_mask.shape)}"
            )

        mask_bool = mask.bool()
        mask_values = mask_bool.to(dtype=fixed.dtype)

        observations = (
            observations
            * channel_mask.bool().to(dtype=observations.dtype)
        )

        token = torch.cat(
            [
                fixed.unsqueeze(1).expand(
                    -1, sequence_length, -1
                ),
                observations,
            ],
            dim=-1,
        )
        token = token * mask_values.unsqueeze(-1)

        # Fig. 8: Masked CNN -> Pooling -> Flatten.
        conv_flat = self.conv(token, mask_values)

        # Yan Eqs. (24)-(25): masked recurrent processing inside the current
        # padded X_bar[k] sequence. No h/c is carried to the next fusion epoch.
        lstm, next_recurrent_state = self.lstm(
            conv_flat,
            mask_values,
            recurrent_state,
        )
        context, attention = self.attention(
            lstm,
            mask_values,
        )

        # Shared head: one full 15-state gain column per sequence slot. Unlike a
        # flattened [15*Nmax] FC output, these weights are trained on every valid
        # satellite slot and can therefore be reused when N changes at inference.
        context_per_slot = context.unsqueeze(1).expand(-1, sequence_length, -1)
        gain_features = torch.cat([lstm, context_per_slot], dim=-1)
        learned_gain = self.gain_head(gain_features).transpose(1, 2)
        learned_gain = (
            learned_gain
            * mask_values.unsqueeze(1)
        )

        gain = learned_gain

        imu_error = self.imu_error_head(context)

        return MaskedCLAOutput(
            kalman_gain=gain,
            imu_error=imu_error,
            attention=attention,
            recurrent_state=next_recurrent_state,
        )


def fig8_state_update(
    network_output: MaskedCLAOutput,
    innovation: torch.Tensor,
) -> torch.Tensor:
    """Fig. 8 ``KG_k -> State Update`` block.

    Yan et al. place the learned Kalman gain before the state-update block and
    train the network end-to-end through the resulting state estimate.  For the
    current pseudorange-only error-state implementation, the update is

        xhat_k = K_k * innovation_k

    after the closed-loop error-state mean has been reset to zero.  Keeping this
    as an explicit operation prevents the training code from treating KG_k as a
    directly supervised label.
    """
    if network_output.kalman_gain.ndim != 3:
        raise ValueError("kalman_gain must have shape [B,state,N]")
    if innovation.ndim != 2:
        raise ValueError("innovation must have shape [B,N]")
    if network_output.kalman_gain.shape[0] != innovation.shape[0]:
        raise ValueError("gain/innovation batch dimensions do not match")
    if network_output.kalman_gain.shape[2] != innovation.shape[1]:
        raise ValueError("gain/innovation measurement dimensions do not match")
    return torch.bmm(
        network_output.kalman_gain,
        innovation.unsqueeze(-1),
    ).squeeze(-1)




def fig2_eta_forward_state_update_9(
    network_output: MaskedCLAOutput,
    eta_forward_state_sensitivity_9: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced future state effect of Fig. 2 eta_k -> Eq. (8) feedback.

    The Jacobian is used only during offline training. Online, eta_k is applied
    directly to the IMU compensation/mechanization path.
    """
    if (
        network_output.imu_error.ndim != 2
        or network_output.imu_error.shape[1] != IMU_ERROR_DIM
    ):
        raise ValueError("imu_error/eta_k must have shape [B,6]")
    if (
        eta_forward_state_sensitivity_9.ndim != 3
        or eta_forward_state_sensitivity_9.shape[1:]
        != (SUPERVISED_STATE_DIM, IMU_ERROR_DIM)
        or eta_forward_state_sensitivity_9.shape[0]
        != network_output.imu_error.shape[0]
    ):
        raise ValueError(
            "eta_forward_state_sensitivity_9 must have shape [B,9,6]"
        )

    sensitivity = eta_forward_state_sensitivity_9.to(
        dtype=network_output.imu_error.dtype,
        device=network_output.imu_error.device,
    )
    return torch.bmm(
        sensitivity,
        network_output.imu_error.unsqueeze(-1),
    ).squeeze(-1)


def fig8_post_update_error_state_9(
    estimated_error_state_15: torch.Tensor,
    true_error_state_9: torch.Tensor,
) -> torch.Tensor:
    """Direct error-state residual used by Yan et al. Eq. (30).

    Yan Eq. (7) defines a 15-state error vector, including b_a and b_g, and
    Eq. (30) uses squared state error. The attached truth exports expose only
    position, velocity, and attitude, so this helper evaluates Eq. (30) on the
    nine available coordinates without fabricating six bias labels. The learned
    update itself remains 15-dimensional. The supplied dataset does not expose
    direct accelerometer/gyro-bias truth, so no additional bias-label loss is
    fabricated.
    """
    if estimated_error_state_15.ndim != 2:
        raise ValueError("estimated_error_state_15 must have shape [B,15]")
    if true_error_state_9.ndim != 2 or true_error_state_9.shape[1] != 9:
        raise ValueError("true_error_state_9 must have shape [B,9]")
    if estimated_error_state_15.shape[0] != true_error_state_9.shape[0]:
        raise ValueError("estimated/true state batch dimensions do not match")
    estimated = estimated_error_state_15[:, :SUPERVISED_STATE_DIM].double()
    truth = true_error_state_9.double()
    return truth - estimated


# =============================================================================
# END-TO-END SIMULATION / TRAINING / ONLINE EVALUATION
# =============================================================================
# Allow direct path execution as well as `python -m ...`.





if __name__ == "__main__":

    _run_wall_start = perf_counter()

    # =========================================================================
    # 0. USER SETTINGS
    # =========================================================================
    OUTPUT_DIR = _default_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MAX_FUSION_EPOCHS = _env_optional_int("MKNET_MAX_FUSION_EPOCHS", 100)
    MAX_TEST_FUSION_EPOCHS = _env_optional_int("MKNET_MAX_TEST_FUSION_EPOCHS", 100)

    # Table III explicitly gives Adam, initial LR=0.01, Conv=24,
    # LSTM=64 x 5, and dropout=0.2. Fig. 15 displays learning curves extending
    # to roughly 500 training epochs, but the text does not publish an exact
    # stopping epoch. Therefore 500 is used only as a maximum figure-guided
    # training horizon.
    TRAINING_EPOCHS = _env_int("MKNET_TRAINING_EPOCHS", 250)
    # Yan et al. do not publish a mini-batch size. BATCH_SIZE is therefore an
    # explicit implementation completion used only for gradient accumulation.
    # The masked LSTM sequence itself is the padded X_bar[k] inside each epoch.
    BATCH_SIZE = _env_int("MKNET_BATCH_SIZE", 16)
    LEARNING_RATE = 0.01
    # Ref. [15] permits separate learning rates for its two alternating blocks,
    # but Yan et al. publish only one initial learning rate (0.01). To avoid an
    # unsupported extra hyperparameter, both alternating blocks use the same
    # paper-supported initial learning rate.
    FILTER_BLOCK_LEARNING_RATE = LEARNING_RATE
    REPRESENTATION_BLOCK_LEARNING_RATE = LEARNING_RATE

    # Eq. (32) explicitly includes gamma*||Theta||^2 but does not publish gamma.
    GAMMA_L2 = 1e-6  # explicit reproducibility completion

    # Yan et al. use one dataset for training and a distinct dataset for testing;
    # no internal validation split or early-stopping rule is published.

    # Yan et al. Fig. 2 / Sec. II-D: FDE is an ONLINE stage and the raw
    # INS-predicted innovation is tested before the learned Kalman gain.
    # FDE_DIA_ON is the single explicit ablation switch defined above.
    # Yan et al. state that Td is selected from false-alarm probability but do
    # not publish its value. Ref. [33] uses alpha=1e-3 in its quantitative DIA
    # analysis, so this value is an explicit reproducibility completion.
    FDE_SIGNIFICANCE_ALPHA = 1e-3
    FDE_STATISTICAL_SELF_CHECK = (
        _validate_ref33_statistical_core(FDE_SIGNIFICANCE_ALPHA)
        if FDE_DIA_ON
        else None
    )

    SEED = 0
    TEST_LEO_SEED = LEO_SEED + 1  # independent reproducible LEO-error realization for test
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== DIRECT GNSS/LEO/INS + MASKED KALMANNET ===")
    print("runtime:", "Kaggle" if IN_KAGGLE else "local")
    print("project root:", KAGGLE_PROJECT_ROOT)
    print("training dataset:", TRAIN_DATASET_DIR)
    print("test dataset:", TEST_DATASET_DIR)
    print("TLE directory:", LEO_TLE_DIR)
    print("output dir:", OUTPUT_DIR.resolve())
    print("device:", DEVICE)
    print(
        "FDE/DIA block:",
        "ON (paper run)" if FDE_DIA_ON else "OFF (ablation run)",
    )
    print(
        "eta_k -> Eq.(8) feedback:",
        "ON (paper path)" if ETA_FEEDBACK_ON else "OFF (diagnostic ablation)",
    )
    print(
        "classical Data02 diagnostic baseline:",
        "ON" if RUN_CLASSICAL_TEST_BASELINE else "OFF",
    )
    print(
        "NN input standardization:",
        "ON (Data01-only numerical conditioning)"
        if FEATURE_STANDARDIZATION_ON else "OFF (identity ablation)",
    )
    print(
        "NN temporal training:",
        f"fixed Data01 X/M/Y; intra-epoch masked LSTM; gradient batch={BATCH_SIZE}; "
        "no cross-fusion h/c carry",
    )
    if MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None:
        print(
            "WARNING: fusion-epoch cap active; this is a shortened diagnostic "
            "run, not the full available Data01/Data02 experiment."
        )
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("PyTorch CUDA runtime:", torch.version.cuda)
    else:
        print("GPU: not available to PyTorch")

    # =========================================================================
    # 1. LOAD RAW DATA
    # =========================================================================
    required = [
        README_XML_PATH, IMU_ERROR_MODEL_PATH, ROVE_GROUND_TRUTH_PATH, IMU_GROUND_TRUTH_PATH,
        RINEX_OBS_PATH, IMR_PATH, SP3_PATH, CLK_PATH, NAV_PATH, LEO_TLE_DIR,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    rover = load_smartpnt_metadata(README_XML_PATH, "01")
    antenna_truth = load_ie_ground_truth(ROVE_GROUND_TRUTH_PATH)
    imu_truth = load_ie_ground_truth(IMU_GROUND_TRUTH_PATH)
    imu_models = read_imu_error_models(IMU_ERROR_MODEL_PATH)
    imu_noise = imu_model_to_si(imu_models[rover.imu_type])

    # Parse only the first raw RINEX epoch to anchor IMR TOW to GPS week.  The
    # bounded/capped fusion epochs are parsed after the common time span is known.
    rinex = RINEXObservationFile.open(RINEX_OBS_PATH)
    first_rinex_epoch = next(
        rinex.iter_epochs(allowed_constellations={"G", "C"}),
        None,
    )
    if first_rinex_epoch is None:
        raise ValueError("RINEX file contains no GPS/BDS epochs")

    imr_header, imr_tow_all, imr_record_count = read_imr_tow_only(IMR_PATH)
    if imr_record_count < 2:
        raise ValueError("IMR file contains fewer than two usable samples")
    if len(antenna_truth.tow_s) < 2 or len(imu_truth.tow_s) < 2:
        raise ValueError("Ground-truth file contains fewer than two usable epochs")

    # =========================================================================
    # 2. GPST SYNCHRONIZATION
    # =========================================================================
    anchor_time = float(first_rinex_epoch.time_gpst_s)
    imr_time_all = anchor_imr_tow_to_gpst_seconds(imr_tow_all, anchor_time)

    antenna_truth_time = validate_strict_time_axis(
        "training antenna truth",
        antenna_truth.week.astype(float) * GPS_WEEK_S + antenna_truth.tow_s,
    )
    imu_truth_time = validate_strict_time_axis(
        "training IMU truth",
        imu_truth.week.astype(float) * GPS_WEEK_S + imu_truth.tow_s,
    )

    common_start = max(
        float(imr_time_all[0]),
        float(antenna_truth_time[0]),
        float(imu_truth_time[0]),
    )
    common_end = min(
        float(imr_time_all[-1]),
        float(antenna_truth_time[-1]),
        float(imu_truth_time[-1]),
    )
    start = int(np.searchsorted(imr_time_all, common_start, side="left"))
    if start >= len(imr_time_all):
        raise ValueError("Training common time span starts after the IMR file")
    usable_imu_start = float(imr_time_all[start])

    # This yields exactly the same first MAX_FUSION_EPOCHS usable epochs that the
    # previous full-file parse retained after filtering, but stops parsing once the
    # requested subset has been collected.
    gnss_epochs = tuple(
        rinex.iter_epochs(
            allowed_constellations={"G", "C"},
            start_time_gpst_s=usable_imu_start,
            end_time_gpst_s=common_end,
            max_epochs=MAX_FUSION_EPOCHS,
            require_measurements=True,
        )
    )
    if not gnss_epochs:
        raise ValueError("RINEX file contains no usable GPS/BDS pseudorange epochs")

    fusion_time = validate_strict_time_axis(
        "training fusion",
        np.asarray([epoch.time_gpst_s for epoch in gnss_epochs], dtype=float),
    )
    if len(fusion_time) < 3:
        raise ValueError(
            "Fewer than three synchronized GNSS fusion epochs remain; "
            "cannot build lagged Masked KalmanNet features"
        )

    # Read and scale only the exact contiguous IMU samples that can affect the
    # selected fusion epochs.  No IMU sample inside that interval is dropped.
    common_stop = min(
        int(np.searchsorted(imr_time_all, common_end, side="left")) + 1,
        len(imr_time_all),
    )
    requested_stop = min(
        int(np.searchsorted(imr_time_all, fusion_time[-1], side="left")) + 1,
        common_stop,
    )
    imr = read_imr(
        IMR_PATH,
        scaling_mode="cpp_exact",
        start_record=start,
        stop_record=requested_stop,
    )
    imr_time = imr_time_all[start:requested_stop].copy()
    del imr_tow_all, imr_time_all

    if len(imr.tow_s) < 2:
        raise ValueError("Selected training IMR window contains fewer than two samples")

    print("IMU records in file:", imr_record_count)
    print("IMU samples used:", len(imr.tow_s))
    print("RINEX fusion epochs selected:", len(gnss_epochs))

    # ROVE truth is the GNSS-antenna reference point; ISA-100C truth is the IMU
    # reference point. Both exports provide position, velocity and attitude, but
    # neither export contains accelerometer/gyro bias truth.
    query_time = np.concatenate(([imr_time[0]], fusion_time))
    antenna_position, _, _, _, _ = interpolate_ground_truth(
        antenna_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S
    )
    imu_position, imu_velocity, imu_heading, imu_pitch, imu_roll = interpolate_ground_truth(
        imu_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S
    )

    initial_imu_truth_position = imu_position[0]
    initial_imu_truth_velocity = imu_velocity[0]
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]

    print("Synchronized fusion epochs:", len(gnss_epochs))

    # =========================================================================
    # 3. INITIAL 15-STATE INS
    # =========================================================================
    C_b_e = body_to_ecef_from_ie_hpr(
        initial_imu_truth_position, imu_heading[0], imu_pitch[0], imu_roll[0], rover.mounting_xyz_deg
    )
    lever_arm_b_m = transform_lever_arm_vehicle_to_body(
        rover.lever_arm_vehicle_m, *rover.mounting_xyz_deg
    )

    # Dataset check: ROVE truth must equal ISA-100C IMU truth plus the documented
    # IMU->GNSS lever arm. This also verifies that the two truth files were assigned
    # to the correct reference points.
    initial_antenna_from_imu = initial_imu_truth_position + C_b_e @ lever_arm_b_m
    truth_reference_error_m = np.linalg.norm(antenna_position[0] - initial_antenna_from_imu)
    if truth_reference_error_m > 0.05:
        raise ValueError(
            f"ROVE/ISA-100C ground-truth reference points are inconsistent with the lever arm: "
            f"{truth_reference_error_m:.3f} m"
        )

    fusion_truth_body_to_ecef = np.stack([
        body_to_ecef_from_ie_hpr(p, h, pt, r, rover.mounting_xyz_deg)
        for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])
    ])

    initial_nav = NavigationState(
        initial_imu_truth_position.copy(),
        initial_imu_truth_velocity.copy(),
        C_b_e,
        np.zeros(3),
        np.zeros(3),
    )

    P0 = initial_covariance_from_imu_model(imu_noise)
    Qc = continuous_process_covariance_from_imu_model(imu_noise)

    # =========================================================================
    # 4. GNSS + TLE/SGP4 LEO MODELS
    # =========================================================================
    ionosphere_coefficients = None
    if USE_IONOSPHERE:
        iono_header = read_rinex_navigation_header(NAV_PATH)
        ionosphere_coefficients = {
            "G": (np.asarray(iono_header["GPSA"]), np.asarray(iono_header["GPSB"])),
            "C": (np.asarray(iono_header["BDSA"]), np.asarray(iono_header["BDSB"])),
        }

    gnss_preprocessor = GNSSPreprocessor(
        SP3Orbit(SP3_PATH),
        RINEXClock(CLK_PATH),
        min_elevation_deg=MIN_GNSS_ELEVATION_DEG,
        use_ionosphere=USE_IONOSPHERE,
        use_troposphere=USE_TROPOSPHERE,
        broadcast_ionosphere_coefficients=ionosphere_coefficients,
    )
    tle_provider = TLESGP4Provider(
        LEO_TLE_DIR,
        max_tle_age_days=TLE_MAX_AGE_DAYS,
        allow_degraded_eop=TLE_ALLOW_DEGRADED_EOP,
        allow_non_tle_files=TLE_ALLOW_NON_TLE_FILES,
    )
    leo_klobuchar = (
        None
        if ionosphere_coefficients is None
        else ionosphere_coefficients["G"]
    )
    leo_simulator = LEODownlinkSimulator(
        tle_provider,
        leo_klobuchar,
        seed=LEO_SEED,
        tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M,
        tx_max_iterations=LEO_TX_MAX_ITERATIONS,
        minimum_elevation_deg=LEO_MIN_ELEVATION_DEG,
        prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG,
        use_ionosphere=USE_IONOSPHERE,
        use_troposphere=USE_TROPOSPHERE,
    )
    print("TLE satellites:", len(tle_provider.satellite_ids))
    print(
        "LEO residual model: v6 variance retained -- Ref.[35] ionosphere/"
        "troposphere/MP Eq.(18), independent Gaussian; URA/receiver-noise omitted"
    )

    # =========================================================================
    # 5. CLASSICAL TC PASS -> TRAINING HISTORY
    # =========================================================================
    nav = initial_nav.copy()
    P = P0.copy()
    history_rows = []

    # Propagation pieces since the previous usable fusion epoch.  A current
    # eta_k affects the following interval in the online Fig. 2 feedback path;
    # retaining these exact pieces lets the offline completion supervise eta_k
    # causally without fabricating a direct IMU-error label.
    interval_segments_since_previous_usable_fusion = []

    # Current corrected IMU values are needed by the feature vector.
    last_gyro, last_accel = compensate_imu(
        imr.angular_rate_body_radps[0],
        imr.acceleration_body_mps2[0],
        nav.gyroscope_bias_body_radps,
        nav.accelerometer_bias_body_mps2,
    )

    training_timeline = build_exact_fusion_timeline(
        imr_time,
        fusion_time,
        through_last_fusion=True,
    )

    for event in training_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(
                imr.angular_rate_body_radps[imu_index],
                imr.acceleration_body_mps2[imu_index],
                nav.gyroscope_bias_body_radps,
                nav.accelerometer_bias_body_mps2,
            )
            nav = mechanize_ecef(nav, last_gyro, last_accel, dt)
            interval_segments_since_previous_usable_fusion.append(
                (imu_index, dt)
            )
            F_error = build_error_state_dynamics(nav, last_accel)
            Phi, Qd = discretize_process_noise_van_loan(F_error, Qc, dt)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue

        fusion_index = event.fusion_index
        t = event.time_gpst_s
        epoch = gnss_epochs[fusion_index]
        if abs(epoch.time_gpst_s - t) > 1e-9:
            raise RuntimeError("training fusion timeline/epoch timestamp mismatch")

        # -------- GNSS + LEO pseudoranges --------
        gnss_measurements = gnss_preprocessor.prepare_epoch(
            epoch, nav, lever_arm_b_m
        )
        # Truth is used only as the physical receiver trajectory that generates
        # the simulated LEO observation. It is not passed to the TC predictor.
        leo_measurements = leo_simulator.simulate_epoch(
            t, fusion_antenna_truth_position[fusion_index]
        )
        measurements = retain_clock_observable_measurements(
            tuple(gnss_measurements) + tuple(leo_measurements)
        )

        if not measurements:
            continue

        prior_nav = nav.copy()

        measurement_model = build_measurement_model(
            nav, measurements, lever_arm_b_m
        )

        # Closed-loop 15-state error-state convention: after feedback/reset,
        # the propagated error-state mean is zero.
        error_state_pred = np.zeros(INS_STATE_DIM)
        correction, P, _ = kalman_measurement_update(
            P,
            measurement_model.innovation,
            measurement_model.H,
            measurement_model.R,
        )
        error_state_post = error_state_pred + correction
        nav = inject_error_state(nav, correction)

        posterior_position = gnss_antenna_position(nav, lever_arm_b_m)
        posterior_residual = build_innovation_only(
            nav, measurements, lever_arm_b_m
        )
        target_state_9 = np.concatenate([
            fusion_imu_truth_position[fusion_index] - prior_nav.position_ecef_m,
            fusion_imu_truth_velocity[fusion_index] - prior_nav.velocity_ecef_mps,
            attitude_error_state_target(
                prior_nav.body_to_ecef_dcm,
                fusion_truth_body_to_ecef[fusion_index],
            ),
        ])

        history_rows.append({
            "time": t,
            "sat_ids": measurement_model.sat_ids,
            "innovation": measurement_model.innovation.copy(),
            "residual": posterior_residual.copy(),
            "H": measurement_model.H.copy(),
            "x_pred": error_state_pred.copy(),
            "x_post": error_state_post.copy(),
            "accel": last_accel.copy(),
            "gyro": last_gyro.copy(),
            "target_state_9": target_state_9.copy(),
            "fusion_index": int(fusion_index),
            "preceding_interval_segments": tuple(
                interval_segments_since_previous_usable_fusion
            ),
        })

        # A new eta_k would become available at this usable fusion epoch.
        interval_segments_since_previous_usable_fusion = []

    print("Usable classical fusion rows:", len(history_rows))
    if len(history_rows) < 4:
        raise ValueError(
            "Classical TC pass produced fewer than four usable measurement rows; "
            "check GNSS products, TLE coverage, masks, and time synchronization"
        )

    # =========================================================================
    # 6. PAPER Eqs. (10)-(21): DIRECT CAUSAL FEATURES + PADDING/MASKS
    # =========================================================================
    # The first usable fusion row supplies the lagged context required by
    # Eqs. (11)-(14). No future-epoch target is required by Yan Eq. (30)/(32),
    # so every subsequent usable Data01 row participates in training.
    nmax = max(len(row["sat_ids"]) for row in history_rows[1:])
    feature_time = []
    fixed = []
    observations = []
    channel_masks = []
    innovation_padded = []
    target_states_9 = []

    # First classical row supplies lagged Eq. (11)-(14) context only.
    for k in range(1, len(history_rows)):
        current = history_rows[k]
        previous = history_rows[k - 1]

        delta_accel = current["accel"] - previous["accel"]
        delta_gyro = current["gyro"] - previous["gyro"]
        # Paper Eq. (12): posterior-minus-prior state innovation. The current
        # posterior cannot be used as an input to the same forward pass without
        # circular information leakage, so the previous completed fusion state is
        # the declared causal timing completion documented at the top of this file.
        previous_state_innovation = previous["x_post"] - previous["x_pred"]
        previous_state_residual = (
            np.zeros(INS_STATE_DIM)
            if k == 1
            else previous["x_post"] - history_rows[k - 2]["x_post"]
        )
        fixed_k = np.concatenate([
            delta_accel,
            delta_gyro,
            previous_state_residual,
            previous_state_innovation,
        ])  # [36]

        n = len(current["sat_ids"])
        innovation_k = np.zeros(nmax)
        innovation_k[:n] = current["innovation"]
        current_mask = np.zeros(nmax, dtype=bool)
        current_mask[:n] = True

        previous_residual_by_sat = dict(zip(previous["sat_ids"], previous["residual"]))
        residual_k = np.zeros(nmax)
        residual_mask = np.zeros(nmax, dtype=bool)
        for slot, sat_id in enumerate(current["sat_ids"]):
            if sat_id in previous_residual_by_sat:
                residual_k[slot] = previous_residual_by_sat[sat_id]
                residual_mask[slot] = True

        feature_time.append(current["time"])
        fixed.append(fixed_k)
        observations.append(np.stack([residual_k, innovation_k], axis=1))
        channel_masks.append(np.stack([residual_mask, current_mask], axis=1))
        innovation_padded.append(innovation_k)
        target_states_9.append(current["target_state_9"])

    feature_time = np.asarray(feature_time)
    fixed = np.stack(fixed)
    observations = np.stack(observations)
    channel_masks = np.stack(channel_masks)
    innovation_padded = np.stack(innovation_padded)
    target_states_9 = np.stack(target_states_9)

    def _truth_nav_at_fusion(fusion_index: int) -> NavigationState:
        """Postprocessed Data01 truth state used only for teacher-forced eta training."""
        return NavigationState(
            fusion_imu_truth_position[fusion_index].copy(),
            fusion_imu_truth_velocity[fusion_index].copy(),
            fusion_truth_body_to_ecef[fusion_index].copy(),
            np.zeros(3),
            np.zeros(3),
        )

    # Yan Fig. 8 exposes eta_k and Fig. 2 feeds it to the Eq. (8) IMU-error
    # compensation path, but the paper publishes neither a direct eta target nor
    # a separate eta loss.  To avoid adding such an unpublished loss, supervise
    # eta_k only through its navigation-state consequence at the NEXT usable
    # fusion epoch.  This is a declared timing completion; the criterion itself
    # remains the same Eq. (30)-type state error used by the paper.
    eta_forward_state_sensitivities_9 = np.zeros(
        (len(feature_time), SUPERVISED_STATE_DIM, IMU_ERROR_DIM),
        dtype=float,
    )
    eta_forward_target_states_9 = np.zeros(
        (len(feature_time), SUPERVISED_STATE_DIM),
        dtype=float,
    )
    eta_forward_valid = np.zeros(len(feature_time), dtype=bool)
    eta_future_end_time = np.full(len(feature_time), np.nan, dtype=float)

    # feature sample s corresponds to history row k=s+1.
    for s in range(len(feature_time) - 1):
        k = s + 1
        current = history_rows[k]
        future = history_rows[k + 1]
        current_fusion_index = int(current["fusion_index"])
        future_fusion_index = int(future["fusion_index"])
        interval_segments = future["preceding_interval_segments"]

        truth_start_nav = _truth_nav_at_fusion(current_fusion_index)
        truth_future_nav = _truth_nav_at_fusion(future_fusion_index)

        # Baseline future state from measured IMU with eta=0.
        base_future_nav = _replay_imu_interval_with_eta(
            truth_start_nav,
            interval_segments,
            imr,
            np.zeros(IMU_ERROR_DIM),
        )
        eta_forward_target_states_9[s] = _navigation_delta_9(
            base_future_nav,
            truth_future_nav,
        )
        eta_forward_state_sensitivities_9[s] = (
            eta_interval_state_sensitivity_9(
                truth_start_nav,
                interval_segments,
                imr,
                reference_end_nav=base_future_nav,
            )
        )
        eta_forward_valid[s] = True
        eta_future_end_time[s] = float(future["time"])

    # features.py-style structural checks: fail early rather than silently
    # accepting malformed padding, masks, time ordering, or non-finite values.
    expected_t = len(feature_time)
    if fixed.shape != (expected_t, FIXED_FEATURE_DIM):
        raise ValueError(f"fixed feature shape must be {(expected_t, FIXED_FEATURE_DIM)}, got {fixed.shape}")
    if innovation_padded.shape != (expected_t, nmax):
        raise ValueError("innovation_padded must have shape [T,Nmax]")
    if eta_forward_state_sensitivities_9.shape != (
        expected_t, SUPERVISED_STATE_DIM, IMU_ERROR_DIM
    ):
        raise ValueError(
            "eta_forward_state_sensitivities_9 must have shape [T,9,6]"
        )
    if eta_forward_target_states_9.shape != (
        expected_t, SUPERVISED_STATE_DIM
    ):
        raise ValueError("eta_forward_target_states_9 must have shape [T,9]")
    if eta_forward_valid.shape != (expected_t,):
        raise ValueError("eta_forward_valid must have shape [T]")
    if np.any(
        eta_forward_valid
        & (
            ~np.isfinite(eta_future_end_time)
            | (eta_future_end_time <= feature_time)
        )
    ):
        raise ValueError("valid eta_k supervision must end strictly after epoch k")
    if not np.all(np.isfinite(eta_forward_state_sensitivities_9)):
        raise ValueError(
            "eta_forward_state_sensitivities_9 contains non-finite values"
        )
    if not np.all(np.isfinite(eta_forward_target_states_9)):
        raise ValueError("eta_forward_target_states_9 contains non-finite values")
    if channel_masks.shape != (expected_t, nmax, OBSERVATION_FEATURE_DIM):
        raise ValueError("channel_masks must have shape [T,Nmax,2]")
    if expected_t > 1 and not np.all(np.diff(feature_time) > 0.0):
        raise ValueError("feature times must be strictly increasing")
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(innovation_padded)) or not np.all(np.isfinite(observations)):
        raise ValueError("feature arrays contain non-finite values")
    if np.any(innovation_padded[~channel_masks[:, :, 1]] != 0.0):
        raise ValueError("masked innovation padding must be exactly zero")
    satellite_masks = channel_masks[:, :, 1]

    # Diagnostic-only reference envelope for Data01 raw features.  These values
    # are never fed back into the network or used to rescale Data02; they only
    # quantify when the recursive online trajectory leaves the feature range
    # encountered in the fixed offline training set.
    train_fixed_min = np.min(fixed, axis=0)
    train_fixed_max = np.max(fixed, axis=0)
    train_fixed_abs_max = np.maximum(np.max(np.abs(fixed), axis=0), 1e-12)
    train_obs_min = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_abs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    for diagnostic_channel in range(OBSERVATION_FEATURE_DIM):
        valid_values = observations[:, :, diagnostic_channel][
            channel_masks[:, :, diagnostic_channel]
        ]
        if valid_values.size == 0:
            train_obs_min[diagnostic_channel] = -np.inf
            train_obs_max[diagnostic_channel] = np.inf
            train_obs_abs_max[diagnostic_channel] = 1.0
        else:
            train_obs_min[diagnostic_channel] = float(np.min(valid_values))
            train_obs_max[diagnostic_channel] = float(np.max(valid_values))
            train_obs_abs_max[diagnostic_channel] = max(
                float(np.max(np.abs(valid_values))), 1e-12
            )


    # =========================================================================
    # =========================================================================
    # 7. YAN Eqs. (18)-(20): FULL DATA01 OFFLINE TRAINING SET + NORMALIZATION
    # =========================================================================
    # Yan et al. designate one complete real-world dataset for training and a
    # distinct dataset for testing. They do not publish an internal validation
    # split or early-stopping rule. Therefore every usable Data01 feature sample
    # participates in training. Normalization is fitted from Data01 only; Data02
    # remains completely independent.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Yan et al. Eqs. (15)-(21) define the physical feature information,
    # zero-padding, and mask but do not publish feature scaling. The recursive
    # Data01/Data02 diagnostics showed severe raw feature-scale departure, while
    # the classical navigation branch remained stable. Apply a fixed affine
    # numerical conditioning fitted ONLY on Data01. This does not change any
    # feature definition, measurement, state target, FDE statistic, INS equation,
    # or K*innovation state update; only the neural-network input coordinates are
    # reparameterized. Masked padding is restored to exact zero after scaling.
    if FEATURE_STANDARDIZATION_ON:
        fixed_mean = fixed.mean(axis=0)
        fixed_std = fixed.std(axis=0)
        fixed_std[fixed_std < 1e-8] = 1.0

        obs_mean = np.zeros(OBSERVATION_FEATURE_DIM, dtype=float)
        obs_std = np.ones(OBSERVATION_FEATURE_DIM, dtype=float)
        for channel in range(OBSERVATION_FEATURE_DIM):
            values = observations[:, :, channel][
                channel_masks[:, :, channel]
            ]
            if values.size:
                obs_mean[channel] = float(values.mean())
                std = float(values.std())
                obs_std[channel] = 1.0 if std < 1e-8 else std

        fixed_normalized = (fixed - fixed_mean) / fixed_std
        observations_normalized = (
            observations - obs_mean.reshape(1, 1, OBSERVATION_FEATURE_DIM)
        ) / obs_std.reshape(1, 1, OBSERVATION_FEATURE_DIM)
        observations_normalized = np.where(
            channel_masks, observations_normalized, 0.0
        )
        normalizer_source = (
            "Data01_only_zscore_numerical_conditioning_"
            "paper_feature_information_unchanged"
        )
    else:
        fixed_mean = np.zeros(FIXED_FEATURE_DIM, dtype=float)
        fixed_std = np.ones(FIXED_FEATURE_DIM, dtype=float)
        obs_mean = np.zeros(OBSERVATION_FEATURE_DIM, dtype=float)
        obs_std = np.ones(OBSERVATION_FEATURE_DIM, dtype=float)
        fixed_normalized = fixed.copy()
        observations_normalized = np.where(
            channel_masks, observations, 0.0
        )
        normalizer_source = "identity_ablation"

    dataset = TensorDataset(
        torch.tensor(fixed_normalized, dtype=torch.float32),
        torch.tensor(observations_normalized, dtype=torch.float32),
        torch.tensor(satellite_masks, dtype=torch.bool),
        torch.tensor(channel_masks, dtype=torch.bool),
        torch.tensor(innovation_padded, dtype=torch.float32),
        torch.tensor(target_states_9, dtype=torch.float64),
        torch.tensor(eta_forward_state_sensitivities_9, dtype=torch.float32),
        torch.tensor(eta_forward_target_states_9, dtype=torch.float64),
        torch.tensor(eta_forward_valid, dtype=torch.bool),
    )

    use_pinned_memory = DEVICE == "cuda"
    # Keep Yan's fixed X/M/Y samples unchanged. Each X_bar[k] is one masked
    # intra-epoch LSTM sequence. BATCH_SIZE only controls gradient accumulation.

    # Retained for shared online feature construction and checkpoint metadata.
    normalizer = (fixed_mean, fixed_std, obs_mean, obs_std)

    np.savez(
        OUTPUT_DIR / "normalizer.npz",
        fixed_mean=fixed_mean,
        fixed_std=fixed_std,
        obs_mean=obs_mean,
        obs_std=obs_std,
        source=np.asarray(normalizer_source),
        feature_standardization_on=np.asarray(FEATURE_STANDARDIZATION_ON),
    )

    valid_eta_future_target = eta_forward_target_states_9[eta_forward_valid]
    eta_future_state_target_rms = (
        np.sqrt(np.mean(valid_eta_future_target**2, axis=0))
        if len(valid_eta_future_target)
        else np.zeros(SUPERVISED_STATE_DIM)
    )
    print(f"Data01 training samples={len(dataset)}, Nmax={nmax}")
    print(
        "NN input standardization:",
        "ON (Data01-only affine conditioning)"
        if FEATURE_STANDARDIZATION_ON else "OFF (identity ablation)",
    )
    print(
        "eta_k future-state target RMS [dp(3), dv(3), dtheta(3)]:",
        eta_future_state_target_rms,
    )

    # 8. FIG. 8 MASKED CLA TRAINING + EQ. (30)/(32)
    # =========================================================================
    # Paper-supported structure:
    #   - Eqs. (18)-(20): fixed offline labeled training data.
    #   - Fig. 8: Masked CNN -> Masked LSTM/Attention -> FC -> learned KG.
    #   - KG_k enters the State Update path; Eq. (30) supervises the resulting
    #     state estimate, so no ground-truth Kalman gain is required.
    #   - Eq. (32): mean prediction error plus gamma*||Theta||^2.
    #   - Table III: Adam with initial learning rate 0.01.
    #
    # Fig. 8 also depicts eta_k / IMU-error estimation and Fig. 2 feeds it to
    # Eq. (8) error compensation. Yan et al. do not publish a direct eta label or
    # loss. Therefore eta_k is trained through its NEXT-epoch navigation-state
    # consequence, preserving the published Eq. (30)/(32) state-error criterion
    # instead of appending an unrelated eta-MSE term.
    #
    # Ref. [15] Algorithm 2 alternates filter theta and representation psi using
    # the SAME final state-estimation objective. Yan et al. do not publish the
    # exact Masked-CLA parameter partition. The closest decomposable mapping is:
    #   psi   : Masked CNN + Masked LSTM + Masked Attention
    #   theta : KG FC head + eta_k FC head
    # This partition is explicitly a paper-guided completion. Ref. [15]'s
    # separate encoder-only warm start is not copied because Yan already publishes
    # its own fixed X/M/Y construction and no intermediate encoder target.
    # -------------------------------------------------------------------------
    model = MaskedCLA(nmax=nmax, dropout=0.2).to(DEVICE)
    print(
        "Masked CLA CNN:",
        "Conv1D 24x3/stride1 + masked same-length max-pool3 + per-slot flatten",
    )
    print(
        "Masked LSTM:",
        "64 units x 5 layers, dropout=0.2; packed masked sequence inside each fusion epoch",
    )
    print(
        "LSTM cross-fusion state carry:",
        "OFF; masked LSTM resets at every fusion epoch and runs inside X_bar[k]",
    )
    print(
        "Kalman gain output:",
        "full 15-state Eq.(7) rows; b_a/b_g rows zero-initialized; no fabricated bias labels",
    )

    representation_modules = (
        model.conv,
        model.lstm,
        model.attention,
    )
    filter_modules = (
        model.gain_head,
        model.imu_error_head,
    )

    representation_parameters = [
        parameter for module in representation_modules for parameter in module.parameters()
    ]
    filter_parameters = [
        parameter for module in filter_modules for parameter in module.parameters()
    ]
    if {id(p) for p in representation_parameters} & {id(p) for p in filter_parameters}:
        raise RuntimeError("alternating parameter blocks overlap")
    if {id(p) for p in representation_parameters + filter_parameters} != {
        id(p) for p in model.parameters()
    }:
        raise RuntimeError("alternating blocks must cover the entire MaskedCLA network")

    filter_optimizer = torch.optim.Adam(
        filter_parameters, lr=FILTER_BLOCK_LEARNING_RATE
    )
    representation_optimizer = torch.optim.Adam(
        representation_parameters, lr=REPRESENTATION_BLOCK_LEARNING_RATE
    )
    trainable_eq32_parameters = representation_parameters + filter_parameters
    training_history = []

    def _state_prediction_loss(predicted_update, true_error_state_9):
        if predicted_update.ndim != 2:
            raise ValueError("predicted_update must have shape [B,9] or [B,15]")
        if predicted_update.shape[1] == SUPERVISED_STATE_DIM:
            predicted_update = F.pad(
                predicted_update,
                (0, INS_STATE_DIM - SUPERVISED_STATE_DIM),
            )
        elif predicted_update.shape[1] != INS_STATE_DIM:
            raise ValueError("predicted_update must have 9 or 15 state columns")

        # Yan Eq. (30) is a direct squared Euclidean error in the declared state
        # coordinates. This dataset directly labels only the first nine Eq. (7)
        # coordinates. The remaining gain rows stay in the 15-state update and
        # have no fabricated direct bias labels in this reproduction.
        residual_9 = fig8_post_update_error_state_9(
            predicted_update,
            true_error_state_9,
        )
        per_sample_squared_norm = torch.sum(residual_9**2, dim=1)
        state_loss = torch.mean(per_sample_squared_norm)
        position_rmse = torch.sqrt(
            torch.mean(torch.sum(residual_9[:, :3] ** 2, dim=1))
        )
        return state_loss, position_rmse

    def _regularized_eq32_objective(
        network_output,
        kg_state_update,
        current_target_9,
        eta_forward_sensitivity_9,
        eta_future_target_9,
        eta_valid,
    ):
        """Published Eq. (30)/(32) state criterion for KG and eta_k paths.

        Current epoch:
            KG_k * innovation_k -> current navigation state error.

        Following interval (declared eta timing completion):
            eta_k -> Eq. (8) -> INS mechanization -> next navigation state error.

        Yan et al. do not publish a direct eta target/loss, so no eta-MSE is
        appended. Both supervised terms are navigation-state instances of the
        same Eq. (30) squared state error. The finite-record final sample simply
        lacks the following-interval eta term.
        """
        current_eq30_loss, current_position_rmse = _state_prediction_loss(
            kg_state_update, current_target_9
        )
        batch_count = int(current_target_9.shape[0])

        eta_state_update_9 = fig2_eta_forward_state_update_9(
            network_output,
            eta_forward_sensitivity_9,
        )

        valid_count = int(torch.count_nonzero(eta_valid).item())
        if valid_count:
            future_eq30_loss, future_position_rmse = _state_prediction_loss(
                eta_state_update_9[eta_valid],
                eta_future_target_9[eta_valid],
            )
            # Eq. (32) is a mean over supervised state instants. Weight by the
            # number of current/future instants rather than introducing an
            # unpublished KG-vs-eta branch coefficient.
            temporal_state_loss = (
                current_eq30_loss * batch_count
                + future_eq30_loss * valid_count
            ) / float(batch_count + valid_count)
        else:
            future_eq30_loss = current_eq30_loss.new_zeros(())
            future_position_rmse = current_position_rmse.new_zeros(())
            temporal_state_loss = current_eq30_loss

        if GAMMA_L2:
            l2_all = sum(
                torch.sum(parameter * parameter)
                for parameter in trainable_eq32_parameters
            )
            objective = temporal_state_loss + GAMMA_L2 * l2_all
        else:
            objective = temporal_state_loss

        return (
            objective,
            temporal_state_loss,
            current_eq30_loss,
            future_eq30_loss,
            current_position_rmse,
            future_position_rmse,
        )


    def _set_alternating_phase(phase: str) -> None:
        """Freeze one block and train the other, following Ref. [15] Algorithm 2."""
        if phase not in {"filter", "representation"}:
            raise ValueError("phase must be 'filter' or 'representation'")

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model.train()

        if phase == "filter":
            for module in representation_modules:
                module.eval()
            for module in filter_modules:
                module.train()
            for parameter in filter_parameters:
                parameter.requires_grad_(True)
        else:
            for module in representation_modules:
                module.train()
            for module in filter_modules:
                module.eval()
            for parameter in representation_parameters:
                parameter.requires_grad_(True)


    def _sample_on_device(sample_index: int):
        return [
            tensor.unsqueeze(0).to(
                DEVICE,
                non_blocking=use_pinned_memory,
            )
            for tensor in dataset[sample_index]
        ]

    def _run_training_phase(
        phase: str,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ):
        """One Ref.[15]-style alternating pass over Yan's fixed X/M/Y data.

        Each sample X_bar[k] is one masked LSTM sequence. There is no hidden-state
        recurrence between fusion epochs. BATCH_SIZE only controls how many
        per-epoch objectives are averaged before an optimizer step.
        """
        _set_alternating_phase(phase)

        objective_sum = 0.0
        temporal_state_sum = 0.0
        current_eq30_sum = 0.0
        future_eq30_weighted_sum = 0.0
        current_position_rmse_squared_sum = 0.0
        future_position_rmse_squared_sum = 0.0
        future_count = 0
        count = 0

        chunk_objectives = []
        optimizer.zero_grad(set_to_none=True)

        for sample_index in range(len(dataset)):
            (
                fixed_b,
                obs_b,
                mask_b,
                channel_b,
                innovation_b,
                target_b,
                eta_sensitivity_b,
                eta_future_target_b,
                eta_valid_b,
            ) = _sample_on_device(sample_index)

            output = model(
                fixed_b,
                obs_b,
                mask_b,
                channel_b,
                recurrent_state=None,
            )

            kg_state_update = fig8_state_update(output, innovation_b)
            (
                objective,
                temporal_state_loss,
                current_eq30_loss,
                future_eq30_loss,
                current_position_rmse_b,
                future_position_rmse_b,
            ) = _regularized_eq32_objective(
                output,
                kg_state_update,
                target_b,
                eta_sensitivity_b,
                eta_future_target_b,
                eta_valid_b,
            )

            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"non-finite {phase} alternating objective at epoch {epoch}, "
                    f"sample {sample_index}"
                )

            chunk_objectives.append(objective)

            valid_future_now = int(torch.count_nonzero(eta_valid_b).item())
            objective_sum += float(objective.detach().cpu())
            temporal_state_sum += float(temporal_state_loss.detach().cpu())
            current_eq30_sum += float(current_eq30_loss.detach().cpu())
            current_position_rmse_squared_sum += (
                float(current_position_rmse_b.detach().cpu()) ** 2
            )
            if valid_future_now:
                future_eq30_weighted_sum += (
                    float(future_eq30_loss.detach().cpu()) * valid_future_now
                )
                future_position_rmse_squared_sum += (
                    float(future_position_rmse_b.detach().cpu()) ** 2
                    * valid_future_now
                )
                future_count += valid_future_now
            count += 1

            chunk_end = (
                len(chunk_objectives) >= BATCH_SIZE
                or sample_index == len(dataset) - 1
            )
            if chunk_end:
                chunk_objective = torch.stack(chunk_objectives).mean()
                if not torch.isfinite(chunk_objective):
                    raise FloatingPointError(
                        f"non-finite {phase} chunk objective at epoch {epoch}"
                    )
                chunk_objective.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                chunk_objectives.clear()

        return {
            "objective": objective_sum / count,
            "temporal_state": temporal_state_sum / count,
            "eq30": current_eq30_sum / count,
            "eta_future_eq30": (
                future_eq30_weighted_sum / future_count
                if future_count else 0.0
            ),
            "position_rmse_m": math.sqrt(
                current_position_rmse_squared_sum / count
            ),
            "eta_future_position_rmse_m": (
                math.sqrt(future_position_rmse_squared_sum / future_count)
                if future_count else 0.0
            ),
        }

    def _evaluate_state_objective():
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.eval()

        objective_sum = 0.0
        temporal_state_sum = 0.0
        current_eq30_sum = 0.0
        future_eq30_weighted_sum = 0.0
        current_position_rmse_squared_sum = 0.0
        future_position_rmse_squared_sum = 0.0
        future_count = 0
        count = 0
        with torch.inference_mode():
            for sample_index in range(len(dataset)):
                (
                    fixed_b,
                    obs_b,
                    mask_b,
                    channel_b,
                    innovation_b,
                    target_b,
                    eta_sensitivity_b,
                    eta_future_target_b,
                    eta_valid_b,
                ) = _sample_on_device(sample_index)

                output = model(
                    fixed_b,
                    obs_b,
                    mask_b,
                    channel_b,
                    recurrent_state=None,
                )
                kg_state_update = fig8_state_update(output, innovation_b)
                (
                    objective,
                    temporal_state_loss,
                    current_eq30_loss,
                    future_eq30_loss,
                    current_position_rmse_b,
                    future_position_rmse_b,
                ) = _regularized_eq32_objective(
                    output,
                    kg_state_update,
                    target_b,
                    eta_sensitivity_b,
                    eta_future_target_b,
                    eta_valid_b,
                )

                if (
                    not torch.isfinite(objective)
                    or not torch.isfinite(temporal_state_loss)
                    or not torch.isfinite(current_eq30_loss)
                    or not torch.isfinite(future_eq30_loss)
                ):
                    raise FloatingPointError(
                        "non-finite Data01 training objective"
                    )

                valid_future_now = int(torch.count_nonzero(eta_valid_b).item())
                objective_sum += float(objective.detach().cpu())
                temporal_state_sum += float(temporal_state_loss.detach().cpu())
                current_eq30_sum += float(current_eq30_loss.detach().cpu())
                current_position_rmse_squared_sum += (
                    float(current_position_rmse_b.detach().cpu()) ** 2
                )
                if valid_future_now:
                    future_eq30_weighted_sum += (
                        float(future_eq30_loss.detach().cpu()) * valid_future_now
                    )
                    future_position_rmse_squared_sum += (
                        float(future_position_rmse_b.detach().cpu()) ** 2
                        * valid_future_now
                    )
                    future_count += valid_future_now
                count += 1

        return {
            "objective": objective_sum / count,
            "temporal_state": temporal_state_sum / count,
            "eq30": current_eq30_sum / count,
            "eta_future_eq30": (
                future_eq30_weighted_sum / future_count
                if future_count else 0.0
            ),
            "position_rmse_m": math.sqrt(
                current_position_rmse_squared_sum / count
            ),
            "eta_future_position_rmse_m": (
                math.sqrt(future_position_rmse_squared_sum / future_count)
                if future_count else 0.0
            ),
        }

    print(
        "\n=== YAN EQ.30/EQ.32 + REF.[15] ALTERNATING OPTIMIZATION "
        "ON FIXED OFFLINE DATA01 (Eqs. 18-20; INTRA-EPOCH MASKED LSTM) ==="
    )

    final_training_metrics = None
    epochs_ran = 0
    for epoch in range(1, TRAINING_EPOCHS + 1):
        epochs_ran = epoch

        # Ref. [15] Algorithm 2 ordering: theta first, then psi. Yan's fixed
        # Eqs. (18)-(20) Data01 dataset is kept unchanged. Each X_bar[k] is an
        # independent masked LSTM sequence; no network-generated feature recursion
        # or unpublished cross-fusion hidden-state carry is introduced.
        filter_phase = _run_training_phase(
            "filter", filter_optimizer, epoch
        )
        representation_phase = _run_training_phase(
            "representation", representation_optimizer, epoch
        )

        final_training_metrics = _evaluate_state_objective()
        current_filter_lr = float(filter_optimizer.param_groups[0]["lr"])
        current_representation_lr = float(
            representation_optimizer.param_groups[0]["lr"]
        )
        training_history.append({
            "stage": "yan_fixed_offline_Data01_ref15_intra_epoch_alternating",
            "epoch": epoch,
            "filter_phase_total_objective": filter_phase["objective"],
            "filter_phase_eq30": filter_phase["eq30"],
            "representation_phase_total_objective": representation_phase["objective"],
            "representation_phase_eq30": representation_phase["eq30"],
            "train_total_objective": final_training_metrics["objective"],
            "train_temporal_state_objective": final_training_metrics["temporal_state"],
            "train_eq30": final_training_metrics["eq30"],
            "train_eta_future_eq30": final_training_metrics["eta_future_eq30"],
            "train_position_rmse_m": final_training_metrics["position_rmse_m"],
            "train_eta_future_position_rmse_m": final_training_metrics[
                "eta_future_position_rmse_m"
            ],
            "filter_block_learning_rate": current_filter_lr,
            "representation_block_learning_rate": current_representation_lr,
        })

        print(
            f"epoch {epoch:03d}/{TRAINING_EPOCHS}: "
            f"theta_obj={filter_phase['objective']:.6g}, "
            f"psi_obj={representation_phase['objective']:.6g}, "
            f"train_Eq30={final_training_metrics['eq30']:.6g}, "
            f"train_etaNextEq30={final_training_metrics['eta_future_eq30']:.6g}, "
            f"train_posRMSE={final_training_metrics['position_rmse_m']:.3f} m, "
            f"etaNext_posRMSE={final_training_metrics['eta_future_position_rmse_m']:.3f} m, "
            f"train_totalObj={final_training_metrics['objective']:.6g}, "
            f"lr_theta={current_filter_lr:.3g}, "
            f"lr_psi={current_representation_lr:.3g}"
        )

    if final_training_metrics is None:
        raise RuntimeError("Training completed without producing any epoch metrics")

    # Yan et al. publish the fixed offline X/M/Y training construction and do
    # not report an internal validation checkpoint rule. Keep the final epoch
    # of this fixed-X/M/Y alternating training pass.
    model.eval()
    best_epoch = epochs_ran  # compatibility field: final published-style epoch
    best_val_state_loss = float(final_training_metrics["eq30"])
    best_val_position_rmse_m = float(
        final_training_metrics["position_rmse_m"]
    )
    selected_training_stage = "yan_fixed_offline_Data01_ref15_alternating_eq30_final_epoch"

    # Causal online feature construction for paper Eqs. (11)-(14).
    def _make_online_context(
        sat_ids,
        posterior_residual,
        x_pred,
        x_post,
        accel,
        gyro,
        previous_context=None,
    ):
        """Store the completed fusion state needed by paper Eqs. (11)-(14).

        Eq. (12): state innovation  = xhat_{k,k} - xhat_{k,k-1}
        Eq. (13): state residual    = xhat_{k,k} - xhat_{k-1,k-1}

        The Masked-CLA gain for epoch k has already been applied before this
        function is called.  These completed quantities are therefore consumed
        only when constructing the *next* causal network input.  This avoids an
        off-by-one/circular-use ambiguity in the online loop while preserving the
        same lag convention used by the offline training features.
        """
        x_pred = np.asarray(x_pred, dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(x_post, dtype=float).reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = (
            np.zeros(INS_STATE_DIM)
            if previous_context is None
            else x_post - np.asarray(
                previous_context["x_post"], dtype=float
            ).reshape(INS_STATE_DIM)
        )

        context = {
            "sat_ids": tuple(sat_ids),
            "residual": np.asarray(posterior_residual, dtype=float).copy(),
            "x_pred": x_pred.copy(),
            "x_post": x_post.copy(),
            "state_innovation": state_innovation.copy(),
            "state_residual": state_residual.copy(),
            "accel": np.asarray(accel, dtype=float).reshape(3).copy(),
            "gyro": np.asarray(gyro, dtype=float).reshape(3).copy(),
        }
        if (
            len(context["sat_ids"]) != len(context["residual"])
            or not np.all(np.isfinite(context["residual"]))
            or not np.all(np.isfinite(context["state_innovation"]))
            or not np.all(np.isfinite(context["state_residual"]))
            or not np.all(np.isfinite(context["accel"]))
            or not np.all(np.isfinite(context["gyro"]))
        ):
            raise FloatingPointError("invalid completed online fusion context")
        return context

    def _online_feature_arrays(
        previous_online,
        current_sat_ids,
        innovation_now,
        current_accel,
        current_gyro,
    ):
        """Build one causal online Masked-CLA input.

        Current prior information supplies Eq. (10) and the IMU difference.
        The immediately preceding *completed* fusion context supplies the lagged
        Eq. (11)-(13) quantities.  This is the same causal convention used in
        the offline training set and prevents current-posterior leakage into the
        gain that is about to be predicted.
        """
        n = len(current_sat_ids)
        innovation_k = np.asarray(innovation_now, dtype=float).copy()
        current_mask = np.ones(n, dtype=bool)
        residual_k = np.zeros(n, dtype=float)
        residual_mask = np.zeros(n, dtype=bool)

        delta_accel = (
            np.asarray(current_accel, dtype=float).reshape(3)
            - previous_online["accel"]
        )
        delta_gyro = (
            np.asarray(current_gyro, dtype=float).reshape(3)
            - previous_online["gyro"]
        )
        previous_state_innovation = previous_online["state_innovation"]
        previous_state_residual = previous_online["state_residual"]

        previous_residual = dict(
            zip(previous_online["sat_ids"], previous_online["residual"])
        )
        for slot, sat_id in enumerate(current_sat_ids):
            if sat_id in previous_residual:
                residual_k[slot] = previous_residual[sat_id]
                residual_mask[slot] = True

        fixed_k_raw = np.concatenate([
            delta_accel,
            delta_gyro,
            previous_state_residual,
            previous_state_innovation,
        ])
        obs_k_raw = np.stack([residual_k, innovation_k], axis=1)
        channel_k = np.stack([residual_mask, current_mask], axis=1)

        fixed_k = (fixed_k_raw - fixed_mean) / fixed_std
        obs_k = (
            obs_k_raw - obs_mean.reshape(1, 2)
        ) / obs_std.reshape(1, 2)
        obs_k = np.where(channel_k, obs_k, 0.0)

        if (
            fixed_k_raw.shape != (FIXED_FEATURE_DIM,)
            or obs_k_raw.shape != (n, OBSERVATION_FEATURE_DIM)
            or not np.all(np.isfinite(fixed_k))
            or not np.all(np.isfinite(obs_k))
            or not np.all(np.isfinite(innovation_k))
        ):
            raise FloatingPointError(
                "invalid/non-finite feature generated during online rollout"
            )
        return (
            fixed_k,
            obs_k,
            current_mask,
            channel_k,
            innovation_k,
            fixed_k_raw,
            obs_k_raw,
        )

    def _pad_neural_epoch(
        fixed_k,
        obs_k,
        current_mask,
        channel_k,
        innovation_k,
        slot_capacity: int,
    ):
        """Pad one online epoch without changing its physical observations."""
        fixed_k = np.asarray(fixed_k, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_k = np.asarray(obs_k, dtype=float).reshape(
            -1, OBSERVATION_FEATURE_DIM
        )
        current_mask = np.asarray(current_mask, dtype=bool).reshape(-1)
        channel_k = np.asarray(channel_k, dtype=bool).reshape(
            -1, OBSERVATION_FEATURE_DIM
        )
        innovation_k = np.asarray(innovation_k, dtype=float).reshape(-1)
        count = len(current_mask)
        if (
            slot_capacity < count
            or len(obs_k) != count
            or len(channel_k) != count
            or len(innovation_k) != count
        ):
            raise ValueError("invalid online neural padding dimensions")

        obs_padded = np.zeros(
            (slot_capacity, OBSERVATION_FEATURE_DIM), dtype=float
        )
        mask_padded = np.zeros(slot_capacity, dtype=bool)
        channel_padded = np.zeros(
            (slot_capacity, OBSERVATION_FEATURE_DIM), dtype=bool
        )
        innovation_padded_now = np.zeros(slot_capacity, dtype=float)
        obs_padded[:count] = obs_k
        mask_padded[:count] = current_mask
        channel_padded[:count] = channel_k
        innovation_padded_now[:count] = innovation_k
        return (
            fixed_k,
            obs_padded,
            mask_padded,
            channel_padded,
            innovation_padded_now,
        )

    def _feature_shift_diagnostics(fixed_raw, obs_raw, channel_mask):
        """Compare one Data02 input with the raw Data01 training envelope only."""
        fixed_raw = np.asarray(fixed_raw, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_raw = np.asarray(obs_raw, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        channel_mask = np.asarray(channel_mask, dtype=bool).reshape(
            -1, OBSERVATION_FEATURE_DIM
        )

        fixed_outside = (fixed_raw < train_fixed_min) | (fixed_raw > train_fixed_max)
        fixed_ratio = float(np.max(np.abs(fixed_raw) / train_fixed_abs_max))

        obs_outside_count = 0
        obs_valid_count = 0
        obs_ratio = 0.0
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            if not np.any(valid):
                continue
            values = obs_raw[valid, channel]
            obs_valid_count += int(values.size)
            obs_outside_count += int(
                np.count_nonzero(
                    (values < train_obs_min[channel])
                    | (values > train_obs_max[channel])
                )
            )
            obs_ratio = max(
                obs_ratio,
                float(np.max(np.abs(values) / train_obs_abs_max[channel])),
            )

        return {
            "fixed_ood_fraction": float(np.mean(fixed_outside)),
            "fixed_max_abs_train_ratio": fixed_ratio,
            "obs_ood_fraction": (
                float(obs_outside_count / obs_valid_count)
                if obs_valid_count else 0.0
            ),
            "obs_max_abs_train_ratio": float(obs_ratio),
        }


    # =========================================================================
    # 8B. TRAINING SELECTION
    # =========================================================================
    # No second recursive/full-trajectory training stage is applied. This keeps
    # Yan's published fixed X/M/Y construction as the sole training data source.
    # No unpublished cross-fusion hidden-state carry or recursive training is added.
    recursive_training_metrics = None
    selected_training_stage = (
        "yan_fixed_offline_Data01_ref15_intra_epoch_masked_LSTM"
    )
    model.eval()


    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "checkpoint_schema": "stage1_full_15_state_gain_v1",
            "nmax": nmax,
            "eq7_state_order": "[delta_p,delta_v,delta_theta,b_a,b_g]",
            "kalman_gain_state_dimension": INS_STATE_DIM,
            "direct_state_label_dimension": SUPERVISED_STATE_DIM,
            "bias_gain_training": (
                "full_15_row_output_retained; bias_rows_zero_initialized; "
                "no_fabricated_direct_bias_labels"
            ),
            "fixed_mean": fixed_mean,
            "fixed_std": fixed_std,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "feature_standardization_on": bool(FEATURE_STANDARDIZATION_ON),
            "normalizer_source": normalizer_source,
            "eta_training": (
                "fixed_Data01_next_interval_Eq8_INS_state_consequence_"
                "no_direct_eta_MSE"
            ),
            "alternating_partition": (
                "psi=conv+lstm+attention;theta=gain_head+eta_head"
            ),
            "training_stage_selected": selected_training_stage,
            "fixed_offline_training_epochs": best_epoch,
            "sequence_training": (
                "fixed_XMY_intra_epoch_masked_LSTM; "
                "hidden_state_local_to_each_fusion_epoch"
            ),
            "sequence_chunk_length": int(BATCH_SIZE),
            "lstm_trajectory_state": (
                "no_cross_fusion_hc_state_each_Xbar_k_is_one_sequence"
            ),
            "final_epoch": best_epoch,
            "final_train_eq30_state_loss": best_val_state_loss,
            "final_train_position_rmse_m": best_val_position_rmse_m,
            "state_target_scaling": "none_raw_physical_units_per_Yan_Eq30",
            "measurement_mode": "pseudorange_only",
        },
        OUTPUT_DIR / "best_model.pt",
    )
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )

    # =========================================================================
    # 8A. POST-TRAINING RECURSIVE DATA01 DIAGNOSTIC
    # =========================================================================
    # DIAGNOSTIC ONLY. The Yan fixed-X/M/Y training has finished
    # before this point. This replay uses model.eval()+inference_mode(), never updates a
    # parameter, and is not a model-selection stage. Its purpose is only to test
    # whether the final trained
    # network remains stable when its own posterior history recursively generates
    # the next Eqs. (11)-(14) inputs on the SAME Data01 trajectory.
    if RUN_RECURSIVE_DATA01_DIAGNOSTIC:
        print("\n=== RECURSIVE DATA01 CLOSED-LOOP DIAGNOSTIC ===")
        print("mode: trained Masked-CLA replay on Data01; weights frozen")
        print("FDE/DIA:", "ON" if FDE_DIA_ON else "OFF")
        print("eta_k -> Eq.(8) feedback:", "ON" if ETA_FEEDBACK_ON else "OFF")

        # A fresh simulator with the original training seed reproduces the same
        # stochastic LEO realization family without consuming/changing the
        # training or Data02 simulator RNG states.
        data01_diag_leo = LEODownlinkSimulator(
            tle_provider,
            leo_klobuchar,
            seed=LEO_SEED,
            tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M,
            tx_max_iterations=LEO_TX_MAX_ITERATIONS,
            minimum_elevation_deg=LEO_MIN_ELEVATION_DEG,
            prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG,
            use_ionosphere=USE_IONOSPHERE,
            use_troposphere=USE_TROPOSPHERE,
        )

        diag_nav = initial_nav.copy()
        diag_P = P0.copy()
        diag_eta = np.zeros(IMU_ERROR_DIM)
        diag_rows = []
        diag_fde_stats = _new_fde_stats()
        diag_previous = None
        diag_warm_start = None

        diag_last_gyro, diag_last_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            diag_nav.gyroscope_bias_body_radps,
            diag_nav.accelerometer_bias_body_mps2,
            gyroscope_measurement_error_body_radps=diag_eta[ETA_GYRO_SLICE],
            accelerometer_measurement_error_body_mps2=diag_eta[ETA_ACCEL_SLICE],
        )
        diag_feature_gyro, diag_feature_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            diag_nav.gyroscope_bias_body_radps,
            diag_nav.accelerometer_bias_body_mps2,
        )
        # Keep the main-run Van Loan counters unchanged: this replay is diagnostic
        # and must not contaminate the reported Data01+Data02 numerical statistics.
        van_loan_stats_before_data01_diag = dict(_VAN_LOAN_STATS)

        try:
            for event in build_exact_fusion_timeline(
                imr_time, fusion_time, through_last_fusion=True
            ):
                if isinstance(event, PropagationSegment):
                    imu_index = event.imu_index
                    dt = event.end_time_gpst_s - event.start_time_gpst_s
                    diag_last_gyro, diag_last_accel = compensate_imu(
                        imr.angular_rate_body_radps[imu_index],
                        imr.acceleration_body_mps2[imu_index],
                        diag_nav.gyroscope_bias_body_radps,
                        diag_nav.accelerometer_bias_body_mps2,
                        gyroscope_measurement_error_body_radps=diag_eta[ETA_GYRO_SLICE],
                        accelerometer_measurement_error_body_mps2=diag_eta[ETA_ACCEL_SLICE],
                    )
                    diag_feature_gyro, diag_feature_accel = compensate_imu(
                        imr.angular_rate_body_radps[imu_index],
                        imr.acceleration_body_mps2[imu_index],
                        diag_nav.gyroscope_bias_body_radps,
                        diag_nav.accelerometer_bias_body_mps2,
                    )
                    diag_nav = mechanize_ecef(
                        diag_nav, diag_last_gyro, diag_last_accel, dt
                    )
                    diag_F = build_error_state_dynamics(diag_nav, diag_last_accel)
                    diag_Phi, diag_Qd = discretize_process_noise_van_loan(
                        diag_F, Qc, dt
                    )
                    diag_P = diag_Phi @ diag_P @ diag_Phi.T + diag_Qd
                    diag_P = 0.5 * (diag_P + diag_P.T)
                    continue

                fusion_index = event.fusion_index
                t = event.time_gpst_s
                epoch = gnss_epochs[fusion_index]
                if abs(epoch.time_gpst_s - t) > 1e-9:
                    raise RuntimeError(
                        "Data01 recursive diagnostic fusion timestamp mismatch"
                    )

                diag_gnss = gnss_preprocessor.prepare_epoch(
                    epoch, diag_nav, lever_arm_b_m
                )
                diag_leo = data01_diag_leo.simulate_epoch(
                    t, fusion_antenna_truth_position[fusion_index]
                )
                diag_measurements = retain_clock_observable_measurements(
                    tuple(diag_gnss) + tuple(diag_leo)
                )
                if not diag_measurements:
                    continue

                truth_position_now = fusion_antenna_truth_position[fusion_index]
                prior_position = gnss_antenna_position(diag_nav, lever_arm_b_m)
                prior_error_3d_m = float(
                    np.linalg.norm(prior_position - truth_position_now)
                )

                # The first usable fusion row is the causal context row, exactly
                # as in the fixed Data01 construction (history_rows[0]).  It is an
                # ordinary TC/KF update, is excluded from learned metrics, and
                # does not advance the neural recurrent state.
                if diag_previous is None:
                    warm_model = build_measurement_model(
                        diag_nav,
                        diag_measurements,
                        lever_arm_b_m,
                    )
                    warm_correction, diag_P, _ = kalman_measurement_update(
                        diag_P,
                        warm_model.innovation,
                        warm_model.H,
                        warm_model.R,
                    )
                    diag_nav = inject_error_state(diag_nav, warm_correction)
                    warm_residual = build_innovation_only(
                        diag_nav,
                        diag_measurements,
                        lever_arm_b_m,
                    )
                    diag_previous = _make_online_context(
                        warm_model.sat_ids,
                        warm_residual,
                        np.zeros(INS_STATE_DIM),
                        warm_correction,
                        diag_feature_accel,
                        diag_feature_gyro,
                        previous_context=None,
                    )
                    diag_warm_start = {
                        "method": "ordinary_TC_KF_context_only",
                        "time_gpst_s": float(t),
                        "excluded_from_learned_metrics": True,
                    }
                    continue

                if FDE_DIA_ON:
                    diag_fde = ref33_fde_dia_decision(
                        diag_nav,
                        diag_P,
                        diag_measurements,
                        lever_arm_b_m,
                        FDE_SIGNIFICANCE_ALPHA,
                    )
                    _accumulate_fde_stats(diag_fde_stats, diag_fde)
                    diag_prefde_model = diag_fde.tested_measurement_model
                    diag_measurements = diag_fde.measurements
                    diag_model = diag_fde.measurement_model
                else:
                    diag_fde = None
                    diag_model = build_measurement_model(
                        diag_nav, diag_measurements, lever_arm_b_m
                    )
                    diag_prefde_model = diag_model

                # Mirror the online Data02 safety behavior: if FDE leaves no usable
                # observations, keep the propagated INS state and advance only the
                # causal history. No suspect innovation is presented to Masked-CLA.
                if FDE_DIA_ON and not diag_measurements:
                    posterior_position = gnss_antenna_position(
                        diag_nav, lever_arm_b_m
                    )
                    diag_rows.append({
                        "time": t,
                        "position": posterior_position.copy(),
                        "truth": truth_position_now.copy(),
                        "prior_error_3d_m": prior_error_3d_m,
                        "posterior_error_3d_m": prior_error_3d_m,
                        "spectral_radius_knet": float("nan"),
                        "gain_fro_norm": float("nan"),
                        "correction_position_norm_m": 0.0,
                        "prefde_innovation_max_abs_m": (
                            float(np.max(np.abs(diag_prefde_model.innovation)))
                            if len(diag_prefde_model.innovation) else 0.0
                        ),
                        "obs_max_abs_train_ratio": float("nan"),
                    })
                    diag_previous = _make_online_context(
                        (),
                        np.empty(0),
                        np.zeros(INS_STATE_DIM),
                        np.zeros(INS_STATE_DIM),
                        diag_feature_accel,
                        diag_feature_gyro,
                        previous_context=diag_previous,
                    )
                    continue

                current_sat_ids = diag_model.sat_ids
                (
                    fixed_k,
                    obs_k,
                    current_mask,
                    channel_k,
                    innovation_k,
                    fixed_k_raw,
                    obs_k_raw,
                ) = _online_feature_arrays(
                    diag_previous,
                    current_sat_ids,
                    diag_model.innovation,
                    diag_feature_accel,
                    diag_feature_gyro,
                )
                feature_shift = _feature_shift_diagnostics(
                    fixed_k_raw, obs_k_raw, channel_k
                )
                (
                    fixed_nn,
                    obs_nn,
                    mask_nn,
                    channel_nn,
                    innovation_nn,
                ) = _pad_neural_epoch(
                    fixed_k,
                    obs_k,
                    current_mask,
                    channel_k,
                    innovation_k,
                    nmax,
                )

                with torch.inference_mode():
                    diag_output = model(
                        torch.tensor(
                            fixed_nn[None], dtype=torch.float32, device=DEVICE
                        ),
                        torch.tensor(
                            obs_nn[None], dtype=torch.float32, device=DEVICE
                        ),
                        torch.tensor(
                            mask_nn[None], dtype=torch.bool, device=DEVICE
                        ),
                        torch.tensor(
                            channel_nn[None], dtype=torch.bool, device=DEVICE
                        ),
                        recurrent_state=None,
                    )
                    diag_correction = fig8_state_update(
                        diag_output,
                        torch.tensor(
                            innovation_nn[None],
                            dtype=torch.float32,
                            device=DEVICE,
                        ),
                    )[0]

                diag_gain = (
                    diag_output.kalman_gain[0].cpu().numpy().astype(float)
                )
                diag_eta_pred = (
                    diag_output.imu_error[0].cpu().numpy().astype(float)
                )
                diag_correction = diag_correction.cpu().numpy().astype(float)
                n_diag = len(diag_measurements)
                active_gain = diag_gain[:, :n_diag]
                diag_gain_bias_rows_fro_norm = float(
                    np.linalg.norm(active_gain[SUPERVISED_STATE_DIM:, :])
                )
                diag_correction_accel_bias_norm_mps2 = float(
                    np.linalg.norm(
                        diag_correction[ACCELEROMETER_BIAS_STATE_SLICE]
                    )
                )
                diag_correction_gyro_bias_norm_radps = float(
                    np.linalg.norm(
                        diag_correction[GYROSCOPE_BIAS_STATE_SLICE]
                    )
                )

                if (
                    not np.all(np.isfinite(active_gain))
                    or not np.all(np.isfinite(diag_correction))
                    or not np.all(np.isfinite(diag_eta_pred))
                ):
                    raise FloatingPointError(
                        "non-finite learned output in recursive Data01 diagnostic"
                    )

                rho_knet = closed_loop_spectral_radius(
                    active_gain, diag_model.H
                )
                diag_P = learned_gain_covariance_update(
                    diag_P,
                    active_gain,
                    diag_model.H,
                    diag_model.R,
                    diag_correction,
                )
                diag_nav = inject_error_state(diag_nav, diag_correction)
                diag_estimated_accel_bias_norm_mps2 = float(
                    np.linalg.norm(diag_nav.accelerometer_bias_body_mps2)
                )
                diag_estimated_gyro_bias_norm_radps = float(
                    np.linalg.norm(diag_nav.gyroscope_bias_body_radps)
                )
                diag_eta = (
                    diag_eta_pred.copy()
                    if ETA_FEEDBACK_ON
                    else np.zeros(IMU_ERROR_DIM)
                )

                posterior_position = gnss_antenna_position(
                    diag_nav, lever_arm_b_m
                )
                posterior_error_3d_m = float(
                    np.linalg.norm(posterior_position - truth_position_now)
                )
                posterior_residual = build_innovation_only(
                    diag_nav, diag_measurements, lever_arm_b_m
                )
                diag_previous = _make_online_context(
                    current_sat_ids,
                    posterior_residual,
                    np.zeros(INS_STATE_DIM),
                    diag_correction,
                    diag_feature_accel,
                    diag_feature_gyro,
                    previous_context=diag_previous,
                )

                diag_rows.append({
                    "time": t,
                    "position": posterior_position.copy(),
                    "truth": truth_position_now.copy(),
                    "prior_error_3d_m": prior_error_3d_m,
                    "posterior_error_3d_m": posterior_error_3d_m,
                    "spectral_radius_knet": rho_knet,
                    "gain_fro_norm": float(np.linalg.norm(active_gain)),
                    "gain_bias_rows_fro_norm": diag_gain_bias_rows_fro_norm,
                    "correction_position_norm_m": float(
                        np.linalg.norm(diag_correction[:3])
                    ),
                    "correction_accel_bias_norm_mps2": (
                        diag_correction_accel_bias_norm_mps2
                    ),
                    "correction_gyro_bias_norm_radps": (
                        diag_correction_gyro_bias_norm_radps
                    ),
                    "estimated_accel_bias_norm_mps2": (
                        diag_estimated_accel_bias_norm_mps2
                    ),
                    "estimated_gyro_bias_norm_radps": (
                        diag_estimated_gyro_bias_norm_radps
                    ),
                    "prefde_innovation_max_abs_m": (
                        float(np.max(np.abs(diag_prefde_model.innovation)))
                        if len(diag_prefde_model.innovation) else 0.0
                    ),
                    "obs_max_abs_train_ratio": float(
                        feature_shift["obs_max_abs_train_ratio"]
                    ),
                })
        finally:
            _VAN_LOAN_STATS.clear()
            _VAN_LOAN_STATS.update(van_loan_stats_before_data01_diag)

        if not diag_rows:
            raise RuntimeError(
                "Recursive Data01 diagnostic produced no usable trajectory rows"
            )

        diag_estimate = np.stack([row["position"] for row in diag_rows])
        diag_truth = np.stack([row["truth"] for row in diag_rows])
        diag_ned_error = np.empty_like(diag_estimate)
        for i, (est, truth_i) in enumerate(zip(diag_estimate, diag_truth)):
            lat, lon, _ = ecef_to_llh(truth_i)
            diag_ned_error[i] = (
                c_ecef_to_ned(lat, lon) @ (est - truth_i)
            )

        diag_error_3d = np.linalg.norm(diag_ned_error, axis=1)
        diag_rmse_ned = np.sqrt(np.mean(diag_ned_error**2, axis=0))
        diag_rmse_3d = float(np.sqrt(np.mean(diag_error_3d**2)))
        diag_rmse = np.append(diag_rmse_ned, diag_rmse_3d)
        diag_time = np.asarray([row["time"] for row in diag_rows], dtype=float)
        diag_rho = np.asarray(
            [row["spectral_radius_knet"] for row in diag_rows], dtype=float
        )
        diag_obs_ratio = np.asarray(
            [row["obs_max_abs_train_ratio"] for row in diag_rows], dtype=float
        )

        def _diag_first(mask):
            indices = np.flatnonzero(np.asarray(mask, dtype=bool))
            if not len(indices):
                return None
            i = int(indices[0])
            return {
                "row_index": i,
                "time_gpst_s": float(diag_time[i]),
                "posterior_error_3d_m": float(diag_error_3d[i]),
            }

        data01_recursive_summary = {
            "epochs": int(len(diag_rows)),
            "causal_warm_start": diag_warm_start,
            "rmse_n_e_d_3d_m": [float(x) for x in diag_rmse],
            "first_spectral_radius_gt_1": _diag_first(diag_rho > 1.0),
            "first_error_gt_100m": _diag_first(diag_error_3d > 100.0),
            "first_error_gt_1km": _diag_first(diag_error_3d > 1_000.0),
            "first_error_gt_100km": _diag_first(diag_error_3d > 100_000.0),
            "first_obs_feature_gt_10x_training_absmax": _diag_first(
                diag_obs_ratio > 10.0
            ),
            "max_prefde_innovation_abs_m": float(np.nanmax([
                row["prefde_innovation_max_abs_m"] for row in diag_rows
            ])),
            "max_gain_fro_norm": float(np.nanmax([
                row["gain_fro_norm"] for row in diag_rows
            ])),
            "max_bias_gain_rows_fro_norm": float(np.nanmax([
                row["gain_bias_rows_fro_norm"] for row in diag_rows
            ])),
            "max_position_correction_norm_m": float(np.nanmax([
                row["correction_position_norm_m"] for row in diag_rows
            ])),
            "max_accel_bias_correction_norm_mps2": float(np.nanmax([
                row["correction_accel_bias_norm_mps2"] for row in diag_rows
            ])),
            "max_gyro_bias_correction_norm_radps": float(np.nanmax([
                row["correction_gyro_bias_norm_radps"] for row in diag_rows
            ])),
            "max_estimated_accel_bias_norm_mps2": float(np.nanmax([
                row["estimated_accel_bias_norm_mps2"] for row in diag_rows
            ])),
            "max_estimated_gyro_bias_norm_radps": float(np.nanmax([
                row["estimated_gyro_bias_norm_radps"] for row in diag_rows
            ])),
            "fde_checked_detected_unresolved": [
                int(diag_fde_stats["epochs_checked"]),
                int(diag_fde_stats["epochs_detected"]),
                int(diag_fde_stats["unresolved_epochs"]),
            ] if FDE_DIA_ON else None,
            "scope": (
                "diagnostic_only_frozen_weights_same_Data01_recursive_online_rollout"
            ),
        }

        diag_csv_path = OUTPUT_DIR / "recursive_data01_diagnostics.csv"
        diag_fields = (
            "time",
            "prior_error_3d_m",
            "posterior_error_3d_m",
            "spectral_radius_knet",
            "gain_fro_norm",
            "gain_bias_rows_fro_norm",
            "correction_position_norm_m",
            "correction_accel_bias_norm_mps2",
            "correction_gyro_bias_norm_radps",
            "estimated_accel_bias_norm_mps2",
            "estimated_gyro_bias_norm_radps",
            "prefde_innovation_max_abs_m",
            "obs_max_abs_train_ratio",
        )
        with diag_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=diag_fields)
            writer.writeheader()
            for row in diag_rows:
                writer.writerow({name: row[name] for name in diag_fields})

        (OUTPUT_DIR / "recursive_data01_summary.json").write_text(
            json.dumps(data01_recursive_summary, indent=2),
            encoding="utf-8",
        )

        print(
            "recursive Data01 RMSE [N, E, D, 3D] m:",
            np.asarray(diag_rmse),
        )
        print(
            "recursive Data01 diagnostics:",
            json.dumps(data01_recursive_summary, indent=2),
        )
        print("recursive Data01 CSV:", diag_csv_path)

    # =========================================================================
    # 9. LOAD AND SYNCHRONIZE THE INDEPENDENT TEST DATASET
    # =========================================================================
    # Important: no test sample is used above for fitting the model or normalizers.
    test_dataset_dir = resolve_test_dataset_dir()
    if test_dataset_dir.resolve() == TRAIN_DATASET_DIR.resolve():
        raise ValueError(
            "The online test dataset must be distinct from the training dataset"
        )
    test_files, test_rover = resolve_dataset_files(test_dataset_dir, "01")

    test_required = [IMU_ERROR_MODEL_PATH, LEO_TLE_DIR]
    test_missing = [str(path) for path in test_required if not Path(path).exists()]
    if test_missing:
        raise FileNotFoundError("Missing shared test inputs:\n" + "\n".join(test_missing))

    test_antenna_truth = load_ie_ground_truth(test_files.rover_ground_truth)
    test_imu_truth = load_ie_ground_truth(test_files.imu_ground_truth)
    test_imu_models = read_imu_error_models(IMU_ERROR_MODEL_PATH)
    if test_rover.imu_type not in test_imu_models:
        raise KeyError(f"No IMU noise model for test IMU type {test_rover.imu_type!r}")
    test_imu_noise = imu_model_to_si(test_imu_models[test_rover.imu_type])

    test_rinex = RINEXObservationFile.open(test_files.rinex_obs)
    first_test_rinex_epoch = next(
        test_rinex.iter_epochs(allowed_constellations={"G", "C"}),
        None,
    )
    if first_test_rinex_epoch is None:
        raise ValueError("Test RINEX file contains no GPS/BDS epochs")

    test_imr_header, test_imr_tow_all, test_imr_record_count = read_imr_tow_only(
        test_files.imr
    )
    if test_imr_record_count < 2:
        raise ValueError("Test IMR file contains fewer than two usable samples")
    if len(test_antenna_truth.tow_s) < 2 or len(test_imu_truth.tow_s) < 2:
        raise ValueError("Test ground-truth file contains fewer than two usable epochs")

    # Same validated GPST anchoring used by the training dataset.
    test_anchor_time = float(first_test_rinex_epoch.time_gpst_s)
    test_imr_time_all = anchor_imr_tow_to_gpst_seconds(
        test_imr_tow_all,
        test_anchor_time,
    )

    test_antenna_truth_time = validate_strict_time_axis(
        "test antenna truth",
        test_antenna_truth.week.astype(float) * GPS_WEEK_S
        + test_antenna_truth.tow_s,
    )
    test_imu_truth_time = validate_strict_time_axis(
        "test IMU truth",
        test_imu_truth.week.astype(float) * GPS_WEEK_S
        + test_imu_truth.tow_s,
    )

    test_common_start = max(
        float(test_imr_time_all[0]),
        float(test_antenna_truth_time[0]),
        float(test_imu_truth_time[0]),
    )
    test_common_end = min(
        float(test_imr_time_all[-1]),
        float(test_antenna_truth_time[-1]),
        float(test_imu_truth_time[-1]),
    )
    test_start = int(
        np.searchsorted(test_imr_time_all, test_common_start, side="left")
    )
    if test_start >= len(test_imr_time_all):
        raise ValueError("Test common time span starts after the IMR file")
    test_usable_imu_start = float(test_imr_time_all[test_start])

    test_gnss_epochs = tuple(
        test_rinex.iter_epochs(
            allowed_constellations={"G", "C"},
            start_time_gpst_s=test_usable_imu_start,
            end_time_gpst_s=test_common_end,
            max_epochs=MAX_TEST_FUSION_EPOCHS,
            require_measurements=True,
        )
    )
    if not test_gnss_epochs:
        raise ValueError("Test RINEX file contains no usable GPS/BDS pseudorange epochs")

    test_fusion_time = validate_strict_time_axis(
        "test fusion",
        np.asarray(
            [epoch.time_gpst_s for epoch in test_gnss_epochs], dtype=float
        ),
    )
    if len(test_fusion_time) < 1:
        raise ValueError("No synchronized test GNSS fusion epochs remain")

    test_common_stop = min(
        int(np.searchsorted(test_imr_time_all, test_common_end, side="left")) + 1,
        len(test_imr_time_all),
    )
    test_requested_stop = min(
        int(np.searchsorted(test_imr_time_all, test_fusion_time[-1], side="left")) + 1,
        test_common_stop,
    )
    test_imr = read_imr(
        test_files.imr,
        scaling_mode="cpp_exact",
        start_record=test_start,
        stop_record=test_requested_stop,
    )
    test_imr_time = test_imr_time_all[test_start:test_requested_stop].copy()
    del test_imr_tow_all, test_imr_time_all

    if len(test_imr.tow_s) < 2:
        raise ValueError("Selected test IMR window contains fewer than two samples")

    test_query_time = np.concatenate(([test_imr_time[0]], test_fusion_time))
    test_antenna_position, _, _, _, _ = interpolate_ground_truth(
        test_antenna_truth,
        test_query_time,
        MAX_TRUTH_INTERPOLATION_GAP_S,
    )
    (
        test_imu_position,
        test_imu_velocity,
        test_imu_heading,
        test_imu_pitch,
        test_imu_roll,
    ) = interpolate_ground_truth(
        test_imu_truth,
        test_query_time,
        MAX_TRUTH_INTERPOLATION_GAP_S,
    )

    test_fusion_antenna_truth_position = test_antenna_position[1:]

    test_C_b_e = body_to_ecef_from_ie_hpr(
        test_imu_position[0],
        test_imu_heading[0],
        test_imu_pitch[0],
        test_imu_roll[0],
        test_rover.mounting_xyz_deg,
    )
    test_lever_arm_b_m = transform_lever_arm_vehicle_to_body(
        test_rover.lever_arm_vehicle_m,
        *test_rover.mounting_xyz_deg,
    )

    # Same reference-point consistency check used for the training dataset.
    test_initial_antenna_from_imu = (
        test_imu_position[0] + test_C_b_e @ test_lever_arm_b_m
    )
    test_truth_reference_error_m = np.linalg.norm(
        test_antenna_position[0] - test_initial_antenna_from_imu
    )
    if test_truth_reference_error_m > 0.05:
        raise ValueError(
            "Test ROVE/IMU ground-truth reference points are inconsistent with "
            f"the lever arm: {test_truth_reference_error_m:.3f} m"
        )

    test_initial_nav = NavigationState(
        test_imu_position[0].copy(),
        test_imu_velocity[0].copy(),
        test_C_b_e,
        np.zeros(3),
        np.zeros(3),
    )
    test_P0 = initial_covariance_from_imu_model(test_imu_noise)
    test_Qc = continuous_process_covariance_from_imu_model(test_imu_noise)

    test_ionosphere_coefficients = None
    if USE_IONOSPHERE:
        test_iono_header = read_rinex_navigation_header(test_files.nav)
        test_ionosphere_coefficients = {
            "G": (
                np.asarray(test_iono_header["GPSA"]),
                np.asarray(test_iono_header["GPSB"]),
            ),
            "C": (
                np.asarray(test_iono_header["BDSA"]),
                np.asarray(test_iono_header["BDSB"]),
            ),
        }

    test_gnss_preprocessor = GNSSPreprocessor(
        SP3Orbit(test_files.sp3),
        RINEXClock(test_files.clk),
        min_elevation_deg=MIN_GNSS_ELEVATION_DEG,
        use_ionosphere=USE_IONOSPHERE,
        use_troposphere=USE_TROPOSPHERE,
        broadcast_ionosphere_coefficients=test_ionosphere_coefficients,
    )
    test_leo_klobuchar = (
        None
        if test_ionosphere_coefficients is None
        else test_ionosphere_coefficients["G"]
    )
    test_leo_simulator = LEODownlinkSimulator(
        tle_provider,
        test_leo_klobuchar,
        seed=TEST_LEO_SEED,
        tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M,
        tx_max_iterations=LEO_TX_MAX_ITERATIONS,
        minimum_elevation_deg=LEO_MIN_ELEVATION_DEG,
        prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG,
        use_ionosphere=USE_IONOSPHERE,
        use_troposphere=USE_TROPOSPHERE,
    )

    print("\n=== INDEPENDENT TEST DATASET ===")
    print("dataset:", test_dataset_dir)
    print("test IMU records in file:", test_imr_record_count)
    print("test IMU samples used:", len(test_imr.tow_s))
    print("test RINEX fusion epochs selected:", len(test_gnss_epochs))

    # =========================================================================
    # 10. ONLINE MASKED KALMANNET PASS ON TEST DATASET ONLY
    # =========================================================================
    model = model.to(DEVICE)
    model.eval()

    nav = test_initial_nav.copy()
    P = test_P0.copy()
    current_eta_test = np.zeros(IMU_ERROR_DIM)
    online_rows = []
    previous_online = None
    test_recurrent_capacity = int(nmax)
    test_warm_start = None
    test_fde_stats = _new_fde_stats()

    last_gyro, last_accel = compensate_imu(
        test_imr.angular_rate_body_radps[0],
        test_imr.acceleration_body_mps2[0],
        nav.gyroscope_bias_body_radps,
        nav.accelerometer_bias_body_mps2,
        gyroscope_measurement_error_body_radps=current_eta_test[ETA_GYRO_SLICE],
        accelerometer_measurement_error_body_mps2=current_eta_test[ETA_ACCEL_SLICE],
    )
    last_feature_gyro, last_feature_accel = compensate_imu(
        test_imr.angular_rate_body_radps[0],
        test_imr.acceleration_body_mps2[0],
        nav.gyroscope_bias_body_radps,
        nav.accelerometer_bias_body_mps2,
    )
    test_timeline = build_exact_fusion_timeline(
        test_imr_time,
        test_fusion_time,
        through_last_fusion=True,
    )

    for event in test_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(
                test_imr.angular_rate_body_radps[imu_index],
                test_imr.acceleration_body_mps2[imu_index],
                nav.gyroscope_bias_body_radps,
                nav.accelerometer_bias_body_mps2,
                gyroscope_measurement_error_body_radps=current_eta_test[ETA_GYRO_SLICE],
                accelerometer_measurement_error_body_mps2=current_eta_test[ETA_ACCEL_SLICE],
            )
            last_feature_gyro, last_feature_accel = compensate_imu(
                test_imr.angular_rate_body_radps[imu_index],
                test_imr.acceleration_body_mps2[imu_index],
                nav.gyroscope_bias_body_radps,
                nav.accelerometer_bias_body_mps2,
            )
            nav = mechanize_ecef(nav, last_gyro, last_accel, dt)
            F_error = build_error_state_dynamics(nav, last_accel)
            Phi, Qd = discretize_process_noise_van_loan(
                F_error, test_Qc, dt
            )
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue

        fusion_index = event.fusion_index
        t = event.time_gpst_s
        epoch = test_gnss_epochs[fusion_index]
        if abs(epoch.time_gpst_s - t) > 1e-9:
            raise RuntimeError("test fusion timeline/epoch timestamp mismatch")

        gnss_measurements = test_gnss_preprocessor.prepare_epoch(
            epoch, nav, test_lever_arm_b_m
        )
        # As in training, test truth is used only to generate the simulated LEO
        # measurement; it is not an input to the learned navigation update.
        leo_measurements = test_leo_simulator.simulate_epoch(
            t, test_fusion_antenna_truth_position[fusion_index]
        )
        measurements = retain_clock_observable_measurements(
            tuple(gnss_measurements) + tuple(leo_measurements)
        )

        if not measurements:
            continue

        truth_position_now = test_fusion_antenna_truth_position[fusion_index]
        prior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        prior_error_3d_m = float(np.linalg.norm(prior_position - truth_position_now))

        # Eqs. (11)-(14) require a completed preceding fusion epoch.  Data01
        # training obtains that context from history_rows[0], so Data02 must not
        # manufacture an all-zero predecessor and feed it to the learned model.
        # One ordinary TC/KF update establishes the causal context, exactly like
        # the original KalmanNet InitSequence boundary.  It is excluded from all
        # learned test metrics and does not advance the neural recurrent state.
        if previous_online is None:
            warm_measurement_model = build_measurement_model(
                nav,
                measurements,
                test_lever_arm_b_m,
            )
            warm_correction, P, _ = kalman_measurement_update(
                P,
                warm_measurement_model.innovation,
                warm_measurement_model.H,
                warm_measurement_model.R,
            )
            nav = inject_error_state(nav, warm_correction)
            warm_posterior_residual = build_innovation_only(
                nav,
                measurements,
                test_lever_arm_b_m,
            )
            previous_online = _make_online_context(
                warm_measurement_model.sat_ids,
                warm_posterior_residual,
                np.zeros(INS_STATE_DIM),
                warm_correction,
                last_feature_accel,
                last_feature_gyro,
                previous_context=None,
            )
            test_warm_start = {
                "method": "ordinary_TC_KF_context_only",
                "time_gpst_s": float(t),
                "excluded_from_learned_metrics": True,
                "neural_state_advanced": False,
            }
            continue

        # ================================================================
        # OPTIONAL FDE / DIA BLOCK — the only algorithmic ON/OFF branch.
        # ================================================================
        if FDE_DIA_ON:
            # Paper path: raw INS-predicted innovation -> Eq. (33) detection /
            # Ref. [33] identification -> fault elimination -> Eq. (34) DIA.
            fde_result = ref33_fde_dia_decision(
                nav, P, measurements, test_lever_arm_b_m, FDE_SIGNIFICANCE_ALPHA
            )
            _accumulate_fde_stats(test_fde_stats, fde_result)
            prefde_measurement_model = fde_result.tested_measurement_model
            measurements = fde_result.measurements
            measurement_model = fde_result.measurement_model
        else:
            # Ablation path: the entire FDE/DIA block is absent. The untouched
            # measurement set enters the same Masked-CLA/KalmanNet update.
            fde_result = None
            measurement_model = build_measurement_model(
                nav, measurements, test_lever_arm_b_m
            )
            prefde_measurement_model = measurement_model

        # A detected/identified large fault is removed before any Masked-CLA
        # evaluation. If Identification is unresolved, or exclusion leaves no
        # usable clock-observable measurements, keep the INS-propagated state and
        # do not feed the suspect innovation to KalmanNet. Once causal history
        # exists, retain this INS-only epoch in the reported trajectory so the
        # evaluation is not biased by silently dropping difficult epochs.
        if FDE_DIA_ON and not measurements:
            if previous_online is not None:
                posterior_position = gnss_antenna_position(
                    nav, test_lever_arm_b_m
                )
                hpl_m, vpl_m = protection_levels_from_covariance(
                    fde_result.dia_covariance, posterior_position
                )
                online_rows.append({
                    "time": t,
                    "position": posterior_position.copy(),
                    "truth": test_fusion_antenna_truth_position[fusion_index].copy(),
                    "hpl_m": hpl_m,
                    "vpl_m": vpl_m,
                    "spectral_radius_knet": float("nan"),
                    "spectral_radius_classical": float("nan"),
                    "n_input": len(fde_result.tested_measurements),
                    "n_total": 0,
                    "n_gnss": 0,
                    "n_leo": 0,
                    "sat_ids": (),
                    "eta_gyro_radps": current_eta_test[ETA_GYRO_SLICE].copy(),
                    "eta_accel_mps2": current_eta_test[ETA_ACCEL_SLICE].copy(),
                    "eta_feedback_enabled": bool(ETA_FEEDBACK_ON),
                    "prior_error_3d_m": prior_error_3d_m,
                    "posterior_error_3d_m": prior_error_3d_m,
                    "prefde_innovation_rms_m": float(
                        np.sqrt(np.mean(prefde_measurement_model.innovation**2))
                    ) if len(prefde_measurement_model.innovation) else 0.0,
                    "prefde_innovation_max_abs_m": float(
                        np.max(np.abs(prefde_measurement_model.innovation))
                    ) if len(prefde_measurement_model.innovation) else 0.0,
                    "knet_innovation_rms_m": float("nan"),
                    "knet_innovation_max_abs_m": float("nan"),
                    "gain_fro_norm": float("nan"),
                    "gain_max_abs": float("nan"),
                    "correction_position_norm_m": 0.0,
                    "correction_velocity_norm_mps": 0.0,
                    "correction_attitude_norm_rad": 0.0,
                    "eta_gyro_norm_radps": float(np.linalg.norm(current_eta_test[ETA_GYRO_SLICE])),
                    "eta_accel_norm_mps2": float(np.linalg.norm(current_eta_test[ETA_ACCEL_SLICE])),
                    "fixed_ood_fraction": float("nan"),
                    "fixed_max_abs_train_ratio": float("nan"),
                    "obs_ood_fraction": float("nan"),
                    "obs_max_abs_train_ratio": float("nan"),
                    "fde_detected": bool(fde_result.detected),
                    "fde_identified_sat_ids": tuple(fde_result.identified_sat_ids),
                    "fde_excluded_sat_ids": tuple(fde_result.excluded_sat_ids),
                    "fde_ambiguous_sat_ids": tuple(fde_result.ambiguous_sat_ids),
                    "fde_statistic": float(fde_result.statistic),
                    "fde_threshold": float(fde_result.threshold),
                    "fde_dof": int(fde_result.dof),
                    "fde_post_exclusion_statistic": float(
                        fde_result.post_exclusion_statistic
                    ),
                    "fde_post_exclusion_threshold": float(
                        fde_result.post_exclusion_threshold
                    ),
                    "fde_post_exclusion_dof": int(
                        fde_result.post_exclusion_dof
                    ),
                    "fde_post_exclusion_consistent": bool(
                        fde_result.post_exclusion_consistent
                    ),
                    "fde_local_identification_score": float(
                        fde_result.local_identification_score
                    ),
                    "fde_estimated_fault_m": float(fde_result.estimated_fault_m),
                    "fde_hard_exclusion_applied": bool(fde_result.excluded_sat_ids),
                    "fde_state_correction_norm": float(np.linalg.norm(fde_result.dia_state_correction)),
                    "fde_unresolved": bool(fde_result.unresolved),
                    "fusion_mode": "INS_only_after_empty_FDE",
                })
                previous_online = _make_online_context(
                    (),
                    np.empty(0),
                    np.zeros(INS_STATE_DIM),
                    np.zeros(INS_STATE_DIM),
                    last_feature_accel,
                    last_feature_gyro,
                    previous_context=previous_online,
                )
            continue

        # The first learned update now sees the same kind of completed preceding
        # context as the first Data01 training sample.  Neural h/c is still zero
        # here and begins its recurrence at this first learned epoch.

        # Training normalization is still fixed from training only, but the
        # Masked CLA gain head is slot-shared and therefore accepts the current
        # measurement count directly.  No test observation is truncated, and no
        # test statistic is used to refit normalization or retrain the model.
        n = len(measurements)

        current_sat_ids = measurement_model.sat_ids
        innovation_now = measurement_model.innovation

        (
            fixed_k,
            obs_k,
            current_mask,
            channel_k,
            innovation_k,
            fixed_k_raw,
            obs_k_raw,
        ) = _online_feature_arrays(
            previous_online,
            current_sat_ids,
            innovation_now,
            last_feature_accel,
            last_feature_gyro,
        )

        feature_shift = _feature_shift_diagnostics(
            fixed_k_raw, obs_k_raw, channel_k
        )
        prefde_innovation = np.asarray(
            prefde_measurement_model.innovation, dtype=float
        )
        prefde_innovation_rms_m = (
            float(np.sqrt(np.mean(prefde_innovation**2)))
            if prefde_innovation.size else 0.0
        )
        prefde_innovation_max_abs_m = (
            float(np.max(np.abs(prefde_innovation)))
            if prefde_innovation.size else 0.0
        )
        knet_innovation_rms_m = (
            float(np.sqrt(np.mean(innovation_k**2)))
            if innovation_k.size else 0.0
        )
        knet_innovation_max_abs_m = (
            float(np.max(np.abs(innovation_k)))
            if innovation_k.size else 0.0
        )

        if n > test_recurrent_capacity:
            test_recurrent_capacity = int(n)
        (
            fixed_nn,
            obs_nn,
            mask_nn,
            channel_nn,
            innovation_nn,
        ) = _pad_neural_epoch(
            fixed_k,
            obs_k,
            current_mask,
            channel_k,
            innovation_k,
            test_recurrent_capacity,
        )

        with torch.inference_mode():
            output = model(
                torch.tensor(
                    fixed_nn[None],
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                torch.tensor(
                    obs_nn[None],
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                torch.tensor(
                    mask_nn[None],
                    dtype=torch.bool,
                    device=DEVICE,
                ),
                torch.tensor(
                    channel_nn[None],
                    dtype=torch.bool,
                    device=DEVICE,
                ),
                recurrent_state=None,
            )
            innovation_tensor = torch.tensor(
                innovation_nn[None],
                dtype=torch.float32,
                device=DEVICE,
            )
            correction = fig8_state_update(
                output,
                innovation_tensor,
            )[0]

        gain = output.kalman_gain[0].cpu().numpy().astype(float)
        eta = output.imu_error[0].cpu().numpy().astype(float)
        correction = correction.cpu().numpy().astype(float)
        learned_error_state_pred = np.zeros(INS_STATE_DIM)
        learned_error_state_post = learned_error_state_pred + correction
        active_gain = gain[:, :n]
        gain_fro_norm = float(np.linalg.norm(active_gain))
        gain_max_abs = float(np.max(np.abs(active_gain))) if active_gain.size else 0.0
        gain_navigation_rows_fro_norm = float(
            np.linalg.norm(active_gain[:SUPERVISED_STATE_DIM, :])
        )
        gain_accel_bias_rows_fro_norm = float(
            np.linalg.norm(active_gain[ACCELEROMETER_BIAS_STATE_SLICE, :])
        )
        gain_gyro_bias_rows_fro_norm = float(
            np.linalg.norm(active_gain[GYROSCOPE_BIAS_STATE_SLICE, :])
        )
        correction_position_norm_m = float(np.linalg.norm(correction[0:3]))
        correction_velocity_norm_mps = float(np.linalg.norm(correction[3:6]))
        correction_attitude_norm_rad = float(np.linalg.norm(correction[6:9]))
        correction_accel_bias_norm_mps2 = float(
            np.linalg.norm(correction[ACCELEROMETER_BIAS_STATE_SLICE])
        )
        correction_gyro_bias_norm_radps = float(
            np.linalg.norm(correction[GYROSCOPE_BIAS_STATE_SLICE])
        )
        eta_gyro_norm_radps = float(np.linalg.norm(eta[ETA_GYRO_SLICE]))
        eta_accel_norm_mps2 = float(np.linalg.norm(eta[ETA_ACCEL_SLICE]))

        # Preserve the complete Eq. (7) state update, including b_a and b_g.
        # Their rows have no fabricated direct labels: they are trained through
        # their propagated effect on future labeled navigation errors.
        if (
            not np.all(np.isfinite(active_gain))
            or not np.all(np.isfinite(correction))
            or not np.all(np.isfinite(eta))
        ):
            raise FloatingPointError(
                "non-finite learned gain/correction/eta during online fusion"
            )

        # Detection/Identification used the untouched INS-predicted innovation.
        # Yan Eq. (34) has already been evaluated inside the classical DIA branch
        # for integrity bookkeeping; the learned navigation update uses only the
        # measurements left after fault elimination.
        P_prior_update = P.copy()
        Q_classical = (
            measurement_model.H @ P_prior_update @ measurement_model.H.T
            + measurement_model.R
        )
        K_classical = (
            P_prior_update @ measurement_model.H.T
            @ np.linalg.pinv(Q_classical, rcond=1e-12)
        )
        rho_knet = closed_loop_spectral_radius(active_gain, measurement_model.H)
        rho_classical = closed_loop_spectral_radius(K_classical, measurement_model.H)

        P = learned_gain_covariance_update(
            P_prior_update,
            active_gain,
            measurement_model.H,
            measurement_model.R,
            correction,
        )
        learned_error_state_post = learned_error_state_pred + correction
        nav = inject_error_state(nav, correction)
        estimated_accel_bias_norm_mps2 = float(
            np.linalg.norm(nav.accelerometer_bias_body_mps2)
        )
        estimated_gyro_bias_norm_radps = float(
            np.linalg.norm(nav.gyroscope_bias_body_radps)
        )
        # Fig. 2 / Eq. (8): learned residual gyro/accelerometer measurement
        # errors are fed back to IMU compensation for the following interval.
        # ETA_FEEDBACK_ON=False is diagnostic-only and leaves the predicted eta
        # untouched while bypassing its application to the next IMU interval.
        current_eta_test = (
            eta.copy() if ETA_FEEDBACK_ON else np.zeros(IMU_ERROR_DIM)
        )

        posterior_position = gnss_antenna_position(
            nav, test_lever_arm_b_m
        )
        posterior_error_3d_m = float(
            np.linalg.norm(posterior_position - truth_position_now)
        )
        pl_covariance = fde_result.dia_covariance if FDE_DIA_ON else P
        hpl_m, vpl_m = protection_levels_from_covariance(
            pl_covariance, posterior_position
        )
        posterior_residual = build_innovation_only(
            nav, measurements, test_lever_arm_b_m
        )

        previous_online = _make_online_context(
            current_sat_ids,
            posterior_residual,
            learned_error_state_pred,
            learned_error_state_post,
            last_feature_accel,
            last_feature_gyro,
            previous_context=previous_online,
        )

        counts = Counter(m.constellation for m in measurements)
        online_rows.append({
            "time": t,
            "position": posterior_position.copy(),
            "truth": test_fusion_antenna_truth_position[fusion_index].copy(),
            "hpl_m": hpl_m, "vpl_m": vpl_m,
            "spectral_radius_knet": rho_knet,
            "spectral_radius_classical": rho_classical,
            "n_input": (len(fde_result.tested_measurements) if FDE_DIA_ON else len(measurements)),
            "n_total": n,
            "n_gnss": counts["G"] + counts["C"], "n_leo": counts["L"],
            "sat_ids": current_sat_ids,
            "eta_gyro_radps": eta[ETA_GYRO_SLICE].copy(),
            "eta_accel_mps2": eta[ETA_ACCEL_SLICE].copy(),
            "eta_feedback_enabled": bool(ETA_FEEDBACK_ON),
            "prior_error_3d_m": prior_error_3d_m,
            "posterior_error_3d_m": posterior_error_3d_m,
            "prefde_innovation_rms_m": prefde_innovation_rms_m,
            "prefde_innovation_max_abs_m": prefde_innovation_max_abs_m,
            "knet_innovation_rms_m": knet_innovation_rms_m,
            "knet_innovation_max_abs_m": knet_innovation_max_abs_m,
            "gain_fro_norm": gain_fro_norm,
            "gain_max_abs": gain_max_abs,
            "gain_navigation_rows_fro_norm": gain_navigation_rows_fro_norm,
            "gain_accel_bias_rows_fro_norm": gain_accel_bias_rows_fro_norm,
            "gain_gyro_bias_rows_fro_norm": gain_gyro_bias_rows_fro_norm,
            "correction_position_norm_m": correction_position_norm_m,
            "correction_velocity_norm_mps": correction_velocity_norm_mps,
            "correction_attitude_norm_rad": correction_attitude_norm_rad,
            "correction_accel_bias_norm_mps2": correction_accel_bias_norm_mps2,
            "correction_gyro_bias_norm_radps": correction_gyro_bias_norm_radps,
            "estimated_accel_bias_norm_mps2": estimated_accel_bias_norm_mps2,
            "estimated_gyro_bias_norm_radps": estimated_gyro_bias_norm_radps,
            "eta_gyro_norm_radps": eta_gyro_norm_radps,
            "eta_accel_norm_mps2": eta_accel_norm_mps2,
            **feature_shift,
            **_fde_export_fields(fde_result),
            "fusion_mode": "MaskedCLA_post_FDE_DIA" if FDE_DIA_ON else "MaskedCLA_no_FDE_DIA",
        })


    # =========================================================================
    # 11. TEST-DATASET EVALUATION
    # =========================================================================
    if not online_rows:
        raise ValueError(
            "Online Masked KalmanNet pass produced no usable test epochs; "
            "cannot compute test navigation metrics"
        )

    online_time = np.asarray([row["time"] for row in online_rows])
    estimate = np.stack([row["position"] for row in online_rows])
    truth_aligned = np.stack([row["truth"] for row in online_rows])

    ned_error = np.empty_like(estimate)
    for i, (est, truth_i) in enumerate(zip(estimate, truth_aligned)):
        lat, lon, _ = ecef_to_llh(truth_i)
        ned_error[i] = c_ecef_to_ned(lat, lon) @ (est - truth_i)

    error_3d = np.linalg.norm(ned_error, axis=1)
    horizontal_error = np.linalg.norm(ned_error[:, :2], axis=1)
    vertical_error = np.abs(ned_error[:, 2])
    hpl = np.asarray([row["hpl_m"] for row in online_rows], dtype=float)
    vpl = np.asarray([row["vpl_m"] for row in online_rows], dtype=float)
    spectral_radius_knet = np.asarray(
        [row["spectral_radius_knet"] for row in online_rows], dtype=float
    )
    spectral_radius_classical = np.asarray(
        [row["spectral_radius_classical"] for row in online_rows], dtype=float
    )
    rmse_ned = np.sqrt(np.mean(ned_error**2, axis=0))
    rmse_3d = float(np.sqrt(np.mean(error_3d**2)))
    rmse = np.append(rmse_ned, rmse_3d)
    cdf_probability = np.arange(1, len(error_3d) + 1) / len(error_3d)

    diagnostic_fields = (
        "prior_error_3d_m", "posterior_error_3d_m",
        "prefde_innovation_rms_m", "prefde_innovation_max_abs_m",
        "knet_innovation_rms_m", "knet_innovation_max_abs_m",
        "gain_fro_norm", "gain_max_abs",
        "gain_navigation_rows_fro_norm", "gain_accel_bias_rows_fro_norm",
        "gain_gyro_bias_rows_fro_norm",
        "correction_position_norm_m", "correction_velocity_norm_mps",
        "correction_attitude_norm_rad", "correction_accel_bias_norm_mps2",
        "correction_gyro_bias_norm_radps", "estimated_accel_bias_norm_mps2",
        "estimated_gyro_bias_norm_radps", "eta_gyro_norm_radps",
        "eta_accel_norm_mps2", "fixed_ood_fraction",
        "fixed_max_abs_train_ratio", "obs_ood_fraction",
        "obs_max_abs_train_ratio",
    )
    diagnostic_arrays = {
        name: np.asarray([row[name] for row in online_rows], dtype=float)
        for name in diagnostic_fields
    }

    def _first_true_event(mask):
        index = np.flatnonzero(np.asarray(mask, dtype=bool))
        if index.size == 0:
            return None
        i = int(index[0])
        return {
            "row_index": i,
            "time_gpst_s": float(online_time[i]),
            "posterior_error_3d_m": float(error_3d[i]),
        }

    growth_ratio = np.full(len(error_3d), np.nan, dtype=float)
    if len(error_3d) > 1:
        growth_ratio[1:] = error_3d[1:] / np.maximum(error_3d[:-1], 1e-9)

    divergence_diagnostics = {
        "eta_feedback_enabled": bool(ETA_FEEDBACK_ON),
        "first_spectral_radius_gt_1": _first_true_event(
            spectral_radius_knet > 1.0
        ),
        "first_error_gt_100m": _first_true_event(error_3d > 100.0),
        "first_error_gt_1km": _first_true_event(error_3d > 1_000.0),
        "first_error_gt_100km": _first_true_event(error_3d > 100_000.0),
        "first_error_growth_gt_10x": _first_true_event(growth_ratio > 10.0),
        "first_obs_feature_gt_10x_training_absmax": _first_true_event(
            diagnostic_arrays["obs_max_abs_train_ratio"] > 10.0
        ),
        "max_prefde_innovation_abs_m": float(
            np.nanmax(diagnostic_arrays["prefde_innovation_max_abs_m"])
        ),
        "max_knet_innovation_abs_m": float(
            np.nanmax(diagnostic_arrays["knet_innovation_max_abs_m"])
        ),
        "max_gain_fro_norm": float(np.nanmax(diagnostic_arrays["gain_fro_norm"])),
        "max_accel_bias_gain_rows_fro_norm": float(
            np.nanmax(diagnostic_arrays["gain_accel_bias_rows_fro_norm"])
        ),
        "max_gyro_bias_gain_rows_fro_norm": float(
            np.nanmax(diagnostic_arrays["gain_gyro_bias_rows_fro_norm"])
        ),
        "max_position_correction_norm_m": float(
            np.nanmax(diagnostic_arrays["correction_position_norm_m"])
        ),
        "max_accel_bias_correction_norm_mps2": float(
            np.nanmax(diagnostic_arrays["correction_accel_bias_norm_mps2"])
        ),
        "max_gyro_bias_correction_norm_radps": float(
            np.nanmax(diagnostic_arrays["correction_gyro_bias_norm_radps"])
        ),
        "max_estimated_accel_bias_norm_mps2": float(
            np.nanmax(diagnostic_arrays["estimated_accel_bias_norm_mps2"])
        ),
        "max_estimated_gyro_bias_norm_radps": float(
            np.nanmax(diagnostic_arrays["estimated_gyro_bias_norm_radps"])
        ),
        "max_eta_gyro_norm_radps": float(
            np.nanmax(diagnostic_arrays["eta_gyro_norm_radps"])
        ),
        "max_eta_accel_norm_mps2": float(
            np.nanmax(diagnostic_arrays["eta_accel_norm_mps2"])
        ),
        "max_fixed_feature_abs_training_ratio": float(
            np.nanmax(diagnostic_arrays["fixed_max_abs_train_ratio"])
        ),
        "max_observation_feature_abs_training_ratio": float(
            np.nanmax(diagnostic_arrays["obs_max_abs_train_ratio"])
        ),
    }

    np.savez(
        OUTPUT_DIR / "test_evaluation.npz",
        dataset_dir=np.asarray(str(test_dataset_dir)),
        time_gpst_s=online_time,
        estimate_ecef_m=estimate,
        truth_ecef_m=truth_aligned,
        ned_error_m=ned_error,
        error_3d_m=error_3d,
        horizontal_error_m=horizontal_error,
        vertical_error_m=vertical_error,
        hpl_m=hpl,
        vpl_m=vpl,
        spectral_radius_knet=spectral_radius_knet,
        spectral_radius_classical=spectral_radius_classical,
        **diagnostic_arrays,
        error_growth_ratio=growth_ratio,
        rmse_north_east_down_3d_m=rmse,
        cdf_probability=cdf_probability,
        north_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 0])),
        east_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 1])),
        down_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 2])),
        three_d_cdf_error_m=np.sort(error_3d),
    )

    with (
        OUTPUT_DIR / "test_trajectory.csv"
    ).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time_gpst_s",
            "n_input_before_fde",
            "n_k_used_after_fde",
            "n_gnss",
            "n_leo",
            "sat_ids_used",
            "fde_detected",
            "fde_identified_sat_ids",
            "fde_excluded_sat_ids",
            "fde_ambiguous_sat_ids",
            "fde_statistic",
            "fde_threshold",
            "fde_dof",
            "fde_post_exclusion_statistic",
            "fde_post_exclusion_threshold",
            "fde_post_exclusion_dof",
            "fde_post_exclusion_consistent",
            "fde_local_identification_score",
            "fde_estimated_fault_m",
            "fde_hard_exclusion_applied",
            "fde_state_correction_norm",
            "fde_unresolved",
            "fusion_mode",
            "hpl_m",
            "vpl_m",
            "spectral_radius_knet",
            "spectral_radius_classical",
            "x_ecef_m",
            "y_ecef_m",
            "z_ecef_m",
        ])
        for row in online_rows:
            writer.writerow([
                row["time"],
                row["n_input"],
                row["n_total"],
                row["n_gnss"],
                row["n_leo"],
                ";".join(row["sat_ids"]),
                int(row["fde_detected"]),
                ";".join(row["fde_identified_sat_ids"]),
                ";".join(row["fde_excluded_sat_ids"]),
                ";".join(row["fde_ambiguous_sat_ids"]),
                row["fde_statistic"],
                row["fde_threshold"],
                row["fde_dof"],
                row["fde_post_exclusion_statistic"],
                row["fde_post_exclusion_threshold"],
                row["fde_post_exclusion_dof"],
                int(row["fde_post_exclusion_consistent"]),
                row["fde_local_identification_score"],
                row["fde_estimated_fault_m"],
                int(row["fde_hard_exclusion_applied"]),
                row["fde_state_correction_norm"],
                int(row["fde_unresolved"]),
                row["fusion_mode"],
                row["hpl_m"],
                row["vpl_m"],
                row["spectral_radius_knet"],
                row["spectral_radius_classical"],
                *row["position"].tolist(),
            ])

    with (OUTPUT_DIR / "test_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time_gpst_s", "prior_error_3d_m", "posterior_error_3d_m",
            "prefde_innovation_rms_m", "prefde_innovation_max_abs_m",
            "knet_innovation_rms_m", "knet_innovation_max_abs_m",
            "gain_fro_norm", "gain_max_abs",
            "gain_navigation_rows_fro_norm", "gain_accel_bias_rows_fro_norm",
            "gain_gyro_bias_rows_fro_norm",
            "correction_position_norm_m", "correction_velocity_norm_mps",
            "correction_attitude_norm_rad", "correction_accel_bias_norm_mps2",
            "correction_gyro_bias_norm_radps", "estimated_accel_bias_norm_mps2",
            "estimated_gyro_bias_norm_radps", "eta_gyro_norm_radps",
            "eta_accel_norm_mps2", "fixed_ood_fraction",
            "fixed_max_abs_train_ratio", "obs_ood_fraction",
            "obs_max_abs_train_ratio", "spectral_radius_knet",
            "fde_detected", "fde_excluded_sat_ids",
        ])
        for row in online_rows:
            writer.writerow([
                row["time"],
                *[row[name] for name in diagnostic_fields],
                row["spectral_radius_knet"],
                int(row["fde_detected"]),
                ";".join(row["fde_excluded_sat_ids"]),
            ])

    # Snapshot the proposed-method runtime counters before the optional classical
    # diagnostic baseline adds another independent propagation pass.
    proposed_van_loan_stats = dict(_VAN_LOAN_STATS)

    # -------------------------------------------------------------------------
    # Diagnostic-only independent classical TC baseline on the same Data02.
    # It uses the same raw RINEX, TLE realization seed, INS mechanization, noise
    # models, and current FDE ON/OFF setting, but replaces Masked KalmanNet with
    # the conventional Kalman measurement update.  It never feeds the learned
    # trajectory and is not part of Yan's proposed-method result.
    # -------------------------------------------------------------------------
    classical_baseline_rmse = None
    if RUN_CLASSICAL_TEST_BASELINE:
        baseline_leo_simulator = LEODownlinkSimulator(
            tle_provider,
            test_leo_klobuchar,
            seed=TEST_LEO_SEED,
            tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M,
            tx_max_iterations=LEO_TX_MAX_ITERATIONS,
            minimum_elevation_deg=LEO_MIN_ELEVATION_DEG,
            prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG,
            use_ionosphere=USE_IONOSPHERE,
            use_troposphere=USE_TROPOSPHERE,
        )
        baseline_nav = test_initial_nav.copy()
        baseline_P = test_P0.copy()
        baseline_rows = []
        baseline_last_gyro, baseline_last_accel = compensate_imu(
            test_imr.angular_rate_body_radps[0],
            test_imr.acceleration_body_mps2[0],
            baseline_nav.gyroscope_bias_body_radps,
            baseline_nav.accelerometer_bias_body_mps2,
        )
        for baseline_event in build_exact_fusion_timeline(
            test_imr_time, test_fusion_time, through_last_fusion=True
        ):
            if isinstance(baseline_event, PropagationSegment):
                imu_index = baseline_event.imu_index
                dt = baseline_event.end_time_gpst_s - baseline_event.start_time_gpst_s
                baseline_last_gyro, baseline_last_accel = compensate_imu(
                    test_imr.angular_rate_body_radps[imu_index],
                    test_imr.acceleration_body_mps2[imu_index],
                    baseline_nav.gyroscope_bias_body_radps,
                    baseline_nav.accelerometer_bias_body_mps2,
                )
                baseline_nav = mechanize_ecef(
                    baseline_nav, baseline_last_gyro, baseline_last_accel, dt
                )
                F_baseline = build_error_state_dynamics(
                    baseline_nav, baseline_last_accel
                )
                Phi_baseline, Qd_baseline = discretize_process_noise_van_loan(
                    F_baseline, test_Qc, dt
                )
                baseline_P = (
                    Phi_baseline @ baseline_P @ Phi_baseline.T + Qd_baseline
                )
                baseline_P = 0.5 * (baseline_P + baseline_P.T)
                continue

            fusion_index = baseline_event.fusion_index
            t = baseline_event.time_gpst_s
            epoch = test_gnss_epochs[fusion_index]
            baseline_gnss = test_gnss_preprocessor.prepare_epoch(
                epoch, baseline_nav, test_lever_arm_b_m
            )
            baseline_leo = baseline_leo_simulator.simulate_epoch(
                t, test_fusion_antenna_truth_position[fusion_index]
            )
            baseline_measurements = retain_clock_observable_measurements(
                tuple(baseline_gnss) + tuple(baseline_leo)
            )
            if baseline_measurements:
                if FDE_DIA_ON:
                    baseline_fde = ref33_fde_dia_decision(
                        baseline_nav, baseline_P, baseline_measurements,
                        test_lever_arm_b_m, FDE_SIGNIFICANCE_ALPHA
                    )
                    baseline_measurements = baseline_fde.measurements
                    baseline_model = baseline_fde.measurement_model
                else:
                    baseline_model = build_measurement_model(
                        baseline_nav, baseline_measurements, test_lever_arm_b_m
                    )
                if baseline_measurements:
                    baseline_dx, baseline_P, _ = kalman_measurement_update(
                        baseline_P, baseline_model.innovation,
                        baseline_model.H, baseline_model.R
                    )
                    baseline_nav = inject_error_state(baseline_nav, baseline_dx)

            baseline_position = gnss_antenna_position(
                baseline_nav, test_lever_arm_b_m
            )
            baseline_rows.append((
                t, baseline_position.copy(),
                test_fusion_antenna_truth_position[fusion_index].copy(),
                baseline_nav.accelerometer_bias_body_mps2.copy(),
                baseline_nav.gyroscope_bias_body_radps.copy(),
            ))

        if baseline_rows:
            baseline_time = np.asarray([row[0] for row in baseline_rows], dtype=float)
            baseline_estimate = np.stack([row[1] for row in baseline_rows])
            baseline_truth = np.stack([row[2] for row in baseline_rows])
            baseline_accel_bias = np.stack([row[3] for row in baseline_rows])
            baseline_gyro_bias = np.stack([row[4] for row in baseline_rows])
            baseline_ned_error = np.empty_like(baseline_estimate)
            for i, (est, truth_i) in enumerate(zip(baseline_estimate, baseline_truth)):
                lat, lon, _ = ecef_to_llh(truth_i)
                baseline_ned_error[i] = c_ecef_to_ned(lat, lon) @ (est - truth_i)
            baseline_error_3d = np.linalg.norm(baseline_ned_error, axis=1)
            baseline_rmse_ned = np.sqrt(np.mean(baseline_ned_error**2, axis=0))
            baseline_rmse_3d = float(np.sqrt(np.mean(baseline_error_3d**2)))
            classical_baseline_rmse = np.append(
                baseline_rmse_ned, baseline_rmse_3d
            )
            np.savez(
                OUTPUT_DIR / "classical_test_baseline.npz",
                time_gpst_s=baseline_time,
                estimate_ecef_m=baseline_estimate,
                truth_ecef_m=baseline_truth,
                ned_error_m=baseline_ned_error,
                error_3d_m=baseline_error_3d,
                estimated_accelerometer_bias_body_mps2=baseline_accel_bias,
                estimated_gyroscope_bias_body_radps=baseline_gyro_bias,
                rmse_north_east_down_3d_m=classical_baseline_rmse,
            )

    # Yan Fig. 20 branch: Stanford data. The paper does not publish HAL/VAL,
    # therefore no alert-limit classification is invented here.
    with (OUTPUT_DIR / "stanford_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_gpst_s", "horizontal_error_m", "hpl_m", "vertical_error_m", "vpl_m"])
        writer.writerows(zip(online_time, horizontal_error, hpl, vertical_error, vpl))

    finite_rho_knet = spectral_radius_knet[np.isfinite(spectral_radius_knet)]
    finite_rho_classical = spectral_radius_classical[np.isfinite(spectral_radius_classical)]

    summary = {
        "paper_exact": False,
        "intentional_project_differences": {
            "measurement_mode": "pseudorange_only",
            "leo_orbit": "TLE/SGP4_instead_of_STK_HPOP",
            "leo_clock": "ideal_zero_unpublished_simulated_clock_generator",
        },
        "paper_components": {
            "leo_Eq1_to_Eq5": {
                "deterministic_iono_tropo_and_light_time": True,
                "stochastic_variance_policy": "unchanged_from_v6_by_user_request",
                "variance_terms_used": ["ionosphere", "troposphere", "MP_NLOS"],
                "MP_NLOS": "Ref35_Eq18_elevation_only_same_as_v6",
                "sampling": "independent_zero_mean_Gaussian_same_as_v6",
                "not_added_to_LEO_variance": ["URA", "receiver_noise"],
                "paper_difference": "Yan_Eq4_CN0_dependent_MP_NLOS_not_used_in_LEO_branch",
            },
            "INS_Eq6_to_Eq9": (
                "full_15_state_ECEF_error_model_and_measurement_feedback_plus_"
                "Eq8_estimated_bias_compensation_and_lever_arm"
            ),
            "Eq7_Fig8_state_update": {
                "state_order": "[delta_p,delta_v,delta_theta,b_a,b_g]",
                "classical_TC_gain_rows": INS_STATE_DIM,
                "masked_CLA_gain_rows": INS_STATE_DIM,
                "direct_truth_label_rows": SUPERVISED_STATE_DIM,
                "bias_rows_zero_initialized": True,
                "bias_rows_frozen": False,
                "bias_supervision": (
                    "no_direct_bias_truth_available_no_invented_bias_loss"
                ),
            },
            "MaskedCLA_Eq10_to_Eq29": (
                "implemented_with_Fig8_pool_flatten_and_intra_epoch_masked_LSTM"
            ),
            "training_Eq30_to_Eq32": "Yan_fixed_Data01_intra_epoch_masked_LSTM_ref15_alternating",
            "Fig8_eta": {
                "enabled": True,
                "feedback_enabled_in_this_run": bool(ETA_FEEDBACK_ON),
                "dimension": 6,
                "interpretation": "[epsilon_g(3),epsilon_a(3)]",
                "training": "future_navigation_state_Eq30_supervision_through_Eq8_INS_sensitivity_completion",
            },
            "FDE_Eq33_to_Eq34": {
                "enabled": bool(FDE_DIA_ON),
                "single_code_switch": "FDE_DIA_ON = True/False",
                "raw_innovation_before_KG": bool(FDE_DIA_ON),
                "detection_identification": "Ref33_DIA",
                "hard_exclusion_before_KNet": True,
                "Eq34_DIA_state_covariance": "computed_for_identified_fault_mode",
                "old_nonpaper_post_exclusion_rejection_loop": False,
                "alpha": FDE_SIGNIFICANCE_ALPHA,
            },
            "integrity": {
                "HPL_VPL": (
                    "computed_from_DIA_parameter_estimation_covariance_in_NED"
                    if FDE_DIA_ON else "computed_from_learned_update_covariance_in_NED"
                ),
                "stanford_data_csv": True,
                "alert_limits": "not_invented_paper_does_not_publish_numeric_HAL_VAL",
            },
            "stability": {
                "online_closed_loop_spectral_radius": True,
                "exact_75_25_stress_generator": "not_reconstructed_distribution_parameters_unpublished",
            },
        },
        "unavoidable_unpublished_completions": [
            "offline_Eq10_to_Eq14_history_generator_conventional_TC_reference_pass",
            "causal_boundary_timing_for_state_innovation_and_state_residual",
            "exact_CNN_tensorization_pooling_and_FC_dimensions",
            "Data01_only_feature_standardization_Yan_does_not_publish_input_scaling",
            "eta_timing_Yan_unpublished_future_state_supervision_completion_no_direct_eta_MSE",
            "mini_batch_gradient_accumulation_size_Yan_batching_unpublished",
            "intra_epoch_LSTM_sequence_interpretation_of_Yan_masked_Xbar_k",
            "LEO_variance_intentionally_kept_as_v6_instead_of_Yan_Eq4",
            "random_LEO_masking_angle_distribution_from_Ref43",
            "FDE_false_alarm_probability_and_exact_fault_mode_matrices",
            "learned_gain_covariance_bookkeeping",
            "direct_bias_truth_for_the_six_IMU_bias_states_unavailable_"
            "no_fabricated_bias_supervision_added",
        ],
        "dataset_protocol": {
            "training_dataset": str(TRAIN_DATASET_DIR),
            "test_dataset": str(test_dataset_dir),
            "test_used_for_training": False,
        },
        "training": {
            "samples": int(len(dataset)),
            "Nmax": int(nmax),
            "epochs": int(TRAINING_EPOCHS),
            "sequence_chunk_length_completion": int(BATCH_SIZE),
            "gamma_l2_completion": float(GAMMA_L2),
            "feature_standardization": {
                "enabled": bool(FEATURE_STANDARDIZATION_ON),
                "fit_scope": "Data01_only",
                "targets_normalized": False,
                "innovation_for_state_update_normalized": False,
                "paper_status": "unpublished_numerical_conditioning_completion",
            },
            "alternating_partition": {
                "psi_representation": "masked_conv_masked_lstm_masked_attention",
                "theta_filter_outputs": "gain_head_eta_head",
                "paper_status": "Yan_partition_unpublished_Ref15_guided_completion",
            },
            "sequence_training": {
                "scope": "Data01_fixed_Eqs18_to_20",
                "gradient": "per_epoch_masked_LSTM_sequence_backprop",
                "chunk_length": int(BATCH_SIZE),
                "neural_recurrent_state": (
                    "reset_each_fusion_epoch_intra_epoch_masked_sequence"
                ),
                "feature_recursion": "none_fixed_Yan_X_M_Y_only",
                "paper_status": (
                    "Ref15_guided_training_completion_Yan_batching_unpublished"
                ),
            },
            "final_Eq30": float(final_training_metrics["eq30"]),
            "final_eta_future_Eq30": float(
                final_training_metrics["eta_future_eq30"]
            ),
            "final_temporal_state_objective": float(
                final_training_metrics["temporal_state"]
            ),
            "final_position_rmse_m": float(final_training_metrics["position_rmse_m"]),
            "final_eta_future_position_rmse_m": float(
                final_training_metrics["eta_future_position_rmse_m"]
            ),
            "final_selected_training_stage": selected_training_stage,
            "final_selected_training_eq30": float(best_val_state_loss),
            "final_selected_training_position_rmse_m": float(
                best_val_position_rmse_m
            ),
        },
        "test": {
            "epochs": int(len(online_rows)),
            "causal_warm_start": test_warm_start,
            "neural_recurrent_state": (
                "reset_each_online_fusion_epoch_intra_epoch_masked_sequence"
            ),
            "rmse_ned3d_m": [float(v) for v in rmse],
            "eta_feedback_enabled": bool(ETA_FEEDBACK_ON),
            "full_15_state_bias_feedback_enabled": True,
            "divergence_diagnostics": divergence_diagnostics,
            "classical_baseline_rmse_ned3d_m": (
                [float(v) for v in classical_baseline_rmse]
                if classical_baseline_rmse is not None else None
            ),
            "fde_stats": test_fde_stats,
            "spectral_radius_knet_median": (
                float(np.median(finite_rho_knet)) if finite_rho_knet.size else None
            ),
            "spectral_radius_classical_median": (
                float(np.median(finite_rho_classical))
                if finite_rho_classical.size else None
            ),
        },
        "runtime_checks": {
            "van_loan_fast_calls": int(proposed_van_loan_stats["taylor_calls"]),
            "van_loan_exact_calls": int(proposed_van_loan_stats["exact_fallback_calls"]),
            "van_loan_max_validation_phi_abs": float(
                proposed_van_loan_stats["max_validation_phi_abs"]
            ),
            "van_loan_max_validation_qd_abs": float(
                proposed_van_loan_stats["max_validation_qd_abs"]
            ),
        },
    }
    (
        OUTPUT_DIR / "summary.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINISHED INDEPENDENT TEST ===")
    print("test dataset:", test_dataset_dir)
    print("test RMSE [N, E, D, 3D] m:", rmse)
    print("eta_k -> Eq.(8) feedback:", "ON" if ETA_FEEDBACK_ON else "OFF")
    print("divergence diagnostics:", json.dumps(divergence_diagnostics, indent=2))
    if classical_baseline_rmse is not None:
        print(
            "classical Data02 baseline RMSE [N, E, D, 3D] m:",
            classical_baseline_rmse,
        )
    print("outputs:", OUTPUT_DIR.resolve())
    print(
        "Van Loan fast/fallback calls:",
        proposed_van_loan_stats["taylor_calls"],
        "/",
        proposed_van_loan_stats["exact_fallback_calls"],
    )
    print(
        "Van Loan max ||A*dt||_1 / remainder bound:",
        f"{proposed_van_loan_stats['max_norm_1']:.6g}",
        "/",
        f"{proposed_van_loan_stats['max_taylor_remainder_bound']:.3e}",
    )
    print(
        "Van Loan validation max |dPhi| / |dQd|:",
        f"{proposed_van_loan_stats['max_validation_phi_abs']:.3e}",
        "/",
        f"{proposed_van_loan_stats['max_validation_qd_abs']:.3e}",
    )
    print(
        "LEO prefilter rejected train/test:",
        f"{leo_simulator.prefilter_rejected}/{leo_simulator.prefilter_checked}",
        "/",
        f"{test_leo_simulator.prefilter_rejected}/{test_leo_simulator.prefilter_checked}",
    )
    if FDE_DIA_ON:
        print(
            "FDE synthetic core self-check empirical/requested false alarm:",
            f"{FDE_STATISTICAL_SELF_CHECK['empirical_false_alarm']:.6g}",
            "/",
            f"{FDE_STATISTICAL_SELF_CHECK['requested_alpha']:.6g}",
            "ambiguity_check=",
            FDE_STATISTICAL_SELF_CHECK["two_measurement_ambiguity_detected"],
        )
        print(
            "FDE test checked/detected/identified/hard-exclusion/unresolved:",
            test_fde_stats["epochs_checked"],
            "/",
            test_fde_stats["epochs_detected"],
            "/",
            test_fde_stats["identified_fault_modes"],
            "/",
            test_fde_stats["hard_exclusion_epochs"],
            "/",
            test_fde_stats["unresolved_epochs"],
        )
        print(
            "FDE test max statistic/threshold ratio / max estimated fault / max local ID score:",
            f"{test_fde_stats['max_statistic_to_threshold_ratio']:.6g}",
            "/",
            f"{test_fde_stats['max_abs_estimated_fault_m']:.3f} m",
            "/",
            f"{test_fde_stats['max_local_identification_score']:.6g}",
        )
    else:
        print("FDE ablation: OFF -- detection, identification, exclusion and DIA were bypassed.")
    if finite_rho_knet.size:
        print(
            "median spectral radius KNet/classical:",
            f"{np.median(finite_rho_knet):.6g}",
            "/",
            (
                f"{np.median(finite_rho_classical):.6g}"
                if finite_rho_classical.size else "nan"
            ),
        )
    print(
        "median HPL/VPL [m]:",
        f"{np.nanmedian(hpl):.3f}",
        "/",
        f"{np.nanmedian(vpl):.3f}",
    )
    print(f"total wall runtime: {perf_counter() - _run_wall_start:.3f} s")
