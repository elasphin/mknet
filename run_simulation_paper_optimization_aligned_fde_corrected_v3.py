"""Single-file GNSS/LEO/INS + Masked KalmanNet reproduction.

This file inlines the former config.py, dataset_io.py, tc.py, and masked_cla.py
so the complete current experiment can be inspected and executed from one file.

Dataset protocol:
- Data01 is split chronologically into training and validation subsets.
- A distinct second SmartPNT-Pos dataset is used only for online testing/evaluation.
  Test statistics never enter training, validation, normalization, or model selection.
- Validation/early stopping are reproducibility safeguards: the paper does not publish
  an internal validation protocol or early-stopping rule, so these are explicitly marked
  project completions rather than paper-exact training details.

Intentional project differences from Yan et al. (2026) that are still retained:
- pseudorange-only (no pseudorange-rate/Doppler)
- TLE/SGP4 LEO orbit instead of STK/HPOP
- ideal LEO satellite/receiver clock terms because the paper does not publish a
  reproducible LEO clock-error generator for the simulated constellation
- FDE is enabled in the online closed-loop path. Detection/identification follow
  Yan et al. Eq. (33) together with the DIA hypothesis-testing formulation of
  Ref. [33] Eqs. (4)-(8). The INS-predicted innovation is tested before the
  learned KG is applied. Each pseudorange is represented by a 1-D
  measurement-fault hypothesis.
- A detected and uniquely identified large fault is hard-excluded BEFORE the
  Masked-CLA/KalmanNet forward pass, and the measurement model is rebuilt from
  the remaining observations. This follows Yan et al.'s stated online sequence
  (fault detection -> fault elimination -> state update) and their description
  of FDE as the hard-exclusion layer. The previous implementation that applied
  Yan Eq. (34) directly to a learned-KG posterior has been removed: Ref. [33]'s
  Eq. (34)/Appendix Eq. (39) covariance identity is derived for the classical
  Kalman posterior, for which the nominal estimation error is uncorrelated with
  the innovation. That identity does not hold in general for an arbitrary
  learned gain without additional cross-covariance terms.
- If the receiver-clock projection makes two or more single-pseudorange fault
  hypotheses structurally indistinguishable (for example, exactly two
  observations from one projected GNSS constellation), the FDE result is marked
  unresolved. No suspect observation from that epoch is sent to KalmanNet; the
  navigation solution remains INS-propagated for that fusion epoch.
- Because this project eliminates GPS/BDS receiver clocks by a WLS projection,
  Ref. [33]'s full-rank chi-square test is evaluated in the effective projected
  residual subspace (degrees of freedom = rank(Q_nu_nu)). Yan et al. do not
  publish the false-alarm probability; alpha=1e-3 is adopted from the numerical
  DIA examples of Ref. [33] as an explicit completion. Numerical rank and local
  fault-direction tests use machine-precision-relative tolerances only; no
  dimensioned absolute cutoff is used in the Ref. [33] statistics.
- GPS/BDS receiver clocks are epoch-wise WLS nuisance parameters; their fitted
  directions are projected consistently out of innovation, H, and R
- the supplied IE truth exports do not contain accelerometer/gyro bias truth;
  both classical history/warm-start and learned updates therefore correct only the
  9 position/velocity/attitude states and keep the 6 nominal bias states frozen
- Eqs. (10)-(15) are implemented causally: current innovation/IMU increments are
  combined with the previous completed fusion state's residual/innovation terms.
  The paper does not publish enough implementation detail to remove this causal
  timing completion without introducing current-posterior information leakage.
- The Masked CNN/LSTM/attention core follows Eqs. (22)-(29) and Table III.
  The exact CNN tensorization, the pooling block drawn in Fig. 8, and the exact FC
  head dimensions are not fully specified by the paper and remain explicit
  implementation choices.
- Fig. 8 explicitly requires a second network output eta_k for inertial-navigation
  measurement-error estimation. Fig. 2 shows this quantity on a feedback path from
  KF-NET Online to the IMU ``Error compensation`` block, while the learned KG follows
  the separate ``State Update`` / corrected-P,V,A path. Eq. (8) is the only explicit
  IMU compensation equation in the paper and subtracts residual gyro/accelerometer
  error terms epsilon_g and epsilon_a. This reproduction therefore uses the declared,
  paper-guided completion
      eta_k = [epsilon_gx, epsilon_gy, epsilon_gz,
               epsilon_ax, epsilon_ay, epsilon_az],
  with gyro components in rad/s and accelerometer components in m/s^2.
  The paper does not publish eta_k sample timing between fusion epochs. To preserve
  the feedback topology of Fig. 2, eta_k is NOT added to the same-epoch Kalman state
  correction and is NOT replayed backward over the preceding IMU interval. Instead,
  after fusion epoch k it is held constant (zero-order hold) and supplied to Eq. (8)
  for every IMU propagation sample until the next usable network output. Offline,
  the eta branch is constrained by the navigation-state error at the following usable
  fusion epoch; no direct sensor-error pseudo-label is fabricated.

- Yan et al. state that the network parameters are optimized alternately and cite
  Ref. [15]. Ref. [15] alternates two differentiable trainable blocks while using the
  SAME final state-estimation objective: first optimize the filter block while the
  representation block is frozen, then optimize the representation block while the
  filter block is frozen. Yan et al. do not publish the exact partition of their
  Masked-CLA parameters. The closest reproducible mapping used here is therefore:
      representation block psi = Masked CNN + Masked LSTM + Masked Attention
      filter/output block theta = Masked FC gain head + eta head
  Each outer epoch performs theta then psi, matching Algorithm 2 ordering in Ref. [15].
  Both phases minimize the same Eq. (30)-type navigation-state error averaged over the
  available current/following supervised state instants plus Yan Eq. (32) L2 penalty.
  No separate KG-vs-eta loss weights, LR scheduler, or gradient clipping are introduced.
  This parameter partition is an explicit paper-guided completion because Yan et al.
  do not publish which Masked-CLA submodules form the alternating blocks.
- The supplied truth does not contain accelerometer/gyro bias labels. Therefore
  Eq. (30) is reproduced as a partial-state supervised loss on the available
  9 position/velocity/attitude error-state components; the 6 bias-gain rows are
  deterministically zero rather than trained against invented targets.
- orchestration.py was audited as a workflow reference. Its strict GPST checks
  and exact fusion-event scheduler are adopted here. Its stale single-truth,
  single-dataset online path, old ideal-atmosphere/iid-LEO-noise path, and
  position-only supervised pipeline are NOT adopted because they regress the
  scientific fixes already present in this run.

The LEO pseudorange atmosphere/error model follows Yan et al. Eq. (1)-(3).
Per the current project choice, Ref. [35] supplies the ionospheric, tropospheric,
and elevation-dependent multipath residual-error standard deviations. URA and
receiver noise are intentionally omitted. Independent zero-mean stochastic
realizations with those standard deviations are injected into the simulated LEO
pseudorange so that its injected-error covariance is consistent with sigma_code_m.

Runtime optimizations preserve the scientific state definition, units, masks, stochastic
seeds/draw ordering, and float precision.  Most changes are algebraically exact.  The
one deliberate speed/accuracy trade-off is Van Loan matrix-exponential evaluation: for
small ||A*dt||_1 it uses a 10th-order Taylor series with a conservative norm gate and
SciPy expm fallback; the configured gate keeps the exponential truncation bound near
machine precision for normal 200 Hz IMU steps.  LEO receive-time elevation prefiltering
uses a guard band, followed by the original exact transmit-time/final-elevation test for
all candidates that could plausibly cross the mask.
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
from torch.utils.data import DataLoader, TensorDataset
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
# The available truth supervises position/velocity/attitude only. Biases remain
# part of the propagated 15-state uncertainty model but are not measurement-updated
# in either the classical history/warm-start or the learned online pass.
NAVIGATION_CORRECTION_DIM = 9
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
    ordinary bias-only compensation used by the classical history/reference pass.
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
    """Propagate one IMU interval with a constant eta_k correction.

    eta_k is interpreted from Eq. (8) as
        [epsilon_gx, epsilon_gy, epsilon_gz,
         epsilon_ax, epsilon_ay, epsilon_az].
    In the online path eta_k is produced at fusion epoch k and held over the
    subsequent IMU propagation interval until the next usable KF-NET output.
    Zero-order hold is a declared timing completion because Yan et al. do not
    publish the eta_k sample/update rate between fusion epochs.
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
    """Local d([dp,dv,dtheta])/d(eta) through Eq. (8) + INS mechanization.

    This Jacobian is used only in offline teacher-forced training of the forward
    eta feedback branch. It maps a constant eta_k applied over the *future* IMU
    interval k->k+1 to the resulting navigation-state change at the interval end.
    The online navigation path does not use this Jacobian: it applies eta_k
    directly in Eq. (8) before every subsequent INS mechanization step.

    A one-sided local numerical derivative is sufficient because the project uses
    a first-order error-state approximation for this auxiliary training path. The
    perturbation magnitudes are numerical differentiation steps only; they do not
    constrain or regularize eta_k.
    """
    if not interval_segments:
        return np.zeros((SUPERVISED_STATE_DIM, IMU_ERROR_DIM), dtype=float)

    zero_eta = np.zeros(IMU_ERROR_DIM)
    base_end = _replay_imu_interval_with_eta(
        start_nav,
        interval_segments,
        imr,
        zero_eta,
    )

    if reference_end_nav is not None:
        replay_mismatch = _navigation_delta_9(reference_end_nav, base_end)
        if (
            np.linalg.norm(replay_mismatch[:3]) > 1e-5
            or np.linalg.norm(replay_mismatch[3:6]) > 1e-8
            or np.linalg.norm(replay_mismatch[6:9]) > 1e-10
        ):
            raise RuntimeError(
                "zero-eta IMU interval replay does not reproduce the current INS prior; "
                f"delta9={replay_mismatch}"
            )

    steps = np.array(
        [ETA_FD_GYRO_STEP_RADPS] * 3 + [ETA_FD_ACCEL_STEP_MPS2] * 3,
        dtype=float,
    )
    jacobian = np.zeros((SUPERVISED_STATE_DIM, IMU_ERROR_DIM), dtype=float)

    for axis in range(IMU_ERROR_DIM):
        eta = np.zeros(IMU_ERROR_DIM)
        eta[axis] = steps[axis]
        perturbed_end = _replay_imu_interval_with_eta(
            start_nav,
            interval_segments,
            imr,
            eta,
        )
        jacobian[:, axis] = (
            _navigation_delta_9(base_end, perturbed_end) / steps[axis]
        )

    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("eta interval state sensitivity contains non-finite values")
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

        return PseudorangeMeasurement(
            raw.sat_id,
            raw.constellation,
            float(raw.pseudorange_m),
            satellite_position_rx,
            satellite_clock_bias_s,
            float(ionosphere),
            float(troposphere),
            5.0,  # GNSS R weighting remains a declared project completion; Yan et al. use real GNSS data.
            float(elevation),
            raw.cn0_dbhz,
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
    F[3:6, 9:12] = -C
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    F[6:9, 12:15] = C
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
    """FDE decision from the pre-update INS-predicted innovation.

    ``tested_*`` contains the raw pre-KalmanNet set on which Detection and
    Identification are performed. ``measurements``/``measurement_model`` contain
    only the observations allowed to reach the estimator after hard exclusion.
    """

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


def _ref33_psd_pinv_and_rank(matrix: Array) -> tuple[Array, int]:
    """Moore-Penrose inverse/rank in the effective residual subspace.

    Ref. [33] assumes nonsingular Q_nu_nu. This project has already projected
    GPS/BDS receiver-clock nuisance directions out of innovation, H and R, so
    Q_nu_nu is intentionally singular in those removed directions. The same
    weighted-norm test is therefore evaluated on rank(Q_nu_nu).

    This is a project-specific completion required by the existing clock model,
    not a claim about the implementation used by Yan et al.
    """
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
    tolerance = (
        100.0
        * np.finfo(float).eps
        * max(matrix.shape)
        * scale
    )
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise FloatingPointError(
            "Q_nu_nu is materially indefinite in Ref. [33] FDE"
        )

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
    """Global Detection test from Ref. [33] Eq. (6).

    Q_nu_nu = H P^- H^T + R
    T_D      = nu^T Q_nu_nu^+ nu
    H0 is rejected when T_D exceeds the upper-tail chi-square threshold.

    Yan et al. Eq. (33), as printed, omits the inverse on the innovation
    covariance. Taken literally that expression is not the chi-square weighted
    residual test needed to set Td from a false-alarm probability. Yan et al.
    explicitly refer to Ref. [33] for Q_nu_nu, nu and Li, so this implementation
    follows the mathematically consistent Ref. [33] weighted norm.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("FDE significance alpha must lie strictly between 0 and 1")

    P = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(measurement_model.H, dtype=float)
    R = np.asarray(measurement_model.R, dtype=float)
    nu = np.asarray(measurement_model.innovation, dtype=float)

    if P.shape != (INS_STATE_DIM, INS_STATE_DIM):
        raise ValueError("prior covariance has wrong shape for FDE")
    if H.shape != (len(nu), INS_STATE_DIM) or R.shape != (len(nu), len(nu)):
        raise ValueError("measurement dimensions are inconsistent in FDE")

    Q_nu_nu = H @ P @ H.T + R
    Q_nu_nu = 0.5 * (Q_nu_nu + Q_nu_nu.T)
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
    """Identification from Ref. [33] Eq. (7), q_i=1 per pseudorange.

    Yan et al. do not publish their exact fault-mode matrices C_i. For this
    pseudorange-only reproduction the transparent completion is one 1-D
    measurement-fault hypothesis per pseudorange. After receiver-clock nuisance
    elimination a raw fault e_j appears in the tested residual space as

        C_j = clock_projector @ e_j.

    For q_j=1 the local statistic is

        t_j = (C_j^T Q^+ nu)^2 / (C_j^T Q^+ C_j),

    and Ref. [33] Eq. (7) uses CDF_chi2_1(t_j) as the comparable identification
    score. The score is *not* a posterior probability that satellite j is faulty.

    Structural/numerical ties are returned as ambiguous rather than silently
    selecting the first satellite. This is essential after clock projection:
    with exactly two observations from one projected GNSS constellation, the two
    single-measurement fault directions span the same 1-D residual subspace and
    cannot be uniquely identified.
    """
    nu = np.asarray(measurement_model.innovation, dtype=float)
    projector = np.asarray(measurement_model.clock_projector, dtype=float)

    if projector.shape != (len(nu), len(nu)):
        raise ValueError("clock projector has wrong shape in FDE")

    qpinv_norm_2 = float(np.linalg.norm(Q_pinv, ord=2))
    machine_eps = np.finfo(float).eps
    candidates = []

    for j in range(len(nu)):
        C_i = projector[:, j].astype(float, copy=False)
        denominator = float(C_i @ Q_pinv @ C_i)
        denominator_scale = qpinv_norm_2 * float(C_i @ C_i)
        denominator_tolerance = (
            100.0
            * machine_eps
            * max(denominator_scale, np.finfo(float).tiny)
        )
        if (
            not math.isfinite(denominator)
            or denominator <= denominator_tolerance
        ):
            continue

        numerator = float(C_i @ Q_pinv @ nu)
        local_statistic = float((numerator * numerator) / denominator)
        local_score = float(chi2.cdf(local_statistic, df=1))
        estimated_fault_m = float(numerator / denominator)

        candidates.append({
            "index": int(j),
            "C_i": C_i.copy(),
            "denominator": denominator,
            "local_statistic": local_statistic,
            "local_score": local_score,
            "estimated_fault_m": estimated_fault_m,
        })

    if not candidates:
        return None

    # chi-square CDF is monotonic but saturates numerically near one. Compare the
    # underlying local statistic when selecting the maximum, while still reporting
    # the Eq. (7) transformed score.
    best = max(candidates, key=lambda item: item["local_statistic"])
    best_stat = float(best["local_statistic"])
    best_C = np.asarray(best["C_i"], dtype=float)
    best_den = float(best["denominator"])

    subspace_tolerance = 1000.0 * machine_eps * max(1, len(nu))
    statistic_tolerance = (
        1000.0 * machine_eps * max(1.0, abs(best_stat))
    )
    ambiguous_indices = []
    for candidate in candidates:
        C_j = np.asarray(candidate["C_i"], dtype=float)
        den_j = float(candidate["denominator"])
        cross = abs(float(best_C @ Q_pinv @ C_j))
        cosine = cross / math.sqrt(max(best_den * den_j, np.finfo(float).tiny))
        cosine = min(max(cosine, 0.0), 1.0)
        same_subspace = (1.0 - cosine) <= subspace_tolerance
        tied_statistic = (
            abs(float(candidate["local_statistic"]) - best_stat)
            <= statistic_tolerance
        )
        if same_subspace or tied_statistic:
            ambiguous_indices.append(int(candidate["index"]))

    best = dict(best)
    best["ambiguous_indices"] = tuple(sorted(set(ambiguous_indices)))
    best["unique"] = len(best["ambiguous_indices"]) == 1
    return best


def ref33_fde_dia_decision(
    nav: NavigationState,
    prior_covariance: Array,
    measurements,
    lever_arm_b_m: Array,
    alpha: float,
    enabled: bool = True,
) -> Ref33FDEResult:
    """Run Ref. [33] D/I on raw innovation and hard-exclude before KalmanNet.

    Detection/Identification are computed from the untouched INS-predicted
    innovation, as required by Yan et al. If H0 is rejected and a single fault
    hypothesis is uniquely identified, that pseudorange is removed and the
    measurement model is rebuilt before any learned gain is evaluated.

    The former post-hoc application of Yan Eq. (34) to a KalmanNet posterior is
    deliberately not used. Ref. [33]'s Eq. (34)/Appendix Eq. (39) covariance
    identity is tied to the classical Kalman posterior. Combining its L_i with an
    arbitrary learned gain generally requires nonzero cross-covariance terms and
    is therefore not justified by either source paper.

    If Identification is structurally ambiguous, the epoch is marked unresolved
    and no suspect measurement is passed to the estimator. The navigation state
    then remains INS-propagated for that fusion epoch.
    """
    current = tuple(retain_clock_observable_measurements(measurements))
    tested_model = build_measurement_model(nav, current, lever_arm_b_m)
    empty_model = build_measurement_model(nav, (), lever_arm_b_m)

    if not current:
        return Ref33FDEResult(
            (), tested_model, (), empty_model,
            False, (), (), (), (),
            0.0, float("inf"), 0, False,
            0.0, 0.0, np.zeros((0, 0)),
        )

    statistic, threshold, dof, Q_nu_nu, Q_pinv = _ref33_detection_terms(
        prior_covariance, tested_model, alpha
    )

    if not enabled or dof == 0 or statistic <= threshold:
        return Ref33FDEResult(
            current, tested_model, current, tested_model,
            False, (), (), (), (),
            statistic, threshold, dof, False,
            0.0, 0.0, Q_nu_nu,
        )

    identification = _ref33_identify_single_pseudorange_fault(
        tested_model, Q_pinv
    )
    if identification is None:
        return Ref33FDEResult(
            current, tested_model, (), empty_model,
            True, (), (), tuple(m.sat_id for m in current), (),
            statistic, threshold, dof, True,
            0.0, 0.0, Q_nu_nu,
        )

    ambiguous_indices = tuple(identification["ambiguous_indices"])
    if not bool(identification["unique"]):
        ambiguous_sat_ids = tuple(current[j].sat_id for j in ambiguous_indices)
        return Ref33FDEResult(
            current, tested_model, (), empty_model,
            True, (), (), ambiguous_sat_ids, (),
            statistic, threshold, dof, True,
            float(identification["local_score"]),
            float(identification["estimated_fault_m"]),
            Q_nu_nu,
        )

    j = int(identification["index"])
    selected = current[j]
    remaining = current[:j] + current[j + 1:]
    usable = tuple(retain_clock_observable_measurements(remaining))
    filtered_model = build_measurement_model(nav, usable, lever_arm_b_m)

    return Ref33FDEResult(
        current,
        tested_model,
        usable,
        filtered_model,
        True,
        (selected.sat_id,),
        (selected.constellation,),
        (),
        (selected.sat_id,),
        statistic,
        threshold,
        dof,
        False,
        float(identification["local_score"]),
        float(identification["estimated_fault_m"]),
        Q_nu_nu,
    )


def _new_fde_stats() -> dict:
    return {
        "epochs_checked": 0,
        "epochs_detected": 0,
        "identified_fault_modes": 0,
        "hard_exclusion_epochs": 0,
        "excluded_measurements": 0,
        "unresolved_epochs": 0,
        "max_statistic_to_threshold_ratio": 0.0,
        "max_abs_estimated_fault_m": 0.0,
        "max_local_identification_score": 0.0,
        "identified_by_constellation": {"G": 0, "C": 0, "L": 0},
    }


def _accumulate_fde_stats(stats: dict, result: Ref33FDEResult) -> None:
    stats["epochs_checked"] += 1
    stats["epochs_detected"] += int(result.detected)
    stats["identified_fault_modes"] += len(result.identified_sat_ids)
    stats["hard_exclusion_epochs"] += int(bool(result.excluded_sat_ids))
    stats["excluded_measurements"] += len(result.excluded_sat_ids)
    stats["unresolved_epochs"] += int(result.unresolved)

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
    """Classical history/warm-start update consistent with the learned 9-row policy.

    The GPS/BDS clock projection makes S singular in the removed clock directions,
    so a Moore-Penrose inverse is required. Bias gain rows are forced to zero so
    the classical history and warm-start use the same nominal bias policy as the
    learned online pass; the full 15-state covariance is still propagated.
    """
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)

    # No accelerometer/gyro bias truth is available for the learned model. Keep
    # nominal bias estimates frozen in both classical and learned measurement
    # updates instead of creating a train/test bias-state mismatch.
    K[NAVIGATION_CORRECTION_DIM:, :] = 0.0

    dx = K @ innovation
    I_KH = np.eye(INS_STATE_DIM) - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    return dx, 0.5 * (P_post + P_post.T), K


def learned_gain_covariance_update(
    prior_covariance: Array,
    learned_gain: Array,
    measurement_jacobian: Array,
    measurement_covariance: Array,
) -> Array:
    """Joseph covariance bookkeeping for a KalmanNet-produced gain.

    IMPORTANT:
    - The Masked CLA does NOT use ``R`` to compute its Kalman gain.
    - The navigation correction remains ``dx = K_net @ innovation``.
    - ``R`` appears here only to propagate a covariance estimate for diagnostics
      and future FDE/integrity work. Yan et al. do not publish an explicit
      learned-gain covariance-update equation, so this is a declared
      STANDARD-COMPLETION rather than a paper-exact step.
    """
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    learned_gain = np.asarray(learned_gain, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)

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
    ):
        raise ValueError("non-finite P/K/H/R in learned covariance update")

    update_matrix = np.eye(INS_STATE_DIM) - learned_gain @ measurement_jacobian
    posterior_covariance = (
        update_matrix @ prior_covariance @ update_matrix.T
        + learned_gain @ measurement_covariance @ learned_gain.T
    )
    return 0.5 * (posterior_covariance + posterior_covariance.T)


def inject_error_state(nav: NavigationState, dx: Array) -> NavigationState:
    dx = np.asarray(dx, dtype=float).reshape(15)
    out = nav.copy()
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _rotation(
        _so3_exponential(ATTITUDE_FEEDBACK_SIGN * dx[6:9])
        @ out.body_to_ecef_dcm
    )
    out.accelerometer_bias_body_mps2 += dx[9:12]
    out.gyroscope_bias_body_radps += dx[12:15]
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
    """Ref. [35], Eqs. (15)-(16): 1-sigma ionospheric residual error [m]."""
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    if latitude_deg <= 20.0:
        sigma_vertical_m = 9.0
    elif latitude_deg <= 55.0:
        sigma_vertical_m = 4.5
    else:
        sigma_vertical_m = 6.0
    earth_mean_radius_m = 6_378_140.0
    ionosphere_mean_height_m = 350_000.0
    mapping_denominator = 1.0 - (
        earth_mean_radius_m * math.cos(float(elevation_rad))
        / (earth_mean_radius_m + ionosphere_mean_height_m)
    ) ** 2
    return float(sigma_vertical_m / math.sqrt(max(mapping_denominator, 1e-15)))


def ref35_troposphere_sigma_m(elevation_rad: float) -> float:
    """Ref. [35], Eq. (17): Black-Eisner mapped 1-sigma tropo error [m]."""
    mapping = 1.001 / math.sqrt(0.002001 + math.sin(float(elevation_rad)) ** 2)
    return float(mapping * 0.12)


def ref35_multipath_sigma_m(elevation_rad: float) -> float:
    """Ref. [35], Eq. (18): elevation-dependent multipath 1-sigma error [m].

    The equation in Ref. [35] is
        sigma_mp = 0.13 + 0.53 * exp(-psi / 10),
    where psi is the satellite elevation angle in degrees.

    Ref. [35] models multipath, not NLOS separately. In this project this term is
    used as the requested MP/NLOS stochastic-error model. Receiver noise is not
    modeled.
    """
    elevation_deg = float(np.rad2deg(elevation_rad))
    if not math.isfinite(elevation_deg):
        raise ValueError("finite elevation is required")
    if elevation_deg < 0.0:
        raise ValueError("multipath model requires a nonnegative elevation angle")
    return float(0.13 + 0.53 * math.exp(-elevation_deg / 10.0))


class LEODownlinkSimulator:
    """Yan et al. Eqs. (1)-(5) with TLE/SGP4 orbit and ideal LEO clocks.

    Ref. [35] supplies ionospheric, tropospheric, and multipath residual-error
    standard deviations. URA and receiver noise are intentionally omitted.

    The source papers specify standard deviations but do not publish a unique
    stochastic sampling law for these residuals. This implementation therefore
    uses independent zero-mean Gaussian realizations as an explicit simulation
    assumption. The same component variances are summed to form sigma_code_m,
    making the simulated residual covariance and R internally consistent.
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

        # Residual-error standard deviations. Ref. [35] defines these as
        # pseudorange estimation-error sigmas, not as the deterministic modeled
        # atmospheric delays themselves. The LEO-height factor follows Yan et al.
        # Eq. (2) for the ionospheric path inside the ionosphere.
        iono_sigma_m = (
            ionosphere_path_scale * ref35_ionosphere_sigma_m(elevation, receiver_lat)
            if self.use_ionosphere
            else 0.0
        )
        tropo_sigma_m = (
            ref35_troposphere_sigma_m(elevation)
            if self.use_troposphere
            else 0.0
        )

        # User-requested simplification: Ref. [35], Eq. (18), supplies the
        # elevation-dependent multipath sigma. Ref. [35] does not model NLOS
        # separately, so this term is used as the combined MP/NLOS simplification.
        mp_sigma_m = ref35_multipath_sigma_m(elevation)

        # Yan et al. Eq. (3) motivates summing component variances. URA and
        # receiver-noise terms are intentionally omitted in this project.
        #
        # The references provide sigma models but not a unique sampling
        # distribution. Independent N(0, sigma^2) realizations are therefore an
        # explicit simulation assumption. Injecting the same three components
        # whose variances form R removes the previous inconsistency in which
        # ionosphere/troposphere variances were present in R but no corresponding
        # stochastic residuals appeared in the simulated pseudorange.
        ionosphere_residual_m = float(self.rng.normal(0.0, iono_sigma_m)) if iono_sigma_m > 0.0 else 0.0
        troposphere_residual_m = float(self.rng.normal(0.0, tropo_sigma_m)) if tropo_sigma_m > 0.0 else 0.0
        mp_nlos_error_m = float(self.rng.normal(0.0, mp_sigma_m)) if mp_sigma_m > 0.0 else 0.0

        total_variance_m2 = (
            iono_sigma_m**2
            + tropo_sigma_m**2
            + mp_sigma_m**2
        )
        sigma_code_m = math.sqrt(max(total_variance_m2, 1e-12))

        # Yan et al. Eq. (1), with ideal LEO receiver/satellite clock terms.
        # The modeled ionosphere/troposphere terms are also used by the predictor;
        # only their stochastic residual errors remain in the innovation.
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
#   Table III: 24 Conv filters, stride 1, ReLU, 64 LSTM units, 5 layers,
#              dropout 0.2
#
# Explicit project completions:
#   - one pseudorange-only satellite slot is represented by
#       [previous residual, current innovation]
#   - the fixed 36 IMU/state features are broadcast to each valid satellite slot
#   - Fig. 8 shows pooling, but its type/kernel/stride are unpublished; no guessed
#     pooling operator is inserted
#   - Fig. 8's KG_k -> State Update path is explicit in training; Eq. (30) is
#     evaluated after composing that state update with the postprocessed truth
#   - only the 9 position/velocity/attitude rows can be supervised because the
#     supplied truth has no accelerometer/gyro bias labels
#   - the six bias-gain rows are exactly zero rather than trained against invented
#     bias truth
#   - Fig. 8 also shows eta_k (inertial-navigation measurement-error estimation).
#     Fig. 2 routes this as a feedback correction to the IMU Error-compensation
#     block, separate from the KG -> State Update -> corrected-P,V,A path. Eq. (8)
#     subtracts residual gyro/accelerometer errors epsilon_g and epsilon_a, so eta_k
#     is implemented as [epsilon_g(3), epsilon_a(3)]. At runtime eta_k is held over
#     the *following* IMU interval and applied directly in Eq. (8). It is never
#     added to the same-epoch K*innovation correction.
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 36
OBSERVATION_FEATURE_DIM = 2                 # [previous residual, current innovation]
IMU_ERROR_DIM = 6
ETA_GYRO_SLICE = slice(0, 3)
ETA_ACCEL_SLICE = slice(3, 6)
# Numerical differentiation steps used only for the offline teacher-forced
# forward-interval eta training Jacobian d(error-state_{k+1})/d(eta_k).
# They are not physical bounds or priors.
ETA_FD_GYRO_STEP_RADPS = 1e-6
ETA_FD_ACCEL_STEP_MPS2 = 1e-4
SUPERVISED_STATE_DIM = NAVIGATION_CORRECTION_DIM  # [position, velocity, attitude]
MASK_NORMALIZATION_EPS = 1e-6


@dataclass(frozen=True)
class MaskedCLAOutput:
    """One Masked-CLA forward pass.

    Shapes
    ------
    kalman_gain : [B, 15, Nmax]
    imu_error   : [B, 6] -- eta_k = [epsilon_g(3) rad/s, epsilon_a(3) m/s^2]
    attention   : [B, Nmax]
    """

    kalman_gain: torch.Tensor
    imu_error: torch.Tensor
    attention: torch.Tensor


class MaskedConv1d(nn.Module):
    """Paper Eqs. (22)-(23): masked same-length Conv1D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 24,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
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

        m = mask.to(dtype=x.dtype).unsqueeze(1)

        x_masked = x.transpose(1, 2) * m
        z = F.conv1d(
            x_masked,
            self.weight,
            bias=None,
            stride=1,
            padding=self.padding,
        )

        mask_kernel = self._mask_kernel.to(dtype=x.dtype)
        local_count = F.conv1d(
            m, mask_kernel, stride=1, padding=self.padding
        )
        valid_window = (local_count > 0).to(dtype=x.dtype)
        denom = local_count.clamp_min(MASK_NORMALIZATION_EPS)

        y = F.relu(
            z / denom
            + self.bias.view(1, -1, 1) * valid_window
        )

        y = y * m
        return y.transpose(1, 2)


class MaskedStackedLSTM(nn.Module):
    """Paper Eqs. (24)-(25): five masked LSTM layers."""

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

        self.cells = nn.ModuleList([
            nn.LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
            )
            for layer in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "MaskedStackedLSTM expects x=[B,N,C] and mask=[B,N]"
            )
        if x.shape[:2] != mask.shape:
            raise ValueError(
                "MaskedStackedLSTM x/mask sequence shapes do not match"
            )

        batch_size, sequence_length = x.shape[:2]
        h = [
            x.new_zeros(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        c = [
            x.new_zeros(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        outputs: list[torch.Tensor] = []

        for t in range(sequence_length):
            mt = mask[:, t].to(dtype=x.dtype).unsqueeze(-1)
            keep = 1.0 - mt
            layer_input = x[:, t, :]

            for layer, cell in enumerate(self.cells):
                h_candidate, c_candidate = cell(
                    layer_input,
                    (h[layer], c[layer]),
                )
                h[layer] = mt * h_candidate + keep * h[layer]
                c[layer] = mt * c_candidate + keep * c[layer]

                layer_input = h[layer]
                if (
                    layer < self.num_layers - 1
                    and self.dropout > 0.0
                ):
                    layer_input = F.dropout(
                        layer_input,
                        p=self.dropout,
                        training=self.training,
                    )

            outputs.append(h[-1])

        return torch.stack(outputs, dim=1)


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

    The feature extractor follows Yan et al. Eqs. (22)-(29) and Table III.
    Fig. 8 is implemented structurally as

        features -> Masked CLA -> KG_k -> State Update -> Eq. (30) loss.

    The training set itself remains the offline labeled set constructed from
    Eqs. (10)-(21), consistent with Eqs. (18)-(20).  The paper does not state
    that those training features are recursively regenerated from the network's
    previous outputs, and it does not publish a multi-epoch BPTT schedule.

    Because the supplied postprocessed truth does not contain IMU-bias labels,
    only the 9 position/velocity/attitude gain rows are supervised and the six
    bias-gain rows are padded with exact zeros. eta_k does not use an invented
    direct sensor-error label. Following the Fig. 2 feedback topology, eta_k is
    interpreted as the six residual IMU measurement-error terms of Eq. (8) and is
    applied to the subsequent IMU interval. Offline, that branch is trained by a
    teacher-forced one-step forward navigation-state loss at the next usable fusion
    epoch. This forward eta supervision is a declared completion because the paper
    does not publish eta_k timing or its separate training target.

    Exact CNN tensorization, pooling parameters, FC-head dimensions, and behavior
    when N_test > Nmax_train are unpublished; the shared per-satellite gain head
    remains the declared variable-measurement-count completion.
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
            SUPERVISED_STATE_DIM,
        )

        # Fig. 8 explicitly produces eta_k in addition to KG_k.  Fig. 2 feeds an
        # IMU-error correction into the INS error-compensation path and Eq. (8)
        # subtracts residual gyro/accelerometer errors epsilon_g and epsilon_a.
        # We therefore use the paper-guided 6-D interpretation
        #   eta_k = [epsilon_g(3) rad/s, epsilon_a(3) m/s^2].
        #
        # The exact eta timing is unpublished. Fig. 2 nevertheless places eta_k
        # on the feedback path to IMU Error compensation, so the online path holds
        # the current output over the following IMU interval. Offline, the branch
        # is trained through a teacher-forced future-interval mechanization
        # sensitivity. Zero initialization starts from the no-eta baseline.
        self.imu_error_head = nn.Linear(
            64,
            IMU_ERROR_DIM,
        )
        nn.init.zeros_(self.imu_error_head.weight)
        nn.init.zeros_(self.imu_error_head.bias)

    def forward(
        self,
        fixed: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        channel_mask: torch.Tensor,
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

        conv = self.conv(token, mask_values)
        lstm = self.lstm(conv, mask_values)
        context, attention = self.attention(
            lstm,
            mask_values,
        )

        # Shared head: one gain column per current sequence slot.  Unlike a
        # flattened [9*Nmax] FC output, these weights are trained on every valid
        # satellite slot and can therefore be reused when N changes at inference.
        context_per_slot = context.unsqueeze(1).expand(-1, sequence_length, -1)
        gain_features = torch.cat([lstm, context_per_slot], dim=-1)
        learned_gain = self.gain_head(gain_features).transpose(1, 2)
        learned_gain = (
            learned_gain
            * mask_values.unsqueeze(1)
        )

        gain = F.pad(
            learned_gain,
            (
                0,
                0,
                0,
                INS_STATE_DIM - SUPERVISED_STATE_DIM,
            ),
        )

        imu_error = self.imu_error_head(context)

        return MaskedCLAOutput(
            kalman_gain=gain,
            imu_error=imu_error,
            attention=attention,
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
    """Teacher-forced forward state effect of the Fig. 2 eta_k feedback path.

    Parameters
    ----------
    eta_forward_state_sensitivity_9 : [B,9,6]
        Local derivative of the *next* usable fusion-epoch INS state with respect
        to eta_k held over the intervening IMU propagation interval.

    Returns
    -------
    [B,9]
        First-order navigation-state change caused by eta_k over the future IMU
        interval. This helper is used only for offline training. Online, eta_k is
        applied directly in compensate_imu() before each mechanization step.
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


def _torch_so3_exp_batch(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable batched SO(3) exponential used only in the training loss."""
    if rotation_vector.ndim != 2 or rotation_vector.shape[1] != 3:
        raise ValueError("rotation_vector must have shape [B,3]")

    batch = rotation_vector.shape[0]
    x, y, z = rotation_vector.unbind(dim=1)
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=1,
    ).reshape(batch, 3, 3)
    K2 = K @ K

    theta2 = torch.sum(rotation_vector * rotation_vector, dim=1)
    theta = torch.sqrt(theta2.clamp_min(1e-30))
    small = theta2 < 1e-10

    # Stable Rodrigues coefficients.  The series branch avoids 0/0 and keeps
    # gradients finite for the small navigation-attitude corrections expected here.
    theta4 = theta2 * theta2
    a_series = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta4 / 720.0
    a_regular = torch.sin(theta) / theta
    b_regular = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-30)
    a = torch.where(small, a_series, a_regular)
    b = torch.where(small, b_series, b_regular)

    I = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    I = I.unsqueeze(0).expand(batch, -1, -1)
    return I + a[:, None, None] * K + b[:, None, None] * K2


def fig8_post_update_error_state_9(
    estimated_error_state_15: torch.Tensor,
    true_error_state_9: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-update 9-state error used by Yan et al. Eq. (30).

    ``true_error_state_9`` is the postprocessed correction required to move the
    same prior navigation state to truth. ``estimated_error_state_15`` is the
    Fig. 8 state update produced by KG_k. Position and velocity feedback are
    additive. Attitude feedback is composed on SO(3), matching
    ``inject_error_state`` instead of subtracting Euler/rotation-vector entries
    as unrelated scalars.

    Returns
    -------
    linear_error_6 : [B,6]
        Remaining position/velocity error after the learned state update.
    attitude_angle_rad : [B]
        Geodesic attitude error magnitude. Its square equals the squared norm of
        the residual attitude rotation vector required by the 9-state Eq. (30).
    """
    if estimated_error_state_15.ndim != 2:
        raise ValueError("estimated_error_state_15 must have shape [B,15]")
    if true_error_state_9.ndim != 2 or true_error_state_9.shape[1] != 9:
        raise ValueError("true_error_state_9 must have shape [B,9]")
    if estimated_error_state_15.shape[0] != true_error_state_9.shape[0]:
        raise ValueError("estimated/true state batch dimensions do not match")

    estimated = estimated_error_state_15[:, :SUPERVISED_STATE_DIM].double()
    truth = true_error_state_9.double()
    linear_error_6 = truth[:, :6] - estimated[:, :6]

    sign = float(ATTITUDE_FEEDBACK_SIGN)
    true_relative = _torch_so3_exp_batch(sign * truth[:, 6:9])
    estimated_relative = _torch_so3_exp_batch(sign * estimated[:, 6:9])
    residual_relative = true_relative @ estimated_relative.transpose(1, 2)

    # Robust principal rotation angle using atan2(|sin(theta)|, cos(theta)).
    # This is the geodesic norm of the residual attitude rotation vector, so
    # angle^2 is exactly its contribution to ||x_k - xhat_k||_2^2.
    vee = torch.stack(
        [
            residual_relative[:, 2, 1] - residual_relative[:, 1, 2],
            residual_relative[:, 0, 2] - residual_relative[:, 2, 0],
            residual_relative[:, 1, 0] - residual_relative[:, 0, 1],
        ],
        dim=1,
    )
    sin_theta = 0.5 * torch.linalg.vector_norm(vee, dim=1)
    cos_theta = 0.5 * (
        torch.diagonal(residual_relative, dim1=1, dim2=2).sum(dim=1) - 1.0
    )
    attitude_angle = torch.atan2(
        sin_theta,
        cos_theta.clamp(-1.0, 1.0),
    )
    return linear_error_6, attitude_angle


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

    MAX_FUSION_EPOCHS = 100
    MAX_TEST_FUSION_EPOCHS = 100

    # Table III explicitly gives Adam, initial LR=0.01, Conv=24,
    # LSTM=64 x 5, and dropout=0.2. Fig. 15 displays learning curves extending
    # to roughly 500 training epochs, but the text does not publish an exact
    # stopping epoch. Therefore 500 is used only as a maximum figure-guided
    # training horizon.
    TRAINING_EPOCHS = 10
    BATCH_SIZE = 16  # unpublished; explicit reproducibility completion
    LEARNING_RATE = 0.01
    # Ref. [15] permits separate learning rates for its two alternating blocks,
    # but Yan et al. publish only one initial learning rate (0.01). To avoid an
    # unsupported extra hyperparameter, both alternating blocks use the same
    # paper-supported initial learning rate.
    FILTER_BLOCK_LEARNING_RATE = LEARNING_RATE
    REPRESENTATION_BLOCK_LEARNING_RATE = LEARNING_RATE

    # Eq. (32) explicitly includes gamma*||Theta||^2 but does not publish gamma.
    GAMMA_L2 = 1e-6  # explicit reproducibility completion

    # Validation/early-stopping are retained as safeguards against overfitting
    # and wasted long training runs. Yan et al. do not publish these details,
    # so they are explicit reproducibility completions, not paper-exact.
    #
    # The split is chronological rather than random because adjacent navigation
    # epochs are strongly correlated; a random split would leak near-duplicate
    # temporal context between train and validation.
    VALIDATION_FRACTION = 0.20
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 0.0

    # Keep the paper-supported Adam initial LR fixed. No unpublished LR scheduler
    # or baseline gradient clipping is introduced.
    REQUIRE_RECURSIVE_DATA01_DIAGNOSTIC = True

    # Yan et al. Fig. 2 / Sec. II-D: FDE is an ONLINE stage and the raw
    # INS-predicted innovation is tested before the learned Kalman gain.
    ENABLE_FDE = True
    # Yan et al. state that Td is selected from false-alarm probability but do
    # not publish its value. Ref. [33] uses alpha=1e-3 in its quantitative DIA
    # analysis, so this value is an explicit reproducibility completion.
    FDE_SIGNIFICANCE_ALPHA = 1e-3
    FDE_STATISTICAL_SELF_CHECK = _validate_ref33_statistical_core(
        FDE_SIGNIFICANCE_ALPHA
    )

    # Optional recursive experiment retained only for diagnostics/ablation.
    # Eqs. (18)-(20) describe an offline fixed training dataset, and the paper
    # does not publish recursive feature regeneration or BPTT; therefore this
    # block is disabled and is never part of the Fig. 8 / Eq. (30) baseline.
    ENABLE_STABILITY_COMPLETION = False
    CLOSED_LOOP_FINETUNE_PASSES = 2
    CLOSED_LOOP_FINETUNE_LR = 1e-3
    CLOSED_LOOP_RESET_INTERVAL = 100
    STABILITY_GRADIENT_CLIP_NORM = 1.0

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
    print("LEO residual model: Ref. [35] ionosphere/troposphere/MP; URA and receiver noise omitted")

    # =========================================================================
    # 5. CLASSICAL TC PASS -> TRAINING HISTORY
    # =========================================================================
    nav = initial_nav.copy()
    P = P0.copy()
    history_rows = []

    # Store the propagation segments since the previous usable fusion epoch.
    # They are later shifted forward: eta_k produced at usable epoch k is trained
    # against the interval from k to the next usable network epoch, matching the
    # Fig. 2 feedback path. Fusion epochs with no usable measurements do not reset
    # this list because no new eta_k is available there.
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
            interval_segments_since_previous_usable_fusion.append((imu_index, dt))
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
        prior_position = gnss_antenna_position(prior_nav, lever_arm_b_m)

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
        counts = Counter(m.constellation for m in measurements)

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
            "x_pred": error_state_pred.copy(),
            "x_post": error_state_post.copy(),
            "accel": last_accel.copy(),
            "gyro": last_gyro.copy(),
            "prior_position": prior_position.copy(),
            "posterior_position": posterior_position.copy(),
            "truth_position": fusion_antenna_truth_position[fusion_index].copy(),
            "target_state_9": target_state_9.copy(),
            "fusion_index": int(fusion_index),
            "preceding_interval_segments": tuple(
                interval_segments_since_previous_usable_fusion
            ),
            "n_total": len(measurements),
            "n_gnss": counts["G"] + counts["C"],
            "n_leo": counts["L"],
        })

        # A new network output would be available at this usable fusion epoch,
        # so the following propagation interval starts here.
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
    # The last usable fusion row has no future interval in this finite offline
    # record, so it cannot supervise eta_k. The first row supplies lagged context.
    # Training samples are therefore history_rows[1:-1].
    nmax = max(len(row["sat_ids"]) for row in history_rows[1:-1])
    feature_time = []
    eta_future_end_time = []
    fixed = []
    observations = []
    channel_masks = []
    innovation_padded = []
    residual_padded = []
    target_states_9 = []
    eta_forward_state_sensitivities_9 = []
    eta_forward_target_states_9 = []
    prior_positions = []
    truth_positions = []

    def _truth_nav_at_fusion(fusion_index: int) -> NavigationState:
        return NavigationState(
            fusion_imu_truth_position[fusion_index].copy(),
            fusion_imu_truth_velocity[fusion_index].copy(),
            fusion_truth_body_to_ecef[fusion_index].copy(),
            np.zeros(3),
            np.zeros(3),
        )

    # First classical row supplies lagged Eq. (11)-(14) context only. For each
    # current row k, eta_k is paired with the propagation interval ending at the
    # next usable network row k+1. This is the causal Fig. 2 direction.
    for k in range(1, len(history_rows) - 1):
        current = history_rows[k]
        previous = history_rows[k - 1]
        future = history_rows[k + 1]

        delta_accel = current["accel"] - previous["accel"]
        delta_gyro = current["gyro"] - previous["gyro"]
        # Paper Eq. (12): posterior-minus-prior state innovation.
        # In this closed-loop implementation x_pred is normally zero after reset,
        # but retaining it explicitly makes the state semantics unambiguous.
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

        # Fig. 2 eta_k feedback completion:
        #   eta_k is produced at the current usable fusion epoch and affects the
        #   subsequent IMU interval. For offline supervision only, start that
        #   interval from postprocessed truth (teacher forcing), propagate the
        #   measured IMU with eta=0, and ask J_eta*eta_k to explain the state
        #   correction required at the next usable fusion epoch. No direct eta
        #   or sensor-error pseudo-label is fabricated.
        current_fusion_index = int(current["fusion_index"])
        future_fusion_index = int(future["fusion_index"])
        future_interval_segments = future["preceding_interval_segments"]
        truth_start_nav = _truth_nav_at_fusion(current_fusion_index)
        truth_future_nav = _truth_nav_at_fusion(future_fusion_index)
        future_base_end_nav = _replay_imu_interval_with_eta(
            truth_start_nav,
            future_interval_segments,
            imr,
            np.zeros(IMU_ERROR_DIM),
        )
        eta_forward_target_9 = _navigation_delta_9(
            future_base_end_nav,
            truth_future_nav,
        )
        eta_forward_sensitivity_9 = eta_interval_state_sensitivity_9(
            truth_start_nav,
            future_interval_segments,
            imr,
            reference_end_nav=future_base_end_nav,
        )

        feature_time.append(current["time"])
        eta_future_end_time.append(future["time"])
        fixed.append(fixed_k)
        observations.append(np.stack([residual_k, innovation_k], axis=1))
        channel_masks.append(np.stack([residual_mask, current_mask], axis=1))
        innovation_padded.append(innovation_k)
        residual_padded.append(residual_k)
        target_states_9.append(current["target_state_9"])
        eta_forward_state_sensitivities_9.append(eta_forward_sensitivity_9)
        eta_forward_target_states_9.append(eta_forward_target_9)
        prior_positions.append(current["prior_position"])
        truth_positions.append(current["truth_position"])

    feature_time = np.asarray(feature_time)
    eta_future_end_time = np.asarray(eta_future_end_time)
    fixed = np.stack(fixed)
    observations = np.stack(observations)
    channel_masks = np.stack(channel_masks)
    innovation_padded = np.stack(innovation_padded)
    residual_padded = np.stack(residual_padded)
    target_states_9 = np.stack(target_states_9)
    eta_forward_state_sensitivities_9 = np.stack(
        eta_forward_state_sensitivities_9
    )
    eta_forward_target_states_9 = np.stack(eta_forward_target_states_9)

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
        raise ValueError(
            "eta_forward_target_states_9 must have shape [T,9]"
        )
    if not np.all(np.isfinite(eta_forward_state_sensitivities_9)):
        raise ValueError(
            "eta_forward_state_sensitivities_9 contains non-finite values"
        )
    if not np.all(np.isfinite(eta_forward_target_states_9)):
        raise ValueError("eta_forward_target_states_9 contains non-finite values")
    if eta_future_end_time.shape != (expected_t,):
        raise ValueError("eta_future_end_time must have shape [T]")
    if np.any(eta_future_end_time <= feature_time):
        raise ValueError("every eta_k training target must lie strictly after epoch k")
    if residual_padded.shape != (expected_t, nmax):
        raise ValueError("residual_padded must have shape [T,Nmax]")
    if channel_masks.shape != (expected_t, nmax, OBSERVATION_FEATURE_DIM):
        raise ValueError("channel_masks must have shape [T,Nmax,2]")
    if expected_t > 1 and not np.all(np.diff(feature_time) > 0.0):
        raise ValueError("feature times must be strictly increasing")
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(innovation_padded)) or not np.all(np.isfinite(residual_padded)):
        raise ValueError("feature arrays contain non-finite values")
    if np.any(innovation_padded[~channel_masks[:, :, 1]] != 0.0):
        raise ValueError("masked innovation padding must be exactly zero")
    if np.any(residual_padded[~channel_masks[:, :, 0]] != 0.0):
        raise ValueError("masked residual padding must be exactly zero")
    prior_positions = np.stack(prior_positions)
    truth_positions = np.stack(truth_positions)
    satellite_masks = channel_masks[:, :, 1]

    np.savez(
        OUTPUT_DIR / "training_direct.npz",
        time_gpst_s=feature_time,
        fixed=fixed,
        observations=observations,
        channel_mask=channel_masks,
        innovation_padded=innovation_padded,
        residual_padded=residual_padded,
        target_state_position_velocity_attitude=target_states_9,
        eta_forward_end_time_gpst_s=eta_future_end_time,
        eta_forward_state_sensitivity_9=eta_forward_state_sensitivities_9,
        eta_forward_target_state_9=eta_forward_target_states_9,
        eta_definition=np.asarray(
            "[epsilon_gx,epsilon_gy,epsilon_gz,epsilon_ax,epsilon_ay,epsilon_az]"
        ),
        eta_units=np.asarray("[rad/s,rad/s,rad/s,m/s^2,m/s^2,m/s^2]"),
        eta_timing_completion=np.asarray(
            "output_at_fusion_k_zero_order_hold_over_following_IMU_interval"
        ),
        eta_training_completion=np.asarray(
            "teacher_forced_forward_interval_state_loss_no_direct_eta_labels"
        ),
        prior_position_ecef_m=prior_positions,
        truth_position_ecef_m=truth_positions,
        measurement_mode=np.asarray("pseudorange_only"),
        feature_timing=np.asarray("causal_lagged"),
    )

    # =========================================================================
    # 7. CHRONOLOGICAL DATA01 TRAIN/VALIDATION SPLIT + TRAIN-ONLY NORMALIZATION
    # =========================================================================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sample_count = len(feature_time)
    validation_count = max(1, int(round(sample_count * VALIDATION_FRACTION)))
    split = sample_count - validation_count
    if split < 2 or validation_count < 1:
        raise ValueError(
            "Not enough feature samples for a leakage-free chronological "
            "train/validation split with forward eta supervision"
        )

    # Sample i uses truth at eta_future_end_time[i]. The final nominal training
    # sample would therefore cross the chronological split boundary. Drop exactly
    # that boundary sample so no validation-epoch truth enters the training loss.
    train_index = np.arange(split - 1)
    boundary_index = split - 1
    val_index = np.arange(split, sample_count)
    if train_index.size < 1 or val_index.size < 1:
        raise ValueError("Leakage-free train/validation split became empty")
    if eta_future_end_time[train_index[-1]] >= feature_time[val_index[0]]:
        raise RuntimeError(
            "forward eta target still crosses the train/validation boundary"
        )

    # Fit every normalization statistic on Data01-train only.
    # Validation and Data02 therefore cannot leak into feature scaling.
    fixed_mean = fixed[train_index].mean(axis=0)
    fixed_std = fixed[train_index].std(axis=0)
    fixed_std[fixed_std < 1e-8] = 1.0

    obs_mean = np.zeros(2)
    obs_std = np.ones(2)
    valid_channels = (
        channel_masks[train_index]
        & channel_masks[train_index, :, 1, None]
    )
    for channel in range(2):
        values = observations[train_index, :, channel][
            valid_channels[:, :, channel]
        ]
        if values.size:
            obs_mean[channel] = values.mean()
            s = values.std()
            obs_std[channel] = 1.0 if s < 1e-8 else s

    fixed_normalized = (fixed - fixed_mean) / fixed_std
    observations_normalized = (
        observations - obs_mean.reshape(1, 1, 2)
    ) / obs_std.reshape(1, 1, 2)
    observations_normalized = np.where(
        channel_masks, observations_normalized, 0.0
    )

    # Only INPUT features are normalized. Eq. (30) targets remain in physical
    # units; target scaling would change the relative state weighting and is not
    # stated in the paper.
    np.savez(
        OUTPUT_DIR / "normalizer.npz",
        fixed_mean=fixed_mean,
        fixed_std=fixed_std,
        obs_mean=obs_mean,
        obs_std=obs_std,
        validation_fraction=VALIDATION_FRACTION,
        split_index=split,
        dropped_boundary_index=boundary_index,
        dropped_boundary_time_gpst_s=feature_time[boundary_index],
    )

    dataset = TensorDataset(
        torch.tensor(fixed_normalized, dtype=torch.float32),
        torch.tensor(observations_normalized, dtype=torch.float32),
        torch.tensor(satellite_masks, dtype=torch.bool),
        torch.tensor(channel_masks, dtype=torch.bool),
        torch.tensor(innovation_padded, dtype=torch.float32),
        torch.tensor(eta_forward_state_sensitivities_9, dtype=torch.float32),
        torch.tensor(target_states_9, dtype=torch.float64),
        torch.tensor(eta_forward_target_states_9, dtype=torch.float64),
    )
    train_dataset = torch.utils.data.Subset(dataset, train_index.tolist())
    val_dataset = torch.utils.data.Subset(dataset, val_index.tolist())

    use_pinned_memory = DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=use_pinned_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=use_pinned_memory,
    )

    print(
        f"Data01 train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"Nmax={nmax}"
    )

    # =========================================================================
    # 8. FIG. 8 / FIG. 2 MASKED CLA TRAINING + EQ. (30)/(32) + VALIDATION
    # =========================================================================
    # Paper-supported structure:
    #   - Eqs. (18)-(20): offline labeled training data.
    #   - Fig. 8: Masked CLA produces KG_k and eta_k.
    #   - KG_k follows the explicit State Update path and is trained against the
    #     postprocessed current state through Eq. (30).
    #   - Fig. 2 routes the separate IMU-error correction feedback to Error
    #     compensation before INS mechanization; Eq. (8) supplies the physical
    #     subtraction point for epsilon_g / epsilon_a.
    #   - Eq. (32): prediction error plus gamma*||Theta||^2; Table III: Adam,
    #     initial learning rate 0.01.
    #
    # Paper-guided eta completion:
    #   Yan et al. do not publish eta_k timing or a separate eta label/loss. We do
    #   not replay eta_k backward and do not add it to K_k*innovation_k. Instead,
    #   eta_k is constrained on the *following* IMU interval using postprocessed
    #   state truth only at the interval endpoints. The local Eq. (8)+INS sensitivity
    #   maps eta_k to a next-epoch 9-state correction. That next state is treated as
    #   another temporal instance of the same Eq. (30)-type state objective, not as
    #   a separately weighted eta loss. At runtime the Jacobian disappears:
    #   eta_k is held and applied directly in compensate_imu() until the next
    #   usable network output.
    #
    # Current and following navigation-state errors are treated as two temporal
    # samples of the SAME Eq. (30)-type state-estimation objective.  The optimizer
    # is alternated blockwise following Ref. [15], rather than jointly updating all
    # Masked-CLA parameters in one Adam step.
    # -------------------------------------------------------------------------
    # Alternating optimization -- closest reproducible mapping of Yan et al.
    # "optimized alternately [15]" to the Masked-CLA architecture.
    #
    # Ref. [15], Algorithm 2:
    #   (1) optimize filter theta while representation psi is frozen,
    #   (2) optimize representation psi while filter theta is frozen,
    # using the SAME final state-estimation loss in both phases.
    #
    # Yan et al. do not publish their Masked-CLA parameter partition.  Fig. 8
    # nevertheless exposes a natural decomposable structure:
    #   psi   : Masked CNN + Masked LSTM + Masked Attention (feature/representation)
    #   theta : Masked FC gain head + eta head (filter/output mapping)
    #
    # This partition is therefore PAPER-GUIDED, not claimed paper-exact.
    # A separate encoder warm-start from Ref. [15] is intentionally NOT copied,
    # because Yan et al. provide no intermediate representation target for the
    # Masked CNN/LSTM/attention stack.
    # -------------------------------------------------------------------------
    model = MaskedCLA(nmax=nmax, dropout=0.2).to(DEVICE)

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
        parameter
        for module in representation_modules
        for parameter in module.parameters()
    ]
    filter_parameters = [
        parameter
        for module in filter_modules
        for parameter in module.parameters()
    ]

    all_parameter_ids = {id(parameter) for parameter in model.parameters()}
    representation_parameter_ids = {id(parameter) for parameter in representation_parameters}
    filter_parameter_ids = {id(parameter) for parameter in filter_parameters}

    if representation_parameter_ids & filter_parameter_ids:
        raise RuntimeError("alternating parameter blocks overlap")
    if representation_parameter_ids | filter_parameter_ids != all_parameter_ids:
        raise RuntimeError(
            "alternating parameter blocks do not cover all MaskedCLA parameters"
        )

    filter_optimizer = torch.optim.Adam(
        filter_parameters,
        lr=FILTER_BLOCK_LEARNING_RATE,
    )
    representation_optimizer = torch.optim.Adam(
        representation_parameters,
        lr=REPRESENTATION_BLOCK_LEARNING_RATE,
    )
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

        linear_error_6, attitude_angle = fig8_post_update_error_state_9(
            predicted_update,
            true_error_state_9,
        )
        per_sample_squared_norm = (
            torch.sum(linear_error_6**2, dim=1)
            + attitude_angle**2
        )
        state_loss = torch.mean(per_sample_squared_norm)
        position_rmse = torch.sqrt(
            torch.mean(torch.sum(linear_error_6[:, :3] ** 2, dim=1))
        )
        return state_loss, position_rmse

    def _regularized_temporal_state_objective(
        kg_state_update,
        current_target_9,
        eta_forward_state_update_9,
        eta_forward_target_9,
    ):
        """Yan Eq. (30)/(32)-style state objective used in BOTH alternating phases.

        The first state term is the current fusion-epoch Fig. 8 KG update.  The
        second state term is the next usable fusion-epoch navigation consequence
        of the Fig. 2 eta feedback completion.  They are treated as two supervised
        state instants of the same temporal state-estimation objective, rather than
        as independently weighted KG and eta branch losses.

        The future eta term remains a paper-guided completion because Yan et al.
        do not publish eta_k timing/supervision.  No direct eta label is used.
        """
        current_eq30_loss, current_position_rmse = _state_prediction_loss(
            kg_state_update,
            current_target_9,
        )
        following_eq30_loss, following_position_rmse = _state_prediction_loss(
            eta_forward_state_update_9,
            eta_forward_target_9,
        )

        # Same raw physical-state squared-error criterion at both supervised
        # instants. Equal averaging here is simply the time-sample mean analogue
        # of Yan Eq. (32), not a tunable KG-vs-eta branch weight.
        temporal_state_loss = torch.stack(
            [current_eq30_loss, following_eq30_loss]
        ).mean()

        # Yan Eq. (32) regularizes the entire trainable set Theta.  During an
        # alternating phase the frozen block contributes only a constant term;
        # gradients naturally flow only to the active block, matching Ref. [15].
        if GAMMA_L2:
            l2_all = sum(
                torch.sum(parameter * parameter)
                for parameter in model.parameters()
            )
            objective = temporal_state_loss + GAMMA_L2 * l2_all
        else:
            objective = temporal_state_loss

        return (
            objective,
            temporal_state_loss,
            current_eq30_loss,
            following_eq30_loss,
            current_position_rmse,
            following_position_rmse,
        )

    def _set_alternating_phase(phase: str) -> None:
        """Freeze one block and train the other, following Ref. [15] Algorithm 2."""
        if phase not in {"filter", "representation"}:
            raise ValueError("phase must be 'filter' or 'representation'")

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model.train()

        if phase == "filter":
            # Fixed representation should also have deterministic dropout behavior.
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

    def _run_training_phase(
        phase: str,
        optimizer: torch.optim.Optimizer,
        epoch_batches,
        epoch: int,
    ):
        _set_alternating_phase(phase)

        objective_sum = 0.0
        temporal_state_sum = 0.0
        current_eq30_sum = 0.0
        following_eq30_sum = 0.0
        current_position_rmse_squared_sum = 0.0
        following_position_rmse_squared_sum = 0.0
        eta_gyro_sq_sum = 0.0
        eta_accel_sq_sum = 0.0
        count = 0

        for batch in epoch_batches:
            (
                fixed_b,
                obs_b,
                mask_b,
                channel_b,
                innovation_b,
                eta_forward_sensitivity_b,
                target_b,
                eta_forward_target_b,
            ) = [
                x.to(DEVICE, non_blocking=use_pinned_memory) for x in batch
            ]

            optimizer.zero_grad(set_to_none=True)

            output = model(fixed_b, obs_b, mask_b, channel_b)
            kg_state_update = fig8_state_update(output, innovation_b)
            eta_forward_state_update_9 = fig2_eta_forward_state_update_9(
                output,
                eta_forward_sensitivity_b,
            )

            (
                objective,
                temporal_state_loss,
                current_eq30_loss,
                following_eq30_loss,
                current_position_rmse_b,
                following_position_rmse_b,
            ) = _regularized_temporal_state_objective(
                kg_state_update,
                target_b,
                eta_forward_state_update_9,
                eta_forward_target_b,
            )

            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"non-finite {phase} alternating objective at epoch {epoch}"
                )

            objective.backward()
            optimizer.step()

            batch_size_now = len(fixed_b)
            objective_sum += float(objective.detach().cpu()) * batch_size_now
            temporal_state_sum += (
                float(temporal_state_loss.detach().cpu()) * batch_size_now
            )
            current_eq30_sum += (
                float(current_eq30_loss.detach().cpu()) * batch_size_now
            )
            following_eq30_sum += (
                float(following_eq30_loss.detach().cpu()) * batch_size_now
            )
            current_position_rmse_squared_sum += (
                float(current_position_rmse_b.detach().cpu()) ** 2
                * batch_size_now
            )
            following_position_rmse_squared_sum += (
                float(following_position_rmse_b.detach().cpu()) ** 2
                * batch_size_now
            )
            eta_gyro_sq_sum += float(
                torch.mean(
                    output.imu_error[:, ETA_GYRO_SLICE] ** 2
                ).detach().cpu()
            ) * batch_size_now
            eta_accel_sq_sum += float(
                torch.mean(
                    output.imu_error[:, ETA_ACCEL_SLICE] ** 2
                ).detach().cpu()
            ) * batch_size_now
            count += batch_size_now

        return {
            "objective": objective_sum / count,
            "temporal_state_loss": temporal_state_sum / count,
            "current_eq30": current_eq30_sum / count,
            "following_eq30_completion": following_eq30_sum / count,
            "current_position_rmse_m": math.sqrt(
                current_position_rmse_squared_sum / count
            ),
            "following_position_rmse_m": math.sqrt(
                following_position_rmse_squared_sum / count
            ),
            "eta_gyro_rms_radps": math.sqrt(eta_gyro_sq_sum / count),
            "eta_accel_rms_mps2": math.sqrt(eta_accel_sq_sum / count),
        }

    def _evaluate_state_objective(loader):
        # Restore every parameter as trainable for checkpoint consistency, then
        # switch the whole network to deterministic evaluation mode.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.eval()

        objective_sum = 0.0
        temporal_state_sum = 0.0
        current_eq30_sum = 0.0
        following_eq30_sum = 0.0
        current_position_rmse_squared_sum = 0.0
        following_position_rmse_squared_sum = 0.0
        eta_gyro_sq_sum = 0.0
        eta_accel_sq_sum = 0.0
        count = 0

        with torch.inference_mode():
            for batch in loader:
                (
                    fixed_b,
                    obs_b,
                    mask_b,
                    channel_b,
                    innovation_b,
                    eta_forward_sensitivity_b,
                    target_b,
                    eta_forward_target_b,
                ) = [
                    x.to(DEVICE, non_blocking=use_pinned_memory) for x in batch
                ]

                output = model(fixed_b, obs_b, mask_b, channel_b)
                kg_state_update = fig8_state_update(output, innovation_b)
                eta_forward_state_update_9 = fig2_eta_forward_state_update_9(
                    output,
                    eta_forward_sensitivity_b,
                )

                (
                    objective,
                    temporal_state_loss,
                    current_eq30_loss,
                    following_eq30_loss,
                    current_position_rmse_b,
                    following_position_rmse_b,
                ) = _regularized_temporal_state_objective(
                    kg_state_update,
                    target_b,
                    eta_forward_state_update_9,
                    eta_forward_target_b,
                )

                if (
                    not torch.isfinite(objective)
                    or not torch.isfinite(temporal_state_loss)
                ):
                    raise FloatingPointError(
                        "non-finite validation state-estimation objective"
                    )

                batch_size_now = len(fixed_b)
                objective_sum += float(objective.detach().cpu()) * batch_size_now
                temporal_state_sum += (
                    float(temporal_state_loss.detach().cpu()) * batch_size_now
                )
                current_eq30_sum += (
                    float(current_eq30_loss.detach().cpu()) * batch_size_now
                )
                following_eq30_sum += (
                    float(following_eq30_loss.detach().cpu()) * batch_size_now
                )
                current_position_rmse_squared_sum += (
                    float(current_position_rmse_b.detach().cpu()) ** 2
                    * batch_size_now
                )
                following_position_rmse_squared_sum += (
                    float(following_position_rmse_b.detach().cpu()) ** 2
                    * batch_size_now
                )
                eta_gyro_sq_sum += float(
                    torch.mean(
                        output.imu_error[:, ETA_GYRO_SLICE] ** 2
                    ).detach().cpu()
                ) * batch_size_now
                eta_accel_sq_sum += float(
                    torch.mean(
                        output.imu_error[:, ETA_ACCEL_SLICE] ** 2
                    ).detach().cpu()
                ) * batch_size_now
                count += batch_size_now

        return {
            "objective": objective_sum / count,
            "temporal_state_loss": temporal_state_sum / count,
            "current_eq30": current_eq30_sum / count,
            "following_eq30_completion": following_eq30_sum / count,
            "current_position_rmse_m": math.sqrt(
                current_position_rmse_squared_sum / count
            ),
            "following_position_rmse_m": math.sqrt(
                following_position_rmse_squared_sum / count
            ),
            "eta_gyro_rms_radps": math.sqrt(eta_gyro_sq_sum / count),
            "eta_accel_rms_mps2": math.sqrt(eta_accel_sq_sum / count),
        }

    print(
        "\n=== YAN EQ.30/EQ.32 + REF.[15] ALTERNATING OPTIMIZATION "
        "(DATA01) ==="
    )

    best_val_joint_state_loss = float("inf")
    best_val_eq30 = float("inf")
    best_val_eta_forward = float("inf")
    best_val_objective = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch in range(1, TRAINING_EPOCHS + 1):
        epochs_ran = epoch

        # Ref. [15] Algorithm 2 randomly partitions the data once per outer epoch
        # and reuses that partition for the theta and psi phases. Materializing the
        # current epoch's DataLoader batches reproduces that ordering without
        # introducing a second independently shuffled partition.
        epoch_batches = list(train_loader)

        # Algorithm 2 ordering: optimize filter theta first, then representation psi.
        filter_phase = _run_training_phase(
            "filter",
            filter_optimizer,
            epoch_batches,
            epoch,
        )
        representation_phase = _run_training_phase(
            "representation",
            representation_optimizer,
            epoch_batches,
            epoch,
        )

        # Model selection is performed only after a COMPLETE alternating cycle.
        validation = _evaluate_state_objective(val_loader)

        current_filter_lr = float(filter_optimizer.param_groups[0]["lr"])
        current_representation_lr = float(
            representation_optimizer.param_groups[0]["lr"]
        )

        training_history.append({
            "stage": "ref15_alternating_train_validation",
            "epoch": epoch,
            "filter_phase_objective_eq32": filter_phase["objective"],
            "filter_phase_temporal_state_loss": filter_phase["temporal_state_loss"],
            "representation_phase_objective_eq32": representation_phase["objective"],
            "representation_phase_temporal_state_loss": representation_phase[
                "temporal_state_loss"
            ],
            "train_current_eq30_after_representation_phase": representation_phase[
                "current_eq30"
            ],
            "train_following_eq30_eta_completion_after_representation_phase":
                representation_phase["following_eq30_completion"],
            "val_objective_eq32": validation["objective"],
            "val_temporal_state_loss": validation["temporal_state_loss"],
            "val_current_eq30": validation["current_eq30"],
            "val_following_eq30_eta_completion": validation[
                "following_eq30_completion"
            ],
            "val_current_position_rmse_m": validation[
                "current_position_rmse_m"
            ],
            "val_following_position_rmse_m": validation[
                "following_position_rmse_m"
            ],
            "val_eta_gyro_rms_radps": validation["eta_gyro_rms_radps"],
            "val_eta_accel_rms_mps2": validation["eta_accel_rms_mps2"],
            "filter_block_learning_rate": current_filter_lr,
            "representation_block_learning_rate": current_representation_lr,
        })

        print(
            f"epoch {epoch:03d}/{TRAINING_EPOCHS}: "
            f"theta_obj={filter_phase['objective']:.6g}, "
            f"psi_obj={representation_phase['objective']:.6g}, "
            f"val_Eq30mean={validation['temporal_state_loss']:.6g}, "
            f"val_current={validation['current_eq30']:.6g}, "
            f"val_following={validation['following_eq30_completion']:.6g}, "
            f"val_posRMSE={validation['current_position_rmse_m']:.3f} m, "
            f"val_etaFuturePosRMSE={validation['following_position_rmse_m']:.3f} m, "
            f"eta_g_RMS={validation['eta_gyro_rms_radps']:.3g} rad/s, "
            f"eta_a_RMS={validation['eta_accel_rms_mps2']:.3g} m/s^2, "
            f"val_Eq32={validation['objective']:.6g}, "
            f"lr_theta={current_filter_lr:.3g}, "
            f"lr_psi={current_representation_lr:.3g}"
        )

        # Eq. (30)-type validation state error, not the L2 term, drives model
        # selection. Early stopping is a project safeguard and is evaluated only
        # after a full theta->psi alternating cycle.
        val_state_loss = validation["temporal_state_loss"]
        if val_state_loss < best_val_joint_state_loss - EARLY_STOP_MIN_DELTA:
            best_val_joint_state_loss = val_state_loss
            best_val_eq30 = validation["current_eq30"]
            best_val_eta_forward = validation["following_eq30_completion"]
            best_val_objective = validation["objective"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                "early stopping: validation Eq.(30)-type temporal state loss "
                f"did not improve for {EARLY_STOP_PATIENCE} complete alternating "
                f"cycles; best epoch={best_epoch}, "
                f"best val Eq30mean={best_val_joint_state_loss:.6g}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training produced no finite validation checkpoint")

    # Continue with the best validation checkpoint, not the last epoch.
    model.load_state_dict(best_state)
    pretrain_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": pretrain_state,
            "stage": "yan_eq30_eq32_ref15_alternating_best_checkpoint",
            "nmax": nmax,
            "fixed_mean": fixed_mean,
            "fixed_std": fixed_std,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "supervised_state_dim": SUPERVISED_STATE_DIM,
            "loss": "mean_raw_state_Eq30_over_current_and_following_supervised_instants_plus_Eq32_L2",
            "training_features": "offline_fixed_labeled_dataset_Yan_Eqs18_to_20",
            "optimization": "Ref15_Algorithm2_style_theta_then_psi_same_state_objective",
            "alternating_theta_block": "Masked_FC_gain_head_plus_eta_head",
            "alternating_psi_block": "Masked_CNN_plus_Masked_LSTM_plus_Masked_Attention",
            "alternating_partition_status": "paper_guided_completion_exact_Yan_partition_unpublished",
            "ref15_encoder_warm_start_used": False,
            "recursive_training": False,
            "bptt_across_navigation_epochs": False,
            "eta_k_status": "active_trainable_paper_guided_Eq8_residual_IMU_error",
            "eta_k_definition": "[epsilon_g(3)_radps,epsilon_a(3)_mps2]",
            "eta_k_timing_completion": "fusion_k_output_zero_order_hold_over_following_IMU_interval",
            "eta_k_training_path": "teacher_forced_future_Eq8_mechanization_state_loss_no_direct_eta_label",
            "max_training_epochs": TRAINING_EPOCHS,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "best_val_eq30": best_val_eq30,
            "best_val_eta_forward_state_loss_completion": best_val_eta_forward,
            "best_val_joint_state_loss": best_val_joint_state_loss,
            "best_val_eq32": best_val_objective,
            "validation_fraction_completion": VALIDATION_FRACTION,
            "validation_split": "chronological_tail_with_one_boundary_sample_dropped_for_forward_eta_leakage_control",
            "early_stopping_patience_completion": EARLY_STOP_PATIENCE,
            "early_stopping_min_delta_completion": EARLY_STOP_MIN_DELTA,
            "learning_rate": LEARNING_RATE,
            "filter_block_learning_rate": FILTER_BLOCK_LEARNING_RATE,
            "representation_block_learning_rate": REPRESENTATION_BLOCK_LEARNING_RATE,
            "gamma_l2_completion": GAMMA_L2,
            "batch_size_completion": BATCH_SIZE,
        },
        OUTPUT_DIR / "pretrain_model.pt",
    )

    # -------------------------------------------------------------------------
    # Closed-loop feature construction shared by the optional stability completion
    # and the post-training recursive Data01 diagnostic. Eq. (14) IMU-difference
    # features are kept on the same bias-only compensation convention used by the
    # offline training history; the learned eta feedback changes mechanization but
    # is not fed back into its own IMU input feature channels.
    # -------------------------------------------------------------------------
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

    def _truth_target_9(fusion_index, prior_nav):
        return np.concatenate([
            fusion_imu_truth_position[fusion_index] - prior_nav.position_ecef_m,
            fusion_imu_truth_velocity[fusion_index] - prior_nav.velocity_ecef_mps,
            attitude_error_state_target(
                prior_nav.body_to_ecef_dcm,
                fusion_truth_body_to_ecef[fusion_index],
            ),
        ])

    train_end_time = float(feature_time[split - 1])
    validation_start_time = float(feature_time[split])

    # -------------------------------------------------------------------------
    # OPTIONAL NON-PAPER recursive closed-loop experiment on Data01.
    # This deliberately regenerates features from previous learned updates.
    # Yan et al. do not specify this in Eqs. (18)-(20) or Fig. 8, so it remains
    # disabled and must not be reported as the paper's training procedure.
    # -------------------------------------------------------------------------
    def _closed_loop_finetune_pass(pass_index: int):
        model.train()
        finetune_optimizer = torch.optim.Adam(
            model.parameters(), lr=CLOSED_LOOP_FINETUNE_LR
        )
        rollout_leo = LEODownlinkSimulator(
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

        nav_roll = initial_nav.copy()
        P_roll = P0.copy()
        current_eta_roll = np.zeros(IMU_ERROR_DIM)
        previous_online = None
        learned_steps = 0
        loss_sum = 0.0

        current_gyro, current_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            nav_roll.gyroscope_bias_body_radps,
            nav_roll.accelerometer_bias_body_mps2,
            gyroscope_measurement_error_body_radps=current_eta_roll[ETA_GYRO_SLICE],
            accelerometer_measurement_error_body_mps2=current_eta_roll[ETA_ACCEL_SLICE],
        )
        feature_gyro, feature_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            nav_roll.gyroscope_bias_body_radps,
            nav_roll.accelerometer_bias_body_mps2,
        )

        for event in build_exact_fusion_timeline(
            imr_time, fusion_time, through_last_fusion=True
        ):
            if isinstance(event, PropagationSegment):
                imu_index = event.imu_index
                dt = event.end_time_gpst_s - event.start_time_gpst_s
                current_gyro, current_accel = compensate_imu(
                    imr.angular_rate_body_radps[imu_index],
                    imr.acceleration_body_mps2[imu_index],
                    nav_roll.gyroscope_bias_body_radps,
                    nav_roll.accelerometer_bias_body_mps2,
                    gyroscope_measurement_error_body_radps=current_eta_roll[ETA_GYRO_SLICE],
                    accelerometer_measurement_error_body_mps2=current_eta_roll[ETA_ACCEL_SLICE],
                )
                feature_gyro, feature_accel = compensate_imu(
                    imr.angular_rate_body_radps[imu_index],
                    imr.acceleration_body_mps2[imu_index],
                    nav_roll.gyroscope_bias_body_radps,
                    nav_roll.accelerometer_bias_body_mps2,
                )
                nav_roll = mechanize_ecef(
                    nav_roll, current_gyro, current_accel, dt
                )
                F_error = build_error_state_dynamics(nav_roll, current_accel)
                Phi, Qd = discretize_process_noise_van_loan(F_error, Qc, dt)
                P_roll = Phi @ P_roll @ Phi.T + Qd
                P_roll = 0.5 * (P_roll + P_roll.T)
                continue

            fusion_index = event.fusion_index
            t = event.time_gpst_s
            if t > train_end_time + 1e-9:
                break

            epoch_data = gnss_epochs[fusion_index]
            gnss_measurements = gnss_preprocessor.prepare_epoch(
                epoch_data, nav_roll, lever_arm_b_m
            )
            leo_measurements = rollout_leo.simulate_epoch(
                t, fusion_antenna_truth_position[fusion_index]
            )
            measurements = retain_clock_observable_measurements(
                tuple(gnss_measurements) + tuple(leo_measurements)
            )
            if not measurements:
                continue

            # FDE is not inserted into this optional training completion:
            # Yan et al. place FDE in the online-testing branch of Fig. 2.
            measurement_model = build_measurement_model(
                nav_roll, measurements, lever_arm_b_m
            )

            # One classical update is used only to establish the lagged Eq. (11)-
            # (14) context at the start of a rollout window.
            if previous_online is None:
                warm_pred = np.zeros(INS_STATE_DIM)
                warm_correction, P_roll, _ = kalman_measurement_update(
                    P_roll,
                    measurement_model.innovation,
                    measurement_model.H,
                    measurement_model.R,
                )
                warm_post = warm_pred + warm_correction
                nav_roll = inject_error_state(nav_roll, warm_correction)
                warm_residual = build_innovation_only(
                    nav_roll, measurements, lever_arm_b_m
                )
                previous_online = _make_online_context(
                    measurement_model.sat_ids,
                    warm_residual,
                    warm_pred,
                    warm_post,
                    feature_accel,
                    feature_gyro,
                    previous_context=None,
                )
                current_eta_roll = np.zeros(IMU_ERROR_DIM)
                continue

            prior_nav = nav_roll.copy()
            arrays = _online_feature_arrays(
                previous_online,
                measurement_model.sat_ids,
                measurement_model.innovation,
                feature_accel,
                feature_gyro,
            )
            fixed_k, obs_k, current_mask, channel_k, innovation_k, _, _ = arrays
            target_k = _truth_target_9(fusion_index, prior_nav)
            fixed_t = torch.tensor(
                fixed_k[None], dtype=torch.float32, device=DEVICE
            )
            obs_t = torch.tensor(
                obs_k[None], dtype=torch.float32, device=DEVICE
            )
            mask_t = torch.tensor(
                current_mask[None], dtype=torch.bool, device=DEVICE
            )
            channel_t = torch.tensor(
                channel_k[None], dtype=torch.bool, device=DEVICE
            )
            innovation_t = torch.tensor(
                innovation_k[None], dtype=torch.float32, device=DEVICE
            )
            target_t = torch.tensor(
                target_k[None], dtype=torch.float64, device=DEVICE
            )

            finetune_optimizer.zero_grad(set_to_none=True)
            output = model(fixed_t, obs_t, mask_t, channel_t)
            correction_t = fig8_state_update(
                output,
                innovation_t,
            )
            # Optional non-paper fine-tuning optimizes only the current-state KG
            # loss. The eta head keeps its offline forward-interval supervision;
            # its output is still used causally for the next propagation interval.
            kg_loss, _ = _state_prediction_loss(correction_t, target_t)
            if GAMMA_L2:
                l2 = sum(
                    torch.sum(parameter * parameter)
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                loss = kg_loss + GAMMA_L2 * l2
            else:
                loss = kg_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite closed-loop fine-tune loss at pass={pass_index}, "
                    f"fusion_index={fusion_index}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                STABILITY_GRADIENT_CLIP_NORM,
            )
            finetune_optimizer.step()

            gain = output.kalman_gain[0].detach().cpu().numpy().astype(float)
            eta = output.imu_error[0].detach().cpu().numpy().astype(float)
            correction = correction_t[0].detach().cpu().numpy().astype(float)
            active_gain = gain[:, :len(measurements)]
            if (
                not np.all(np.isfinite(active_gain))
                or not np.all(np.isfinite(correction))
                or not np.all(np.isfinite(eta))
            ):
                raise FloatingPointError(
                    f"non-finite learned update during closed-loop fine-tuning "
                    f"at fusion_index={fusion_index}"
                )

            P_roll = learned_gain_covariance_update(
                P_roll,
                active_gain,
                measurement_model.H,
                measurement_model.R,
            )
            nav_roll = inject_error_state(nav_roll, correction)
            current_eta_roll = eta.copy()
            posterior_residual = build_innovation_only(
                nav_roll, measurements, lever_arm_b_m
            )

            previous_online = _make_online_context(
                measurement_model.sat_ids,
                posterior_residual,
                np.zeros(INS_STATE_DIM),
                correction,
                feature_accel,
                feature_gyro,
                previous_context=previous_online,
            )

            learned_steps += 1
            loss_sum += float(loss.detach().cpu())

            # Truncated closed-loop rollout is an explicit stability completion:
            # the exact BPTT/alternating schedule is unpublished.  Resetting only
            # on Data01-train prevents a rare early bad update from corrupting the
            # remainder of the offline fine-tuning pass.  Validation/test never reset.
            if (
                CLOSED_LOOP_RESET_INTERVAL > 0
                and learned_steps % CLOSED_LOOP_RESET_INTERVAL == 0
            ):
                nav_roll = NavigationState(
                    fusion_imu_truth_position[fusion_index].copy(),
                    fusion_imu_truth_velocity[fusion_index].copy(),
                    fusion_truth_body_to_ecef[fusion_index].copy(),
                    np.zeros(3),
                    np.zeros(3),
                )
                P_roll = P0.copy()
                current_eta_roll = np.zeros(IMU_ERROR_DIM)
                previous_online = None

        if learned_steps == 0:
            raise RuntimeError("closed-loop fine-tuning produced no learned steps")
        mean_loss = loss_sum / learned_steps
        print(
            f"closed-loop fine-tune pass {pass_index}/"
            f"{CLOSED_LOOP_FINETUNE_PASSES}: steps={learned_steps}, "
            f"mean_loss={mean_loss:.6g}"
        )
        training_history.append({
            "stage": "closed_loop_finetune",
            "pass": pass_index,
            "steps": learned_steps,
            "mean_loss": mean_loss,
            "learning_rate": CLOSED_LOOP_FINETUNE_LR,
            "reset_interval": CLOSED_LOOP_RESET_INTERVAL,
        })

    if ENABLE_STABILITY_COMPLETION:
        print(
            "\n=== OPTIONAL NON-PAPER CLOSED-LOOP STABILITY COMPLETION "
            "(DATA01) ==="
        )
        for pass_index in range(1, CLOSED_LOOP_FINETUNE_PASSES + 1):
            _closed_loop_finetune_pass(pass_index)
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "stage": "optional_closed_loop_stability_completion",
                    "pass": pass_index,
                    "nmax": nmax,
                },
                OUTPUT_DIR / f"closed_loop_pass_{pass_index:02d}.pt",
            )
    else:
        print(
            "\nOptional closed-loop stability completion: DISABLED "
            "(paper-baseline mode)"
        )

    # -------------------------------------------------------------------------
    # Recursive Data01 stability diagnostic (no optimization).
    # -------------------------------------------------------------------------
    # After restoring the best validation checkpoint, run one full Data01
    # closed-loop stability diagnostic before the expensive independent Data02 run.
    # This diagnostic never changes model weights or normalization statistics and
    # is NOT a paper training stage or a model-selection signal.
    # One classical TC update is used only to establish the lagged Eq. (11)-(14)
    # context, matching the causal warm-start policy used for Data02.
    def _recursive_data01_diagnostic_rollout():
        model.eval()
        diagnostic_leo = LEODownlinkSimulator(
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
        nav_diag = initial_nav.copy()
        P_diag = P0.copy()
        current_eta_diag = np.zeros(IMU_ERROR_DIM)
        previous_online = None
        position_errors = []
        diagnostic_steps = 0
        max_abs_innovation = 0.0
        max_abs_normalized_feature = 0.0
        max_correction_norm = 0.0
        max_eta_gyro_abs = 0.0
        max_eta_accel_abs = 0.0
        fde_stats = _new_fde_stats()

        current_gyro, current_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            nav_diag.gyroscope_bias_body_radps,
            nav_diag.accelerometer_bias_body_mps2,
            gyroscope_measurement_error_body_radps=current_eta_diag[ETA_GYRO_SLICE],
            accelerometer_measurement_error_body_mps2=current_eta_diag[ETA_ACCEL_SLICE],
        )
        feature_gyro, feature_accel = compensate_imu(
            imr.angular_rate_body_radps[0],
            imr.acceleration_body_mps2[0],
            nav_diag.gyroscope_bias_body_radps,
            nav_diag.accelerometer_bias_body_mps2,
        )

        for event in build_exact_fusion_timeline(
            imr_time, fusion_time, through_last_fusion=True
        ):
            if isinstance(event, PropagationSegment):
                imu_index = event.imu_index
                dt = event.end_time_gpst_s - event.start_time_gpst_s
                current_gyro, current_accel = compensate_imu(
                    imr.angular_rate_body_radps[imu_index],
                    imr.acceleration_body_mps2[imu_index],
                    nav_diag.gyroscope_bias_body_radps,
                    nav_diag.accelerometer_bias_body_mps2,
                    gyroscope_measurement_error_body_radps=current_eta_diag[ETA_GYRO_SLICE],
                    accelerometer_measurement_error_body_mps2=current_eta_diag[ETA_ACCEL_SLICE],
                )
                feature_gyro, feature_accel = compensate_imu(
                    imr.angular_rate_body_radps[imu_index],
                    imr.acceleration_body_mps2[imu_index],
                    nav_diag.gyroscope_bias_body_radps,
                    nav_diag.accelerometer_bias_body_mps2,
                )
                nav_diag = mechanize_ecef(
                    nav_diag, current_gyro, current_accel, dt
                )
                F_error = build_error_state_dynamics(nav_diag, current_accel)
                Phi, Qd = discretize_process_noise_van_loan(F_error, Qc, dt)
                P_diag = Phi @ P_diag @ Phi.T + Qd
                P_diag = 0.5 * (P_diag + P_diag.T)
                continue

            fusion_index = event.fusion_index
            t = event.time_gpst_s
            epoch_data = gnss_epochs[fusion_index]
            gnss_measurements = gnss_preprocessor.prepare_epoch(
                epoch_data, nav_diag, lever_arm_b_m
            )
            leo_measurements = diagnostic_leo.simulate_epoch(
                t, fusion_antenna_truth_position[fusion_index]
            )
            measurements = retain_clock_observable_measurements(
                tuple(gnss_measurements) + tuple(leo_measurements)
            )
            if not measurements:
                continue

            fde_result = ref33_fde_dia_decision(
                nav_diag,
                P_diag,
                measurements,
                lever_arm_b_m,
                FDE_SIGNIFICANCE_ALPHA,
                enabled=ENABLE_FDE,
            )
            _accumulate_fde_stats(fde_stats, fde_result)
            measurements = fde_result.measurements
            measurement_model = fde_result.measurement_model

            # Hard-FDE policy: a uniquely identified fault has already been removed
            # before this point. If Identification is unresolved (or exclusion leaves
            # no clock-observable measurements), do not expose the suspect innovation
            # to the learned gain. Keep the INS-propagated state for this epoch.
            if not measurements:
                if previous_online is not None:
                    posterior_position = gnss_antenna_position(
                        nav_diag, lever_arm_b_m
                    )
                    error = (
                        posterior_position
                        - fusion_antenna_truth_position[fusion_index]
                    )
                    if not np.all(np.isfinite(error)):
                        raise FloatingPointError(
                            "recursive Data01 diagnostic INS-only position became "
                            f"non-finite at fusion_index={fusion_index}"
                        )
                    position_errors.append(error)
                    diagnostic_steps += 1
                    previous_online = _make_online_context(
                        (),
                        np.empty(0),
                        np.zeros(INS_STATE_DIM),
                        np.zeros(INS_STATE_DIM),
                        feature_accel,
                        feature_gyro,
                        previous_context=previous_online,
                    )
                continue

            # Warm start: one classical update only to establish causal history.
            # It uses the post-FDE measurement set; no Yan Eq. (34) term is mixed
            # with a nonclassical/learned posterior.
            if previous_online is None:
                pred = np.zeros(INS_STATE_DIM)
                correction, P_diag, _ = kalman_measurement_update(
                    P_diag,
                    measurement_model.innovation,
                    measurement_model.H,
                    measurement_model.R,
                )
                post = pred + correction
                nav_diag = inject_error_state(nav_diag, correction)
                residual = build_innovation_only(
                    nav_diag, measurements, lever_arm_b_m
                )
                previous_online = _make_online_context(
                    measurement_model.sat_ids,
                    residual,
                    pred,
                    post,
                    feature_accel,
                    feature_gyro,
                    previous_context=None,
                )
                current_eta_diag = np.zeros(IMU_ERROR_DIM)
                continue

            arrays = _online_feature_arrays(
                previous_online,
                measurement_model.sat_ids,
                measurement_model.innovation,
                feature_accel,
                feature_gyro,
            )
            fixed_k, obs_k, current_mask, channel_k, innovation_k, _, _ = arrays
            max_abs_innovation = max(
                max_abs_innovation,
                float(np.max(np.abs(innovation_k))),
            )
            max_abs_normalized_feature = max(
                max_abs_normalized_feature,
                float(np.max(np.abs(fixed_k))),
                float(np.max(np.abs(obs_k))),
            )
            with torch.inference_mode():
                output = model(
                    torch.tensor(
                        fixed_k[None], dtype=torch.float32, device=DEVICE
                    ),
                    torch.tensor(
                        obs_k[None], dtype=torch.float32, device=DEVICE
                    ),
                    torch.tensor(
                        current_mask[None], dtype=torch.bool, device=DEVICE
                    ),
                    torch.tensor(
                        channel_k[None], dtype=torch.bool, device=DEVICE
                    ),
                )
                innovation_t = torch.tensor(
                    innovation_k[None], dtype=torch.float32, device=DEVICE
                )
                correction_t = fig8_state_update(
                    output,
                    innovation_t,
                )[0]

            gain = output.kalman_gain[0].cpu().numpy().astype(float)
            eta = output.imu_error[0].cpu().numpy().astype(float)
            correction = correction_t.cpu().numpy().astype(float)
            max_eta_gyro_abs = max(
                max_eta_gyro_abs, float(np.max(np.abs(eta[ETA_GYRO_SLICE])))
            )
            max_eta_accel_abs = max(
                max_eta_accel_abs, float(np.max(np.abs(eta[ETA_ACCEL_SLICE])))
            )
            active_gain = gain[:, :len(measurements)]
            if (
                not np.all(np.isfinite(active_gain))
                or not np.all(np.isfinite(correction))
                or not np.all(np.isfinite(eta))
            ):
                raise FloatingPointError(
                    "recursive Data01 diagnostic became non-finite at "
                    f"fusion_index={fusion_index}, time={t}, "
                    f"max|innovation|={max_abs_innovation:.6g}, "
                    f"max|normalized feature|={max_abs_normalized_feature:.6g}"
                )

            P_diag = learned_gain_covariance_update(
                P_diag,
                active_gain,
                measurement_model.H,
                measurement_model.R,
            )
            correction_norm = float(np.linalg.norm(correction))
            max_correction_norm = max(max_correction_norm, correction_norm)
            nav_diag = inject_error_state(nav_diag, correction)
            current_eta_diag = eta.copy()
            residual = build_innovation_only(
                nav_diag, measurements, lever_arm_b_m
            )

            previous_online = _make_online_context(
                measurement_model.sat_ids,
                residual,
                np.zeros(INS_STATE_DIM),
                correction,
                feature_accel,
                feature_gyro,
                previous_context=previous_online,
            )

            posterior_position = gnss_antenna_position(
                nav_diag, lever_arm_b_m
            )
            error = (
                posterior_position
                - fusion_antenna_truth_position[fusion_index]
            )
            if not np.all(np.isfinite(error)):
                raise FloatingPointError(
                    "recursive Data01 diagnostic position became non-finite at "
                    f"fusion_index={fusion_index}"
                )
            position_errors.append(error)
            diagnostic_steps += 1

        if diagnostic_steps == 0:
            raise RuntimeError(
                "recursive Data01 diagnostic produced no learned epochs"
            )

        errors = np.stack(position_errors)
        rmse_xyz = np.sqrt(np.mean(errors**2, axis=0))
        rmse_3d = float(
            np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        )
        return {
            "steps": diagnostic_steps,
            "rmse_ecef_xyz_m": rmse_xyz.tolist(),
            "rmse_3d_m": rmse_3d,
            "max_abs_innovation_m": max_abs_innovation,
            "max_abs_normalized_feature": max_abs_normalized_feature,
            "max_correction_norm": max_correction_norm,
            "max_abs_eta_gyro_radps": max_eta_gyro_abs,
            "max_abs_eta_accel_mps2": max_eta_accel_abs,
            "fde": fde_stats,
            "changes_weights": False,
            "purpose": "pre_Data02_recursive_stability_diagnostic",
        }

    selected_training_stage = (
        "optional_closed_loop_stability_completion"
        if ENABLE_STABILITY_COMPLETION
        else "yan_eq30_eq32_ref15_alternating_best_checkpoint"
    )

    recursive_data01_diagnostic = None
    if REQUIRE_RECURSIVE_DATA01_DIAGNOSTIC:
        print("\n=== RECURSIVE DATA01 STABILITY DIAGNOSTIC ===")
        try:
            recursive_data01_diagnostic = (
                _recursive_data01_diagnostic_rollout()
            )
        except Exception as diagnostic_error:
            raise RuntimeError(
                "The recursive Data01 diagnostic could not be completed. "
                "This exception alone does NOT prove that the trained model is "
                "unstable; it may originate from the online/FDE numerical path. "
                "Independent Data02 testing is blocked until the diagnostic "
                "failure is resolved. "
                f"Diagnostic error: {diagnostic_error!r}"
            ) from diagnostic_error

        recursive_data01_diagnostic["selected_training_stage"] = (
            selected_training_stage
        )
        print(
            "recursive Data01 diagnostic: "
            f"steps={recursive_data01_diagnostic['steps']}, "
            f"3D_RMSE={recursive_data01_diagnostic['rmse_3d_m']:.3f} m, "
            f"max|innovation|="
            f"{recursive_data01_diagnostic['max_abs_innovation_m']:.3f} m, "
            f"max|norm feature|="
            f"{recursive_data01_diagnostic['max_abs_normalized_feature']:.3g}, "
            f"max|correction|="
            f"{recursive_data01_diagnostic['max_correction_norm']:.3f}, "
            f"max|eta_g|="
            f"{recursive_data01_diagnostic['max_abs_eta_gyro_radps']:.3g} rad/s, "
            f"max|eta_a|="
            f"{recursive_data01_diagnostic['max_abs_eta_accel_mps2']:.3g} m/s^2"
        )
        print(
            "recursive Data01 FDE: "
            f"checked={recursive_data01_diagnostic['fde']['epochs_checked']}, "
            f"detected={recursive_data01_diagnostic['fde']['epochs_detected']}, "
            f"identified={recursive_data01_diagnostic['fde']['identified_fault_modes']}, "
            f"hard_exclusion={recursive_data01_diagnostic['fde']['hard_exclusion_epochs']}, "
            f"unresolved={recursive_data01_diagnostic['fde']['unresolved_epochs']}, "
            f"max(stat/Td)="
            f"{recursive_data01_diagnostic['fde']['max_statistic_to_threshold_ratio']:.3g}"
        )

    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "nmax": nmax,
            "training_nmax": nmax,
            "variable_length_gain_head": True,
            "measurement_mode": "pseudorange_only",
            "feature_timing": "causal_lagged",
            "fixed_mean": fixed_mean,
            "fixed_std": fixed_std,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "supervised_state_dim": SUPERVISED_STATE_DIM,
            "bias_gain_rows_forced_zero": True,
            "classical_bias_gain_rows_forced_zero": True,
            "masked_cla_core": "Yan_Eqs_22_to_29_Table_III",
            "cnn_tensorization": "project_specific_fixed36_broadcast_plus_two_PR_channels",
            "gain_head": "shared_per_satellite_[local_LSTM,global_attention_context]_to_9_state_column",
            "pooling": "identity_completion_because_parameters_unpublished",
            "imu_error_head": "trainable_6D_[epsilon_g_radps,epsilon_a_mps2]_future_navigation_state_Eq30_completion",
            "eta_k_timing_completion": "fusion_k_output_zero_order_hold_over_following_IMU_interval",
            "eta_k_application": "direct_Eq8_IMU_compensation_during_following_online_propagation",
            "eta_k_direct_label_used": False,
            "eta_fd_gyro_step_radps": ETA_FD_GYRO_STEP_RADPS,
            "eta_fd_accel_step_mps2": ETA_FD_ACCEL_STEP_MPS2,
            "training_mode": selected_training_stage,
            "paper_baseline_loss": "mean_raw_state_Eq30_over_current_and_following_supervised_instants_plus_Eq32_L2",
            "optimization": "Ref15_Algorithm2_style_theta_then_psi_same_state_objective",
            "alternating_theta_block": "Masked_FC_gain_head_plus_eta_head",
            "alternating_psi_block": "Masked_CNN_plus_Masked_LSTM_plus_Masked_Attention",
            "alternating_partition_status": "paper_guided_completion_exact_Yan_partition_unpublished",
            "offline_training_dataset": "Yan_Eqs18_to_20_fixed_labeled_features",
            "recursive_training_claimed_by_paper": False,
            "bptt_across_navigation_epochs": False,
            "max_training_epochs": TRAINING_EPOCHS,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "best_val_eq30": best_val_eq30,
            "best_val_eta_forward_state_loss_completion": best_val_eta_forward,
            "best_val_joint_state_loss": best_val_joint_state_loss,
            "best_val_eq32": best_val_objective,
            "training_epoch_status": "500_max_figure_guided_with_validation_early_stop",
            "validation_fraction_completion": VALIDATION_FRACTION,
            "validation_split": "chronological_tail_with_boundary_sample_dropped_for_forward_eta_target_leakage_control",
            "early_stopping_patience_completion": EARLY_STOP_PATIENCE,
            "early_stopping_min_delta_completion": EARLY_STOP_MIN_DELTA,
            "paper_initial_learning_rate": LEARNING_RATE,
            "gamma_l2_completion": GAMMA_L2,
            "batch_size_completion": BATCH_SIZE,
            "paper_baseline_gradient_clipping": False,
            "optional_stability_completion_enabled": ENABLE_STABILITY_COMPLETION,
            "closed_loop_finetune_passes_if_enabled": CLOSED_LOOP_FINETUNE_PASSES,
            "closed_loop_finetune_lr_completion": CLOSED_LOOP_FINETUNE_LR,
            "closed_loop_reset_interval_completion": CLOSED_LOOP_RESET_INTERVAL,
            "recursive_data01_diagnostic": recursive_data01_diagnostic,
        },
        OUTPUT_DIR / "best_model.pt",
    )
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )
    if recursive_data01_diagnostic is not None:
        (OUTPUT_DIR / "recursive_data01_diagnostic.json").write_text(
            json.dumps(recursive_data01_diagnostic, indent=2),
            encoding="utf-8",
        )

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
    online_warm_start_time = None
    previous_online = None
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

        fde_result = ref33_fde_dia_decision(
            nav,
            P,
            measurements,
            test_lever_arm_b_m,
            FDE_SIGNIFICANCE_ALPHA,
            enabled=ENABLE_FDE,
        )
        _accumulate_fde_stats(test_fde_stats, fde_result)
        measurements = fde_result.measurements
        measurement_model = fde_result.measurement_model

        # A detected/identified large fault is removed before any Masked-CLA
        # evaluation. If Identification is unresolved, or exclusion leaves no
        # usable clock-observable measurements, keep the INS-propagated state and
        # do not feed the suspect innovation to KalmanNet. Once causal history
        # exists, retain this INS-only epoch in the reported trajectory so the
        # evaluation is not biased by silently dropping difficult epochs.
        if not measurements:
            if previous_online is not None:
                posterior_position = gnss_antenna_position(
                    nav, test_lever_arm_b_m
                )
                online_rows.append({
                    "time": t,
                    "position": posterior_position.copy(),
                    "truth": test_fusion_antenna_truth_position[fusion_index].copy(),
                    "n_input": len(fde_result.tested_measurements),
                    "n_total": 0,
                    "n_gnss": 0,
                    "n_leo": 0,
                    "sat_ids": (),
                    "eta_gyro_radps": current_eta_test[ETA_GYRO_SLICE].copy(),
                    "eta_accel_mps2": current_eta_test[ETA_ACCEL_SLICE].copy(),
                    "fde_detected": bool(fde_result.detected),
                    "fde_identified_sat_ids": tuple(fde_result.identified_sat_ids),
                    "fde_excluded_sat_ids": tuple(fde_result.excluded_sat_ids),
                    "fde_ambiguous_sat_ids": tuple(fde_result.ambiguous_sat_ids),
                    "fde_statistic": float(fde_result.statistic),
                    "fde_threshold": float(fde_result.threshold),
                    "fde_dof": int(fde_result.dof),
                    "fde_local_identification_score": float(
                        fde_result.local_identification_score
                    ),
                    "fde_estimated_fault_m": float(fde_result.estimated_fault_m),
                    "fde_hard_exclusion_applied": bool(fde_result.excluded_sat_ids),
                    "fde_state_correction_norm": 0.0,
                    "fde_unresolved": bool(fde_result.unresolved),
                    "fusion_mode": "INS_only_after_unresolved_or_empty_FDE",
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

        # Paper Eqs. (11)-(14) require previous-epoch residual/state context.
        # Training drops its first history row, so one ordinary TC/KF update is
        # used only to establish matching causal history. It uses the post-FDE
        # observations and is excluded from reported test metrics.
        if previous_online is None:
            warm_error_state_pred = np.zeros(INS_STATE_DIM)
            warm_correction, P, _ = kalman_measurement_update(
                P,
                measurement_model.innovation,
                measurement_model.H,
                measurement_model.R,
            )
            warm_error_state_post = warm_error_state_pred + warm_correction
            nav = inject_error_state(nav, warm_correction)
            warm_posterior_position = gnss_antenna_position(
                nav, test_lever_arm_b_m
            )
            warm_posterior_residual = build_innovation_only(
                nav, measurements, test_lever_arm_b_m
            )
            previous_online = _make_online_context(
                measurement_model.sat_ids,
                warm_posterior_residual,
                warm_error_state_pred,
                warm_error_state_post,
                last_feature_accel,
                last_feature_gyro,
                previous_context=None,
            )
            current_eta_test = np.zeros(IMU_ERROR_DIM)
            online_warm_start_time = t
            continue

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
            _,
            _,
        ) = _online_feature_arrays(
            previous_online,
            current_sat_ids,
            innovation_now,
            last_feature_accel,
            last_feature_gyro,
        )

        with torch.inference_mode():
            output = model(
                torch.tensor(
                    fixed_k[None],
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                torch.tensor(
                    obs_k[None],
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                torch.tensor(
                    current_mask[None],
                    dtype=torch.bool,
                    device=DEVICE,
                ),
                torch.tensor(
                    channel_k[None],
                    dtype=torch.bool,
                    device=DEVICE,
                ),
            )
            innovation_tensor = torch.tensor(
                innovation_k[None],
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

        # This run intentionally trains only rows 0:9. Never inject
        # unsupervised accelerometer/gyro-bias corrections.
        if (
            not np.all(np.isfinite(active_gain))
            or not np.all(np.isfinite(correction))
            or not np.all(np.isfinite(eta))
        ):
            raise FloatingPointError(
                "non-finite learned gain/correction/eta during online fusion"
            )
        if not np.allclose(
            active_gain[SUPERVISED_STATE_DIM:, :],
            0.0,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(
                "online model produced nonzero bias-gain rows although rows "
                "9:15 must remain zero in the current supervised-state completion"
            )

        # Detection/Identification used the untouched INS-predicted innovation,
        # and any uniquely identified hard fault has already been excluded. The
        # learned posterior is therefore formed only from the post-FDE set. No
        # classical DIA L_i term is mixed into the arbitrary learned-KG posterior.
        P = learned_gain_covariance_update(
            P,
            active_gain,
            measurement_model.H,
            measurement_model.R,
        )
        learned_error_state_post = learned_error_state_pred + correction
        nav = inject_error_state(nav, correction)
        # Fig. 2 feedback path: eta_k is not part of the same-epoch state update.
        # It becomes the Eq. (8) IMU correction for the following propagation
        # interval and is held until the next usable KF-NET output.
        current_eta_test = eta.copy()

        posterior_position = gnss_antenna_position(
            nav, test_lever_arm_b_m
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
            "n_input": len(fde_result.tested_measurements),
            "n_total": n,
            "n_gnss": counts["G"] + counts["C"],
            "n_leo": counts["L"],
            "sat_ids": current_sat_ids,
            "eta_gyro_radps": eta[ETA_GYRO_SLICE].copy(),
            "eta_accel_mps2": eta[ETA_ACCEL_SLICE].copy(),
            "fde_detected": bool(fde_result.detected),
            "fde_identified_sat_ids": tuple(fde_result.identified_sat_ids),
            "fde_excluded_sat_ids": tuple(fde_result.excluded_sat_ids),
            "fde_ambiguous_sat_ids": tuple(fde_result.ambiguous_sat_ids),
            "fde_statistic": float(fde_result.statistic),
            "fde_threshold": float(fde_result.threshold),
            "fde_dof": int(fde_result.dof),
            "fde_local_identification_score": float(
                fde_result.local_identification_score
            ),
            "fde_estimated_fault_m": float(fde_result.estimated_fault_m),
            "fde_hard_exclusion_applied": bool(fde_result.excluded_sat_ids),
            "fde_state_correction_norm": 0.0,
            "fde_unresolved": bool(fde_result.unresolved),
            "fusion_mode": "MaskedCLA_post_hard_FDE",
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
    rmse_ned = np.sqrt(np.mean(ned_error**2, axis=0))
    rmse_3d = float(np.sqrt(np.mean(error_3d**2)))
    rmse = np.append(rmse_ned, rmse_3d)
    cdf_probability = np.arange(1, len(error_3d) + 1) / len(error_3d)

    np.savez(
        OUTPUT_DIR / "test_evaluation.npz",
        dataset_dir=np.asarray(str(test_dataset_dir)),
        time_gpst_s=online_time,
        estimate_ecef_m=estimate,
        truth_ecef_m=truth_aligned,
        ned_error_m=ned_error,
        error_3d_m=error_3d,
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
            "fde_local_identification_score",
            "fde_estimated_fault_m",
            "fde_hard_exclusion_applied",
            "fde_state_correction_norm",
            "fde_unresolved",
            "fusion_mode",
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
                row["fde_local_identification_score"],
                row["fde_estimated_fault_m"],
                int(row["fde_hard_exclusion_applied"]),
                row["fde_state_correction_norm"],
                int(row["fde_unresolved"]),
                row["fusion_mode"],
                *row["position"].tolist(),
            ])

    summary = {
        "paper_exact": False,
        "dataset_protocol": {
            "training_dataset": str(TRAIN_DATASET_DIR),
            "test_dataset": str(test_dataset_dir),
            "test_used_for_training": False,
            "normalization_source": "training_dataset_only",
        },
        "measurement_mode": "pseudorange_only",
        "orbit_source": "TLE/SGP4",
        "leo_pseudorange_error_model": (
            "Yan_Eqs_1_to_3_Ref35_iono_tropo_MP_residuals_no_URA_no_receiver_noise"
        ),
        "leo_ura_enabled": False,
        "leo_receiver_noise_enabled": False,
        "fde_enabled": bool(ENABLE_FDE),
        "ref33_dia_detection_identification_enabled": bool(ENABLE_FDE),
        "fde_protocol": {
            "paper_stage": "online_before_learned_KG",
            "detection": "Yan_Eq33_Ref33_Eq6_global_chi_square_on_INS_predicted_innovation",
            "identification": "Ref33_Eq7_single_pseudorange_measurement_fault_hypotheses",
            "fault_elimination": "hard_exclusion_before_MaskedCLA_state_update_per_Yan_online_FDE_text",
            "yan_eq34_hybrid_policy": (
                "not_applied_to_KalmanNet_posterior_because_Ref33_classical_"
                "orthogonality_does_not_hold_for_arbitrary_learned_gain"
            ),
            "unresolved_policy": "INS_only_epoch_no_suspect_measurement_sent_to_KalmanNet",
            "bias_state_policy": "no_FDE_state_injection_bias_rows_remain_frozen",
            "alpha": FDE_SIGNIFICANCE_ALPHA,
            "alpha_status": "completion_from_Ref33_numerical_examples_Yan_does_not_publish_false_alarm_probability",
            "statistical_core_self_check": FDE_STATISTICAL_SELF_CHECK,
            "projected_clock_dof": "rank_Q_nu_nu_due_to_WLS_receiver_clock_projection",
            "fault_mode_completion": "one_1D_measurement_fault_hypothesis_per_pseudorange_Yan_fault_modes_unpublished",
            "multiple_fault_policy": "single_selected_hypothesis_per_Ref33_Eq6_Eq8_no_iterative_retest",
            "ambiguity_policy": "structural_or_machine_precision_ties_are_unresolved_not_first_index_selected",
            "training_fixed_dataset_fde": False,
            "recursive_data01_stats": recursive_data01_diagnostic.get("fde", {}),
            "test_stats": test_fde_stats,
        },
        "training_nmax": int(nmax),
        "test_sequence_length_policy": "dynamic_all_valid_measurements_no_truncation",
        "feature_protocol": {
            "timing": "causal_lagged",
            "state_innovation": "stored_previous_completed_x_post_minus_x_pred_Eq12",
            "state_residual": "stored_previous_completed_x_post_minus_prior_completed_x_post_Eq13",
            "closed_loop_error_state_prior": "zero_after_feedback_reset",
            "observation_layout": "fixed_36_plus_per_satellite_[previous_residual,current_innovation]",
            "observation_layout_status": "project_specific_tensorization; paper does not publish exact reshape",
            "test_nmax_policy": "dynamic_sequence_length_with_shared_gain_head",
        },
        "masked_cla_protocol": {
            "paper_supported_core": "masked_Conv1D_then_5_layer_LSTM_then_masked_attention",
            "table_III": {
                "conv_filters": 24,
                "kernel_completion": 3,
                "stride": 1,
                "activation": "ReLU",
                "lstm_units": 64,
                "lstm_hidden_layers": 5,
                "dropout": 0.2,
                "optimizer": "Adam",
                "initial_learning_rate": LEARNING_RATE,
            },
            "pooling": "identity_completion_because_Fig8_parameters_are_not_published",
            "gain_output_shape": "[15,N_current] at inference; [15,training_Nmax] for padded training batches",
            "gain_head_completion": "shared_per_satellite_head_to_avoid_test_Nmax_leakage_or_truncation",
            "learned_gain_rows": "0:9_position_velocity_attitude",
            "forced_zero_gain_rows": "9:15_accelerometer_and_gyro_bias",
            "imu_error_output": "trainable_6D_eta_[epsilon_g,epsilon_a]_forward_feedback_completion",
            "eta_online_timing": "fusion_k_output_held_until_next_usable_network_output",
            "eta_online_application": "Eq8_subtraction_before_each_following_INS_mechanization_step",
            "eta_same_epoch_state_update": False,
            "eta_training_completion": "following_fusion_navigation_state_term_inside_common_Eq30_temporal_objective",
            "eta_direct_sensor_error_labels": False,
            "training_status": "Yan_Eq30_Eq32_common_state_objective_with_Ref15_style_alternating_blocks",
            "training_features": "offline_fixed_labeled_dataset_Yan_Eqs18_to_20",
            "recursive_training_claimed_by_paper": False,
            "bptt_across_navigation_epochs": False,
            "loss": "mean_raw_state_Eq30_over_current_and_following_supervised_instants_plus_Eq32_L2",
            "alternating_optimization": "Ref15_Algorithm2_order_theta_then_psi_same_objective",
            "alternating_theta_block": "Masked_FC_gain_head_plus_eta_head",
            "alternating_psi_block": "Masked_CNN_plus_Masked_LSTM_plus_Masked_Attention",
            "alternating_partition_status": "paper_guided_completion_exact_Yan_partition_unpublished",
            "ref15_encoder_warm_start_used": False,
            "training_epochs": TRAINING_EPOCHS,
            "training_epoch_status": "figure_guided_from_Fig15_not_text_exact",
            "batch_size_completion": BATCH_SIZE,
            "gamma_l2_completion": GAMMA_L2,
            "paper_baseline_lr_scheduler": False,
            "paper_baseline_early_stopping": True,
            "early_stopping_status": "project_reproducibility_safeguard_not_published_by_paper",
            "paper_baseline_gradient_clipping": False,
            "internal_validation_split": True,
            "internal_validation_status": "chronological_project_completion_with_boundary_drop_for_forward_eta_no_leakage",
            "optional_stability_completion_enabled": ENABLE_STABILITY_COMPLETION,
            "closed_loop_finetune_passes_if_enabled": CLOSED_LOOP_FINETUNE_PASSES,
            "closed_loop_finetune_lr_if_enabled": CLOSED_LOOP_FINETUNE_LR,
            "closed_loop_reset_interval_if_enabled": CLOSED_LOOP_RESET_INTERVAL,
            "recursive_data01_diagnostic": recursive_data01_diagnostic,
        },
        "test_causal_warm_start": {
            "method": "one_classical_TC_KF_update",
            "time_gpst_s": float(online_warm_start_time),
            "included_in_reported_test_metrics": False,
        },
        "orchestration_audit": {
            "adopted": [
                "validated_IMR_TOW_to_GPST_week_anchoring",
                "strict_monotonic_time_axis_checks",
                "central_exact_fusion_event_scheduler",
            ],
            "not_adopted": [
                "old_single_truth_reference_handling",
                "old_iid_LEO_measurement_noise_and_ideal_atmosphere_path",
                "position_only_supervised_arrays_with_full_15_row_gain",
                "single_config_online_evaluation_without_explicit_second_dataset",
                "zero_history_first_learned_update_from_standalone_online_module",
            ],
        },
        "online_update_protocol": {
            "state_correction": "nominal_dx_equals_Knet_times_raw_projected_innovation_then_if_fault_dx_equals_nominal_dx_minus_Li_nu_per_Yan_Eq34",
            "fde_before_Knet": bool(ENABLE_FDE),
            "eta_feedback": "eta_k_applied_to_following_IMU_samples_in_Eq8_zero_order_hold",
            "eta_added_to_same_epoch_state_correction": False,
            "R_used_to_compute_Knet": False,
            "covariance_bookkeeping": "Joseph_update_with_learned_gain_STANDARD_COMPLETION",
            "covariance_bookkeeping_affects_learned_state_correction": False,
            "bias_gain_runtime_guard": "rows_9_to_15_must_be_exact_zero",
            "receiver_clock_handling": "epochwise_WLS_nuisance_projected_from_innovation_H_R",
            "standalone_online_zero_history_first_epoch_adopted": False,
        },
        "runtime_accuracy_tradeoff": {
            "so3_exponential": "closed_form_Rodrigues_exact",
            "van_loan": {
                "taylor_order": VAN_LOAN_TAYLOR_ORDER,
                "max_norm_1_fast_path": VAN_LOAN_TAYLOR_MAX_NORM_1,
                "taylor_calls": int(_VAN_LOAN_STATS["taylor_calls"]),
                "exact_fallback_calls": int(_VAN_LOAN_STATS["exact_fallback_calls"]),
                "max_norm_1_seen": float(_VAN_LOAN_STATS["max_norm_1"]),
                "max_taylor_remainder_bound": float(_VAN_LOAN_STATS["max_taylor_remainder_bound"]),
                "validation_calls": int(_VAN_LOAN_STATS["validation_calls"]),
                "max_validation_phi_abs": float(_VAN_LOAN_STATS["max_validation_phi_abs"]),
                "max_validation_qd_abs": float(_VAN_LOAN_STATS["max_validation_qd_abs"]),
            },
            "leo_receive_time_prefilter": {
                "guard_deg": LEO_PREFILTER_GUARD_DEG,
                "training_checked": int(leo_simulator.prefilter_checked),
                "training_rejected": int(leo_simulator.prefilter_rejected),
                "test_checked": int(test_leo_simulator.prefilter_checked),
                "test_rejected": int(test_leo_simulator.prefilter_rejected),
                "final_mask_deg_unchanged": LEO_MIN_ELEVATION_DEG,
            },
        },
        "test_online_epochs": len(online_rows),
        "test_rmse_m": {
            "north": float(rmse[0]),
            "east": float(rmse[1]),
            "down": float(rmse[2]),
            "three_d": float(rmse[3]),
        },
        "limitations": [
            "Paper uses pseudorange + pseudorange-rate; this project is pseudorange-only.",
            "Paper uses STK/HPOP LEO orbits; this project uses TLE/SGP4.",
            "LEO clock terms remain ideal zero because the paper does not publish a reproducible simulated LEO clock-error generator.",
            "By project choice, LEO MP/NLOS uses Ref. [35] Eq. (18) multipath sigma rather than Yan et al. Eq. (4)/empirical non-Gaussian injection; URA and receiver noise are omitted.",
            "Independent zero-mean Gaussian draws realize the Ref. [35] ionosphere, troposphere, and multipath residual sigmas.",
            "The paper uses a random LEO masking-angle model [43]; this project still uses a fixed minimum elevation mask.",
            "Yan et al. do not publish the FDE false-alarm probability. The online implementation uses alpha=1e-3 from Ref. [33] quantitative DIA examples; this threshold choice is a declared completion.",
            "Ref. [33] assumes a full-rank predicted-residual covariance. Because this project projects epoch-wise GPS/BDS clock nuisance directions out of innovation/H/R, chi-square degrees of freedom use rank(Q_nu_nu), a necessary project-specific completion.",
            "Yan Eq. (34) is not applied directly to the KalmanNet posterior. Ref. [33] derives that adaptation for the classical Kalman posterior; with an arbitrary learned gain, the nominal estimation error is generally correlated with the innovation and extra cross-covariance terms are required. The online hybrid therefore uses hard fault exclusion before Masked CLA, matching Yan et al.'s stated fault-elimination ordering.",
            "When clock projection makes competing single-pseudorange fault directions structurally indistinguishable, Identification is marked unresolved rather than selecting the first index. No suspect observation from that epoch is sent to KalmanNet; the epoch is reported as INS-only.",
            "Yan Eq. (33), as printed, omits the inverse of the innovation covariance. The implementation follows Ref. [33] Eq. (6), i.e. nu^T Q_nu_nu^{-1} nu, because that is the chi-square statistic consistent with a threshold derived from false-alarm probability and with Yan's explicit reference to [33].",
            "Yan et al. do not publish the fault-mode matrices C_i. This pseudorange-only reproduction uses one 1D measurement-fault hypothesis per pseudorange and selects exactly one hypothesis per Ref. [33] Eq. (6)-(8); simultaneous multiple-fault handling is therefore not claimed to be paper-exact.",
            "The IE postprocessed truth provides position, velocity, and attitude but not accelerometer/gyro bias truth; both classical history/warm-start and Masked CLA measurement updates therefore keep nominal bias states frozen while propagating the full 15-state covariance.",
            "The first test fusion epoch is a classical TC/KF warm-start used only to form the previous-epoch causal features required by Eqs. (11)-(14); reported test metrics begin with the following learned-update epoch.",
            "The standalone online.py applies the learned model even when no previous causal snapshot exists by allowing zero history features; this behavior is intentionally not adopted because training discards the first history row.",
            "Yan et al. do not publish an explicit posterior covariance equation for a learned KG. Joseph covariance propagation with K_net and R is retained only as a declared bookkeeping completion; R is not an input to the learned gain or learned state correction.",
            "The causal-lagged feature timing and fixed-36 plus per-satellite two-channel tensorization are implementation completions because Yan et al. do not publish enough tensor-layout detail to reconstruct them uniquely.",
            "Fig. 8 depicts pooling inside the masked CNN block, but the paper publishes no pooling type, kernel, or stride; no pooling operator is guessed in this reproduction.",
            "Fig. 8 outputs eta_k as inertial-navigation measurement-error estimation and Fig. 2 feeds IMU-error correction back to Error compensation. Eq. (8) is the only explicit compensation equation, so this run uses eta_k=[epsilon_g(3),epsilon_a(3)] and applies it to the following IMU interval. The six-dimensional interpretation, zero-order hold timing, and teacher-forced forward eta loss remain paper-guided completions because the paper does not publish eta_k dimension, timing, or a separate eta target/loss.",
            "Fig. 8 is implemented as KG_k -> State Update -> Eq. (30) state loss. Eqs. (18)-(20) define an offline labeled training dataset; the paper does not state that training features are recursively regenerated from previous network outputs or that gradients are propagated through multiple navigation epochs, so recursive/BPTT training is not attributed to the paper.",
            "The eta branch has no direct pseudo-label. Its following-interval navigation consequence is treated as another supervised state instant in the same raw Eq. (30)-type temporal state objective. The future-state construction is still a paper-guided completion because Yan et al. do not publish eta_k timing/supervision. One chronological boundary sample is dropped so its future target cannot leak validation truth into training.",
            "Yan et al. explicitly state alternating optimization and cite Ref. [15]. This run follows Ref. [15] Algorithm 2 ordering (filter/output block theta, then representation block psi) with the same final state-estimation objective in both phases. The exact Yan Masked-CLA partition is unpublished, so mapping theta to the two Masked-FC output heads and psi to Masked CNN/LSTM/attention is a declared paper-guided completion.",
            "Because the supplied postprocessed truth contains position/velocity/attitude but not IMU-bias labels, Eq. (30) is reproduced on the available 9 state components and bias gain rows 9:15 are forced to zero rather than trained against invented labels.",
            "Input features are normalized, but Eq. (30) state targets are not normalized because the paper does not specify target/state-error scaling.",
            "The 500-epoch horizon is guided by Fig. 15; the exact stopping epoch is not published in the text.",
            "The paper defines zero-padding with Nmax from the training dataset but does not specify behavior when an independent test epoch has more observations than that Nmax. This run uses a declared shared per-satellite gain-head completion so every valid test observation is retained without using test labels/statistics for training or normalization.",
            "Exact fusion scheduling now uses the validated orchestration.py event scheduler; its current-sample IMU zero-order-hold convention remains a documented standard completion because the paper does not publish the raw IMU interpolation/hold rule.",
            "Runtime trade-off: Van Loan exp(A*dt) uses Taylor order 10 only when ||A*dt||_1 <= 0.2, with exact SciPy expm fallback otherwise; several early fast-path calls are regression-checked against SciPy expm.",
            "Runtime trade-off: LEOs clearly below the final 10-degree mask at receive time are prefiltered using a 0.5-degree guard; all remaining candidates still use the original transmit-time iteration and exact final mask.",
        ],
    }
    (
        OUTPUT_DIR / "summary.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINISHED INDEPENDENT TEST ===")
    print("test dataset:", test_dataset_dir)
    print("test RMSE [N, E, D, 3D] m:", rmse)
    print("outputs:", OUTPUT_DIR.resolve())
    print(
        "Van Loan fast/fallback calls:",
        _VAN_LOAN_STATS["taylor_calls"],
        "/",
        _VAN_LOAN_STATS["exact_fallback_calls"],
    )
    print(
        "Van Loan max ||A*dt||_1 / remainder bound:",
        f"{_VAN_LOAN_STATS['max_norm_1']:.6g}",
        "/",
        f"{_VAN_LOAN_STATS['max_taylor_remainder_bound']:.3e}",
    )
    print(
        "Van Loan validation max |dPhi| / |dQd|:",
        f"{_VAN_LOAN_STATS['max_validation_phi_abs']:.3e}",
        "/",
        f"{_VAN_LOAN_STATS['max_validation_qd_abs']:.3e}",
    )
    print(
        "LEO prefilter rejected train/test:",
        f"{leo_simulator.prefilter_rejected}/{leo_simulator.prefilter_checked}",
        "/",
        f"{test_leo_simulator.prefilter_rejected}/{test_leo_simulator.prefilter_checked}",
    )
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
    print(f"total wall runtime: {perf_counter() - _run_wall_start:.3f} s")
