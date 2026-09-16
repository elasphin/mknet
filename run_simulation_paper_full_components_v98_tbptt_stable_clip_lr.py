'GNSS/LEO/INS + Masked KalmanNet simulation.\n\nUses Yan et al. where details are published. Project deviations are\npseudorange-only, TLE/SGP4 LEO, and a 9-state no-bias/no-eta filter.\nData01 trains the network; Data02 is an independent, non-blocking evaluation.'
from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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
from torch.utils.checkpoint import checkpoint
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
# --- CONFIGURATION, CONSTANTS, AND COORDINATE/TIME UTILITIES ---
EARTH_SEMI_MAJOR_AXIS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SEMI_MINOR_AXIS_M = EARTH_SEMI_MAJOR_AXIS_M * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_ROTATION_RATE_RADPS = 7.292115e-5
EARTH_GRAVITATIONAL_PARAMETER_M3PS2 = 3.986004418e14
SPEED_OF_LIGHT_MPS = 299792458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0
IN_KAGGLE = Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
KAGGLE_PROJECT_ROOT = Path(os.environ.get('MKNET_PROJECT_ROOT', '/kaggle/input/datasets/elasphin/mknet-project'))
DATASET_ROOT = KAGGLE_PROJECT_ROOT
TRAIN_DATASET_DIR = DATASET_ROOT / 'Data01_20230102_ISA-100C_Vehicle_Complex'
TEST_DATASET_DIR: Path | None = DATASET_ROOT / 'Data02_20220309_ISA-100C_Vehicle_Complex'
README_XML_PATH = TRAIN_DATASET_DIR / "README.xml"
IMU_ERROR_MODEL_PATH = KAGGLE_PROJECT_ROOT / "IMUErrorModel.txt"
ROVE_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ROVE_GroundTruth.txt"
IMU_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ISA-100C_GroundTruth.txt"
RINEX_OBS_PATH = TRAIN_DATASET_DIR / "ROVE.23O"
IMR_PATH = TRAIN_DATASET_DIR / "ISA-100C.imr"
SP3_PATH = KAGGLE_PROJECT_ROOT / "WUM0MGXFIN_20230020000_01D_05M_ORB.SP3"
CLK_PATH = KAGGLE_PROJECT_ROOT / "WUM0MGXFIN_20230020000_01D_30S_CLK.CLK"
NAV_PATH = TRAIN_DATASET_DIR / "brdm0020.23p"
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
def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return value

def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
def _default_output_dir() -> Path:
    override = os.environ.get("MKNET_OUTPUT_DIR")
    if override:
        return Path(override).expanduser()
    if IN_KAGGLE:
        return Path("/kaggle/working/direct_run")
    return Path("direct_run")
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
# --- OPTIONAL FDE / DIA + INTEGRITY MODULE ---
# Both switches False -> fde_dia_v2.py is not imported or required.
FDE_DIA_ON = True
INTEGRITY_ON = True
if FDE_DIA_ON or INTEGRITY_ON:
    try:
        from fde_dia_v2 import FDEDIAEngine, IntegrityMonitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("FDE_DIA_ON=True or INTEGRITY_ON=True requires fde_dia_v2.py beside the main simulation file") from exc
# PROJECT SIMPLIFICATION: 9-state [delta-p,delta-v,delta-theta], no bias states or Fig.8 eta_k. Read-only Data01 closed-loop replay compares learned vs classical TC/KF; Data02 always runs independently.
RUN_RECURSIVE_DATA01_DIAGNOSTIC = True
RUN_CLASSICAL_TEST_BASELINE = False
# KalmanNet-family feature normalization:
# normalize each causal feature group online with its own L2 norm.
# This avoids fitting mean/std on the classical TC/KF trajectory and then
# applying those fixed statistics to recursively generated KalmanNet features.
# Yan does not publish an input-normalization rule; this is an explicitly
# documented KalmanNet-family implementation choice.
FEATURE_NORMALIZATION_MODE = 'online_group_l2'
FEATURE_L2_EPS = 1e-12
# Recursive-training hyperparameters are configured in __main__ after the environment helpers are available. No synthetic noisy-prior augmentation is used. Runtime/accuracy trade-off controls. The final scientific LEO mask remains exactly LEO_MIN_ELEVATION_DEG.  This guard is used only to avoid expensive transmit-time iteration for satellites clearly below it.
LEO_PREFILTER_GUARD_DEG = 0.5
# For ||A*dt||_1 <= 0.2, the Taylor-10 remainder bound is O(1e-15) in matrix norm. Larger steps/states fall back to scipy.linalg.expm exactly.
VAN_LOAN_TAYLOR_ORDER = 10
VAN_LOAN_TAYLOR_MAX_NORM_1 = 0.20
VAN_LOAN_VALIDATE_CALLS = 8
# Yan Eq. (2) places the ionosphere roughly at 100-1000 km; use those stated bounds.
LEO_IONOSPHERE_LOWER_HEIGHT_M = 100_000.0
LEO_IONOSPHERE_UPPER_HEIGHT_M = 1_000_000.0
# Ref. [35] error-model constants used by Yan Eq. (3). Ref. [35] states typical URA values of 1-2 m; the midpoint is used because Yan et al. do not publish the exact simulated LEO URA index.
LEO_URA_SIGMA_M = 1.5
LEO_CODE_LOOP_BANDWIDTH_HZ = 2.0
LEO_CORRELATOR_SPACING_CHIPS = 0.1
LEO_CORRELATOR_ACCUMULATION_S = 0.02
LEO_CODE_CHIPPING_RATE_HZ = 1.023e6
# Yan Eq. (4) cites goGPS [37]. These are the sample parameters published by goGPS for its elevation/CN0 weighting law.
GOGPS_A = 30.0
GOGPS_S0_DBHZ = 10.0
GOGPS_S1_DBHZ = 50.0
GOGPS_A_DB = 20.0
@dataclass(frozen=True)
class GPSTime:
    week: int
    tow_s: float
def calendar_to_gpst_seconds(year: int, month: int, day: int, hour: int, minute: int, second: float, time_system: str='GPS') -> float:
    """Calendar epoch -> continuous GPST seconds."""
    sec_int = int(math.floor(second))
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    base = (dt - GPS_EPOCH).total_seconds() + sec_int + (second - sec_int)
    system = time_system.upper()
    if system in {'GPS', 'GPST', 'GAL', 'GST', 'QZS', 'QZSST', 'IRN'}:
        return float(base)
    if system in {'BDT', 'BDS'}:
        return float(base + 14.0)
    if system in {'UTC', 'GLO'}:
        return float(base + 18.0)
    raise ValueError(f'Unsupported time system: {time_system}')
def gpst_seconds_to_week_tow(time_gpst_s: float) -> GPSTime:
    week = int(math.floor(float(time_gpst_s) / GPS_WEEK_S))
    return GPSTime(week, float(time_gpst_s) - week * GPS_WEEK_S)
def anchor_imr_tow_to_gpst_seconds(tow_s: np.ndarray, anchor_time_gpst_s: float) -> np.ndarray:
    """Attach GPS week numbers to IMR time-of-week samples."""
    tow_s = np.asarray(tow_s, dtype=float).reshape(-1)
    if tow_s.size == 0:
        return tow_s.copy()
    if not np.all(np.isfinite(tow_s)):
        raise ValueError('IMR TOW contains non-finite values')
    if np.any((tow_s < 0.0) | (tow_s >= GPS_WEEK_S)):
        raise ValueError('IMR TOW must lie in [0, 604800) seconds')
    if not math.isfinite(float(anchor_time_gpst_s)):
        raise ValueError('RINEX anchor time must be finite')
    anchor_week = int(math.floor(float(anchor_time_gpst_s) / GPS_WEEK_S))
    candidates = np.array([(anchor_week + offset) * GPS_WEEK_S + tow_s[0] for offset in (-1, 0, 1)])
    first = float(candidates[np.argmin(np.abs(candidates - float(anchor_time_gpst_s)))])
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
        raise ValueError('anchored IMR GPST tags must be strictly increasing')
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
def build_exact_fusion_timeline(imu_time_gpst_s: np.ndarray, fusion_time_gpst_s: np.ndarray, *, through_last_fusion: bool=True):
    """Yield ordered IMU-propagation and fusion events."""
    imu_time = validate_strict_time_axis('IMU', imu_time_gpst_s)
    fusion_time = np.asarray(fusion_time_gpst_s, dtype=float).reshape(-1)
    if fusion_time.size == 0:
        return
    validate_strict_time_axis('fusion', fusion_time)
    if fusion_time[0] < imu_time[0] or fusion_time[-1] > imu_time[-1]:
        raise ValueError('fusion epochs must lie inside the IMU time span')
    if through_last_fusion:
        stop = int(np.searchsorted(imu_time, fusion_time[-1], side='left')) + 1
        imu_time = imu_time[:min(max(stop, 2), len(imu_time))]
    fusion_index = 0
    segment_start = float(imu_time[0])
    for imu_index in range(1, len(imu_time)):
        interval_end = float(imu_time[imu_index])
        while fusion_index < len(fusion_time) and fusion_time[fusion_index] <= interval_end + 1e-12:
            t = float(fusion_time[fusion_index])
            if t < segment_start - 1e-12:
                raise RuntimeError('fusion scheduler encountered a fusion epoch before the current propagation segment')
            if t > segment_start + 1e-12:
                yield PropagationSegment(segment_start, t, imu_index)
            yield FusionMarker(t, fusion_index)
            segment_start = t
            fusion_index += 1
        if interval_end > segment_start + 1e-12:
            yield PropagationSegment(segment_start, interval_end, imu_index)
        segment_start = interval_end
    if fusion_index != len(fusion_time):
        raise RuntimeError('not all requested fusion epochs were scheduled')
def ecef_to_llh(position_ecef_m: np.ndarray) -> tuple[float, float, float]:
    """ECEF [m] -> WGS-84 latitude [rad], longitude [rad], height [m]."""
    x, y, z = np.asarray(position_ecef_m, dtype=float).reshape(3)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    if p < 1e-08:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        return (float(lat), lon, float(abs(z) - EARTH_SEMI_MINOR_AXIS_M))
    lat = float(np.arctan2(z, p * (1.0 - EARTH_ECCENTRICITY_SQUARED)))
    for _ in range(15):
        sin_lat = np.sin(lat)
        N = EARTH_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - EARTH_ECCENTRICITY_SQUARED * sin_lat * sin_lat)
        h = p / np.cos(lat) - N
        new_lat = float(np.arctan2(z, p * (1.0 - EARTH_ECCENTRICITY_SQUARED * N / (N + h))))
        if abs(new_lat - lat) < 1e-13:
            lat = new_lat
            break
        lat = new_lat
    sin_lat = np.sin(lat)
    N = EARTH_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - EARTH_ECCENTRICITY_SQUARED * sin_lat * sin_lat)
    h = p / np.cos(lat) - N
    return (float(lat), lon, float(h))
def c_ecef_to_ned(lat_rad: float, lon_rad: float) -> np.ndarray:
    slat, clat = (np.sin(lat_rad), np.cos(lat_rad))
    slon, clon = (np.sin(lon_rad), np.cos(lon_rad))
    return np.array([[-slat * clon, -slat * slon, clat], [-slon, clon, 0.0], [-clat * clon, -clat * slon, -slat]])
@lru_cache(maxsize=8)
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
def transform_lever_arm_vehicle_to_body(lever_vehicle_m: np.ndarray, x_rot_deg: float, y_rot_deg: float, z_rot_deg: float) -> np.ndarray:
    return c_vehicle_to_body_zxy(x_rot_deg, y_rot_deg, z_rot_deg) @ np.asarray(lever_vehicle_m, dtype=float).reshape(3)
def saastamoinen_delay_m(height_m: float, elevation_rad: float) -> float:
    """Standard Saastamoinen tropospheric delay used by the current GNSS path."""
    if elevation_rad <= 0.0:
        return float('inf')
    h = max(-100.0, min(float(height_m), 10000.0))
    temperature_k = 15.0 - 0.0065 * h + 273.15
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-05 * h) ** 5.2568
    water_vapor_hpa = 6.108 * 0.7 * math.exp((17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45))
    z = math.pi / 2.0 - elevation_rad
    return 0.002277 / math.cos(z) * (pressure_hpa + (1255.0 / temperature_k + 0.05) * water_vapor_hpa - 1.16 * math.tan(z) ** 2)
def klobuchar_delay_m(time_gps_tow_s: float, latitude_rad: float, longitude_rad: float, elevation_rad: float, azimuth_rad: float, alpha_s, beta_s) -> float:
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
    basis = np.array([1.0, phi_m, phi_m ** 2, phi_m ** 3])
    amplitude_s = max(0.0, float(alpha @ basis))
    period_s = max(72000.0, float(beta @ basis))
    phase = 2.0 * math.pi * (local_time_s - 50400.0) / period_s
    F = 1.0 + 16.0 * (0.53 - elev_sc) ** 3
    if abs(phase) < 1.57:
        delay_s = F * (5e-09 + amplitude_s * (1.0 - phase ** 2 / 2.0 + phase ** 4 / 24.0))
    else:
        delay_s = F * 5e-09
    return SPEED_OF_LIGHT_MPS * delay_s
def geometric_range(receiver_position_ecef_m: np.ndarray, satellite_position_tx_ecef_m: np.ndarray, transit_s: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Earth-rotation corrected one-way range and satellite->receiver LOS."""
    angle = EARTH_ROTATION_RATE_RADPS * float(transit_s)
    c, s = (np.cos(angle), np.sin(angle))
    C_rx_tx = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sat_rx = C_rx_tx @ np.asarray(satellite_position_tx_ecef_m, dtype=float).reshape(3)
    los = np.asarray(receiver_position_ecef_m, dtype=float).reshape(3) - sat_rx
    rho = float(np.linalg.norm(los))
    return (rho, los / rho, sat_rx)
def elevation_azimuth_from_ned_matrix(c_ecef_ned: np.ndarray, los_satellite_to_receiver_ecef: np.ndarray) -> tuple[float, float]:
    """Elevation/azimuth using a receiver NED matrix already computed for the epoch."""
    receiver_to_satellite_ned = np.asarray(c_ecef_ned, dtype=float) @ -np.asarray(los_satellite_to_receiver_ecef, dtype=float)
    elevation = float(np.arcsin(np.clip(-receiver_to_satellite_ned[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(receiver_to_satellite_ned[1], receiver_to_satellite_ned[0]) % (2.0 * np.pi))
    return (elevation, azimuth)
# --- DATASET READERS --- --- IMU noise model ---
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
def read_imu_error_models(path: str | Path) -> dict[str, IMUNoiseModelSI]:
    """Read SmartPNT IMUErrorModel.txt and return SI-unit models by IMU type."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    keys = ('ISDV_Pos', 'ISDV_Vel', 'ISDV_Att', 'ISDV_AccelBias', 'ISDV_GyrosBias', 'PNSD_Pos', 'PNSD_Vel', 'PNSD_Att', 'PNSD_AccelBias', 'PNSD_GyrosBias')
    models = {}
    d2r = np.pi / 180.0
    for match in re.finditer('IMU\\s*\\{(.*?)\\}', text, flags=re.S):
        block = match.group(1)
        type_match = re.search('IMU_Type\\s*=\\s*"([^"]+)"', block)
        if not type_match:
            continue
        values = {}
        for key in keys:
            value_match = re.search(f'{key}\\s*=\\s*([^\\r\\n]+)', block)
            if value_match:
                values[key] = np.fromstring(value_match.group(1), sep=' ', dtype=float)
        if len(values) != len(keys):
            continue
        imu_type = type_match.group(1)
        models[imu_type] = IMUNoiseModelSI(
            imu_type,
            values['ISDV_Pos'].copy(),
            values['ISDV_Vel'].copy(),
            values['ISDV_Att'] * d2r,
            values['ISDV_AccelBias'].copy(),
            values['ISDV_GyrosBias'] * d2r,
            values['PNSD_Pos'].copy(),
            values['PNSD_Vel'].copy(),
            values['PNSD_Att'] * d2r,
            values['PNSD_AccelBias'].copy(),
            values['PNSD_GyrosBias'] * d2r,
        )
    return models
# --- RINEX observation: one pseudorange signal per GPS/BDS satellite ---
_FREQUENCY_HZ = {('G', '1'): 1575420000.0, ('G', '2'): 1227600000.0, ('G', '5'): 1176450000.0, ('C', '1'): 1575420000.0, ('C', '2'): 1561098000.0, ('C', '5'): 1176450000.0, ('C', '7'): 1207140000.0, ('C', '8'): 1191795000.0, ('C', '6'): 1268520000.0}
_SIGNAL_PREFS = {'G': ('1C', '1W', '1P', '2W', '2L', '2X', '5Q', '5X', '5I'), 'C': ('2I', '1I', '2X', '1X', '1P', '1D', '5X', '5P', '5D', '7I', '7X', '6I', '6X')}
@dataclass(frozen=True)
class SatelliteMeasurement:
    sat_id: str
    constellation: str
    signal_suffix: str
    frequency_hz: float
    pseudorange_m: float
    cn0_dbhz: float | None = None
@dataclass(frozen=True)
class ObservationEpoch:
    time_gpst_s: float
    measurements: tuple[SatelliteMeasurement, ...]
class RINEXObservationFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.version = 0.0
        self.time_scale = 'GPS'
        self._observation_count: dict[str, int] = {}
        self._index_by_type: dict[str, dict[str, int]] = {}
        self._read_header()
    def _read_header(self) -> None:
        types: dict[str, list[str]] = {}
        current_system = None
        expected_count = 0
        with self.path.open('r', encoding='ascii', errors='replace') as stream:
            self.version = float(stream.readline()[:9])
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'TIME OF FIRST OBS':
                    self.time_scale = line[48:51].strip() or 'GPS'
                elif label == 'SYS / # / OBS TYPES':
                    if line[0:1].strip():
                        current_system = line[0]
                        expected_count = int(line[3:6])
                        types[current_system] = []
                    if current_system:
                        types[current_system].extend(line[7:60].split())
                        types[current_system] = types[current_system][:expected_count]
        self._observation_count = {system: len(values) for system, values in types.items()}
        self._index_by_type = {system: {obs_type: i for i, obs_type in enumerate(values)} for system, values in types.items()}
    def _decode_measurement(self, sat_id: str, fields: list[str]) -> SatelliteMeasurement | None:
        constellation = sat_id[0]
        index_by_type = self._index_by_type.get(constellation, {})
        for suffix in _SIGNAL_PREFS.get(constellation, ()):
            index = index_by_type.get('C' + suffix)
            if index is None or index >= len(fields):
                continue
            raw = fields[index].ljust(16)[:14].strip()
            if not raw:
                continue
            try:
                pseudorange = float(raw.replace('D', 'E'))
            except ValueError:
                continue
            if not math.isfinite(pseudorange) or pseudorange <= 0.0:
                continue
            if constellation == 'C' and self.version < 3.04 and suffix in {'1I', '1Q', '1X'}:
                frequency = 1561098000.0
            else:
                frequency = _FREQUENCY_HZ.get((constellation, suffix[0]))
            if frequency is None:
                continue
            cn0 = None
            snr_index = index_by_type.get('S' + suffix)
            if snr_index is not None and snr_index < len(fields):
                raw_snr = fields[snr_index].ljust(16)[:14].strip()
                if raw_snr:
                    value = float(raw_snr.replace('D', 'E'))
                    cn0 = value if math.isfinite(value) else None
            return SatelliteMeasurement(sat_id, constellation, suffix, float(frequency), float(pseudorange), cn0)
        return None
    @staticmethod
    def _fields(first_payload: str, stream, count: int) -> list[str]:
        payload = first_payload.rstrip('\n')
        fields = [payload[i:i + 16] for i in range(0, len(payload), 16)]
        while len(fields) < count:
            line = stream.readline()
            if not line:
                break
            payload = line[3:].rstrip('\n')
            fields.extend(payload[i:i + 16] for i in range(0, len(payload), 16))
        return fields[:count]
    def iter_epochs(self, allowed_constellations: set[str] | None=None, *, start_time_gpst_s: float | None=None, end_time_gpst_s: float | None=None, max_epochs: int | None=None, require_measurements: bool=False):
        """Iterate RINEX epochs with optional exact time/counter pruning."""
        yielded = 0
        with self.path.open('r', encoding='ascii', errors='replace') as stream:
            for line in stream:
                if len(line) >= 60 and line[60:80].strip() == 'END OF HEADER':
                    break
            for line in stream:
                if not line.startswith('>'):
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
                time_gpst_s = calendar_to_gpst_seconds(year, month, day, hour, minute, second, self.time_scale)
                if end_time_gpst_s is not None and time_gpst_s > float(end_time_gpst_s):
                    break
                measurements = []
                for _ in range(satellite_count):
                    sat_line = stream.readline()
                    if not sat_line:
                        break
                    sat_id = sat_line[:3].strip()
                    if not sat_id:
                        continue
                    constellation = sat_id[0]
                    fields = self._fields(sat_line[3:], stream, self._observation_count.get(constellation, 0))
                    if allowed_constellations and constellation not in allowed_constellations:
                        continue
                    measurement = self._decode_measurement(sat_id, fields)
                    if measurement is not None:
                        measurements.append(measurement)
                if start_time_gpst_s is not None and time_gpst_s < float(start_time_gpst_s):
                    continue
                if require_measurements and not measurements:
                    continue
                yield ObservationEpoch(time_gpst_s, tuple(measurements))
                yielded += 1
                if max_epochs is not None and yielded >= int(max_epochs):
                    break
# --- RINEX precise clock ---
class RINEXClock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.time_scale = 'GPS'
        self._series = self._parse()
    def _parse(self):
        records = defaultdict(lambda: [[], []])
        with self.path.open('r', encoding='ascii', errors='replace') as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'TIME SYSTEM ID':
                    fields = line[:10].split()
                    if fields:
                        self.time_scale = fields[0]
            for line in stream:
                if line[:2] != 'AS':
                    continue
                fields = line.split()
                if len(fields) < 10:
                    continue
                sat_id = fields[1]
                year, month, day, hour, minute = map(int, fields[2:7])
                second = float(fields[7])
                value_count = int(fields[8])
                values = [float(x.replace('D', 'E')) for x in fields[9:]]
                while len(values) < value_count:
                    values.extend(float(x.replace('D', 'E')) for x in stream.readline().split())
                t = calendar_to_gpst_seconds(year, month, day, hour, minute, second, self.time_scale)
                records[sat_id][0].append(t)
                records[sat_id][1].append(values[0])
        out = {}
        for sat_id, (times, bias) in records.items():
            order = np.argsort(times)
            out[sat_id] = (np.asarray(times)[order], np.asarray(bias)[order])
        return out
    def bias(self, sat_id: str, time_gpst_s: float) -> float:
        times, bias = self._series[sat_id]
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f'CLK time outside product span for {sat_id}')
        return float(np.interp(time_gpst_s, times, bias))
# --- RINEX navigation header: Klobuchar coefficients only ---
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
# --- SP3 precise orbit --- SP3 orbit interpolation; interpolation equations and units unchanged.
class SP3Orbit:
    def __init__(self, path: str | Path, interpolation_points: int=9):
        self.path = Path(path)
        self.interpolation_points = interpolation_points
        self.series = self._parse()
    def _parse(self):
        temporary = defaultdict(lambda: [[], [], []])
        current_time = None
        time_scale = 'GPS'
        with self.path.open('r', encoding='ascii', errors='replace') as stream:
            for line in stream:
                if line.startswith('%c') and len(line) >= 12:
                    candidate = line[9:12].strip()
                    if candidate.upper() in {'GPS', 'GPST', 'GAL', 'GST', 'QZS', 'IRN', 'BDT', 'BDS', 'UTC', 'GLO'}:
                        time_scale = candidate
                elif line.startswith('*'):
                    f = line[1:].split()
                    current_time = calendar_to_gpst_seconds(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[5]), time_scale)
                elif current_time is not None and line.startswith('P'):
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
                    temporary[sat_id][2].append(np.nan if abs(clock_us) >= 999999.0 else clock_us * 1e-06)
        series = {}
        for sat_id, (times, positions, clocks) in temporary.items():
            times = np.asarray(times, dtype=float)
            order = np.argsort(times)
            times = times[order]
            clock = np.asarray(clocks, dtype=float)[order]
            finite_clock = np.isfinite(clock)
            series[sat_id] = {'time': times, 'position': np.asarray(positions, dtype=float)[order], 'clock_time': times[finite_clock], 'clock_value': clock[finite_clock]}
        return series
    @staticmethod
    def _interpolate_values_only(times: np.ndarray, values: np.ndarray, query: float):
        """Barycentric position interpolation without unused derivatives."""
        scale = max(float(np.max(np.abs(times - query))), 1.0)
        x = (times - query) / scale
        y = []
        for column in range(values.shape[1]):
            p = BarycentricInterpolator(x, values[:, column], rng=0)
            y.append(float(p(0.0)))
        return np.asarray(y)
    def position(self, sat_id: str, time_gpst_s: float) -> np.ndarray:
        s = self.series[sat_id]
        times = s['time']
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f'SP3 time outside product span for {sat_id}')
        count = min(self.interpolation_points, len(times))
        i = int(np.searchsorted(times, time_gpst_s))
        start = max(0, min(len(times) - count, i - count // 2))
        w = slice(start, start + count)
        return self._interpolate_values_only(times[w], s['position'][w], time_gpst_s)
    def clock_bias(self, sat_id: str, time_gpst_s: float) -> float | None:
        s = self.series[sat_id]
        times = s['time']
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f'SP3 time outside product span for {sat_id}')
        if s['clock_time'].size == 0:
            return None
        return float(np.interp(time_gpst_s, s['clock_time'], s['clock_value']))
# --- SmartPNT IMR ---
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
    tow_s: np.ndarray
    angular_rate_body_radps: np.ndarray
    acceleration_body_mps2: np.ndarray
@lru_cache(maxsize=8)
def _read_imr_layout(path: str | Path) -> tuple[Path, IMRHeader, np.dtype, int, int]:
    """Read only the fixed IMR header and binary-record layout metadata."""
    path = Path(path)
    with path.open('rb') as stream:
        buffer = stream.read(512)
    endian = '<' if buffer[8] == 0 else '>'
    values = struct.unpack(endian + IMR_HEADER_FORMAT_BODY, buffer)
    header = IMRHeader(endian=endian, delta_theta=int(values[3]), delta_velocity=int(values[4]), data_rate_hz=float(values[5]), gyro_scale=float(values[6]), accel_scale=float(values[7]), utc_or_gps_time=int(values[8]), receiver_or_corrected_time=int(values[9]), time_tag_bias_ms=float(values[10]))
    record_bytes = struct.calcsize(endian + IMR_RECORD_FORMAT_BODY)
    payload_bytes = max(0, path.stat().st_size - 512)
    record_count = payload_bytes // record_bytes
    record_dtype = np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))], align=False)
    if record_dtype.itemsize != record_bytes:
        raise RuntimeError('NumPy IMR dtype does not match the documented record size')
    return (path, header, record_dtype, record_count, record_bytes)
def _adjust_imr_tow(tow: np.ndarray, header: IMRHeader) -> np.ndarray:
    tow = np.asarray(tow, dtype=np.float64)
    tow[tow > 604800.0] -= 604800.0
    tow -= header.time_tag_bias_ms * 1e-3
    return tow
def read_imr_tow_only(path: str | Path) -> tuple[IMRHeader, np.ndarray, int]:
    """Read IMR time tags without converting sensor channels."""
    path, header, record_dtype, record_count, _ = _read_imr_layout(path)
    records = np.memmap(path, dtype=record_dtype, mode='r', offset=512, shape=(record_count,))
    tow = np.asarray(records['tow'], dtype=np.float64).copy()
    del records
    return (header, _adjust_imr_tow(tow, header), record_count)
def read_imr(path: str | Path, scaling_mode: str='cpp_exact', *, start_record: int=0, stop_record: int | None=None) -> IMRData:
    """Read and scale a contiguous SmartPNT IMR record window."""
    path, header, record_dtype, record_count, record_bytes = _read_imr_layout(path)
    start_record = int(start_record)
    if stop_record is None:
        stop_record = record_count
    stop_record = int(stop_record)
    if not 0 <= start_record <= stop_record <= record_count:
        raise ValueError(f'invalid IMR record window [{start_record}, {stop_record}) for {record_count} records')
    count = stop_record - start_record
    records = np.fromfile(path, dtype=record_dtype, count=count, offset=512 + start_record * record_bytes)
    tow = _adjust_imr_tow(records['tow'].astype(np.float64, copy=True), header)
    gyro = records['counts'][:, :3].astype(float) * header.gyro_scale
    accel = records['counts'][:, 3:6].astype(float) * header.accel_scale
    if scaling_mode == 'cpp_exact':
        gyro *= header.data_rate_hz
        accel *= header.data_rate_hz
    else:
        if header.delta_theta:
            gyro *= header.data_rate_hz
        if header.delta_velocity:
            accel *= header.data_rate_hz
    return IMRData(tow, np.deg2rad(gyro), accel)
# --- SmartPNT README.xml ---
@dataclass(frozen=True)
class RoverMetadata:
    imu_type: str
    lever_arm_vehicle_m: np.ndarray
    mounting_xyz_deg: np.ndarray
def load_smartpnt_metadata(path: str | Path, rover_id: str='01') -> RoverMetadata:
    root = ET.fromstring(Path(path).read_text(encoding='utf-8', errors='replace'))
    for rover in root.findall('ROVE'):
        if (rover.findtext('ID') or '').strip() == str(rover_id):
            return RoverMetadata((rover.findtext('SINS_IMUType') or '').strip(), np.fromstring(rover.findtext('SINS_LeverArm_GNSS') or '', sep=' '), np.fromstring(rover.findtext('SINS_RotAngle_IMU') or '', sep=' '))
    raise KeyError(f'Rover {rover_id} not found')
@dataclass(frozen=True)
class DatasetFiles:
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
        matches.extend((path for path in directory.glob(pattern) if path.is_file()))
    matches = sorted(set(matches))
    if len(matches) != 1:
        names = ', '.join((path.name for path in matches)) or 'none'
        raise FileNotFoundError(f'Expected exactly one {label} file in {directory}, found {len(matches)}: {names}')
    return matches[0]
def resolve_dataset_files(dataset_dir: str | Path, rover_id: str='01') -> tuple[DatasetFiles, RoverMetadata]:
    """Resolve one SmartPNT-Pos dataset without assuming date-specific product filenames."""
    root = Path(dataset_dir)
    readme = root / 'README.xml'
    if not readme.exists():
        raise FileNotFoundError(f'Missing README.xml in dataset: {root}')
    rover = load_smartpnt_metadata(readme, rover_id)
    imu_truth = root / f'{rover.imu_type}_GroundTruth.txt'
    imr = root / f'{rover.imu_type}.imr'
    rover_truth = _unique_file(root, ('ROVE_GroundTruth.txt', f'ROVE_{rover_id}_GroundTruth.txt', f'Rove_{rover_id}_GroundTruth.txt'), f'rover {rover_id} ground truth')
    rover_observation = _unique_file(root, ('ROVE.*O', 'ROVE.*o', f'ROVE_{rover_id}.*O', f'ROVE_{rover_id}.*o'), f'rover {rover_id} RINEX observation')
    files = DatasetFiles(rover_ground_truth=rover_truth, imu_ground_truth=imu_truth, rinex_obs=rover_observation, imr=imr, sp3=_unique_file(root, ('*.SP3', '*.sp3'), 'SP3 precise-orbit'), clk=_unique_file(root, ('*.CLK', '*.clk'), 'RINEX clock'), nav=_unique_file(root, ('brdm*.*p', 'brdm*.*P', 'brdm*.rnx', 'BRDM*.RNX'), 'broadcast navigation'))
    missing = [str(path) for path in (imu_truth, imr) if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing dataset files:\n' + '\n'.join(missing))
    return (files, rover)
def resolve_test_dataset_dir() -> Path:
    """Resolve the independent Data02 directory."""
    if TEST_DATASET_DIR is not None:
        return Path(TEST_DATASET_DIR)
    candidates = sorted((path for path in DATASET_ROOT.iterdir() if path.is_dir() and path.resolve() != TRAIN_DATASET_DIR.resolve()))
    if len(candidates) != 1:
        names = ', '.join((path.name for path in candidates)) or 'none'
        raise RuntimeError(f'TEST_DATASET_DIR is not set and automatic selection is ambiguous. Found {len(candidates)} non-training dataset directories: {names}. Set TEST_DATASET_DIR to the exact second SmartPNT-Pos dataset.')
    return candidates[0]
# --- Inertial Explorer ground truth: only fields used by this project ---
@dataclass
class GroundTruth:
    time_gpst_s: np.ndarray
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray
    heading_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
def load_ie_ground_truth(path: str | Path) -> GroundTruth:
    lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    rows = []
    started = False
    for line in lines:
        fields = line.split()
        if len(fields) < 24:
            if started:
                break
            continue
        try:
            row = (int(fields[0]), float(fields[1]), float(fields[9]), float(fields[10]), float(fields[11]), float(fields[15]), float(fields[16]), float(fields[17]), float(fields[21]), float(fields[22]), float(fields[23]))
        except (ValueError, IndexError):
            if started:
                break
            continue
        started = True
        rows.append(row)
    a = np.asarray(rows, dtype=float)
    week = a[:, 0].astype(int)
    time_gpst_s = week.astype(float) * GPS_WEEK_S + a[:, 1]
    return GroundTruth(time_gpst_s, a[:, 2:5], a[:, 5:8], a[:, 8], a[:, 9], a[:, 10])
def interpolate_ground_truth(truth: GroundTruth, query_time_gpst_s: np.ndarray, max_gap_s: float, *, position_only: bool=False):
    """Interpolate IE position/velocity/HPR at requested GPST epochs."""
    truth_time = truth.time_gpst_s
    query = np.asarray(query_time_gpst_s, dtype=float)
    upper = np.searchsorted(truth_time, query, side='left')
    upper = np.clip(upper, 0, len(truth_time) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = truth_time[upper] == query
    lower[exact] = upper[exact]
    gap = truth_time[upper] - truth_time[lower]
    if np.any(gap > float(max_gap_s)):
        raise ValueError('Ground-truth interpolation gap is too large')
    if np.any(query < truth_time[0]) or np.any(query > truth_time[-1]):
        raise ValueError('Requested epoch is outside ground-truth time span')
    weight = np.zeros_like(query)
    nz = gap > 0.0
    weight[nz] = (query[nz] - truth_time[lower[nz]]) / gap[nz]
    w0 = 1.0 - weight
    position = w0[:, None] * truth.position_ecef_m[lower] + weight[:, None] * truth.position_ecef_m[upper]
    if position_only:
        return position
    velocity = w0[:, None] * truth.velocity_ecef_mps[lower] + weight[:, None] * truth.velocity_ecef_mps[upper]
    heading_unwrapped = np.unwrap(np.deg2rad(truth.heading_deg))
    heading = np.rad2deg(w0 * heading_unwrapped[lower] + weight * heading_unwrapped[upper]) % 360.0
    pitch = w0 * truth.pitch_deg[lower] + weight * truth.pitch_deg[upper]
    roll = w0 * truth.roll_deg[lower] + weight * truth.roll_deg[upper]
    return (position, velocity, heading, pitch, roll)
def body_to_ecef_from_ie_hpr(position_ecef_m: np.ndarray, heading_deg: float, pitch_deg: float, roll_deg: float, mounting_xyz_deg: np.ndarray) -> np.ndarray:
    """Build body->ECEF DCM with the same IE/SmartPNT convention used at initialization."""
    lat, lon, _ = ecef_to_llh(position_ecef_m)
    C_e_n = c_ecef_to_ned(lat, lon).T
    heading, pitch, roll = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh = (np.cos(heading), np.sin(heading))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cr, sr = (np.cos(roll), np.sin(roll))
    C_n_f = np.array([[cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh], [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch], [-sp, sr * cp, cr * cp]])
    C_b_v = c_vehicle_to_body_zxy(*mounting_xyz_deg)
    return _rotation(C_e_n @ (C_n_f @ C_F_V) @ C_b_v.T)
def attitude_error_state_target(prior_body_to_ecef: np.ndarray, truth_body_to_ecef: np.ndarray) -> np.ndarray:
    """Error-state attitude correction consistent with inject_error_state()."""
    relative = truth_body_to_ecef @ prior_body_to_ecef.T
    rotvec = SpatialRotation.from_matrix(relative).as_rotvec()
    return rotvec / ATTITUDE_FEEDBACK_SIGN
# --- TLE files ---
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
# --- ECEF INS, TC MEASUREMENT MODEL, AND TLE/SGP4 LEO SIMULATION ---
Array = np.ndarray
# PROJECT SIMPLIFICATION: use only position, velocity, and attitude errors. Yan Eq. (7) uses a 15-state error vector.
INS_STATE_DIM = 9
DIRECT_STATE_LABEL_DIM = 9
ATTITUDE_FEEDBACK_SIGN = -1.0
J2_UNITLESS = 1.08262668e-3
OMEGA_IE_E = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RADPS])
IDENTITY_3 = np.eye(3)
IDENTITY_STATE = np.eye(INS_STATE_DIM)
C_F_V = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
def _skew(v: Array) -> Array:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
def _so3_exponential(rotation_vector_rad: Array) -> Array:
    """SO(3) exponential via Rodrigues' formula."""
    v = np.asarray(rotation_vector_rad, dtype=float).reshape(3)
    theta2 = float(v @ v)
    K = _skew(v)
    K2 = K @ K
    if theta2 < 1e-12:
        theta4 = theta2 * theta2
        a = 1.0 - theta2 / 6.0 + theta4 / 120.0
        b = 0.5 - theta2 / 24.0 + theta4 / 720.0
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta2
    return IDENTITY_3 + a * K + b * K2
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
    def copy(self):
        return NavigationState(self.position_ecef_m.copy(), self.velocity_ecef_mps.copy(), self.body_to_ecef_dcm.copy())
def compensate_imu(measured_angular_rate_body_radps: Array, measured_specific_force_body_mps2: Array) -> tuple[Array, Array]:
    return (np.asarray(measured_angular_rate_body_radps, dtype=float).reshape(3), np.asarray(measured_specific_force_body_mps2, dtype=float).reshape(3))
def _gravitation_j2_ecef(position_ecef_m: Array) -> Array:
    x, y, z = np.asarray(position_ecef_m, dtype=float).reshape(3)
    r = float(np.linalg.norm([x, y, z]))
    z2_r2 = z * z / (r * r)
    j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / r) ** 2
    xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / r**3
    return scale * np.array([x * xy_factor, y * xy_factor, z * z_factor])
def _earth_rate_cross(vector: Array) -> Array:
    x, y, _ = np.asarray(vector, dtype=float).reshape(3)
    return np.array([-EARTH_ROTATION_RATE_RADPS * y, EARTH_ROTATION_RATE_RADPS * x, 0.0])
def effective_gravity_ecef(position_ecef_m: Array) -> Array:
    r = np.asarray(position_ecef_m, dtype=float).reshape(3)
    return _gravitation_j2_ecef(r) - _earth_rate_cross(_earth_rate_cross(r))
@lru_cache(maxsize=128)
def _earth_rotation_transition(dt_s: float) -> Array:
    """Exact closed-form Earth-rotation transition for repeated IMU step sizes."""
    matrix = _so3_exponential(-OMEGA_IE_E * float(dt_s))
    matrix.setflags(write=False)
    return matrix
def mechanize_ecef(nav: NavigationState, angular_rate_body_radps: Array, specific_force_body_mps2: Array, dt_s: float) -> NavigationState:
    """Project ECEF strapdown INS mechanization."""
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    C1 = _rotation(_earth_rotation_transition(dt) @ C0 @ _so3_exponential(np.asarray(angular_rate_body_radps, dtype=float) * dt))
    Cmid = 0.5 * (C0 + C1)
    acceleration = Cmid @ np.asarray(specific_force_body_mps2) + effective_gravity_ecef(nav.position_ecef_m) - 2.0 * _earth_rate_cross(nav.velocity_ecef_mps)
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavigationState(position, velocity, C1)
def gnss_antenna_position(nav: NavigationState, lever_arm_b_m: Array) -> Array:
    """Paper Eq. (9): IMU reference position -> GNSS antenna position."""
    return nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
# --- Pseudorange preparation ---
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
class TransmitTimeConvergenceError(RuntimeError):
    """Raised when one-satellite light-time iteration does not converge."""
def _iterate_transmit_time(position_at, sat_id: str, reception_time_gpst_s: float, receiver_position_ecef_m: Array, initial_position_ecef_m: Array, initial_transit_s: float, epsilon_position_m: float, max_iterations: int):
    previous_position = np.asarray(initial_position_ecef_m, dtype=float).reshape(3)
    transit_s = float(initial_transit_s)
    reception_time = float(reception_time_gpst_s)
    epsilon_position_m = float(epsilon_position_m)
    last_position_delta_m = float('inf')
    last_light_time_residual_m = float('inf')
    for _ in range(int(max_iterations)):
        transmit_time = reception_time - transit_s
        position = np.asarray(position_at(sat_id, transmit_time), dtype=float).reshape(3)
        rho, _, _ = geometric_range(receiver_position_ecef_m, position, transit_s)
        next_transit_s = float(rho / SPEED_OF_LIGHT_MPS)
        last_position_delta_m = float(np.linalg.norm(position - previous_position))
        last_light_time_residual_m = float(abs(next_transit_s - transit_s) * SPEED_OF_LIGHT_MPS)
        if last_position_delta_m < epsilon_position_m:
            return (transmit_time, transit_s, position)
        previous_position = position
        transit_s = next_transit_s
    raise TransmitTimeConvergenceError(f'Transmit-time iteration did not converge for {sat_id}: iterations={max_iterations}, last_satellite_position_delta_m={last_position_delta_m:.6g}, last_light_time_residual_m={last_light_time_residual_m:.6g}, last_transit_s={transit_s:.12g}')
def _gnss_transmit_time_two_pass(position_at, sat_id: str, reception_time_gpst_s: float, receiver_position_ecef_m: Array, satellite_position_reception_ecef_m: Array):
    """Project GNSS two-pass transmit-time correction with Earth rotation."""
    receiver = np.asarray(receiver_position_ecef_m, dtype=float).reshape(3)
    sat_rx = np.asarray(satellite_position_reception_ecef_m, dtype=float).reshape(3)
    reception_time = float(reception_time_gpst_s)
    rho0, _, _ = geometric_range(receiver, sat_rx, 0.0)
    tau0 = float(rho0 / SPEED_OF_LIGHT_MPS)
    tx0 = reception_time - tau0
    sat_tx0 = np.asarray(position_at(sat_id, tx0), dtype=float).reshape(3)
    rho1, _, _ = geometric_range(receiver, sat_tx0, tau0)
    tau1 = float(rho1 / SPEED_OF_LIGHT_MPS)
    tx1 = reception_time - tau1
    sat_tx1 = np.asarray(position_at(sat_id, tx1), dtype=float).reshape(3)
    return (tx1, tau1, sat_tx1)
class GNSSPreprocessor:
    def __init__(self, orbit: SP3Orbit, clock: RINEXClock, min_elevation_deg: float=5.0, use_troposphere: bool=True, use_ionosphere: bool=True, broadcast_ionosphere_coefficients: dict[str, tuple[Array, Array]] | None=None):
        self.orbit = orbit
        self.clock = clock
        self.clock_satellites = frozenset(clock._series)
        self.min_elevation_rad = np.deg2rad(min_elevation_deg)
        self.use_troposphere = use_troposphere
        self.use_ionosphere = use_ionosphere
        self.iono = broadcast_ionosphere_coefficients or {}
    def _clock_bias(self, sat_id: str, time_gpst_s: float) -> float:
        """Return the same clock source without computing unused SP3 velocity."""
        if sat_id in self.clock_satellites:
            return float(self.clock.bias(sat_id, time_gpst_s))
        clock_bias_s = self.orbit.clock_bias(sat_id, time_gpst_s)
        if clock_bias_s is None:
            raise KeyError(sat_id)
        return float(clock_bias_s)
    def prepare_measurement(self, reception_time_gpst_s: float, raw: SatelliteMeasurement, receiver_position_ecef_m: Array, receiver_llh: tuple[float, float, float] | None=None, c_ecef_ned: Array | None=None) -> PseudorangeMeasurement | None:
        try:
            initial_position = self.orbit.position(raw.sat_id, reception_time_gpst_s)
        except (KeyError, ValueError):
            return None
        try:
            transmit_time, transit_s, state_tx_position = _gnss_transmit_time_two_pass(self.orbit.position, raw.sat_id, reception_time_gpst_s, receiver_position_ecef_m, initial_position)
        except (KeyError, ValueError):
            return None
        satellite_clock_bias_s = self._clock_bias(raw.sat_id, transmit_time)
        rho, los, satellite_position_rx = geometric_range(receiver_position_ecef_m, state_tx_position, transit_s)
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
            if raw.constellation == 'G':
                reference_frequency_hz = 1575420000.0
            else:
                if raw.signal_suffix not in {'2I', '2X', '1I', '1X'}:
                    return None
                tow = (tow - 14.0) % 604800.0
                reference_frequency_hz = 1561098000.0
            lat, lon, height_m = receiver_llh
            ionosphere = klobuchar_delay_m(tow, lat, lon, elevation, azimuth, alpha, beta) * (reference_frequency_hz / raw.frequency_hz) ** 2
        troposphere = 0.0
        if self.use_troposphere:
            if height_m is None:
                height_m = receiver_llh[2]
            troposphere = saastamoinen_delay_m(height_m, elevation)
        cn0_dbhz = float(raw.cn0_dbhz) if raw.cn0_dbhz is not None else 45.0
        iono_reference_frequency_hz = 1575420000.0 if raw.constellation == 'G' else 1561098000.0
        iono_sigma = ref35_ionosphere_sigma_m(elevation, receiver_llh[0]) * (iono_reference_frequency_hz / raw.frequency_hz) ** 2 if self.use_ionosphere else 0.0
        tropo_sigma = ref35_troposphere_sigma_m(elevation) if self.use_troposphere else 0.0
        mp_sigma = yan_goGPS_mp_nlos_sigma_m(elevation, cn0_dbhz)
        receiver_sigma = ref35_receiver_noise_sigma_m(cn0_dbhz)
        sigma_code_m = math.sqrt(LEO_URA_SIGMA_M ** 2 + iono_sigma ** 2 + tropo_sigma ** 2 + mp_sigma ** 2 + receiver_sigma ** 2)
        return PseudorangeMeasurement(raw.sat_id, raw.constellation, float(raw.pseudorange_m), satellite_position_rx, satellite_clock_bias_s, float(ionosphere), float(troposphere), float(sigma_code_m))
    def prepare_epoch(self, epoch: ObservationEpoch, nav: NavigationState, lever_arm_b_m: Array) -> tuple[PseudorangeMeasurement, ...]:
        antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
        receiver_llh = ecef_to_llh(antenna_position)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch.measurements:
            measurement = self.prepare_measurement(epoch.time_gpst_s, raw, antenna_position, receiver_llh=receiver_llh, c_ecef_ned=c_ecef_ned)
            if measurement is not None:
                out.append(measurement)
        return tuple(out)
# --- 9-state error dynamics / KF (user-requested simplification) ---
def build_error_state_dynamics(nav: NavigationState, specific_force_body_mps2: Array) -> Array:
    """Project 9-state ECEF error dynamics; Yan Eq. (7) uses 15 states."""
    F = np.zeros((INS_STATE_DIM, INS_STATE_DIM))
    r_e = nav.position_ecef_m
    radius = float(np.linalg.norm(r_e))
    gravity = _gravitation_j2_ecef(r_e)
    radial = r_e / radius
    C = nav.body_to_ecef_dcm
    F[0:3, 3:6] = IDENTITY_3
    F[3:6, 0:3] = -(2.0 / radius) * np.outer(gravity, radial)
    F[3:6, 3:6] = -2.0 * OMEGA_IE_SKEW
    F[3:6, 6:9] = _skew(C @ np.asarray(specific_force_body_mps2))
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    return F
def initial_covariance_from_imu_model(model: IMUNoiseModelSI) -> Array:
    sigma = np.concatenate([model.isdv_pos_m, model.isdv_vel_mps, model.isdv_att_rad])
    return np.diag(sigma ** 2)
def continuous_process_covariance_from_imu_model(model: IMUNoiseModelSI) -> Array:
    density = np.concatenate([model.pnsd_pos_m_sqrt_s, model.pnsd_vel_mps_sqrt_s, model.pnsd_att_rad_sqrt_s])
    return np.diag(density ** 2)
_VAN_LOAN_STATS = {'taylor_calls': 0, 'exact_fallback_calls': 0, 'validation_calls': 0, 'max_norm_1': 0.0, 'max_taylor_remainder_bound': 0.0, 'max_validation_phi_abs': 0.0, 'max_validation_qd_abs': 0.0}
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
    """Discretize the 9-state process model with a guarded Van Loan exponential."""
    n = INS_STATE_DIM
    A = np.zeros((2 * n, 2 * n), dtype=float)
    A[:n, :n] = F
    A[:n, n:] = Qc
    A[n:, n:] = -F.T
    B = A * float(dt_s)
    norm_1 = float(np.linalg.norm(B, 1))
    _VAN_LOAN_STATS['max_norm_1'] = max(_VAN_LOAN_STATS['max_norm_1'], norm_1)
    if norm_1 <= VAN_LOAN_TAYLOR_MAX_NORM_1:
        E = _matrix_exponential_taylor(B, VAN_LOAN_TAYLOR_ORDER)
        _VAN_LOAN_STATS['taylor_calls'] += 1
        remainder_bound = math.exp(norm_1) * norm_1 ** (VAN_LOAN_TAYLOR_ORDER + 1) / math.factorial(VAN_LOAN_TAYLOR_ORDER + 1)
        _VAN_LOAN_STATS['max_taylor_remainder_bound'] = max(_VAN_LOAN_STATS['max_taylor_remainder_bound'], float(remainder_bound))
        if _VAN_LOAN_STATS['validation_calls'] < VAN_LOAN_VALIDATE_CALLS:
            E_ref = expm(B)
            Phi_fast = E[:n, :n]
            Qd_fast = E[:n, n:] @ Phi_fast.T
            Phi_ref = E_ref[:n, :n]
            Qd_ref = E_ref[:n, n:] @ Phi_ref.T
            _VAN_LOAN_STATS['max_validation_phi_abs'] = max(_VAN_LOAN_STATS['max_validation_phi_abs'], float(np.max(np.abs(Phi_fast - Phi_ref))))
            _VAN_LOAN_STATS['max_validation_qd_abs'] = max(_VAN_LOAN_STATS['max_validation_qd_abs'], float(np.max(np.abs(Qd_fast - Qd_ref))))
            _VAN_LOAN_STATS['validation_calls'] += 1
    else:
        E = expm(B)
        _VAN_LOAN_STATS['exact_fallback_calls'] += 1
    Phi = E[:n, :n]
    Qd = E[:n, n:] @ Phi.T
    return (Phi, 0.5 * (Qd + Qd.T))
@dataclass(frozen=True)
class TCMeasurementModel:
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]
    clock_projector: Array
def predict_pseudorange(antenna_position_ecef_m: Array, measurement: PseudorangeMeasurement) -> tuple[float, Array]:
    range_vector = np.asarray(antenna_position_ecef_m) - measurement.satellite_position_reception_ecef_m
    geometric_range = float(np.linalg.norm(range_vector))
    los = range_vector / geometric_range
    predicted = geometric_range + 0.0 - SPEED_OF_LIGHT_MPS * measurement.satellite_clock_bias_s + measurement.ionosphere_delay_m + measurement.troposphere_delay_m
    return (float(predicted), los)
def retain_clock_observable_measurements(measurements) -> tuple[PseudorangeMeasurement, ...]:
    """Remove lone GPS/BDS codes that carry no position information after clock projection."""
    measurements = tuple(measurements)
    counts = Counter((m.constellation for m in measurements if m.constellation in {'G', 'C'}))
    return tuple((m for m in measurements if m.constellation not in {'G', 'C'} or counts[m.constellation] >= 2))
def _clock_projector(measurements, variances: Array) -> Array:
    n = len(measurements)
    projector = np.eye(n)
    for system in ('G', 'C'):
        index = np.asarray([i for i, m in enumerate(measurements) if m.constellation == system], dtype=int)
        if index.size == 0:
            continue
        if index.size < 2:
            raise RuntimeError(f'{system} receiver-clock elimination requires at least two measurements')
        weights = 1.0 / variances[index]
        normalized_weight = weights / np.sum(weights)
        block = np.eye(index.size) - np.ones((index.size, 1)) @ normalized_weight[None, :]
        projector[np.ix_(index, index)] = block
    return projector
def build_measurement_model(nav: NavigationState, measurements, lever_arm_b_m: Array) -> TCMeasurementModel:
    """Build the 9-state pseudorange innovation, H, and projected R."""
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return TCMeasurementModel(np.empty(0), np.zeros((0, INS_STATE_DIM)), np.zeros((0, 0)), (), np.zeros((0, 0)))
    raw_innovation = np.empty(n)
    H_raw = np.zeros((n, INS_STATE_DIM))
    variances = np.empty(n)
    lever_e = nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    antenna_position = nav.position_ecef_m + lever_e
    lever_skew = _skew(lever_e)
    sat_ids: list[str] = []
    for i, m in enumerate(measurements):
        predicted, los = predict_pseudorange(antenna_position, m)
        raw_innovation[i] = m.pseudorange_m - predicted
        H_raw[i, 0:3] = los
        H_raw[i, 6:9] = -ATTITUDE_FEEDBACK_SIGN * (los @ lever_skew)
        variance = float(m.sigma_code_m) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError('pseudorange variance must be finite and positive')
        variances[i] = variance
        sat_ids.append(m.sat_id)
    projector = _clock_projector(measurements, variances)
    innovation = projector @ raw_innovation
    H = projector @ H_raw
    R_raw = np.diag(variances)
    R = projector @ R_raw @ projector.T
    R = 0.5 * (R + R.T)
    return TCMeasurementModel(innovation, H, R, tuple(sat_ids), projector)
def build_innovation_only(nav: NavigationState, measurements, lever_arm_b_m: Array) -> Array:
    """Recompute posterior innovation without constructing unused H/R matrices."""
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return np.empty(0)
    antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
    raw_innovation = np.empty(n)
    variances = np.empty(n)
    for i, m in enumerate(measurements):
        predicted, _ = predict_pseudorange(antenna_position, m)
        raw_innovation[i] = m.pseudorange_m - predicted
        variance = float(m.sigma_code_m) ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError('pseudorange variance must be finite and positive')
        variances[i] = variance
    return _clock_projector(measurements, variances) @ raw_innovation
# --- Optional external FDE/DIA binding ---
FDE_DIA = FDEDIAEngine(
    build_measurement_model,
    retain_clock_observable_measurements,
    IDENTITY_STATE,
) if FDE_DIA_ON else None
INTEGRITY = IntegrityMonitor(
    ecef_to_llh,
    c_ecef_to_ned,
) if INTEGRITY_ON else None
NO_FDE_EXPORT_FIELDS = {
    "fde_detected": False,
    "fde_identified_sat_ids": (),
    "fde_excluded_sat_ids": (),
    "fde_ambiguous_sat_ids": (),
    "fde_statistic": float("nan"),
    "fde_threshold": float("nan"),
    "fde_dof": 0,
    "fde_post_exclusion_statistic": float("nan"),
    "fde_post_exclusion_threshold": float("nan"),
    "fde_post_exclusion_dof": 0,
    "fde_post_exclusion_consistent": True,
    "fde_local_identification_score": float("nan"),
    "fde_estimated_fault_m": float("nan"),
    "fde_hard_exclusion_applied": False,
    "fde_state_correction_norm": 0.0,
    "fde_unresolved": False,
}
def kalman_measurement_update(P: Array, innovation: Array, H: Array, R: Array):
    """Conventional 9-state TC/KF update used for feature history and baseline comparison."""
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)
    if not np.all(np.isfinite(K)):
        raise FloatingPointError('non-finite 9-state classical Kalman gain')
    dx = K @ innovation
    I_KH = IDENTITY_STATE - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_reset = reset_error_state_covariance(P_post, dx)
    return (dx, P_reset, K)
def learned_gain_covariance_update(prior_covariance: Array, learned_gain: Array, measurement_jacobian: Array, measurement_covariance: Array, injected_error_state: Array) -> Array:
    """Joseph-form covariance bookkeeping for the learned gain; a project completion."""
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    learned_gain = np.asarray(learned_gain, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    injected_error_state = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    measurement_count = measurement_jacobian.shape[0]
    if prior_covariance.shape != (INS_STATE_DIM, INS_STATE_DIM) or measurement_jacobian.ndim != 2 or measurement_jacobian.shape[1] != INS_STATE_DIM or (learned_gain.shape != (INS_STATE_DIM, measurement_count)) or (measurement_covariance.shape != (measurement_count, measurement_count)):
        raise ValueError('inconsistent P/K/H/R dimensions in learned covariance update')
    if not np.all(np.isfinite(prior_covariance)) or not np.all(np.isfinite(learned_gain)) or (not np.all(np.isfinite(measurement_jacobian))) or (not np.all(np.isfinite(measurement_covariance))) or (not np.all(np.isfinite(injected_error_state))):
        raise ValueError('non-finite P/K/H/R in learned covariance update')
    update_matrix = IDENTITY_STATE - learned_gain @ measurement_jacobian
    posterior_covariance = update_matrix @ prior_covariance @ update_matrix.T + learned_gain @ measurement_covariance @ learned_gain.T
    return reset_error_state_covariance(posterior_covariance, injected_error_state)
def recursive_same_epoch_rmse_gate(learned_error_3d_m: Array, classical_error_3d_m: Array) -> dict:
    """Compare recursive Data01 RMSE with classical TC/KF; non-blocking diagnostic."""
    learned = np.asarray(learned_error_3d_m, dtype=float).reshape(-1)
    classical = np.asarray(classical_error_3d_m, dtype=float).reshape(-1)
    if learned.shape != classical.shape or learned.size == 0:
        return {'passed': False, 'reason': 'missing_or_misaligned_same_epoch_errors', 'epochs_compared': 0, 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
    finite = np.isfinite(learned) & np.isfinite(classical)
    if not np.all(finite):
        return {'passed': False, 'reason': 'nonfinite_or_missing_same_epoch_error', 'epochs_compared': int(np.count_nonzero(finite)), 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
    learned_rmse = float(np.sqrt(np.mean(learned ** 2)))
    classical_rmse = float(np.sqrt(np.mean(classical ** 2)))
    passed = bool(learned_rmse <= classical_rmse)
    return {'passed': passed, 'reason': 'learned_rmse_not_worse_than_classical' if passed else 'learned_rmse_worse_than_classical', 'epochs_compared': int(learned.size), 'learned_rmse_3d_m': learned_rmse, 'classical_rmse_3d_m': classical_rmse, 'criterion': 'learned_same_epoch_rmse_3d_m <= classical_same_epoch_rmse_3d_m', 'paper_basis': 'Yan_Fig9a_qualitative_learned_error_below_traditional_KF', 'absolute_threshold_used': False}
def _so3_left_jacobian(rotation_vector_rad: Array) -> Array:
    """SO(3) left Jacobian used by the attitude-error reset."""
    phi = np.asarray(rotation_vector_rad, dtype=float).reshape(3)
    theta_squared = float(phi @ phi)
    K = _skew(phi)
    K2 = K @ K
    if theta_squared < 1e-12:
        a = 0.5 - theta_squared / 24.0 + theta_squared ** 2 / 720.0
        b = 1.0 / 6.0 - theta_squared / 120.0 + theta_squared ** 2 / 5040.0
    else:
        theta = math.sqrt(theta_squared)
        a = (1.0 - math.cos(theta)) / theta_squared
        b = (theta - math.sin(theta)) / (theta_squared * theta)
    return np.eye(3) + a * K + b * K2
def reset_error_state_covariance(posterior_covariance: Array, injected_error_state: Array) -> Array:
    """Map covariance into the post-feedback 9-state error coordinates."""
    P = np.asarray(posterior_covariance, dtype=float)
    dx = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    if P.shape != (INS_STATE_DIM, INS_STATE_DIM):
        raise ValueError('posterior covariance has wrong shape for error-state reset')
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(dx)):
        raise ValueError('non-finite covariance/error-state in feedback reset')
    reset_jacobian = np.eye(INS_STATE_DIM)
    reset_jacobian[6:9, 6:9] = _so3_left_jacobian(ATTITUDE_FEEDBACK_SIGN * dx[6:9])
    reset_covariance = reset_jacobian @ P @ reset_jacobian.T
    reset_covariance = 0.5 * (reset_covariance + reset_covariance.T)
    if not np.all(np.isfinite(reset_covariance)):
        raise FloatingPointError('non-finite covariance after error-state reset')
    return reset_covariance
def inject_error_state(nav: NavigationState, dx: Array) -> NavigationState:
    dx = np.asarray(dx, dtype=float).reshape(INS_STATE_DIM)
    out = nav.copy()
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _rotation(_so3_exponential(ATTITUDE_FEEDBACK_SIGN * dx[6:9]) @ out.body_to_ecef_dcm)
    return out
# Differentiable Data01-only navigation recurrence, KalmanNet-guided; Data02 keeps the NumPy path.
@dataclass
class TorchNavigationState:
    position_ecef_m: torch.Tensor
    velocity_ecef_mps: torch.Tensor
    body_to_ecef_dcm: torch.Tensor
    @classmethod
    def from_numpy(cls, nav: NavigationState, *, device: str | torch.device, dtype: torch.dtype=torch.float64) -> 'TorchNavigationState':
        return cls(torch.as_tensor(nav.position_ecef_m, dtype=dtype, device=device).clone(), torch.as_tensor(nav.velocity_ecef_mps, dtype=dtype, device=device).clone(), torch.as_tensor(nav.body_to_ecef_dcm, dtype=dtype, device=device).clone())
    def detached_numpy(self) -> NavigationState:
        return NavigationState(self.position_ecef_m.detach().cpu().numpy().copy(), self.velocity_ecef_mps.detach().cpu().numpy().copy(), self.body_to_ecef_dcm.detach().cpu().numpy().copy())
def _torch_skew(v: torch.Tensor) -> torch.Tensor:
    v = v.reshape(3)
    x, y, z = v.unbind()
    zero = v.new_zeros(())
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
def _torch_so3_exponential(rotation_vector_rad: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues exponential matching _so3_exponential()."""
    v = rotation_vector_rad.reshape(3)
    theta2 = torch.dot(v, v)
    theta2_safe = theta2.clamp_min(1e-30)
    theta = torch.sqrt(theta2_safe)
    K = _torch_skew(v)
    K2 = K @ K
    theta4 = theta2 * theta2
    a_series = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta4 / 720.0
    a_exact = torch.sin(theta) / theta
    b_exact = (1.0 - torch.cos(theta)) / theta2_safe
    small = theta2 < 1e-10
    a = torch.where(small, a_series, a_exact)
    b = torch.where(small, b_series, b_exact)
    return torch.eye(3, dtype=v.dtype, device=v.device) + a * K + b * K2
def _torch_so3_log(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) log for the small navigation attitude errors."""
    Rm = rotation_matrix.reshape(3, 3)
    vee = torch.stack((Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]))
    sin_theta = 0.5 * torch.linalg.vector_norm(vee)
    cos_theta = 0.5 * (torch.trace(Rm) - 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    denom = (2.0 * sin_theta).clamp_min(1e-15)
    exact_factor = theta / denom
    small_factor = 0.5 + theta * theta / 12.0
    factor = torch.where(sin_theta < 1e-07, small_factor, exact_factor)
    return factor * vee
def _torch_gravitation_j2_ecef(position_ecef_m: torch.Tensor) -> torch.Tensor:
    r_e = position_ecef_m.reshape(3)
    x, y, z = r_e.unbind()
    radius = torch.linalg.vector_norm(r_e).clamp_min(1.0)
    z2_r2 = z * z / (radius * radius)
    j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / radius) ** 2
    xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / radius ** 3
    return scale * torch.stack((x * xy_factor, y * xy_factor, z * z_factor))
def _torch_effective_gravity_ecef(position_ecef_m: torch.Tensor) -> torch.Tensor:
    omega = position_ecef_m.new_tensor(OMEGA_IE_E)
    r_e = position_ecef_m.reshape(3)
    return _torch_gravitation_j2_ecef(r_e) - torch.linalg.cross(omega, torch.linalg.cross(omega, r_e))
def _torch_compensate_imu(measured_angular_rate_body_radps: torch.Tensor, measured_specific_force_body_mps2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass IMU data unchanged in the 9-state no-bias/no-eta branch."""
    return (measured_angular_rate_body_radps.reshape(3), measured_specific_force_body_mps2.reshape(3))
def _torch_mechanize_ecef(nav: TorchNavigationState, angular_rate_body_radps: torch.Tensor, specific_force_body_mps2: torch.Tensor, dt_s: float) -> TorchNavigationState:
    """Differentiable ECEF mechanization used only for Data01 BPTT."""
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    earth_rotvec = C0.new_tensor(-OMEGA_IE_E * dt)
    C1 = _torch_so3_exponential(earth_rotvec) @ C0 @ _torch_so3_exponential(angular_rate_body_radps.reshape(3) * dt)
    Cmid = 0.5 * (C0 + C1)
    omega = C0.new_tensor(OMEGA_IE_E)
    acceleration = Cmid @ specific_force_body_mps2.reshape(3) + _torch_effective_gravity_ecef(nav.position_ecef_m) - 2.0 * torch.linalg.cross(omega, nav.velocity_ecef_mps)
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return TorchNavigationState(position, velocity, C1)
def _torch_inject_error_state(nav: TorchNavigationState, dx: torch.Tensor) -> TorchNavigationState:
    dx = dx.reshape(INS_STATE_DIM)
    return TorchNavigationState(nav.position_ecef_m + dx[0:3], nav.velocity_ecef_mps + dx[3:6], _torch_so3_exponential(ATTITUDE_FEEDBACK_SIGN * dx[6:9]) @ nav.body_to_ecef_dcm)
def _torch_attitude_error_state_target(prior_body_to_ecef: torch.Tensor, truth_body_to_ecef: torch.Tensor) -> torch.Tensor:
    relative = truth_body_to_ecef.reshape(3, 3) @ prior_body_to_ecef.reshape(3, 3).T
    return _torch_so3_log(relative) / ATTITUDE_FEEDBACK_SIGN
# --- TLE/SGP4 LEO orbit and pseudorange-only simulation ---
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
    """LEO orbit source using TLE/SGP4 (WGS-72) transformed from TEME to ECEF."""
    def __init__(self, path: str | Path, max_tle_age_days: float, allow_degraded_eop: bool=False, allow_non_tle_files: bool=True):
        self.max_tle_age_s = float(max_tle_age_days) * 86400.0
        self.allow_degraded_eop = allow_degraded_eop
        by_id = {}
        for line1, line2 in read_tle_directory(path, allow_non_tle_files):
            satrec = Satrec.twoline2rv(line1, line2, WGS72)
            norad = str(satrec.satnum_str).strip()
            epoch = Time(satrec.jdsatepoch, satrec.jdsatepochF, format='jd', scale='utc')
            by_id.setdefault(f'NORAD-{norad}', []).append(_TLEElement(float(epoch.gps), satrec))
        self._elements = {sat_id: tuple(sorted(elements, key=lambda e: e.epoch_gpst_s)) for sat_id, elements in by_id.items()}
        self._element_times = {sat_id: tuple((element.epoch_gpst_s for element in elements)) for sat_id, elements in self._elements.items()}
        self.satellite_ids = tuple(sorted(self._elements, key=lambda s: int(s.split('-')[1])))
    def state_at(self, time_gpst_s: float, sat_id: str) -> LEOSatelliteState:
        elements = self._elements[sat_id]
        index = bisect_right(self._element_times[sat_id], time_gpst_s) - 1
        if index < 0:
            raise ValueError(f'No prior TLE for {sat_id}')
        element = elements[index]
        if time_gpst_s - element.epoch_gpst_s > self.max_tle_age_s:
            raise ValueError(f'TLE too old for {sat_id}')
        utc = _gpst_to_utc_time_cached(float(time_gpst_s))
        error, position_km, velocity_km_s = element.satrec.sgp4(float(utc.jd1), float(utc.jd2))
        if error:
            raise ValueError(SGP4_ERRORS.get(error, f'SGP4 error {error}'))
        position = CartesianRepresentation(np.asarray(position_km) * u.km)
        velocity = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        teme = TEME(position.with_differentials(velocity), obstime=utc)
        degraded = 'warn' if self.allow_degraded_eop else 'error'
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        return LEOSatelliteState(np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3), np.asarray(itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s)).reshape(3))
def ref35_ionosphere_sigma_m(elevation_rad: float, receiver_latitude_rad: float) -> float:
    """Ref. [35] ionospheric residual sigma [m]."""
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else (4.5 if latitude_deg <= 55.0 else 6.0)
    R = 6_378_140.0
    h_i = 350_000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))
def ref35_troposphere_sigma_m(elevation_rad: float) -> float:
    """Ref. [35] tropospheric residual sigma [m]."""
    return float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation_rad)) ** 2))
def ref35_multipath_sigma_m(elevation_rad: float) -> float:
    """Ref. [35] Eq. (18) multipath sigma used by the project LEO noise model."""
    elevation_deg = float(np.rad2deg(elevation_rad))
    if not math.isfinite(elevation_deg):
        raise ValueError("finite elevation is required")
    if elevation_deg < 0.0:
        raise ValueError("multipath model requires a nonnegative elevation angle")
    return float(0.13 + 0.53 * math.exp(-elevation_deg / 10.0))
def yan_goGPS_mp_nlos_sigma_m(elevation_rad: float, cn0_dbhz: float) -> float:
    """Project implementation of Yan Eq. (4); returns the modeled standard deviation."""
    e = max(float(elevation_rad), 1e-06)
    cn0 = float(cn0_dbhz)
    if cn0 >= GOGPS_S1_DBHZ:
        variance = 1.0
    else:
        ratio = (cn0 - GOGPS_S1_DBHZ) / (GOGPS_S0_DBHZ - GOGPS_S1_DBHZ)
        term = 10.0 ** (-(cn0 - GOGPS_S1_DBHZ) / GOGPS_A_DB) * ((GOGPS_A / 10.0 ** (-(GOGPS_S0_DBHZ - GOGPS_S1_DBHZ) / GOGPS_A_DB) - 1.0) * ratio + 1.0)
        variance = 1.0 / max(math.sin(e) ** 2, 1e-06) * term
    return float(math.sqrt(max(variance, 1e-12)))
def ref35_receiver_noise_sigma_m(cn0_dbhz: float) -> float:
    """Ref. [35] code-tracking noise sigma [m]."""
    cn0_linear = 10.0 ** (float(cn0_dbhz) / 10.0)
    d = LEO_CORRELATOR_SPACING_CHIPS
    bl = LEO_CODE_LOOP_BANDWIDTH_HZ
    tau = LEO_CORRELATOR_ACCUMULATION_S
    sigma_chips = math.sqrt(bl * d / (2.0 * cn0_linear) * (1.0 + 2.0 / ((2.0 - d) * cn0_linear * tau)))
    return float(SPEED_OF_LIGHT_MPS / LEO_CODE_CHIPPING_RATE_HZ * sigma_chips)
class LEODownlinkSimulator:
    """LEO pseudorange simulator using Yan geometry with project TLE/SGP4 and retained noise model."""
    def __init__(self, provider: TLESGP4Provider, klobuchar_coefficients: tuple[Array, Array] | None, seed: int=0, tx_epsilon_position_m: float=0.001, tx_max_iterations: int=20, minimum_elevation_deg: float=10.0, prefilter_guard_deg: float=LEO_PREFILTER_GUARD_DEG, use_ionosphere: bool=True, use_troposphere: bool=True):
        self.provider = provider
        self.klobuchar_coefficients = klobuchar_coefficients
        self.tx_epsilon_position_m = float(tx_epsilon_position_m)
        self.tx_max_iterations = int(tx_max_iterations)
        self.minimum_elevation_rad = np.deg2rad(minimum_elevation_deg)
        self.prefilter_guard_rad = np.deg2rad(float(prefilter_guard_deg))
        if self.prefilter_guard_rad < 0.0:
            raise ValueError('LEO prefilter guard must be nonnegative')
        self.use_ionosphere = bool(use_ionosphere)
        self.use_troposphere = bool(use_troposphere)
        self.prefilter_checked = 0
        self.prefilter_rejected = 0
        self.prefilter_passed = 0
        self.rng = np.random.default_rng(seed)
    def _paper_ionosphere_delay_m(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array, satellite_position_reception_ecef_m: Array, elevation_rad: float, azimuth_rad: float, receiver_llh: tuple[float, float, float] | None=None) -> tuple[float, float]:
        if not self.use_ionosphere:
            return (0.0, 0.0)
        if self.klobuchar_coefficients is None:
            raise ValueError('Klobuchar coefficients are required for Yan et al. Eq. (2)')
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
            scale = (satellite_height_m - LEO_IONOSPHERE_LOWER_HEIGHT_M) / (LEO_IONOSPHERE_UPPER_HEIGHT_M - LEO_IONOSPHERE_LOWER_HEIGHT_M)
        return (float(scale * iklo_m), float(scale))
    def simulate_one(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array, sat_id: str, receiver_llh: tuple[float, float, float] | None=None, c_ecef_ned: Array | None=None):
        if receiver_llh is None:
            receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        if c_ecef_ned is None:
            c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        initial_state = self.provider.state_at(receive_time_gpst_s, sat_id)
        initial_range, initial_los, _ = geometric_range(receiver_position_ecef_m, initial_state.position_ecef_m, 0.0)
        initial_elevation, _ = elevation_azimuth_from_ned_matrix(c_ecef_ned, initial_los)
        self.prefilter_checked += 1
        if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
            self.prefilter_rejected += 1
            return None
        self.prefilter_passed += 1
        transmit_time, transit_s, state_tx_position = _iterate_transmit_time(lambda sid, t: self.provider.state_at(t, sid).position_ecef_m, sat_id, receive_time_gpst_s, receiver_position_ecef_m, initial_state.position_ecef_m, initial_range / SPEED_OF_LIGHT_MPS, self.tx_epsilon_position_m, self.tx_max_iterations)
        rho, los, sat_rx = geometric_range(receiver_position_ecef_m, state_tx_position, transit_s)
        elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
        if elevation < self.minimum_elevation_rad:
            return None
        receiver_lat, _, receiver_height_m = receiver_llh
        ionosphere_m, ionosphere_path_scale = self._paper_ionosphere_delay_m(receive_time_gpst_s, receiver_position_ecef_m, sat_rx, elevation, azimuth, receiver_llh=receiver_llh)
        troposphere_m = saastamoinen_delay_m(receiver_height_m, elevation) if self.use_troposphere else 0.0
        iono_sigma_m = ionosphere_path_scale * ref35_ionosphere_sigma_m(elevation, receiver_lat) if self.use_ionosphere else 0.0
        tropo_sigma_m = ref35_troposphere_sigma_m(elevation) if self.use_troposphere else 0.0
        mp_sigma_m = ref35_multipath_sigma_m(elevation)
        ionosphere_residual_m = float(self.rng.normal(0.0, iono_sigma_m)) if iono_sigma_m > 0.0 else 0.0
        troposphere_residual_m = float(self.rng.normal(0.0, tropo_sigma_m)) if tropo_sigma_m > 0.0 else 0.0
        mp_nlos_error_m = float(self.rng.normal(0.0, mp_sigma_m)) if mp_sigma_m > 0.0 else 0.0
        total_variance_m2 = iono_sigma_m ** 2 + tropo_sigma_m ** 2 + mp_sigma_m ** 2
        sigma_code_m = math.sqrt(max(total_variance_m2, 1e-12))
        pseudorange_m = rho + ionosphere_m + troposphere_m + ionosphere_residual_m + troposphere_residual_m + mp_nlos_error_m
        return PseudorangeMeasurement(sat_id, 'L', float(pseudorange_m), sat_rx, 0.0, float(ionosphere_m), float(troposphere_m), float(sigma_code_m))
    def simulate_epoch(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array):
        receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        measurements = []
        for sat_id in self.provider.satellite_ids:
            try:
                measurement = self.simulate_one(receive_time_gpst_s, receiver_position_ecef_m, sat_id, receiver_llh=receiver_llh, c_ecef_ned=c_ecef_ned)
            except ValueError:
                continue
            except RuntimeError as exc:
                if 'Transmit-time iteration did not converge' in str(exc):
                    continue
                raise
            if measurement is not None:
                measurements.append(measurement)
        return tuple(measurements)
def _scan_test_sequence_capacity(dataset_dir: str | Path, max_epochs: int | None, tle_provider: TLESGP4Provider, leo_seed: int) -> int:
    """Return the maximum raw GNSS + simulated visible-LEO count in Data02."""
    files, _ = resolve_dataset_files(dataset_dir, '01')
    antenna_truth = load_ie_ground_truth(files.rover_ground_truth)
    imu_truth = load_ie_ground_truth(files.imu_ground_truth)
    rinex = RINEXObservationFile(files.rinex_obs)
    first_epoch = next(rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    if first_epoch is None:
        raise ValueError('Test RINEX file contains no GPS/BDS epochs')
    _, imr_tow, imr_record_count = read_imr_tow_only(files.imr)
    if imr_record_count < 2:
        raise ValueError('Test IMR file contains fewer than two usable samples')
    imr_time = anchor_imr_tow_to_gpst_seconds(imr_tow, float(first_epoch.time_gpst_s))
    antenna_time = validate_strict_time_axis('test antenna truth', antenna_truth.time_gpst_s)
    imu_truth_time = validate_strict_time_axis('test IMU truth', imu_truth.time_gpst_s)
    common_start = max(float(imr_time[0]), float(antenna_time[0]), float(imu_truth_time[0]))
    common_end = min(float(imr_time[-1]), float(antenna_time[-1]), float(imu_truth_time[-1]))
    start = int(np.searchsorted(imr_time, common_start, side='left'))
    if start >= len(imr_time):
        raise ValueError('Test common time span starts after the IMR file')
    epochs = tuple(rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=float(imr_time[start]), end_time_gpst_s=common_end, max_epochs=max_epochs, require_measurements=True))
    if not epochs:
        raise ValueError('Test RINEX file contains no usable GPS/BDS epochs')
    fusion_time = np.asarray([epoch.time_gpst_s for epoch in epochs], dtype=float)
    antenna_position = interpolate_ground_truth(antenna_truth, fusion_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    ionosphere_coefficients = None
    if USE_IONOSPHERE:
        iono_header = read_rinex_navigation_header(files.nav)
        ionosphere_coefficients = (np.asarray(iono_header['GPSA']), np.asarray(iono_header['GPSB']))
    leo_simulator = LEODownlinkSimulator(tle_provider, ionosphere_coefficients, seed=leo_seed, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    maximum = 0
    for epoch, receiver_position in zip(epochs, antenna_position):
        leo_measurements = leo_simulator.simulate_epoch(float(epoch.time_gpst_s), receiver_position)
        maximum = max(maximum, len(epoch.measurements) + len(leo_measurements))
    if maximum <= 0:
        raise ValueError('Automatic Data02 Nmax scan found no observations')
    return int(maximum)
# --- MASKED CLA NETWORK --- Yan Eqs. (10)-(29); 9-state/pseudorange-only. Pooling and [9,Nmax] FC reshape are project completions.
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 24 in the user-requested 9-state branch
OBSERVATION_FEATURE_DIM = 2                 # [Delta y_{k-1}, Delta ytilde_k]
# Exact Yan Eq. (15)/(16) network-input order:
# [Delta alpha, Delta omega, Delta xhat, Delta xtilde,
#  Delta y_{k-1}(1:Nk), Delta ytilde_k(1:Nk), zero-padding]
SUPERVISED_STATE_DIM = DIRECT_STATE_LABEL_DIM
MASK_NORMALIZATION_EPS = 1e-6
FIG8_POOL_KERNEL_SIZE = 3
@dataclass(frozen=True)
class MaskedCLAOutput:
    """Masked-CLA output.

    recurrent_state stores the Yan Eq. (25) temporal LSTM memory separately
    for every padded feature position t:
        hidden/cell: [num_layers, B, D, hidden_size]
    where k (successive fusion epochs), not D, is the recurrent time axis.
    """
    kalman_gain: torch.Tensor
    attention: torch.Tensor
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None
# Yan Eqs. (22)-(23): masked convolution/pooling.
class MaskedConv1d(nn.Module):
    """Masked Conv1D over Yan Eq. (15)/(16) position axis t: [B,D] -> [B,D,24]."""
    def __init__(self, out_channels: int=24, kernel_size: int=3, pool_kernel_size: int=FIG8_POOL_KERNEL_SIZE) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError('out_channels must be positive')
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        if pool_kernel_size <= 0 or pool_kernel_size % 2 == 0:
            raise ValueError('pool_kernel_size must be a positive odd integer')
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.pool_kernel_size = int(pool_kernel_size)
        self.pool_padding = self.pool_kernel_size // 2
        self.register_buffer('_mask_kernel', torch.ones(1, 1, self.kernel_size), persistent=False)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or mask.ndim != 2:
            raise ValueError('MaskedConv1d expects x/mask=[B,D]')
        if x.shape != mask.shape:
            raise ValueError('MaskedConv1d x/mask shapes do not match')
        m = mask.to(dtype=x.dtype).unsqueeze(1)
        x_masked = x.unsqueeze(1) * m
        z = F.conv1d(x_masked, self.weight, bias=None, stride=1, padding=self.padding)
        local_count = F.conv1d(m, self._mask_kernel.to(dtype=x.dtype), stride=1, padding=self.padding)
        valid_window = (local_count > 0).to(dtype=x.dtype)
        feature_maps = F.relu(z / local_count.clamp_min(MASK_NORMALIZATION_EPS) + self.bias.view(1, -1, 1) * valid_window)
        feature_maps = feature_maps * m
        if self.pool_kernel_size > 1:
            very_negative = torch.finfo(feature_maps.dtype).min
            pool_input = torch.where(m.bool(), feature_maps, torch.full_like(feature_maps, very_negative))
            pooled = F.max_pool1d(pool_input, kernel_size=self.pool_kernel_size, stride=1, padding=self.pool_padding)
            feature_maps = torch.where(m.bool(), pooled, torch.zeros_like(pooled))
        return feature_maps.transpose(1, 2)
# Yan Eqs. (24)-(25): five-layer masked LSTM.
class MaskedStackedLSTM(nn.Module):
    """Yan temporal masked LSTM with one recurrent stream per feature position.

    Paper indexing:
        k = fusion/training epoch (temporal recurrence axis)
        t = position in the padded/masked feature vector

    Input at one fusion epoch:
        x    : [B, D, F]
        mask : [B, D]

    Recurrent state carried from fusion epoch k-1 to k:
        h, c : [num_layers, B, D, hidden_size]

    The D feature positions are NOT interpreted as the LSTM sequence length.
    They are flattened into the batch axis so every position receives exactly
    one temporal LSTM update per fusion epoch. Yan Eq. (25) is then applied
    explicitly: masked positions retain their previous hidden/cell state.
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
            raise ValueError('input_size, hidden_size and num_layers must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0,1)')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError('MaskedStackedLSTM expects x=[B,D,F] and mask=[B,D]')
        if x.shape[:2] != mask.shape:
            raise ValueError('MaskedStackedLSTM x/mask position shapes do not match')
        if x.shape[2] != self.input_size:
            raise ValueError(
                f'MaskedStackedLSTM feature size must be {self.input_size}, got {x.shape[2]}'
            )

        batch_size, position_count, _ = x.shape
        if position_count <= 0:
            raise ValueError('MaskedStackedLSTM requires at least one feature position')

        state_shape = (
            self.num_layers,
            batch_size,
            position_count,
            self.hidden_size,
        )

        if recurrent_state is None:
            previous_hidden = x.new_zeros(state_shape)
            previous_cell = x.new_zeros(state_shape)
        else:
            if len(recurrent_state) != 2:
                raise ValueError('LSTM recurrent_state must be (hidden, cell)')
            previous_hidden, previous_cell = recurrent_state
            for name, state in (
                ('hidden', previous_hidden),
                ('cell', previous_cell),
            ):
                if tuple(state.shape) != state_shape:
                    raise ValueError(
                        f'recurrent {name} must have shape {state_shape}, '
                        f'got {tuple(state.shape)}'
                    )
                if state.device != x.device or state.dtype != x.dtype:
                    raise ValueError(
                        f'recurrent {name} must match input device/dtype'
                    )

        # Yan Eq. (25) uses k as time. Therefore one model call = one LSTM
        # time step. Feature position t is parallelized in the batch axis:
        # [B,D,F] -> [B*D,1,F].
        step_input = x.reshape(batch_size * position_count, 1, self.input_size)

        h0 = (
            previous_hidden
            .permute(0, 1, 2, 3)
            .reshape(self.num_layers, batch_size * position_count, self.hidden_size)
            .contiguous()
        )
        c0 = (
            previous_cell
            .permute(0, 1, 2, 3)
            .reshape(self.num_layers, batch_size * position_count, self.hidden_size)
            .contiguous()
        )

        _, (candidate_hidden_flat, candidate_cell_flat) = self.lstm(
            step_input,
            (h0, c0),
        )

        candidate_hidden = candidate_hidden_flat.reshape(
            self.num_layers,
            batch_size,
            position_count,
            self.hidden_size,
        )
        candidate_cell = candidate_cell_flat.reshape(
            self.num_layers,
            batch_size,
            position_count,
            self.hidden_size,
        )

        # Yan Eq. (25):
        # h_k^(t) = M_k,t * h_candidate + (1-M_k,t) * h_{k-1}^(t)
        # c_k^(t) = M_k,t * c_candidate + (1-M_k,t) * c_{k-1}^(t)
        valid = mask.bool().unsqueeze(0).unsqueeze(-1)
        next_hidden = torch.where(valid, candidate_hidden, previous_hidden)
        next_cell = torch.where(valid, candidate_cell, previous_cell)

        # Top-layer hidden state for every current-epoch position.
        # Attention (Yan Eqs. 26-29) subsequently fuses over D positions.
        output = next_hidden[-1]  # [B,D,H]
        return output, (next_hidden, next_cell)
# Yan Eqs. (26)-(29): masked attention.
class MaskedAttention(nn.Module):
    """Yan Eqs. (26)-(29): masked attention over position t at the current epoch k."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size must be positive')
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or mask.ndim != 2 or h.shape[:2] != mask.shape:
            raise ValueError('MaskedAttention expects h=[B,D,H], mask=[B,D]')
        valid = mask.bool()
        score = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        score = score.masked_fill(~valid, -torch.inf)
        all_masked = ~valid.any(dim=1)
        safe_score = score.masked_fill(all_masked.unsqueeze(1), 0.0)
        alpha = torch.softmax(safe_score, dim=1)
        alpha = alpha * valid.to(dtype=alpha.dtype)
        denom = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        alpha = torch.where(all_masked.unsqueeze(1), torch.zeros_like(alpha), alpha / denom)
        context = torch.sum(alpha.unsqueeze(-1) * h, dim=1)
        return (context, alpha)
# Yan Fig. 8 Masked-CLA; final fixed-Nmax FC reshape remains the existing project completion.
class MaskedCLA(nn.Module):
    """Yan-indexed Masked CNN-LSTM-attention estimator for the 9-state Kalman gain."""
    def __init__(self, nmax: int, dropout: float=0.2) -> None:
        super().__init__()
        if nmax <= 0:
            raise ValueError('nmax must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0,1)')
        self.nmax = int(nmax)
        self.dropout = float(dropout)
        self.eq15_padded_dim = FIXED_FEATURE_DIM + 2 * self.nmax
        self.conv = MaskedConv1d(out_channels=24, kernel_size=3, pool_kernel_size=FIG8_POOL_KERNEL_SIZE)
        self.lstm = MaskedStackedLSTM(input_size=24, hidden_size=64, num_layers=5, dropout=self.dropout)
        self.attention = MaskedAttention(64)
        self.gain_head = nn.Linear(64, INS_STATE_DIM * self.nmax)
    def forward(self, fixed: torch.Tensor, observations: torch.Tensor, mask: torch.Tensor, channel_mask: torch.Tensor, recurrent_state: tuple[torch.Tensor, torch.Tensor] | None=None) -> MaskedCLAOutput:
        if fixed.ndim != 2 or observations.ndim != 3:
            raise ValueError('fixed must be [B,24] and observations [B,Nmax,2]')
        if mask.ndim != 2 or channel_mask.ndim != 3:
            raise ValueError('mask/channel_mask have invalid ranks')
        batch_size = fixed.shape[0]
        expected_obs = (batch_size, self.nmax, OBSERVATION_FEATURE_DIM)
        if tuple(fixed.shape) != (batch_size, FIXED_FEATURE_DIM):
            raise ValueError(f'fixed must have shape {(batch_size, FIXED_FEATURE_DIM)}, got {tuple(fixed.shape)}')
        if tuple(observations.shape) != expected_obs:
            raise ValueError(f'observations must have fixed paper Nmax shape {expected_obs}, got {tuple(observations.shape)}')
        if tuple(mask.shape) != (batch_size, self.nmax):
            raise ValueError('mask must have shape [B,Nmax]')
        if tuple(channel_mask.shape) != expected_obs:
            raise ValueError('channel_mask must have shape [B,Nmax,2]')
        satellite_valid = mask.bool()
        channel_bool = channel_mask.bool()
        satellite_count = satellite_valid.sum(dim=1)
        satellite_position = torch.arange(self.nmax, device=mask.device).unsqueeze(0)
        expected_prefix = satellite_position < satellite_count.unsqueeze(1)
        if not torch.equal(satellite_valid, expected_prefix):
            raise ValueError('mask must contain a contiguous valid prefix for Eq. (16)')
        if not torch.equal(channel_bool[:, :, 1], satellite_valid):
            raise ValueError('innovation channel_mask must match mask for Eq. (16)')
        observations = observations * channel_bool.to(dtype=observations.dtype)
        fixed_valid = torch.ones((batch_size, FIXED_FEATURE_DIM), dtype=torch.bool, device=fixed.device)
        packed_values = []
        packed_masks = []
        satellite_count_cpu = satellite_count.detach().cpu().tolist()
        for batch_index, count in enumerate(satellite_count_cpu):
            suffix_length = 2 * (self.nmax - count)

            # Yan Eq. (15) -> Eq. (16), exact concatenation order:
            # X_k = [Delta alpha_k, Delta omega_k, Delta xhat_k, Delta xtilde_k,
            #        Delta y_{k-1}, Delta ytilde_k]
            # Xbar_k = [X_k, 0_(2*Nmax - 2*Nk)]
            observation_residual = observations[batch_index, :count, 0]
            observation_innovation = observations[batch_index, :count, 1]
            zero_padding = observations.new_zeros(suffix_length)

            variable_values = torch.cat((
                observation_residual,      # Delta y_{k-1}, all valid/current slots first
                observation_innovation,    # Delta ytilde_k, all valid/current slots second
                zero_padding,              # Eq. (16) padding only after the complete X_k
            ), dim=0)

            variable_mask = torch.cat((
                channel_bool[batch_index, :count, 0],
                channel_bool[batch_index, :count, 1],
                torch.zeros(suffix_length, dtype=torch.bool, device=channel_mask.device),
            ), dim=0)

            packed_values.append(torch.cat((
                fixed[batch_index],         # [Delta alpha, Delta omega, Delta xhat, Delta xtilde]
                variable_values,            # [Delta y, Delta ytilde, zero-padding]
            ), dim=0))
            packed_masks.append(torch.cat((fixed_valid[batch_index], variable_mask), dim=0))
        x_bar = torch.stack(packed_values, dim=0)
        feature_mask = torch.stack(packed_masks, dim=0)
        if x_bar.shape[1] != self.eq15_padded_dim:
            raise RuntimeError('Eq. (15)/(16) padded input dimension mismatch')
        # Yan indexing:
        #   CNN acts locally over padded feature position t within epoch k.
        #   LSTM recurrence is across fusion epochs k for each position t.
        #   Attention fuses the valid position-wise hidden states at epoch k.
        conv_features = self.conv(x_bar, feature_mask)  # [B,D,24]
        lstm_positions, next_lstm_state = self.lstm(
            conv_features,
            feature_mask,
            recurrent_state,
        )  # [B,D,64], state [L,B,D,64]
        context, attention = self.attention(lstm_positions, feature_mask)
        gain = self.gain_head(context).view(batch_size, INS_STATE_DIM, self.nmax)
        gain = gain * satellite_valid.to(dtype=gain.dtype).unsqueeze(1)
        return MaskedCLAOutput(kalman_gain=gain, attention=attention, recurrent_state=next_lstm_state)
def fig8_state_update(network_output: MaskedCLAOutput, innovation: torch.Tensor) -> torch.Tensor:
    """Apply the learned gain to the current innovation: dx = K_net @ innovation."""
    if network_output.kalman_gain.ndim != 3:
        raise ValueError('kalman_gain must have shape [B,state,N]')
    if innovation.ndim != 2:
        raise ValueError('innovation must have shape [B,N]')
    if network_output.kalman_gain.shape[0] != innovation.shape[0]:
        raise ValueError('gain/innovation batch dimensions do not match')
    if network_output.kalman_gain.shape[2] != innovation.shape[1]:
        raise ValueError('gain/innovation measurement dimensions do not match')
    return torch.bmm(network_output.kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)
# --- END-TO-END SIMULATION / TRAINING / ONLINE EVALUATION --- Allow direct path execution as well as `python -m ...`.
if __name__ == '__main__':
    # --- RUNTIME CONFIGURATION + DATA01 INPUTS ---
    _run_wall_start = perf_counter()
    OUTPUT_DIR = _default_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_FUSION_EPOCHS = _env_optional_int('MKNET_MAX_FUSION_EPOCHS', 2000)
    MAX_TEST_FUSION_EPOCHS = _env_optional_int('MKNET_MAX_TEST_FUSION_EPOCHS', 600)
    TRAINING_EPOCHS = _env_int('MKNET_TRAINING_EPOCHS', 10)
    KNET_V2_WARMUP_EPOCHS = _env_int('MKNET_V2_WARMUP_EPOCHS', 10)
    KNET_V1_FINE_TUNE_EPOCHS = _env_nonnegative_int('MKNET_V1_FINE_TUNE_EPOCHS', 5)
    if KNET_V2_WARMUP_EPOCHS <= 0:
        raise ValueError('MKNET_V2_WARMUP_EPOCHS must be positive')
    # Yan reports initial LR=0.01. Public Latent-KalmanNet and
    # KalmanNet4SensorFusion configs use 1e-3; use that as the stability
    # default while keeping the strict Yan value selectable by environment.
    YAN_REPORTED_LEARNING_RATE = 0.01
    LEARNING_RATE = _env_positive_float('MKNET_LEARNING_RATE', 0.001)
    FILTER_BLOCK_LEARNING_RATE = LEARNING_RATE
    REPRESENTATION_BLOCK_LEARNING_RATE = LEARNING_RATE

    # SongJgit/KalmanNet4SensorFusion uses detach_step=2.
    # 0 disables truncation; default 2 preserves forward recursion but
    # truncates the autograd history during the long V1 pass.
    V1_TBPTT_DETACH_STEP = _env_nonnegative_int('MKNET_V1_TBPTT_DETACH_STEP', 2)

    # KalmanNet4SensorFusion training-stability completion:
    # clip the active block's global L2 gradient norm before optimizer.step().
    # Yan does not report gradient clipping, so LR remains the paper value 0.01.
    GRADIENT_CLIP_NORM = 1.0
    KNET_V2_TRAJECTORY_LENGTH = _env_int('MKNET_V2_TRAJECTORY_LENGTH', 50)
    KNET_V2_BATCH_SIZE = _env_int('MKNET_V2_BATCH_SIZE', 8)
    GAMMA_L2 = 1e-06
    FDE_SIGNIFICANCE_ALPHA = 0.001
    FDE_STATISTICAL_SELF_CHECK = FDE_DIA.validate_statistical_core(FDE_SIGNIFICANCE_ALPHA) if FDE_DIA_ON else None
    SEED = 0
    TEST_LEO_SEED = LEO_SEED + 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n=== MASKED KALMANNET PERFORMANCE EVALUATION ===')
    if MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None:
        print('WARNING: fusion-epoch cap active; reported metrics are diagnostic.')
    required = [README_XML_PATH, IMU_ERROR_MODEL_PATH, ROVE_GROUND_TRUTH_PATH, IMU_GROUND_TRUTH_PATH, RINEX_OBS_PATH, IMR_PATH, SP3_PATH, CLK_PATH, NAV_PATH, LEO_TLE_DIR]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Missing input files:\n' + '\n'.join(missing))
    rover = load_smartpnt_metadata(README_XML_PATH, '01')
    antenna_truth = load_ie_ground_truth(ROVE_GROUND_TRUTH_PATH)
    imu_truth = load_ie_ground_truth(IMU_GROUND_TRUTH_PATH)
    imu_models = read_imu_error_models(IMU_ERROR_MODEL_PATH)
    imu_noise = imu_models[rover.imu_type]
    rinex = RINEXObservationFile(RINEX_OBS_PATH)
    first_rinex_epoch = next(rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    if first_rinex_epoch is None:
        raise ValueError('RINEX file contains no GPS/BDS epochs')
    imr_header, imr_tow_all, imr_record_count = read_imr_tow_only(IMR_PATH)
    if imr_record_count < 2:
        raise ValueError('IMR file contains fewer than two usable samples')
    if len(antenna_truth.time_gpst_s) < 2 or len(imu_truth.time_gpst_s) < 2:
        raise ValueError('Ground-truth file contains fewer than two usable epochs')
    anchor_time = float(first_rinex_epoch.time_gpst_s)
    imr_time_all = anchor_imr_tow_to_gpst_seconds(imr_tow_all, anchor_time)
    antenna_truth_time = validate_strict_time_axis('training antenna truth', antenna_truth.time_gpst_s)
    imu_truth_time = validate_strict_time_axis('training IMU truth', imu_truth.time_gpst_s)
    common_start = max(float(imr_time_all[0]), float(antenna_truth_time[0]), float(imu_truth_time[0]))
    common_end = min(float(imr_time_all[-1]), float(antenna_truth_time[-1]), float(imu_truth_time[-1]))
    start = int(np.searchsorted(imr_time_all, common_start, side='left'))
    if start >= len(imr_time_all):
        raise ValueError('Training common time span starts after the IMR file')
    usable_imu_start = float(imr_time_all[start])
    gnss_epochs = tuple(rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=usable_imu_start, end_time_gpst_s=common_end, max_epochs=MAX_FUSION_EPOCHS, require_measurements=True))
    if not gnss_epochs:
        raise ValueError('RINEX file contains no usable GPS/BDS pseudorange epochs')
    fusion_time = validate_strict_time_axis('training fusion', np.asarray([epoch.time_gpst_s for epoch in gnss_epochs], dtype=float))
    if len(fusion_time) < 3:
        raise ValueError('Fewer than three synchronized GNSS fusion epochs remain; cannot build lagged Masked KalmanNet features')
    common_stop = min(int(np.searchsorted(imr_time_all, common_end, side='left')) + 1, len(imr_time_all))
    requested_stop = min(int(np.searchsorted(imr_time_all, fusion_time[-1], side='left')) + 1, common_stop)
    imr = read_imr(IMR_PATH, scaling_mode='cpp_exact', start_record=start, stop_record=requested_stop)
    imr_time = imr_time_all[start:requested_stop].copy()
    del imr_tow_all, imr_time_all
    if len(imr.tow_s) < 2:
        raise ValueError('Selected training IMR window contains fewer than two samples')
    query_time = np.concatenate(([imr_time[0]], fusion_time))
    antenna_position = interpolate_ground_truth(antenna_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    imu_position, imu_velocity, imu_heading, imu_pitch, imu_roll = interpolate_ground_truth(imu_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    initial_imu_truth_position = imu_position[0]
    initial_imu_truth_velocity = imu_velocity[0]
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]
    C_b_e = body_to_ecef_from_ie_hpr(initial_imu_truth_position, imu_heading[0], imu_pitch[0], imu_roll[0], rover.mounting_xyz_deg)
    lever_arm_b_m = transform_lever_arm_vehicle_to_body(rover.lever_arm_vehicle_m, *rover.mounting_xyz_deg)
    initial_antenna_from_imu = initial_imu_truth_position + C_b_e @ lever_arm_b_m
    truth_reference_error_m = np.linalg.norm(antenna_position[0] - initial_antenna_from_imu)
    if truth_reference_error_m > 0.05:
        raise ValueError(f'ROVE/ISA-100C ground-truth reference points are inconsistent with the lever arm: {truth_reference_error_m:.3f} m')
    fusion_truth_body_to_ecef = np.stack([body_to_ecef_from_ie_hpr(p, h, pt, r, rover.mounting_xyz_deg) for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])])
    initial_nav = NavigationState(initial_imu_truth_position.copy(), initial_imu_truth_velocity.copy(), C_b_e)
    P0 = initial_covariance_from_imu_model(imu_noise)
    Qc = continuous_process_covariance_from_imu_model(imu_noise)
    ionosphere_coefficients = None
    if USE_IONOSPHERE:
        iono_header = read_rinex_navigation_header(NAV_PATH)
        ionosphere_coefficients = {'G': (np.asarray(iono_header['GPSA']), np.asarray(iono_header['GPSB'])), 'C': (np.asarray(iono_header['BDSA']), np.asarray(iono_header['BDSB']))}
    gnss_preprocessor = GNSSPreprocessor(SP3Orbit(SP3_PATH), RINEXClock(CLK_PATH), min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=ionosphere_coefficients)
    tle_provider = TLESGP4Provider(LEO_TLE_DIR, max_tle_age_days=TLE_MAX_AGE_DAYS, allow_degraded_eop=TLE_ALLOW_DEGRADED_EOP, allow_non_tle_files=TLE_ALLOW_NON_TLE_FILES)
    leo_klobuchar = None if ionosphere_coefficients is None else ionosphere_coefficients['G']
    leo_simulator = LEODownlinkSimulator(tle_provider, leo_klobuchar, seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    nav = initial_nav.copy()
    P = P0.copy()
    history_rows = []
    data01_sequence_nmax = 0
    interval_segments_since_previous_usable_fusion = []
    last_gyro, last_accel = compensate_imu(imr.angular_rate_body_radps[0], imr.acceleration_body_mps2[0])
    # --- DATA01 CLASSICAL INS/TC HISTORY ---
    training_timeline = build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True)
    for event in training_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(imr.angular_rate_body_radps[imu_index], imr.acceleration_body_mps2[imu_index])
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
        if abs(epoch.time_gpst_s - t) > 1e-09:
            raise RuntimeError('training fusion timeline/epoch timestamp mismatch')
        gnss_measurements = gnss_preprocessor.prepare_epoch(epoch, nav, lever_arm_b_m)
        leo_measurements = leo_simulator.simulate_epoch(t, fusion_antenna_truth_position[fusion_index])
        data01_sequence_nmax = max(data01_sequence_nmax, len(epoch.measurements) + len(leo_measurements))
        measurements = retain_clock_observable_measurements(tuple(gnss_measurements) + tuple(leo_measurements))
        if not measurements:
            continue
        prior_nav = nav.copy()
        measurement_model = build_measurement_model(nav, measurements, lever_arm_b_m)
        error_state_pred = np.zeros(INS_STATE_DIM)
        correction, P, _ = kalman_measurement_update(P, measurement_model.innovation, measurement_model.H, measurement_model.R)
        error_state_post = error_state_pred + correction
        nav = inject_error_state(nav, correction)
        posterior_residual = build_innovation_only(nav, measurements, lever_arm_b_m)
        target_state_9 = np.concatenate([fusion_imu_truth_position[fusion_index] - prior_nav.position_ecef_m, fusion_imu_truth_velocity[fusion_index] - prior_nav.velocity_ecef_mps, attitude_error_state_target(prior_nav.body_to_ecef_dcm, fusion_truth_body_to_ecef[fusion_index])])
        # Fixed observation snapshot for KalmanNet training.
        # GNSS m.pseudorange_m is the observed RINEX pseudorange y_k.
        # LEO m.pseudorange_m is the once-generated simulated pseudorange y_k.
        # The learned recursive state is NOT allowed to regenerate/reselect y_k.
        # Only h(xhat_{k|k-1}) and therefore the innovation are recomputed online.
        fixed_measurements = tuple(measurements)
        fixed_sat_ids = tuple(m.sat_id for m in fixed_measurements)
        if fixed_sat_ids != tuple(measurement_model.sat_ids):
            raise RuntimeError('fixed Data01 measurement ordering does not match measurement-model ordering')
        history_rows.append({
            'time': t,
            'sat_ids': measurement_model.sat_ids,
            'innovation': measurement_model.innovation.copy(),
            'residual': posterior_residual.copy(),
            'x_pred': error_state_pred.copy(),
            'x_post': error_state_post.copy(),
            'accel': last_accel.copy(),
            'gyro': last_gyro.copy(),
            'target_state_9': target_state_9.copy(),
            'posterior_nav': nav.copy(),
            'fusion_index': int(fusion_index),
            'fixed_measurements': fixed_measurements,
            'fixed_y_m': np.asarray([m.pseudorange_m for m in fixed_measurements], dtype=float),
            'leo_measurements': tuple(leo_measurements),
            'preceding_interval_segments': tuple(interval_segments_since_previous_usable_fusion),
        })
        interval_segments_since_previous_usable_fusion = []
    if len(history_rows) < 4:
        raise ValueError('Classical TC pass produced fewer than four usable measurement rows; check GNSS products, TLE coverage, masks, and time synchronization')
    # --- YAN Eqs. (10)-(21): FEATURES, MASKS, FIXED Nmax ---
    nmax = max((len(row['sat_ids']) for row in history_rows[1:]))
    test_dataset_dir = resolve_test_dataset_dir()
    if test_dataset_dir.resolve() == TRAIN_DATASET_DIR.resolve():
        raise ValueError('The online test dataset must be distinct from the training dataset')
    data02_sequence_nmax = _scan_test_sequence_capacity(test_dataset_dir, MAX_TEST_FUSION_EPOCHS, tle_provider, TEST_LEO_SEED)
    network_nmax = max(int(nmax), int(data01_sequence_nmax), int(data02_sequence_nmax))
    fixed = []
    observations = []
    channel_masks = []
    target_states_9 = []
    for k in range(1, len(history_rows)):
        current = history_rows[k]
        previous = history_rows[k - 1]
        delta_accel = current['accel'] - previous['accel']
        delta_gyro = current['gyro'] - previous['gyro']
        previous_state_innovation = previous['x_post'] - previous['x_pred']
        previous_state_residual = np.zeros(INS_STATE_DIM) if k == 1 else previous['x_post'] - history_rows[k - 2]['x_post']
        # Yan Eq. (15) exact order for the fixed-dimensional part:
        # [Delta alpha_k, Delta omega_k, Delta xhat_k, Delta xtilde_k]
        # Project causal implementation uses the available previous-epoch
        # state residual/innovation when forming K_k.
        fixed_k = np.concatenate([
            delta_accel,                 # Delta alpha
            delta_gyro,                  # Delta omega
            previous_state_residual,     # Delta xhat
            previous_state_innovation,   # Delta xtilde
        ])
        n = len(current['sat_ids'])
        innovation_k = np.zeros(nmax)
        innovation_k[:n] = current['innovation']
        current_mask = np.zeros(nmax, dtype=bool)
        current_mask[:n] = True
        previous_residual_by_sat = dict(zip(previous['sat_ids'], previous['residual']))
        residual_k = np.zeros(nmax)
        residual_mask = np.zeros(nmax, dtype=bool)
        for slot, sat_id in enumerate(current['sat_ids']):
            if sat_id in previous_residual_by_sat:
                residual_k[slot] = previous_residual_by_sat[sat_id]
                residual_mask[slot] = True
        fixed.append(fixed_k)
        observations.append(np.stack([residual_k, innovation_k], axis=1))
        channel_masks.append(np.stack([residual_mask, current_mask], axis=1))
        target_states_9.append(current['target_state_9'])
    feature_time = np.asarray([row['time'] for row in history_rows[1:]], dtype=float)
    fixed = np.stack(fixed)
    observations = np.stack(observations)
    channel_masks = np.stack(channel_masks)
    innovation_padded = observations[:, :, 1]
    target_states_9 = np.stack(target_states_9)
    expected_t = len(feature_time)
    if fixed.shape != (expected_t, FIXED_FEATURE_DIM):
        raise ValueError(f'fixed feature shape must be {(expected_t, FIXED_FEATURE_DIM)}, got {fixed.shape}')
    if innovation_padded.shape != (expected_t, nmax):
        raise ValueError('innovation_padded must have shape [T,Nmax]')
    if channel_masks.shape != (expected_t, nmax, OBSERVATION_FEATURE_DIM):
        raise ValueError('channel_masks must have shape [T,Nmax,2]')
    if expected_t > 1 and (not np.all(np.diff(feature_time) > 0.0)):
        raise ValueError('feature times must be strictly increasing')
    if not np.all(np.isfinite(fixed)) or not np.all(np.isfinite(innovation_padded)) or (not np.all(np.isfinite(observations))) or (not np.all(np.isfinite(target_states_9))):
        raise ValueError('feature arrays contain non-finite values')
    if np.any(innovation_padded[~channel_masks[:, :, 1]] != 0.0):
        raise ValueError('masked innovation padding must be exactly zero')
    satellite_masks = channel_masks[:, :, 1]
    train_fixed_min = np.min(fixed, axis=0)
    train_fixed_max = np.max(fixed, axis=0)
    train_fixed_abs_max = np.maximum(np.max(np.abs(fixed), axis=0), 1e-12)
    train_obs_min = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_abs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    for diagnostic_channel in range(OBSERVATION_FEATURE_DIM):
        valid_values = observations[:, :, diagnostic_channel][channel_masks[:, :, diagnostic_channel]]
        if valid_values.size == 0:
            train_obs_min[diagnostic_channel] = -np.inf
            train_obs_max[diagnostic_channel] = np.inf
            train_obs_abs_max[diagnostic_channel] = 1.0
        else:
            train_obs_min[diagnostic_channel] = float(np.min(valid_values))
            train_obs_max[diagnostic_channel] = float(np.max(valid_values))
            train_obs_abs_max[diagnostic_channel] = max(float(np.max(np.abs(valid_values))), 1e-12)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # --- KalmanNet-family online groupwise L2 normalization ---
    # Fixed feature order:
    # [Delta alpha(3), Delta omega(3), Delta xhat(9), Delta xtilde(9)].
    # Each semantic group is normalized independently at the same epoch.
    # Observation residual and innovation are normalized independently across
    # only their valid satellite entries. Invalid/padded entries remain zero.
    def _l2_normalize_np(values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(values))
        if not np.isfinite(norm):
            raise FloatingPointError('non-finite L2 norm in feature normalization')
        if norm <= FEATURE_L2_EPS:
            return np.zeros_like(values)
        return values / norm

    def _normalize_fixed_groups_np(fixed_raw: Array) -> Array:
        fixed_raw = np.asarray(fixed_raw, dtype=float).reshape(FIXED_FEATURE_DIM)
        return np.concatenate([
            _l2_normalize_np(fixed_raw[0:3]),    # Delta alpha
            _l2_normalize_np(fixed_raw[3:6]),    # Delta omega
            _l2_normalize_np(fixed_raw[6:15]),   # Delta xhat
            _l2_normalize_np(fixed_raw[15:24]),  # Delta xtilde
        ])

    def _normalize_observation_channels_np(obs_raw: Array, channel_mask: Array) -> Array:
        obs_raw = np.asarray(obs_raw, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        channel_mask = np.asarray(channel_mask, dtype=bool).reshape(-1, OBSERVATION_FEATURE_DIM)
        normalized = np.zeros_like(obs_raw)
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            if not np.any(valid):
                continue
            normalized[valid, channel] = _l2_normalize_np(obs_raw[valid, channel])
        return normalized

    if FEATURE_NORMALIZATION_MODE != 'online_group_l2':
        raise ValueError(f'unsupported FEATURE_NORMALIZATION_MODE={FEATURE_NORMALIZATION_MODE!r}')

    fixed_normalized = np.stack([
        _normalize_fixed_groups_np(row) for row in fixed
    ])
    observations_normalized = np.stack([
        _normalize_observation_channels_np(obs_row, mask_row)
        for obs_row, mask_row in zip(observations, channel_masks)
    ])
    normalizer_source = 'KalmanNet_family_online_groupwise_L2_mask_aware'
    np.savez(
        OUTPUT_DIR / 'normalizer.npz',
        mode=np.asarray(FEATURE_NORMALIZATION_MODE),
        eps=np.asarray(FEATURE_L2_EPS),
        fixed_groups=np.asarray([
            'Delta_alpha[0:3]',
            'Delta_omega[3:6]',
            'Delta_xhat[6:15]',
            'Delta_xtilde[15:24]',
        ]),
        observation_channels=np.asarray([
            'Delta_y_previous_mask_aware',
            'Delta_ytilde_current_mask_aware',
        ]),
        source=np.asarray(normalizer_source),
    )
    training_sample_count = int(len(feature_time))
    if training_sample_count <= 0:
        raise ValueError('Data01 produced no KalmanNet training samples')
    if KNET_V2_TRAJECTORY_LENGTH <= 0:
        raise ValueError('MKNET_V2_TRAJECTORY_LENGTH must be positive')
    if KNET_V2_BATCH_SIZE <= 0:
        raise ValueError('MKNET_V2_BATCH_SIZE must be positive')
    training_trajectories = []
    v2_start = 0
    while v2_start + KNET_V2_TRAJECTORY_LENGTH <= training_sample_count:
        training_trajectories.append((v2_start, v2_start + KNET_V2_TRAJECTORY_LENGTH))
        v2_start += KNET_V2_TRAJECTORY_LENGTH + 1
    trajectory_count = len(training_trajectories)
    if trajectory_count < 1:
        raise ValueError(f'KalmanNet V2 requires at least one complete short trajectory. Data01 samples={training_sample_count}, MKNET_V2_TRAJECTORY_LENGTH={KNET_V2_TRAJECTORY_LENGTH}. Reduce MKNET_V2_TRAJECTORY_LENGTH for a capped diagnostic run or remove the fusion cap for the scientific run.')
    TRAJECTORY_BATCH_SIZE = min(KNET_V2_BATCH_SIZE, trajectory_count)
    v2_trained_sample_count = int(sum((stop - start for start, stop in training_trajectories)))
    v2_initialization_sample_count = int(trajectory_count)
    v2_unused_sample_count = int(training_sample_count - v2_trained_sample_count)
    print(f"training setup: samples={training_sample_count}, Nmax={network_nmax} (Data01={data01_sequence_nmax}, Data02={data02_sequence_nmax}), mode=KalmanNet_V2_short_trajectories, Tshort={KNET_V2_TRAJECTORY_LENGTH}, trajectories={trajectory_count}, trajectory_batch={TRAJECTORY_BATCH_SIZE}, batching=LatentKalmanNet_Algorithm2_random_partition_all_Q_batches_same_partition_theta_then_psi, init=ground_truth_navigation_state, V2_warmup_epochs={KNET_V2_WARMUP_EPOCHS}, V1_finetune_epochs={KNET_V1_FINE_TUNE_EPOCHS}, lr_theta={FILTER_BLOCK_LEARNING_RATE:g}, lr_psi={REPRESENTATION_BLOCK_LEARNING_RATE:g}, grad_clip_norm={GRADIENT_CLIP_NORM:g}, V1_tbptt_detach_step={V1_TBPTT_DETACH_STEP}, FDE={('ON' if FDE_DIA_ON else 'OFF')}")
    # --- MASKED-CLA TRAINING: KALMANNET V2 WARM-UP -> V1 FULL BPTT ---
    model = MaskedCLA(nmax=network_nmax, dropout=0.2).to(DEVICE)
    # Kalman gain output initialization:
    # Keep PyTorch's default nn.Linear initialization for gain_head.
    # Do not force the complete Kalman-gain head to zero.
    #
    # This follows the initialization pattern used by the original KalmanNet
    # implementations examined for this project: nonzero trainable output
    # weights, while recurrent hidden state is initialized separately.
    representation_modules = (model.conv,)
    filter_modules = (model.lstm, model.attention, model.gain_head)
    representation_parameters = [parameter for module in representation_modules for parameter in module.parameters()]
    filter_parameters = [parameter for module in filter_modules for parameter in module.parameters()]
    if {id(p) for p in representation_parameters} & {id(p) for p in filter_parameters}:
        raise RuntimeError('alternating parameter blocks overlap')
    if {id(p) for p in representation_parameters + filter_parameters} != {id(p) for p in model.parameters()}:
        raise RuntimeError('alternating blocks must cover the entire MaskedCLA network')
    filter_optimizer = torch.optim.Adam(filter_parameters, lr=FILTER_BLOCK_LEARNING_RATE)
    representation_optimizer = torch.optim.Adam(representation_parameters, lr=REPRESENTATION_BLOCK_LEARNING_RATE)
    trainable_eq32_parameters = representation_parameters + filter_parameters
    training_history = []
    target_base_t = torch.tensor(target_states_9, dtype=torch.float64, device=DEVICE)
    def _set_alternating_phase(phase: str) -> None:
        """Freeze inactive parameters and disable their training-time stochastic layers."""
        if phase not in {'filter', 'representation'}:
            raise ValueError("phase must be 'filter' or 'representation'")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Start deterministic, then put only the active block in train mode.
        # In particular, the frozen five-layer LSTM must not apply dropout
        # while psi/CNN is optimized through it.
        model.eval()
        if phase == 'filter':
            for module in filter_modules:
                module.train()
            for parameter in filter_parameters:
                parameter.requires_grad_(True)
        else:
            for module in representation_modules:
                module.train()
            for parameter in representation_parameters:
                parameter.requires_grad_(True)
    def _eq32_regularization() -> torch.Tensor:
        if not GAMMA_L2:
            return torch.zeros((), dtype=torch.float64, device=DEVICE)
        l2_all = sum((torch.sum(parameter.double() * parameter.double()) for parameter in trainable_eq32_parameters))
        return GAMMA_L2 * l2_all
    def _torch_training_context_from_stored(history_row_index: int):
        """Causal stored predecessor context used at a Data01 sequence boundary."""
        if not 0 <= history_row_index < len(history_rows):
            raise ValueError('invalid stored Data01 context row')
        warm = history_rows[history_row_index]
        x_pred = np.asarray(warm['x_pred'], dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(warm['x_post'], dtype=float).reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = np.zeros(INS_STATE_DIM) if history_row_index == 0 else x_post - np.asarray(history_rows[history_row_index - 1]['x_post'], dtype=float).reshape(INS_STATE_DIM)
        residual = np.asarray(warm['residual'], dtype=float).copy()
        accel = np.asarray(warm['accel'], dtype=float).reshape(3).copy()
        gyro = np.asarray(warm['gyro'], dtype=float).reshape(3).copy()
        if len(warm['sat_ids']) != len(residual) or not np.all(np.isfinite(residual)) or (not np.all(np.isfinite(state_innovation))) or (not np.all(np.isfinite(state_residual))) or (not np.all(np.isfinite(accel))) or (not np.all(np.isfinite(gyro))):
            raise FloatingPointError('invalid completed Data01 training context')
        return {'sat_ids': tuple(warm['sat_ids']), 'residual': torch.as_tensor(residual, dtype=torch.float64, device=DEVICE), 'x_pred': torch.as_tensor(x_pred.copy(), dtype=torch.float64, device=DEVICE), 'x_post': torch.as_tensor(x_post.copy(), dtype=torch.float64, device=DEVICE), 'state_innovation': torch.as_tensor(state_innovation.copy(), dtype=torch.float64, device=DEVICE), 'state_residual': torch.as_tensor(state_residual.copy(), dtype=torch.float64, device=DEVICE), 'accel': torch.as_tensor(accel, dtype=torch.float64, device=DEVICE), 'gyro': torch.as_tensor(gyro, dtype=torch.float64, device=DEVICE)}
    training_imr_gyro_t = torch.as_tensor(imr.angular_rate_body_radps, dtype=torch.float64, device=DEVICE)
    training_imr_accel_t = torch.as_tensor(imr.acceleration_body_mps2, dtype=torch.float64, device=DEVICE)
    lever_arm_train_t = torch.as_tensor(lever_arm_b_m, dtype=torch.float64, device=DEVICE).reshape(3)
    fusion_truth_position_t = torch.as_tensor(fusion_imu_truth_position, dtype=torch.float64, device=DEVICE)
    fusion_truth_velocity_t = torch.as_tensor(fusion_imu_truth_velocity, dtype=torch.float64, device=DEVICE)
    fusion_truth_dcm_t = torch.as_tensor(fusion_truth_body_to_ecef, dtype=torch.float64, device=DEVICE)
    zero_error_state_t = torch.zeros(INS_STATE_DIM, dtype=torch.float64, device=DEVICE)
    def _torch_navigation_state_finite(nav_state: TorchNavigationState) -> bool:
        finite = torch.all(torch.isfinite(nav_state.position_ecef_m)) & torch.all(torch.isfinite(nav_state.velocity_ecef_mps)) & torch.all(torch.isfinite(nav_state.body_to_ecef_dcm))
        return bool(finite.detach().cpu())
    def _detach_torch_navigation_state(nav_state: TorchNavigationState) -> TorchNavigationState:
        return TorchNavigationState(
            nav_state.position_ecef_m.detach(),
            nav_state.velocity_ecef_mps.detach(),
            nav_state.body_to_ecef_dcm.detach(),
        )

    def _detach_training_context(context):
        detached = {}
        for key, value in context.items():
            detached[key] = value.detach() if isinstance(value, torch.Tensor) else value
        return detached

    def _detach_recurrent_state(recurrent_state):
        if recurrent_state is None:
            return None
        return tuple(state.detach() for state in recurrent_state)

    def _torch_training_context_from_completed_epoch(sat_ids, posterior_residual: torch.Tensor, x_pred: torch.Tensor, x_post: torch.Tensor, accel: torch.Tensor, gyro: torch.Tensor, previous_context):
        x_pred = x_pred.reshape(INS_STATE_DIM)
        x_post = x_post.reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = torch.zeros_like(x_post) if previous_context is None else x_post - previous_context['x_post']
        context = {'sat_ids': tuple(sat_ids), 'residual': posterior_residual.reshape(-1), 'x_pred': x_pred, 'x_post': x_post, 'state_innovation': state_innovation, 'state_residual': state_residual, 'accel': accel.reshape(3), 'gyro': gyro.reshape(3)}
        finite = torch.all(torch.isfinite(context['residual'])) & torch.all(torch.isfinite(context['state_innovation'])) & torch.all(torch.isfinite(context['state_residual'])) & torch.all(torch.isfinite(context['accel'])) & torch.all(torch.isfinite(context['gyro']))
        if len(context['sat_ids']) != int(context['residual'].numel()) or not bool(finite.detach().cpu()):
            raise FloatingPointError('invalid differentiable Data01 training context')
        return context
    def _prepare_torch_measurements(measurements):
        """Prepare estimator-independent pseudorange tensors once per fusion update."""
        measurements = tuple(retain_clock_observable_measurements(measurements))
        n = len(measurements)
        if n == 0:
            return ((), None, None, None, None, None, None)
        variances = np.asarray([float(m.sigma_code_m) ** 2 for m in measurements], dtype=float)
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError('training pseudorange variances must be finite and positive')
        projector = _clock_projector(measurements, variances)
        return (tuple((m.sat_id for m in measurements)), torch.as_tensor(np.stack([m.satellite_position_reception_ecef_m for m in measurements], axis=0), dtype=torch.float64, device=DEVICE), torch.as_tensor([m.pseudorange_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.satellite_clock_bias_s for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.ionosphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.troposphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor(projector, dtype=torch.float64, device=DEVICE))
    def _torch_measurement_innovation(nav_state: TorchNavigationState, prepared_measurements) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Differentiable clock-projected pseudorange innovation for BPTT."""
        sat_ids, satellite_position, observed, satellite_clock, ionosphere, troposphere, projector_t = prepared_measurements
        if not sat_ids:
            return (nav_state.position_ecef_m.new_empty(0), ())
        lever_e = nav_state.body_to_ecef_dcm @ lever_arm_train_t
        antenna_position = nav_state.position_ecef_m + lever_e
        geometric = torch.linalg.vector_norm(antenna_position.unsqueeze(0) - satellite_position, dim=1)
        predicted_zero_clock = geometric - SPEED_OF_LIGHT_MPS * satellite_clock + ionosphere + troposphere
        return (projector_t @ (observed - predicted_zero_clock), sat_ids)
    def _torch_training_feature_tensors(previous_context, current_sat_ids, innovation_now: torch.Tensor, current_accel: torch.Tensor, current_gyro: torch.Tensor):
        """Yan Eqs. (10)-(14), preserving the autograd history of the estimator."""
        n = len(current_sat_ids)
        innovation_now = innovation_now.reshape(n)
        delta_accel = current_accel.reshape(3) - previous_context['accel']
        delta_gyro = current_gyro.reshape(3) - previous_context['gyro']
        fixed_raw = torch.cat((delta_accel, delta_gyro, previous_context['state_residual'], previous_context['state_innovation']))
        previous_index = {sat_id: index for index, sat_id in enumerate(previous_context['sat_ids'])}
        residual_values = []
        residual_valid = []
        zero = innovation_now.new_zeros(())
        for sat_id in current_sat_ids:
            index = previous_index.get(sat_id)
            if index is None:
                residual_values.append(zero)
                residual_valid.append(False)
            else:
                residual_values.append(previous_context['residual'][index])
                residual_valid.append(True)
        residual_k = torch.stack(residual_values)
        residual_mask = torch.as_tensor(residual_valid, dtype=torch.bool, device=DEVICE)
        current_mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
        obs_raw = torch.stack((residual_k, innovation_now), dim=1)
        channel_mask = torch.stack((residual_mask, current_mask), dim=1)
        # KalmanNet-family online normalization. These operations stay inside
        # autograd because recursive state features depend on learned parameters.
        def _torch_l2_group(values: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.vector_norm(values)
            return values / torch.clamp(norm, min=FEATURE_L2_EPS)

        fixed_nn = torch.cat((
            _torch_l2_group(fixed_raw[0:3]),    # Delta alpha
            _torch_l2_group(fixed_raw[3:6]),    # Delta omega
            _torch_l2_group(fixed_raw[6:15]),   # Delta xhat
            _torch_l2_group(fixed_raw[15:24]),  # Delta xtilde
        ))

        obs_columns = []
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            masked_values = torch.where(
                valid,
                obs_raw[:, channel],
                torch.zeros_like(obs_raw[:, channel]),
            )
            obs_columns.append(_torch_l2_group(masked_values))
        obs_nn = torch.stack(obs_columns, dim=1)
        obs_nn = torch.where(channel_mask, obs_nn, torch.zeros_like(obs_nn))
        if fixed_nn.shape != (FIXED_FEATURE_DIM,) or obs_nn.shape != (n, OBSERVATION_FEATURE_DIM):
            raise RuntimeError('differentiable training feature shape mismatch')
        finite = torch.all(torch.isfinite(fixed_nn)) & torch.all(torch.isfinite(obs_nn))
        if not bool(finite.detach().cpu()):
            raise FloatingPointError('non-finite differentiable feature generated by Data01 recurrence')
        return (fixed_nn, obs_nn, current_mask, channel_mask, innovation_now)
    def _torch_pad_training_epoch(fixed_nn: torch.Tensor, obs_nn: torch.Tensor, current_mask: torch.Tensor, channel_mask: torch.Tensor, innovation_now: torch.Tensor, slot_capacity: int):
        n = int(current_mask.numel())
        if slot_capacity < n:
            raise ValueError('training neural slot capacity is smaller than measurements')
        pad = int(slot_capacity - n)
        if pad == 0:
            return (fixed_nn, obs_nn, current_mask, channel_mask, innovation_now)
        obs_pad = torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=obs_nn.dtype, device=obs_nn.device)
        mask_pad = torch.zeros(pad, dtype=torch.bool, device=DEVICE)
        channel_pad = torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=torch.bool, device=DEVICE)
        innovation_pad = torch.zeros(pad, dtype=innovation_now.dtype, device=innovation_now.device)
        return (fixed_nn, torch.cat((obs_nn, obs_pad), dim=0), torch.cat((current_mask, mask_pad), dim=0), torch.cat((channel_mask, channel_pad), dim=0), torch.cat((innovation_now, innovation_pad), dim=0))
    def _propagate_torch_interval(nav_state: TorchNavigationState, segments, *, use_checkpoint: bool):
        """Propagate one fusion interval with 9-state/no-bias INS BPTT."""
        if not segments:
            feature_gyro = training_imr_gyro_t[0]
            feature_accel = training_imr_accel_t[0]
            return (nav_state, feature_gyro, feature_accel)
        segment_tuple = tuple(((int(i), float(dt)) for i, dt in segments))
        def interval_core(position, velocity, dcm):
            state = TorchNavigationState(position, velocity, dcm)
            feature_gyro = training_imr_gyro_t[segment_tuple[0][0]]
            feature_accel = training_imr_accel_t[segment_tuple[0][0]]
            for imu_index, dt in segment_tuple:
                measured_gyro = training_imr_gyro_t[imu_index]
                measured_accel = training_imr_accel_t[imu_index]
                propagation_gyro, propagation_accel = _torch_compensate_imu(measured_gyro, measured_accel)
                feature_gyro, feature_accel = (propagation_gyro, propagation_accel)
                state = _torch_mechanize_ecef(state, propagation_gyro, propagation_accel, dt)
            return (state.position_ecef_m, state.velocity_ecef_mps, state.body_to_ecef_dcm, feature_gyro, feature_accel)
        inputs = (nav_state.position_ecef_m, nav_state.velocity_ecef_mps, nav_state.body_to_ecef_dcm)
        has_history = any((value.requires_grad for value in inputs))
        if use_checkpoint and has_history:
            position, velocity, dcm, feature_gyro, feature_accel = checkpoint(interval_core, *inputs, use_reentrant=False)
        else:
            position, velocity, dcm, feature_gyro, feature_accel = interval_core(*inputs)
        propagated = TorchNavigationState(position, velocity, dcm)
        return (propagated, feature_gyro, feature_accel)
    def _rollout_actual_closed_loop_trajectory(start_sample: int, stop_sample: int, *, phase: str | None, backward_scale: float | None=None, trajectory_id: int | None=None, optimizer_step_index: int | None=None, initialization_mode: str='stored_causal', tbptt_detach_step: int=0):
        """Differentiable learned closed-loop rollout over a Data01 sequence interval.

        KalmanNet observation semantics:
          - y_k and the epoch measurement set are fixed from the Data01 trajectory.
          - the INS/KalmanNet state remains recursive.
          - h(xhat_{k|k-1}), innovation, residual, and state features are recomputed
            from the recursive learned state.
        """
        if not 0 <= start_sample < stop_sample <= training_sample_count:
            raise ValueError('invalid Data01 sequence bounds')
        training = phase is not None
        if training and (backward_scale is None or backward_scale <= 0.0):
            raise ValueError('training trajectory requires a positive backward_scale')
        if not training and backward_scale is not None:
            raise ValueError('evaluation trajectory must not request backward')
        if initialization_mode not in {'stored_causal', 'kalmannet_v2_truth'}:
            raise ValueError('unsupported training trajectory initialization mode')
        if tbptt_detach_step < 0:
            raise ValueError('tbptt_detach_step must be non-negative')
        if not training and tbptt_detach_step:
            raise ValueError('TBPTT detachment is only valid during training')
        if initialization_mode == 'kalmannet_v2_truth':
            boundary_row = history_rows[start_sample]
            boundary_fusion_index = int(boundary_row['fusion_index'])
            nav_train = TorchNavigationState(fusion_truth_position_t[boundary_fusion_index].clone(), fusion_truth_velocity_t[boundary_fusion_index].clone(), fusion_truth_dcm_t[boundary_fusion_index].clone())
            stored_boundary_context = _torch_training_context_from_stored(start_sample)
            # KalmanNet convention: y_k belongs to the data trajectory and is fixed.
            # Recompute only h(xhat), not y_k or the satellite set.
            boundary_measurements = boundary_row['fixed_measurements']
            if tuple(m.sat_id for m in boundary_measurements) != tuple(boundary_row['sat_ids']):
                raise RuntimeError('fixed boundary observation ordering changed')
            boundary_prepared = _prepare_torch_measurements(boundary_measurements)
            boundary_residual, boundary_sat_ids = _torch_measurement_innovation(nav_train, boundary_prepared)
            previous_context = {'sat_ids': tuple(boundary_sat_ids), 'residual': boundary_residual, 'x_pred': zero_error_state_t.clone(), 'x_post': zero_error_state_t.clone(), 'state_innovation': zero_error_state_t.clone(), 'state_residual': zero_error_state_t.clone(), 'accel': stored_boundary_context['accel'], 'gyro': stored_boundary_context['gyro']}
        else:
            nav_train = TorchNavigationState.from_numpy(history_rows[start_sample]['posterior_nav'], device=DEVICE, dtype=torch.float64)
            previous_context = _torch_training_context_from_stored(start_sample)
        last_feature_accel = previous_context['accel']
        last_feature_gyro = previous_context['gyro']
        recurrent_capacity = int(network_nmax)
        recurrent_state = None
        state_losses: list[torch.Tensor] = []
        total_state_sum_t = zero_error_state_t.new_zeros(())
        total_position_sum_t = zero_error_state_t.new_zeros(())
        total_count = 0
        learned_updates_since_detach = 0
        tbptt_detach_count = 0
        skipped_no_measurements = 0
        max_prior_position_error_t = zero_error_state_t.new_zeros(())
        max_innovation_abs_t = zero_error_state_t.new_zeros(())
        max_gain_fro_t = None
        max_position_correction_t = zero_error_state_t.new_zeros(())
        max_velocity_correction_t = zero_error_state_t.new_zeros(())
        max_attitude_correction_t = zero_error_state_t.new_zeros(())
        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context:
            for sample_index in range(start_sample, stop_sample):
                row_index = sample_index + 1
                row = history_rows[row_index]
                if row['preceding_interval_segments']:
                    nav_train, last_feature_gyro, last_feature_accel = _propagate_torch_interval(nav_train, row['preceding_interval_segments'], use_checkpoint=training)
                if not _torch_navigation_state_finite(nav_train):
                    raise FloatingPointError(f'non-finite differentiable INS state inside training trajectory={trajectory_id}, bounds=[{start_sample},{stop_sample}), history_row={row_index}, optimizer_step={optimizer_step_index}')
                fusion_index = int(row['fusion_index'])
                prior_position_error = fusion_truth_position_t[fusion_index] - nav_train.position_ecef_m
                max_prior_position_error_t = torch.maximum(max_prior_position_error_t, torch.linalg.vector_norm(prior_position_error).detach())
                # Fixed-y KalmanNet training:
                # y_k, satellite IDs, and the epoch observation snapshot come from
                # the prepared Data01 trajectory and do not depend on the learned state.
                # The recursive dependence is retained only in:
                #   yhat_{k|k-1} = h(xhat_{k|k-1})
                #   innovation_k = y_k - yhat_{k|k-1}
                measurements = row['fixed_measurements']
                if not measurements:
                    skipped_no_measurements += 1
                    continue
                if tuple(m.sat_id for m in measurements) != tuple(row['sat_ids']):
                    raise RuntimeError('fixed Data01 observation ordering changed during BPTT')
                fixed_y_check = np.asarray([m.pseudorange_m for m in measurements], dtype=float)
                if not np.array_equal(fixed_y_check, np.asarray(row['fixed_y_m'], dtype=float)):
                    raise RuntimeError('fixed Data01 pseudorange vector y_k changed during BPTT')
                prepared_measurements = _prepare_torch_measurements(measurements)
                innovation_now, current_sat_ids = _torch_measurement_innovation(nav_train, prepared_measurements)
                if innovation_now.numel():
                    max_innovation_abs_t = torch.maximum(max_innovation_abs_t, torch.max(torch.abs(innovation_now)).detach())
                fixed_k, obs_k, current_mask, channel_k, innovation_k = _torch_training_feature_tensors(previous_context, current_sat_ids, innovation_now, last_feature_accel, last_feature_gyro)
                if len(current_sat_ids) > recurrent_capacity:
                    raise RuntimeError(f'current Data01 observation count exceeds the automatically pre-scanned Yan Eq.(16) Nmax: count={len(current_sat_ids)}, network_Nmax={recurrent_capacity}. This indicates inconsistent observation availability between the pre-scan and recursive training path.')
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _torch_pad_training_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, recurrent_capacity)
                output = model(fixed_nn.to(dtype=torch.float32).unsqueeze(0), obs_nn.to(dtype=torch.float32).unsqueeze(0), mask_nn.unsqueeze(0), channel_nn.unsqueeze(0), recurrent_state=recurrent_state)
                recurrent_state = output.recurrent_state
                correction = fig8_state_update(output, innovation_nn.to(dtype=torch.float32).unsqueeze(0))[0]
                correction64 = correction.to(dtype=torch.float64)
                valid_gain = output.kalman_gain[0, :, :len(current_sat_ids)]
                gain_fro = torch.linalg.matrix_norm(valid_gain).detach()
                max_gain_fro_t = gain_fro if max_gain_fro_t is None else torch.maximum(max_gain_fro_t, gain_fro)
                target_state_9_now = torch.cat((fusion_truth_position_t[fusion_index] - nav_train.position_ecef_m, fusion_truth_velocity_t[fusion_index] - nav_train.velocity_ecef_mps, _torch_attitude_error_state_target(nav_train.body_to_ecef_dcm, fusion_truth_dcm_t[fusion_index])))
                state_error_9 = target_state_9_now - correction64[:SUPERVISED_STATE_DIM]
                state_loss = torch.sum(state_error_9 ** 2)
                position_loss = torch.sum(state_error_9[:3] ** 2)
                if not bool(torch.isfinite(state_loss).detach().cpu()):
                    max_innovation_abs_m = float(max_innovation_abs_t.cpu())
                    max_gain_fro_norm = 0.0 if max_gain_fro_t is None else float(max_gain_fro_t.cpu())
                    raise FloatingPointError(f'non-finite Yan Eq.(30) loss inside KalmanNet-style BPTT trajectory={trajectory_id}, bounds=[{start_sample},{stop_sample}), history_row={row_index}, fusion_index={fusion_index}, optimizer_step={optimizer_step_index}, max_innovation={max_innovation_abs_m:.6g}, max_gain={max_gain_fro_norm:.6g}')
                state_losses.append(state_loss)
                total_state_sum_t += state_loss.detach()
                total_position_sum_t += position_loss.detach()
                total_count += 1
                max_position_correction_t = torch.maximum(max_position_correction_t, torch.linalg.vector_norm(correction64[0:3]).detach())
                max_velocity_correction_t = torch.maximum(max_velocity_correction_t, torch.linalg.vector_norm(correction64[3:6]).detach())
                max_attitude_correction_t = torch.maximum(max_attitude_correction_t, torch.linalg.vector_norm(correction64[6:9]).detach())
                nav_train = _torch_inject_error_state(nav_train, correction64)
                posterior_residual, posterior_sat_ids = _torch_measurement_innovation(nav_train, prepared_measurements)
                if posterior_sat_ids != current_sat_ids:
                    raise RuntimeError('training satellite order changed within one fusion update')
                previous_context = _torch_training_context_from_completed_epoch(current_sat_ids, posterior_residual, zero_error_state_t, correction64, last_feature_accel, last_feature_gyro, previous_context=previous_context)

                learned_updates_since_detach += 1
                if (
                    training
                    and tbptt_detach_step > 0
                    and learned_updates_since_detach >= tbptt_detach_step
                    and sample_index < stop_sample - 1
                ):
                    # Truncated BPTT: preserve numerical forward recursion while
                    # severing the autograd graph at a controlled cadence.
                    nav_train = _detach_torch_navigation_state(nav_train)
                    previous_context = _detach_training_context(previous_context)
                    recurrent_state = _detach_recurrent_state(recurrent_state)
                    last_feature_accel = last_feature_accel.detach()
                    last_feature_gyro = last_feature_gyro.detach()
                    learned_updates_since_detach = 0
                    tbptt_detach_count += 1
        if total_count <= 0:
            raise RuntimeError(f'differentiable Data01 training sequence produced no learned updates: trajectory={trajectory_id}, bounds=[{start_sample},{stop_sample})')
        trajectory_mean_loss = torch.stack(state_losses).mean()
        trajectory_eq30 = float(trajectory_mean_loss.detach().cpu())
        if training:
            scaled_loss = trajectory_mean_loss * float(backward_scale)
            if not bool(torch.isfinite(scaled_loss).detach().cpu()):
                raise FloatingPointError(f'non-finite scaled {phase} trajectory loss before backward')
            scaled_loss.backward()
        return {'trajectory_id': None if trajectory_id is None else int(trajectory_id), 'start_sample': int(start_sample), 'stop_sample': int(stop_sample), 'trajectory_eq30': trajectory_eq30, 'state_sum': float(total_state_sum_t.cpu()), 'position_sum': float(total_position_sum_t.cpu()), 'learned_updates': int(total_count), 'skipped_no_measurements': int(skipped_no_measurements), 'max_prior_position_error_m': float(max_prior_position_error_t.cpu()), 'max_innovation_abs_m': float(max_innovation_abs_t.cpu()), 'max_gain_fro_norm': 0.0 if max_gain_fro_t is None else float(max_gain_fro_t.cpu()), 'max_position_correction_norm_m': float(max_position_correction_t.cpu()), 'max_velocity_correction_norm_mps': float(max_velocity_correction_t.cpu()), 'max_attitude_correction_norm_rad': float(max_attitude_correction_t.cpu()), 'tbptt_detach_step': int(tbptt_detach_step), 'tbptt_detach_count': int(tbptt_detach_count)}
    def _active_gradient_finite(phase: str) -> bool:
        parameters = filter_parameters if phase == 'filter' else representation_parameters
        finite = None
        for parameter in parameters:
            if parameter.grad is not None:
                current = torch.all(torch.isfinite(parameter.grad))
                finite = current if finite is None else finite & current
        return True if finite is None else bool(finite.detach().cpu())
    def _active_gradient_l2_norm(phase: str) -> float:
        parameters = filter_parameters if phase == 'filter' else representation_parameters
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                grad = parameter.grad.detach().double()
                total += float(torch.sum(grad * grad).cpu())
        return math.sqrt(total)
    def _clip_active_gradient(phase: str) -> tuple[float, float]:
        """Overflow-safe global L2 gradient clipping.

        KalmanNet4SensorFusion uses global norm clipping with max_norm=1.
        This implementation accumulates the norm in float64, because the
        standard float32 total-norm calculation can overflow even when all
        individual gradient entries remain finite.
        """
        parameters = filter_parameters if phase == 'filter' else representation_parameters
        active_parameters = [
            parameter for parameter in parameters if parameter.grad is not None
        ]
        if not active_parameters:
            return 0.0, 0.0

        for parameter in active_parameters:
            if not bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu()):
                raise FloatingPointError(
                    f'non-finite {phase} gradient element before clipping'
                )

        total_norm_before = _active_gradient_l2_norm(phase)
        if not math.isfinite(total_norm_before):
            raise FloatingPointError(
                f'non-finite {phase} float64 global gradient norm before clipping'
            )

        if total_norm_before > GRADIENT_CLIP_NORM:
            scale = GRADIENT_CLIP_NORM / max(total_norm_before, 1e-300)
            with torch.no_grad():
                for parameter in active_parameters:
                    parameter.grad.mul_(scale)

        total_norm_after = _active_gradient_l2_norm(phase)
        if not math.isfinite(total_norm_after):
            raise FloatingPointError(
                f'non-finite {phase} float64 global gradient norm after clipping'
            )
        return float(total_norm_before), float(total_norm_after)

    def _parameter_l2_norm(parameters) -> float:
        total = 0.0
        with torch.no_grad():
            for parameter in parameters:
                value = parameter.detach().double()
                total += float(torch.sum(value * value).cpu())
        return math.sqrt(total)
    def _snapshot_model_state():
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    def _restore_model_state(state_dict) -> None:
        model.load_state_dict({key: value.to(DEVICE) for key, value in state_dict.items()})
    def _build_v2_epoch_batches():
        """Latent-KalmanNet Algorithm 2: randomly divide D into Q batches once per epoch."""
        if not 1 <= TRAJECTORY_BATCH_SIZE <= trajectory_count:
            raise ValueError('V2 trajectory batch size must satisfy 1 <= N_B <= N_E')
        trajectory_ids = list(range(trajectory_count))
        random.shuffle(trajectory_ids)
        epoch_batches = [
            np.asarray(trajectory_ids[start:start + TRAJECTORY_BATCH_SIZE], dtype=int)
            for start in range(0, trajectory_count, TRAJECTORY_BATCH_SIZE)
        ]
        if not epoch_batches or any(batch.size == 0 for batch in epoch_batches):
            raise RuntimeError('Latent-KalmanNet Algorithm 2 partition produced an empty batch')
        flattened = [int(v) for batch in epoch_batches for v in batch.tolist()]
        if sorted(flattened) != list(range(trajectory_count)):
            raise RuntimeError('Algorithm 2 epoch partition must contain every training trajectory exactly once')
        return epoch_batches

    def _run_training_phase(
        phase: str,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        epoch_batches,
    ):
        """Latent-KalmanNet Algorithm 2 phase over all Q batches of one shared epoch partition."""
        _set_alternating_phase(phase)
        if not epoch_batches:
            raise ValueError('Algorithm 2 requires at least one batch')

        total_state_sum = 0.0
        total_position_sum = 0.0
        total_count = 0
        total_skipped = 0
        all_trajectory_eq30 = []
        batch_objectives = []
        batch_eq30_means = []
        gradient_norms = []
        gradient_norms_after_clip = []
        clipped_batch_count = 0
        batch_partitions = []
        maxima = {
            'max_prior_position_error_m': 0.0,
            'max_innovation_abs_m': 0.0,
            'max_gain_fro_norm': 0.0,
            'max_position_correction_norm_m': 0.0,
            'max_velocity_correction_norm_mps': 0.0,
            'max_attitude_correction_norm_rad': 0.0,
        }

        gain_head_l2_before_epoch = _parameter_l2_norm(model.gain_head.parameters())

        for batch_index, batch_ids in enumerate(epoch_batches):
            batch_ids = np.asarray(batch_ids, dtype=int).reshape(-1)
            actual_batch_size = int(batch_ids.size)
            if actual_batch_size <= 0:
                raise RuntimeError('Algorithm 2 encountered an empty mini-batch')

            optimizer.zero_grad(set_to_none=True)

            batch_state_sum = 0.0
            batch_position_sum = 0.0
            batch_count = 0
            batch_skipped = 0
            batch_trajectory_eq30 = []

            for trajectory_id_raw in batch_ids:
                trajectory_id = int(trajectory_id_raw)
                start_sample, stop_sample = training_trajectories[trajectory_id]

                metrics = _rollout_actual_closed_loop_trajectory(
                    start_sample,
                    stop_sample,
                    phase=phase,
                    backward_scale=1.0 / actual_batch_size,
                    trajectory_id=trajectory_id,
                    optimizer_step_index=batch_index,
                    initialization_mode='kalmannet_v2_truth',
                )

                batch_trajectory_eq30.append(metrics['trajectory_eq30'])
                batch_state_sum += metrics['state_sum']
                batch_position_sum += metrics['position_sum']
                batch_count += metrics['learned_updates']
                batch_skipped += metrics['skipped_no_measurements']

                for key in maxima:
                    maxima[key] = max(maxima[key], metrics[key])

            if batch_count <= 0 or not batch_trajectory_eq30:
                raise RuntimeError(
                    f'KalmanNet-V2 Algorithm 2 {phase} batch {batch_index} produced no learned updates'
                )

            regularization = _eq32_regularization()
            if not torch.isfinite(regularization):
                raise FloatingPointError(
                    f'non-finite {phase} Eq.(32) regularization in Algorithm 2 batch {batch_index}'
                )
            if GAMMA_L2:
                regularization.backward()

            if not _active_gradient_finite(phase):
                raise FloatingPointError(
                    f'non-finite {phase} gradient before Algorithm 2 optimizer step {batch_index}'
                )

            batch_data_loss = float(np.mean(batch_trajectory_eq30))
            batch_objective = batch_data_loss + float(regularization.detach().cpu())
            if not math.isfinite(batch_objective):
                raise FloatingPointError(
                    f'non-finite {phase} Algorithm 2 batch objective before step {batch_index}'
                )

            gradient_l2_norm = _active_gradient_l2_norm(phase)
            gradient_norm_before_clip, gradient_norm_after_clip = _clip_active_gradient(phase)

            # The two independent pre-clip norm calculations should agree
            # up to floating-point roundoff.
            if not math.isclose(
                gradient_l2_norm,
                gradient_norm_before_clip,
                rel_tol=1e-6,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f'{phase} pre-clip gradient norm audit mismatch: '
                    f'{gradient_l2_norm} vs {gradient_norm_before_clip}'
                )

            gradient_norms.append(float(gradient_norm_before_clip))
            gradient_norms_after_clip.append(float(gradient_norm_after_clip))
            if gradient_norm_before_clip > GRADIENT_CLIP_NORM:
                clipped_batch_count += 1

            if not _active_gradient_finite(phase):
                raise FloatingPointError(
                    f'non-finite {phase} gradient after clipping before '
                    f'Algorithm 2 optimizer step {batch_index}'
                )

            optimizer.step()

            total_state_sum += batch_state_sum
            total_position_sum += batch_position_sum
            total_count += batch_count
            total_skipped += batch_skipped
            all_trajectory_eq30.extend(float(v) for v in batch_trajectory_eq30)
            batch_objectives.append(float(batch_objective))
            batch_eq30_means.append(float(batch_data_loss))
            batch_partitions.append([int(v) for v in batch_ids.tolist()])

        if total_count <= 0 or not all_trajectory_eq30:
            raise RuntimeError(f'KalmanNet-V2 Algorithm 2 {phase} phase produced no learned updates')

        flattened = [v for batch in batch_partitions for v in batch]
        if sorted(flattened) != list(range(trajectory_count)):
            raise RuntimeError(
                f'KalmanNet-V2 Algorithm 2 {phase} phase did not cover every trajectory exactly once'
            )

        gain_head_l2_after_epoch = _parameter_l2_norm(model.gain_head.parameters())

        out = {
            'eq30': float(total_state_sum / total_count),
            'position_rmse_m': math.sqrt(total_position_sum / total_count),
            'trajectory_mean_eq30': float(np.mean(all_trajectory_eq30)),
            'objective': float(np.mean(batch_objectives)),
            'mean_batch_eq30': float(np.mean(batch_eq30_means)),
            'learned_updates': int(total_count),
            'skipped_no_measurements': int(total_skipped),
            'optimizer_steps': int(len(epoch_batches)),
            'trajectory_count': int(len(flattened)),
            'available_trajectory_count': int(trajectory_count),
            'sequence_batch_size': int(TRAJECTORY_BATCH_SIZE),
            'sampled_trajectory_ids': [int(v) for v in flattened],
            'batch_partitions': batch_partitions,
            'epoch_dataset_coverage_fraction': float(len(set(flattened)) / trajectory_count),
            'gradient_l2_norm': float(np.mean(gradient_norms)),
            'gradient_l2_norm_max': float(np.max(gradient_norms)),
            'gradient_l2_norm_after_clip': float(np.mean(gradient_norms_after_clip)),
            'gradient_l2_norm_after_clip_max': float(np.max(gradient_norms_after_clip)),
            'gradient_clip_max_norm': float(GRADIENT_CLIP_NORM),
            'gradient_clipped_optimizer_steps': int(clipped_batch_count),
            'gain_head_parameter_l2_before': float(gain_head_l2_before_epoch),
            'gain_head_parameter_l2_after': float(gain_head_l2_after_epoch),
            'epoch': int(epoch),
        }
        out.update(maxima)
        return out

    def _run_v1_training_phase(phase: str, optimizer: torch.optim.Optimizer, epoch: int):
        """Full chronological Data01 forward rollout with truncated BPTT."""
        _set_alternating_phase(phase)
        optimizer.zero_grad(set_to_none=True)
        metrics = _rollout_actual_closed_loop_trajectory(
            0,
            training_sample_count,
            phase=phase,
            backward_scale=1.0,
            trajectory_id=0,
            optimizer_step_index=0,
            initialization_mode='kalmannet_v2_truth',
            tbptt_detach_step=V1_TBPTT_DETACH_STEP,
        )
        regularization = _eq32_regularization()
        if not torch.isfinite(regularization):
            raise FloatingPointError(f'non-finite {phase} V1 Eq.(32) regularization')
        if GAMMA_L2:
            regularization.backward()
        if not _active_gradient_finite(phase):
            raise FloatingPointError(f'non-finite {phase} V1 gradient before optimizer step')
        objective = metrics['trajectory_eq30'] + float(regularization.detach().cpu())
        if not math.isfinite(objective):
            raise FloatingPointError(f'non-finite {phase} V1 full-trajectory objective before step')
        gradient_l2_norm = _active_gradient_l2_norm(phase)
        gradient_norm_before_clip, gradient_norm_after_clip = _clip_active_gradient(phase)
        if not math.isclose(
            gradient_l2_norm,
            gradient_norm_before_clip,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f'{phase} V1 pre-clip gradient norm audit mismatch: '
                f'{gradient_l2_norm} vs {gradient_norm_before_clip}'
            )
        if not _active_gradient_finite(phase):
            raise FloatingPointError(
                f'non-finite {phase} V1 gradient after clipping before optimizer step'
            )
        gain_head_l2_before = _parameter_l2_norm(model.gain_head.parameters())
        optimizer.step()
        gain_head_l2_after = _parameter_l2_norm(model.gain_head.parameters())
        return {'objective': float(objective), 'eq30': float(metrics['trajectory_eq30']), 'trajectory_mean_eq30': float(metrics['trajectory_eq30']), 'position_rmse_m': math.sqrt(metrics['position_sum'] / metrics['learned_updates']), 'learned_updates': int(metrics['learned_updates']), 'skipped_no_measurements': int(metrics['skipped_no_measurements']), 'optimizer_steps': 1, 'trajectory_count': 1, 'sequence_batch_size': 1, 'max_prior_position_error_m': float(metrics['max_prior_position_error_m']), 'max_innovation_abs_m': float(metrics['max_innovation_abs_m']), 'max_gain_fro_norm': float(metrics['max_gain_fro_norm']), 'max_position_correction_norm_m': float(metrics['max_position_correction_norm_m']), 'max_velocity_correction_norm_mps': float(metrics['max_velocity_correction_norm_mps']), 'max_attitude_correction_norm_rad': float(metrics['max_attitude_correction_norm_rad']), 'gradient_l2_norm': float(gradient_norm_before_clip), 'gradient_l2_norm_after_clip': float(gradient_norm_after_clip), 'gradient_clip_max_norm': float(GRADIENT_CLIP_NORM), 'gradient_clipped_optimizer_steps': int(gradient_norm_before_clip > GRADIENT_CLIP_NORM), 'tbptt_detach_step': int(metrics['tbptt_detach_step']), 'tbptt_detach_count': int(metrics['tbptt_detach_count']), 'gain_head_parameter_l2_before': float(gain_head_l2_before), 'gain_head_parameter_l2_after': float(gain_head_l2_after), 'epoch': int(epoch)}
    def _evaluate_actual_closed_loop_training_objective():
        """Frozen-weight continuous Data01 rollout."""
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        metrics = _rollout_actual_closed_loop_trajectory(0, training_sample_count, phase=None, backward_scale=None, trajectory_id=None, optimizer_step_index=None)
        metrics['eq30'] = metrics['state_sum'] / metrics['learned_updates']
        metrics['position_rmse_m'] = math.sqrt(metrics['position_sum'] / metrics['learned_updates'])
        metrics['objective'] = metrics['eq30'] + float(_eq32_regularization().detach().cpu())
        return metrics
    def _evaluate_true_zero_gain_closed_loop_baseline():
        """Evaluate a genuine K=0 recursive baseline without changing initialization.

        The trainable model is snapshotted, the gain head is temporarily set
        to exact zero so Fig. 8 produces dx = K @ innovation = 0, and then
        the original nonzero PyTorch initialization is restored exactly.
        This diagnostic baseline is never a checkpoint candidate.
        """
        saved_state = _snapshot_model_state()
        try:
            with torch.no_grad():
                model.gain_head.weight.zero_()
                if model.gain_head.bias is not None:
                    model.gain_head.bias.zero_()
            metrics = _evaluate_actual_closed_loop_training_objective()
        finally:
            _restore_model_state(saved_state)

        if abs(float(metrics['max_gain_fro_norm'])) > 1e-12:
            raise RuntimeError(
                'true zero-gain baseline produced a nonzero Kalman gain'
            )
        if abs(float(metrics['max_position_correction_norm_m'])) > 1e-9:
            raise RuntimeError(
                'true zero-gain baseline produced a nonzero position correction'
            )
        if abs(float(metrics['max_velocity_correction_norm_mps'])) > 1e-9:
            raise RuntimeError(
                'true zero-gain baseline produced a nonzero velocity correction'
            )
        if abs(float(metrics['max_attitude_correction_norm_rad'])) > 1e-12:
            raise RuntimeError(
                'true zero-gain baseline produced a nonzero attitude correction'
            )
        return metrics

    def _evaluate_teacher_forced_eq30():
        """Clean sequential Yan X/M/Y metric retained only for diagnostics."""
        model.eval()
        state_sum_t = zero_error_state_t.new_zeros(())
        position_sum_t = zero_error_state_t.new_zeros(())
        recurrent_state = None
        with torch.inference_mode():
            for sample_index in range(training_sample_count):
                fixed_eval = torch.as_tensor(fixed_normalized[sample_index], dtype=torch.float32, device=DEVICE)
                obs_eval = torch.as_tensor(observations_normalized[sample_index], dtype=torch.float32, device=DEVICE)
                mask_eval = torch.as_tensor(satellite_masks[sample_index], dtype=torch.bool, device=DEVICE)
                channel_eval = torch.as_tensor(channel_masks[sample_index], dtype=torch.bool, device=DEVICE)
                innovation_eval = torch.as_tensor(innovation_padded[sample_index], dtype=torch.float32, device=DEVICE)
                fixed_eval, obs_eval, mask_eval, channel_eval, innovation_eval = _torch_pad_training_epoch(
                    fixed_eval,
                    obs_eval,
                    mask_eval,
                    channel_eval,
                    innovation_eval,
                    network_nmax,
                )
                output = model(
                    fixed_eval.unsqueeze(0),
                    obs_eval.unsqueeze(0),
                    mask_eval.unsqueeze(0),
                    channel_eval.unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
                recurrent_state = output.recurrent_state
                correction = fig8_state_update(output, innovation_eval.unsqueeze(0))[0].double()
                residual_9 = target_base_t[sample_index] - correction[:SUPERVISED_STATE_DIM]
                state_sum_t += torch.sum(residual_9 ** 2)
                position_sum_t += torch.sum(residual_9[:3] ** 2)
        state_sum = float(state_sum_t.cpu())
        position_sum = float(position_sum_t.cpu())
        return {'eq30': state_sum / training_sample_count, 'position_rmse_m': math.sqrt(position_sum / training_sample_count)}
    pretraining_zero_gain_metrics = _evaluate_true_zero_gain_closed_loop_baseline()
    print(
        f"pre-training TRUE zero-K Data01 baseline: "
        f"Eq30={pretraining_zero_gain_metrics['eq30']:.6g}, "
        f"posRMSE={pretraining_zero_gain_metrics['position_rmse_m']:.3f} m, "
        f"maxInnovation={pretraining_zero_gain_metrics['max_innovation_abs_m']:.3f} m, "
        f"maxK={pretraining_zero_gain_metrics['max_gain_fro_norm']:.3g}, "
        f"maxPosCorr={pretraining_zero_gain_metrics['max_position_correction_norm_m']:.3g} m"
    )

    # KalmanNet/Latent-KalmanNet style checkpoint discipline:
    # an untrained/random model and the K=0 diagnostic baseline are NOT
    # candidates. Only checkpoints produced after an optimizer update may win.
    best_model_state = None
    best_selection_eq30 = float('inf')
    best_selection_rmse_m = float('inf')
    best_selection_stage = None
    best_selection_epoch = None

    print('\n=== DATA01 TRAINING ===')
    training_history.append({
        'stage': 'pretraining_true_zero_gain_data01_baseline_diagnostic_only',
        'eligible_for_checkpoint_selection': False,
        'eq30': float(pretraining_zero_gain_metrics['eq30']),
        'position_rmse_m': float(pretraining_zero_gain_metrics['position_rmse_m']),
        'max_innovation_abs_m': float(pretraining_zero_gain_metrics['max_innovation_abs_m']),
        'max_gain_fro_norm': float(pretraining_zero_gain_metrics['max_gain_fro_norm']),
        'max_position_correction_norm_m': float(pretraining_zero_gain_metrics['max_position_correction_norm_m']),
        'learned_updates': int(pretraining_zero_gain_metrics['learned_updates']),
    })
    final_training_metrics = None
    epochs_ran = 0
    print('\n--- KalmanNet V2 warm-up ---')
    for epoch in range(1, KNET_V2_WARMUP_EPOCHS + 1):
        epochs_ran += 1
        # Latent-KalmanNet Algorithm 2:
        # Randomly divide D into Q batches once, optimize theta over all Q,
        # then optimize psi over the exact same ordered Q batches.
        epoch_batches = _build_v2_epoch_batches()
        filter_phase = _run_training_phase(
            'filter',
            filter_optimizer,
            epoch,
            epoch_batches,
        )
        representation_phase = _run_training_phase(
            'representation',
            representation_optimizer,
            epoch,
            epoch_batches,
        )
        if filter_phase['batch_partitions'] != representation_phase['batch_partitions']:
            raise RuntimeError('Algorithm 2 requires the same epoch partition for theta and psi phases')
        final_training_metrics = representation_phase
        current_filter_lr = float(filter_optimizer.param_groups[0]['lr'])
        current_representation_lr = float(representation_optimizer.param_groups[0]['lr'])
        recursive_monitor = _evaluate_actual_closed_loop_training_objective()
        recursive_eq30 = float(recursive_monitor['eq30'])
        recursive_rmse_m = float(recursive_monitor['position_rmse_m'])
        if (
            math.isfinite(recursive_eq30)
            and math.isfinite(recursive_rmse_m)
            and recursive_eq30 < best_selection_eq30
        ):
            best_model_state = _snapshot_model_state()
            best_selection_eq30 = recursive_eq30
            best_selection_rmse_m = recursive_rmse_m
            best_selection_stage = 'kalmannet_v2_Data01_recursive_monitor'
            best_selection_epoch = int(epoch)
        training_history.append({'stage': 'kalmannet_v2_warmup_yan_masked_cla_latent_alternating', 'epoch': epoch, 'global_training_epoch': epochs_ran, 'sequence_length': int(training_sample_count), 'kalmannet_v2_short_trajectory_length': int(KNET_V2_TRAJECTORY_LENGTH), 'kalmannet_v2_initialization': 'ground_truth_navigation_state', 'kalmannet_v2_trained_sample_count': int(v2_trained_sample_count), 'kalmannet_v2_initialization_sample_count': int(v2_initialization_sample_count), 'kalmannet_v2_unused_tail_sample_count': int(v2_unused_sample_count), 'trajectory_count_available': int(trajectory_count), 'trajectory_batch_size': int(TRAJECTORY_BATCH_SIZE), 'algorithm2_q_batches': int(len(epoch_batches)), 'algorithm2_epoch_batch_partitions': filter_phase['batch_partitions'], 'algorithm2_same_partition_theta_psi': True, 'algorithm2_full_dataset_coverage': float(filter_phase['epoch_dataset_coverage_fraction']), 'filter_phase_sampled_trajectory_ids': filter_phase['sampled_trajectory_ids'], 'filter_phase_total_objective': filter_phase['objective'], 'filter_phase_eq30': filter_phase['eq30'], 'filter_phase_trajectory_mean_eq30': filter_phase['trajectory_mean_eq30'], 'filter_phase_position_rmse_m': filter_phase['position_rmse_m'], 'filter_phase_learned_updates': filter_phase['learned_updates'], 'filter_phase_optimizer_steps': filter_phase['optimizer_steps'], 'filter_phase_gradient_l2_norm': filter_phase['gradient_l2_norm'], 'filter_phase_gradient_l2_norm_after_clip': filter_phase['gradient_l2_norm_after_clip'], 'filter_phase_gradient_clip_max_norm': filter_phase['gradient_clip_max_norm'], 'filter_phase_gradient_clipped_optimizer_steps': filter_phase['gradient_clipped_optimizer_steps'], 'filter_phase_gain_head_l2_before': filter_phase['gain_head_parameter_l2_before'], 'filter_phase_gain_head_l2_after': filter_phase['gain_head_parameter_l2_after'], 'filter_phase_max_innovation_abs_m': filter_phase['max_innovation_abs_m'], 'filter_phase_max_gain_fro_norm': filter_phase['max_gain_fro_norm'], 'filter_phase_max_position_correction_norm_m': filter_phase['max_position_correction_norm_m'], 'representation_phase_sampled_trajectory_ids': representation_phase['sampled_trajectory_ids'], 'representation_phase_total_objective': representation_phase['objective'], 'representation_phase_eq30': representation_phase['eq30'], 'representation_phase_trajectory_mean_eq30': representation_phase['trajectory_mean_eq30'], 'representation_phase_position_rmse_m': representation_phase['position_rmse_m'], 'representation_phase_learned_updates': representation_phase['learned_updates'], 'representation_phase_optimizer_steps': representation_phase['optimizer_steps'], 'representation_phase_gradient_l2_norm': representation_phase['gradient_l2_norm'], 'representation_phase_gradient_l2_norm_after_clip': representation_phase['gradient_l2_norm_after_clip'], 'representation_phase_gradient_clip_max_norm': representation_phase['gradient_clip_max_norm'], 'representation_phase_gradient_clipped_optimizer_steps': representation_phase['gradient_clipped_optimizer_steps'], 'representation_phase_gain_head_l2_before': representation_phase['gain_head_parameter_l2_before'], 'representation_phase_gain_head_l2_after': representation_phase['gain_head_parameter_l2_after'], 'representation_phase_max_innovation_abs_m': representation_phase['max_innovation_abs_m'], 'representation_phase_max_gain_fro_norm': representation_phase['max_gain_fro_norm'], 'representation_phase_max_position_correction_norm_m': representation_phase['max_position_correction_norm_m'], 'Data01_recursive_monitor_eq30': float(recursive_monitor['eq30']), 'Data01_recursive_monitor_position_rmse_m': float(recursive_monitor['position_rmse_m']), 'best_Data01_recursive_monitor_eq30_so_far': float(best_selection_eq30), 'best_Data01_recursive_monitor_position_rmse_m_so_far': float(best_selection_rmse_m), 'filter_block_learning_rate': current_filter_lr, 'representation_block_learning_rate': current_representation_lr})
        print(f"V2 epoch {epoch:03d}/{KNET_V2_WARMUP_EPOCHS}: theta_Eq30={filter_phase['trajectory_mean_eq30']:.6g}, theta_RMSE={filter_phase['position_rmse_m']:.3f}m, theta_grad={filter_phase['gradient_l2_norm']:.3g}->{filter_phase['gradient_l2_norm_after_clip']:.3g}, theta_Kparam={filter_phase['gain_head_parameter_l2_before']:.3g}->{filter_phase['gain_head_parameter_l2_after']:.3g}, theta_maxK={filter_phase['max_gain_fro_norm']:.3g}; psi_Eq30={representation_phase['trajectory_mean_eq30']:.6g}, psi_RMSE={representation_phase['position_rmse_m']:.3f}m, psi_grad={representation_phase['gradient_l2_norm']:.3g}->{representation_phase['gradient_l2_norm_after_clip']:.3g}; recursive_Eq30={recursive_monitor['eq30']:.6g}, recursive_RMSE={recursive_monitor['position_rmse_m']:.3f}m, best_recursive_RMSE={best_selection_rmse_m:.3f}m")
    print('\n--- KalmanNet V1 full-sequence fine-tuning ---')
    if KNET_V1_FINE_TUNE_EPOCHS == 0:
        print('V1 fine-tuning disabled by MKNET_V1_FINE_TUNE_EPOCHS=0 (V2-only diagnostic).')
    for epoch in range(1, KNET_V1_FINE_TUNE_EPOCHS + 1):
        epochs_ran += 1
        filter_phase = _run_v1_training_phase('filter', filter_optimizer, epoch)
        representation_phase = _run_v1_training_phase('representation', representation_optimizer, epoch)
        final_training_metrics = representation_phase
        current_filter_lr = float(filter_optimizer.param_groups[0]['lr'])
        current_representation_lr = float(representation_optimizer.param_groups[0]['lr'])
        recursive_monitor = _evaluate_actual_closed_loop_training_objective()
        recursive_eq30 = float(recursive_monitor['eq30'])
        recursive_rmse_m = float(recursive_monitor['position_rmse_m'])
        if (
            math.isfinite(recursive_eq30)
            and math.isfinite(recursive_rmse_m)
            and recursive_eq30 < best_selection_eq30
        ):
            best_model_state = _snapshot_model_state()
            best_selection_eq30 = recursive_eq30
            best_selection_rmse_m = recursive_rmse_m
            best_selection_stage = 'kalmannet_v1_Data01_recursive_monitor'
            best_selection_epoch = int(epoch)
        training_history.append({'stage': 'kalmannet_v1_full_sequence_finetune_yan_masked_cla_latent_alternating', 'epoch': epoch, 'global_training_epoch': epochs_ran, 'sequence_length': int(training_sample_count), 'initialization': 'ground_truth_navigation_state', 'trajectory_count': 1, 'trajectory_batch_size': 1, 'filter_phase_total_objective': filter_phase['objective'], 'filter_phase_eq30': filter_phase['eq30'], 'filter_phase_trajectory_mean_eq30': filter_phase['trajectory_mean_eq30'], 'filter_phase_position_rmse_m': filter_phase['position_rmse_m'], 'filter_phase_learned_updates': filter_phase['learned_updates'], 'filter_phase_optimizer_steps': filter_phase['optimizer_steps'], 'filter_phase_gradient_l2_norm': filter_phase['gradient_l2_norm'], 'filter_phase_gradient_l2_norm_after_clip': filter_phase['gradient_l2_norm_after_clip'], 'filter_phase_gradient_clip_max_norm': filter_phase['gradient_clip_max_norm'], 'filter_phase_gradient_clipped_optimizer_steps': filter_phase['gradient_clipped_optimizer_steps'], 'filter_phase_gain_head_l2_before': filter_phase['gain_head_parameter_l2_before'], 'filter_phase_gain_head_l2_after': filter_phase['gain_head_parameter_l2_after'], 'filter_phase_max_innovation_abs_m': filter_phase['max_innovation_abs_m'], 'filter_phase_max_gain_fro_norm': filter_phase['max_gain_fro_norm'], 'filter_phase_max_position_correction_norm_m': filter_phase['max_position_correction_norm_m'], 'representation_phase_total_objective': representation_phase['objective'], 'representation_phase_eq30': representation_phase['eq30'], 'representation_phase_trajectory_mean_eq30': representation_phase['trajectory_mean_eq30'], 'representation_phase_position_rmse_m': representation_phase['position_rmse_m'], 'representation_phase_learned_updates': representation_phase['learned_updates'], 'representation_phase_optimizer_steps': representation_phase['optimizer_steps'], 'representation_phase_gradient_l2_norm': representation_phase['gradient_l2_norm'], 'representation_phase_gradient_l2_norm_after_clip': representation_phase['gradient_l2_norm_after_clip'], 'representation_phase_gradient_clip_max_norm': representation_phase['gradient_clip_max_norm'], 'representation_phase_gradient_clipped_optimizer_steps': representation_phase['gradient_clipped_optimizer_steps'], 'representation_phase_gain_head_l2_before': representation_phase['gain_head_parameter_l2_before'], 'representation_phase_gain_head_l2_after': representation_phase['gain_head_parameter_l2_after'], 'representation_phase_max_innovation_abs_m': representation_phase['max_innovation_abs_m'], 'representation_phase_max_gain_fro_norm': representation_phase['max_gain_fro_norm'], 'representation_phase_max_position_correction_norm_m': representation_phase['max_position_correction_norm_m'], 'Data01_recursive_monitor_eq30': float(recursive_monitor['eq30']), 'Data01_recursive_monitor_position_rmse_m': float(recursive_monitor['position_rmse_m']), 'best_Data01_recursive_monitor_eq30_so_far': float(best_selection_eq30), 'best_Data01_recursive_monitor_position_rmse_m_so_far': float(best_selection_rmse_m), 'filter_block_learning_rate': current_filter_lr, 'representation_block_learning_rate': current_representation_lr})
        print(f"V1 epoch {epoch:03d}/{KNET_V1_FINE_TUNE_EPOCHS}: theta_Eq30={filter_phase['trajectory_mean_eq30']:.6g}, theta_RMSE={filter_phase['position_rmse_m']:.3f}m, theta_grad={filter_phase['gradient_l2_norm']:.3g}->{filter_phase['gradient_l2_norm_after_clip']:.3g}, theta_Kparam={filter_phase['gain_head_parameter_l2_before']:.3g}->{filter_phase['gain_head_parameter_l2_after']:.3g}; psi_Eq30={representation_phase['trajectory_mean_eq30']:.6g}, psi_RMSE={representation_phase['position_rmse_m']:.3f}m, psi_grad={representation_phase['gradient_l2_norm']:.3g}->{representation_phase['gradient_l2_norm_after_clip']:.3g}; recursive_Eq30={recursive_monitor['eq30']:.6g}, recursive_RMSE={recursive_monitor['position_rmse_m']:.3f}m, best_recursive_RMSE={best_selection_rmse_m:.3f}m")
    if final_training_metrics is None:
        raise RuntimeError('Training completed without producing any epoch metrics')
    # Match the KalmanNet/Latent-KalmanNet checkpoint-selection discipline:
    # restore only a finite checkpoint created after actual optimizer updates.
    if best_model_state is None or best_selection_stage is None or best_selection_epoch is None:
        raise RuntimeError(
            'Training produced no finite trained checkpoint. '
            'The untrained random model and zero-gain diagnostic baseline are '
            'intentionally ineligible for Data02 evaluation.'
        )
    _restore_model_state(best_model_state)
    print(
        f"restored best TRAINED Data01 recursive-monitor checkpoint: "
        f"stage={best_selection_stage}, epoch={best_selection_epoch}, "
        f"Eq30={best_selection_eq30:.6g}, posRMSE={best_selection_rmse_m:.3f} m"
    )
    final_training_metrics = _evaluate_actual_closed_loop_training_objective()
    zero_gain_rmse = float(pretraining_zero_gain_metrics['position_rmse_m'])
    trained_rmse = float(final_training_metrics['position_rmse_m'])
    final_teacher_forced_metrics = _evaluate_teacher_forced_eq30()
    training_performance_summary = {'zero_gain_closed_loop': {'eq30': float(pretraining_zero_gain_metrics['eq30']), 'position_rmse_m': zero_gain_rmse, 'max_innovation_abs_m': float(pretraining_zero_gain_metrics['max_innovation_abs_m'])}, 'trained_closed_loop': {'eq30': float(final_training_metrics['eq30']), 'position_rmse_m': trained_rmse, 'rmse_ratio_vs_zero_gain': trained_rmse / max(zero_gain_rmse, 1e-12), 'max_innovation_abs_m': float(final_training_metrics['max_innovation_abs_m']), 'max_gain_fro_norm': float(final_training_metrics['max_gain_fro_norm']), 'max_position_correction_norm_m': float(final_training_metrics['max_position_correction_norm_m'])}, 'one_step': {'eq30': float(final_teacher_forced_metrics['eq30']), 'position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m'])}, 'selected_checkpoint': {'stage': best_selection_stage, 'epoch': int(best_selection_epoch), 'Data01_recursive_eq30': float(best_selection_eq30), 'Data01_recursive_position_rmse_m': float(best_selection_rmse_m)}}
    print('Data01 training performance:', json.dumps(training_performance_summary, indent=2))
    model.eval()
    best_epoch = int(best_selection_epoch)
    best_val_state_loss = float(best_selection_eq30)
    best_val_position_rmse_m = float(best_selection_rmse_m)
    selected_training_stage = str(best_selection_stage)
    def _make_online_context(sat_ids, posterior_residual, x_pred, x_post, accel, gyro, previous_context=None):
        """Store the completed fusion context used by the next causal network input."""
        x_pred = np.asarray(x_pred, dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(x_post, dtype=float).reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = np.zeros(INS_STATE_DIM) if previous_context is None else x_post - np.asarray(previous_context['x_post'], dtype=float).reshape(INS_STATE_DIM)
        context = {'sat_ids': tuple(sat_ids), 'residual': np.asarray(posterior_residual, dtype=float).copy(), 'x_pred': x_pred.copy(), 'x_post': x_post.copy(), 'state_innovation': state_innovation.copy(), 'state_residual': state_residual.copy(), 'accel': np.asarray(accel, dtype=float).reshape(3).copy(), 'gyro': np.asarray(gyro, dtype=float).reshape(3).copy()}
        if len(context['sat_ids']) != len(context['residual']) or not np.all(np.isfinite(context['residual'])) or (not np.all(np.isfinite(context['state_innovation']))) or (not np.all(np.isfinite(context['state_residual']))) or (not np.all(np.isfinite(context['accel']))) or (not np.all(np.isfinite(context['gyro']))):
            raise FloatingPointError('invalid completed online fusion context')
        return context
    def _online_feature_arrays(previous_online, current_sat_ids, innovation_now, current_accel, current_gyro):
        """Build one causal online input from current innovation and completed prior context."""
        n = len(current_sat_ids)
        innovation_k = np.asarray(innovation_now, dtype=float).copy()
        current_mask = np.ones(n, dtype=bool)
        residual_k = np.zeros(n, dtype=float)
        residual_mask = np.zeros(n, dtype=bool)
        delta_accel = np.asarray(current_accel, dtype=float).reshape(3) - previous_online['accel']
        delta_gyro = np.asarray(current_gyro, dtype=float).reshape(3) - previous_online['gyro']
        previous_state_innovation = previous_online['state_innovation']
        previous_state_residual = previous_online['state_residual']
        previous_residual = dict(zip(previous_online['sat_ids'], previous_online['residual']))
        for slot, sat_id in enumerate(current_sat_ids):
            if sat_id in previous_residual:
                residual_k[slot] = previous_residual[sat_id]
                residual_mask[slot] = True
        fixed_k_raw = np.concatenate([delta_accel, delta_gyro, previous_state_residual, previous_state_innovation])
        obs_k_raw = np.stack([residual_k, innovation_k], axis=1)
        channel_k = np.stack([residual_mask, current_mask], axis=1)
        fixed_k = _normalize_fixed_groups_np(fixed_k_raw)
        obs_k = _normalize_observation_channels_np(obs_k_raw, channel_k)
        if fixed_k_raw.shape != (FIXED_FEATURE_DIM,) or obs_k_raw.shape != (n, OBSERVATION_FEATURE_DIM) or (not np.all(np.isfinite(fixed_k))) or (not np.all(np.isfinite(obs_k))) or (not np.all(np.isfinite(innovation_k))):
            raise FloatingPointError('invalid/non-finite feature generated during online rollout')
        return (fixed_k, obs_k, current_mask, channel_k, innovation_k, fixed_k_raw, obs_k_raw)
    def _pad_neural_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, slot_capacity: int):
        """Pad one online epoch without changing its physical observations."""
        fixed_k = np.asarray(fixed_k, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_k = np.asarray(obs_k, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        current_mask = np.asarray(current_mask, dtype=bool).reshape(-1)
        channel_k = np.asarray(channel_k, dtype=bool).reshape(-1, OBSERVATION_FEATURE_DIM)
        innovation_k = np.asarray(innovation_k, dtype=float).reshape(-1)
        count = len(current_mask)
        if slot_capacity < count or len(obs_k) != count or len(channel_k) != count or (len(innovation_k) != count):
            raise ValueError('invalid online neural padding dimensions')
        obs_padded = np.zeros((slot_capacity, OBSERVATION_FEATURE_DIM), dtype=float)
        mask_padded = np.zeros(slot_capacity, dtype=bool)
        channel_padded = np.zeros((slot_capacity, OBSERVATION_FEATURE_DIM), dtype=bool)
        innovation_padded_now = np.zeros(slot_capacity, dtype=float)
        obs_padded[:count] = obs_k
        mask_padded[:count] = current_mask
        channel_padded[:count] = channel_k
        innovation_padded_now[:count] = innovation_k
        return (fixed_k, obs_padded, mask_padded, channel_padded, innovation_padded_now)
    def _feature_shift_diagnostics(fixed_raw, obs_raw, channel_mask):
        """Compare one Data02 input with the raw Data01 training envelope only."""
        fixed_raw = np.asarray(fixed_raw, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_raw = np.asarray(obs_raw, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        channel_mask = np.asarray(channel_mask, dtype=bool).reshape(-1, OBSERVATION_FEATURE_DIM)
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
            obs_outside_count += int(np.count_nonzero((values < train_obs_min[channel]) | (values > train_obs_max[channel])))
            obs_ratio = max(obs_ratio, float(np.max(np.abs(values) / train_obs_abs_max[channel])))
        return {'fixed_ood_fraction': float(np.mean(fixed_outside)), 'fixed_max_abs_train_ratio': fixed_ratio, 'obs_ood_fraction': float(obs_outside_count / obs_valid_count) if obs_valid_count else 0.0, 'obs_max_abs_train_ratio': float(obs_ratio)}
    # --- TRAINING CHECKPOINT + RECURSIVE DATA01 DIAGNOSTIC ---
    model.eval()
    torch.save({'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 'checkpoint_schema': 'v98_9_state_masked_cla_tbptt_stable_clip_lr1e3', 'nmax': int(network_nmax), 'training_observed_nmax': int(nmax), 'data01_sequence_nmax': int(data01_sequence_nmax), 'data02_sequence_nmax': int(data02_sequence_nmax), 'nmax_source': 'max_observed_sequence_length_across_selected_Data01_Data02', 'state_order': '[delta_p,delta_v,delta_theta]', 'kalman_gain_state_dimension': INS_STATE_DIM, 'direct_state_label_dimension': SUPERVISED_STATE_DIM, 'feature_normalization_mode': FEATURE_NORMALIZATION_MODE, 'feature_l2_eps': float(FEATURE_L2_EPS), 'feature_normalization_groups': ['Delta_alpha', 'Delta_omega', 'Delta_xhat', 'Delta_xtilde', 'Delta_y_previous_mask_aware', 'Delta_ytilde_current_mask_aware'], 'normalizer_source': normalizer_source, 'alternating_partition': 'psi=masked_conv;theta=masked_lstm+attention+gain_head', 'latent_kalmannet_warm_start': 'not_applied_no_published_supervised_target_for_intermediate_Yan_CNN_features', 'training_stage_selected': selected_training_stage, 'training_epochs': best_epoch, 'requires_recursive_Data01_gate_before_Data02': True, 'sequence_training': 'Yan_Fig7_Fig8_order_adapted_to_user_requested_9_state_branch;KalmanNet_V2_Latent_Algorithm2_random_partition_all_Q_batches_same_partition_theta_then_psi_then_optional_V1_full_sequence;learned_posterior_to_ECEF_INS_to_next_features;no_internal_classical_KF_reset;no_Phi_H_surrogate;fixed_observation_snapshot_recursive_innovation', 'training_sequence_length': int(training_sample_count), 'training_sequence_count': int(trajectory_count), 'sequence_batch_size': int(TRAJECTORY_BATCH_SIZE), 'optimizer_step_inside_sequence': False, 'cross_fusion_navigation_gradient': 'KalmanNet_guided_BPTT_through_9_state_correction_ECEF_INS_and_innovation', 'cross_fusion_neural_gradient': 'V2_short_BPTT;V1_full_forward_with_truncated_BPTT_detach_every_configured_steps', 'synthetic_prior_augmentation': False, 'lstm_sequence_state': 'Yan_Eq25_per_position_hc_carried_across_fusion_epochs_k;feature_position_t_not_used_as_LSTM_time_axis', 'external_filter_state': 'recursive_across_complete_Data01_training_sequence_via_actual_INS;no_internal_classical_KF_reset', 'final_epoch': best_epoch, 'final_train_recursive_eq30_state_loss': best_val_state_loss, 'final_train_recursive_position_rmse_m': best_val_position_rmse_m, 'final_clean_teacher_forced_eq30': float(final_teacher_forced_metrics['eq30']), 'final_clean_teacher_forced_position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m']), 'state_target_scaling': 'none_raw_physical_units_per_Yan_Eq30', 'measurement_mode': 'pseudorange_only', 'observation_residual_feature': 'Yan_Eq11_Delta_y_k_minus_1;same_epoch_Delta_y_k_not_used_because_it_requires_the_current_posterior_and_would_make_K_k_input_circular'}, OUTPUT_DIR / 'best_model.pt')
    (OUTPUT_DIR / 'history.json').write_text(json.dumps(training_history, indent=2), encoding='utf-8')
    if RUN_RECURSIVE_DATA01_DIAGNOSTIC:
        print('\n=== DATA01 RECURSIVE PERFORMANCE ===')
        data01_diag_leo = LEODownlinkSimulator(tle_provider, leo_klobuchar, seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
        diag_nav = initial_nav.copy()
        diag_P = P0.copy()
        diag_rows = []
        diag_fde_stats = FDE_DIA.new_stats() if FDE_DIA_ON else {'enabled': False}
        diag_previous = None
        diag_warm_start = None
        diag_recurrent_state = None
        classical_position_by_fusion_index = {int(row['fusion_index']): gnss_antenna_position(row['posterior_nav'], lever_arm_b_m) for row in history_rows}
        diag_last_gyro, diag_last_accel = compensate_imu(imr.angular_rate_body_radps[0], imr.acceleration_body_mps2[0])
        diag_feature_gyro, diag_feature_accel = (diag_last_gyro, diag_last_accel)
        van_loan_stats_before_data01_diag = dict(_VAN_LOAN_STATS)
        try:
            for event in build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True):
                if isinstance(event, PropagationSegment):
                    imu_index = event.imu_index
                    dt = event.end_time_gpst_s - event.start_time_gpst_s
                    diag_last_gyro, diag_last_accel = compensate_imu(imr.angular_rate_body_radps[imu_index], imr.acceleration_body_mps2[imu_index])
                    diag_feature_gyro, diag_feature_accel = (diag_last_gyro, diag_last_accel)
                    diag_nav = mechanize_ecef(diag_nav, diag_last_gyro, diag_last_accel, dt)
                    diag_F = build_error_state_dynamics(diag_nav, diag_last_accel)
                    diag_Phi, diag_Qd = discretize_process_noise_van_loan(diag_F, Qc, dt)
                    diag_P = diag_Phi @ diag_P @ diag_Phi.T + diag_Qd
                    diag_P = 0.5 * (diag_P + diag_P.T)
                    continue
                fusion_index = event.fusion_index
                t = event.time_gpst_s
                epoch = gnss_epochs[fusion_index]
                if abs(epoch.time_gpst_s - t) > 1e-09:
                    raise RuntimeError('Data01 recursive diagnostic fusion timestamp mismatch')
                diag_gnss = gnss_preprocessor.prepare_epoch(epoch, diag_nav, lever_arm_b_m)
                diag_leo = data01_diag_leo.simulate_epoch(t, fusion_antenna_truth_position[fusion_index])
                diag_measurements = retain_clock_observable_measurements(tuple(diag_gnss) + tuple(diag_leo))
                if not diag_measurements:
                    continue
                truth_position_now = fusion_antenna_truth_position[fusion_index]
                prior_position = gnss_antenna_position(diag_nav, lever_arm_b_m)
                prior_error_3d_m = float(np.linalg.norm(prior_position - truth_position_now))
                classical_position_now = classical_position_by_fusion_index.get(int(fusion_index))
                classical_error_3d_m = float(np.linalg.norm(classical_position_now - truth_position_now)) if classical_position_now is not None else float('nan')
                if diag_previous is None:
                    warm_model = build_measurement_model(diag_nav, diag_measurements, lever_arm_b_m)
                    warm_correction, diag_P, _ = kalman_measurement_update(diag_P, warm_model.innovation, warm_model.H, warm_model.R)
                    diag_nav = inject_error_state(diag_nav, warm_correction)
                    warm_residual = build_innovation_only(diag_nav, diag_measurements, lever_arm_b_m)
                    diag_previous = _make_online_context(warm_model.sat_ids, warm_residual, np.zeros(INS_STATE_DIM), warm_correction, diag_feature_accel, diag_feature_gyro, previous_context=None)
                    diag_warm_start = {'method': 'ordinary_TC_KF_context_only', 'time_gpst_s': float(t), 'excluded_from_learned_metrics': True}
                    continue
                if FDE_DIA_ON:
                    diag_fde = FDE_DIA.decision(diag_nav, diag_P, diag_measurements, lever_arm_b_m, FDE_SIGNIFICANCE_ALPHA)
                    FDE_DIA.accumulate_stats(diag_fde_stats, diag_fde)
                    diag_prefde_model = diag_fde.tested_measurement_model
                    diag_measurements = diag_fde.measurements
                    diag_model = diag_fde.measurement_model
                else:
                    diag_fde = None
                    diag_model = build_measurement_model(diag_nav, diag_measurements, lever_arm_b_m)
                    diag_prefde_model = diag_model
                if FDE_DIA_ON and (not diag_measurements):
                    posterior_position = gnss_antenna_position(diag_nav, lever_arm_b_m)
                    diag_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': truth_position_now.copy(), 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': prior_error_3d_m, 'classical_posterior_error_3d_m': classical_error_3d_m, 'gain_fro_norm': float('nan'), 'correction_position_norm_m': 0.0, 'prefde_innovation_max_abs_m': float(np.max(np.abs(diag_prefde_model.innovation))) if len(diag_prefde_model.innovation) else 0.0, 'obs_max_abs_train_ratio': float('nan')})
                    diag_previous = _make_online_context((), np.empty(0), np.zeros(INS_STATE_DIM), np.zeros(INS_STATE_DIM), diag_feature_accel, diag_feature_gyro, previous_context=diag_previous)
                    continue
                current_sat_ids = diag_model.sat_ids
                fixed_k, obs_k, current_mask, channel_k, innovation_k, fixed_k_raw, obs_k_raw = _online_feature_arrays(diag_previous, current_sat_ids, diag_model.innovation, diag_feature_accel, diag_feature_gyro)
                feature_shift = _feature_shift_diagnostics(fixed_k_raw, obs_k_raw, channel_k)
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _pad_neural_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, network_nmax)
                with torch.inference_mode():
                    diag_output = model(torch.tensor(fixed_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(obs_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(mask_nn[None], dtype=torch.bool, device=DEVICE), torch.tensor(channel_nn[None], dtype=torch.bool, device=DEVICE), recurrent_state=diag_recurrent_state)
                    diag_recurrent_state = diag_output.recurrent_state
                    diag_correction = fig8_state_update(diag_output, torch.tensor(innovation_nn[None], dtype=torch.float32, device=DEVICE))[0]
                diag_gain = diag_output.kalman_gain[0].cpu().numpy().astype(float)
                diag_correction = diag_correction.cpu().numpy().astype(float)
                n_diag = len(diag_measurements)
                active_gain = diag_gain[:, :n_diag]
                if not np.all(np.isfinite(active_gain)) or not np.all(np.isfinite(diag_correction)):
                    raise FloatingPointError('non-finite learned gain/correction in recursive Data01 diagnostic')
                diag_P = learned_gain_covariance_update(diag_P, active_gain, diag_model.H, diag_model.R, diag_correction)
                diag_nav = inject_error_state(diag_nav, diag_correction)
                posterior_position = gnss_antenna_position(diag_nav, lever_arm_b_m)
                posterior_error_3d_m = float(np.linalg.norm(posterior_position - truth_position_now))
                posterior_residual = build_innovation_only(diag_nav, diag_measurements, lever_arm_b_m)
                diag_previous = _make_online_context(current_sat_ids, posterior_residual, np.zeros(INS_STATE_DIM), diag_correction, diag_feature_accel, diag_feature_gyro, previous_context=diag_previous)
                diag_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': truth_position_now.copy(), 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': posterior_error_3d_m, 'classical_posterior_error_3d_m': classical_error_3d_m, 'gain_fro_norm': float(np.linalg.norm(active_gain)), 'correction_position_norm_m': float(np.linalg.norm(diag_correction[:3])), 'prefde_innovation_max_abs_m': float(np.max(np.abs(diag_prefde_model.innovation))) if len(diag_prefde_model.innovation) else 0.0, 'obs_max_abs_train_ratio': float(feature_shift['obs_max_abs_train_ratio'])})
        finally:
            _VAN_LOAN_STATS.clear()
            _VAN_LOAN_STATS.update(van_loan_stats_before_data01_diag)
        if not diag_rows:
            raise RuntimeError('Recursive Data01 diagnostic produced no usable trajectory rows')
        diag_fields = ('time', 'prior_error_3d_m', 'posterior_error_3d_m', 'classical_posterior_error_3d_m', 'gain_fro_norm', 'correction_position_norm_m', 'prefde_innovation_max_abs_m', 'obs_max_abs_train_ratio')
        for row_index, row in enumerate(diag_rows):
            missing = [name for name in diag_fields if name not in row]
            if missing:
                raise RuntimeError(f'recursive Data01 diagnostic row schema mismatch at row {row_index}: missing {missing}')
        diag_estimate = np.stack([row['position'] for row in diag_rows])
        diag_truth = np.stack([row['truth'] for row in diag_rows])
        diag_ned_error = np.empty_like(diag_estimate)
        for i, (est, truth_i) in enumerate(zip(diag_estimate, diag_truth)):
            lat, lon, _ = ecef_to_llh(truth_i)
            diag_ned_error[i] = c_ecef_to_ned(lat, lon) @ (est - truth_i)
        diag_error_3d = np.linalg.norm(diag_ned_error, axis=1)
        diag_classical_error_3d = np.asarray([row['classical_posterior_error_3d_m'] for row in diag_rows], dtype=float)
        diag_rmse_ned = np.sqrt(np.mean(diag_ned_error ** 2, axis=0))
        diag_rmse_3d = float(np.sqrt(np.mean(diag_error_3d ** 2)))
        diag_rmse = np.append(diag_rmse_ned, diag_rmse_3d)
        data01_acceptance_gate = recursive_same_epoch_rmse_gate(diag_error_3d, diag_classical_error_3d)
        diag_time = np.asarray([row['time'] for row in diag_rows], dtype=float)
        diag_obs_ratio = np.asarray([row['obs_max_abs_train_ratio'] for row in diag_rows], dtype=float)
        def _diag_first(mask):
            indices = np.flatnonzero(np.asarray(mask, dtype=bool))
            if not len(indices):
                return None
            i = int(indices[0])
            return {'row_index': i, 'time_gpst_s': float(diag_time[i]), 'posterior_error_3d_m': float(diag_error_3d[i])}
        data01_recursive_summary = {'epochs': int(len(diag_rows)), 'causal_warm_start': diag_warm_start, 'rmse_n_e_d_3d_m': [float(x) for x in diag_rmse], 'recursive_same_epoch_acceptance_gate': data01_acceptance_gate, 'first_error_gt_100m': _diag_first(diag_error_3d > 100.0), 'first_error_gt_1km': _diag_first(diag_error_3d > 1000.0), 'first_error_gt_100km': _diag_first(diag_error_3d > 100000.0), 'first_obs_feature_gt_10x_training_absmax': _diag_first(diag_obs_ratio > 10.0), 'max_prefde_innovation_abs_m': float(np.nanmax([row['prefde_innovation_max_abs_m'] for row in diag_rows])), 'max_gain_fro_norm': float(np.nanmax([row['gain_fro_norm'] for row in diag_rows])), 'max_position_correction_norm_m': float(np.nanmax([row['correction_position_norm_m'] for row in diag_rows])), 'fde_checked_detected_unresolved': [int(diag_fde_stats['epochs_checked']), int(diag_fde_stats['epochs_detected']), int(diag_fde_stats['unresolved_epochs'])] if FDE_DIA_ON else None, 'scope': 'frozen_weights_same_Data01_recursive_diagnostic_before_Data02'}
        diag_csv_path = OUTPUT_DIR / 'recursive_data01_diagnostics.csv'
        with diag_csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=diag_fields)
            writer.writeheader()
            for row in diag_rows:
                writer.writerow({name: row[name] for name in diag_fields})
        (OUTPUT_DIR / 'recursive_data01_summary.json').write_text(json.dumps(data01_recursive_summary, indent=2), encoding='utf-8')
        print('Data01 recursive performance:', json.dumps(data01_recursive_summary, indent=2))
        if not data01_acceptance_gate['passed']:
            print('WARNING: Data01 recursive check failed; Data02 evaluation will continue.')
    else:
        data01_acceptance_gate = {'passed': None, 'reason': 'recursive_Data01_diagnostic_disabled', 'criterion': 'diagnostic_not_run'}
        print('WARNING: Data01 recursive diagnostic disabled; Data02 evaluation will continue.')
    # --- DATA02 INDEPENDENT ONLINE TEST ---
    test_files, test_rover = resolve_dataset_files(test_dataset_dir, '01')
    test_required = [IMU_ERROR_MODEL_PATH, LEO_TLE_DIR]
    test_missing = [str(path) for path in test_required if not Path(path).exists()]
    if test_missing:
        raise FileNotFoundError('Missing shared test inputs:\n' + '\n'.join(test_missing))
    test_antenna_truth = load_ie_ground_truth(test_files.rover_ground_truth)
    test_imu_truth = load_ie_ground_truth(test_files.imu_ground_truth)
    if test_rover.imu_type not in imu_models:
        raise KeyError(f'No IMU noise model for test IMU type {test_rover.imu_type!r}')
    test_imu_noise = imu_models[test_rover.imu_type]
    test_rinex = RINEXObservationFile(test_files.rinex_obs)
    first_test_rinex_epoch = next(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    if first_test_rinex_epoch is None:
        raise ValueError('Test RINEX file contains no GPS/BDS epochs')
    _, test_imr_tow_all, test_imr_record_count = read_imr_tow_only(test_files.imr)
    if test_imr_record_count < 2:
        raise ValueError('Test IMR file contains fewer than two usable samples')
    if len(test_antenna_truth.time_gpst_s) < 2 or len(test_imu_truth.time_gpst_s) < 2:
        raise ValueError('Test ground-truth file contains fewer than two usable epochs')
    test_anchor_time = float(first_test_rinex_epoch.time_gpst_s)
    test_imr_time_all = anchor_imr_tow_to_gpst_seconds(test_imr_tow_all, test_anchor_time)
    test_antenna_truth_time = validate_strict_time_axis('test antenna truth', test_antenna_truth.time_gpst_s)
    test_imu_truth_time = validate_strict_time_axis('test IMU truth', test_imu_truth.time_gpst_s)
    test_common_start = max(float(test_imr_time_all[0]), float(test_antenna_truth_time[0]), float(test_imu_truth_time[0]))
    test_common_end = min(float(test_imr_time_all[-1]), float(test_antenna_truth_time[-1]), float(test_imu_truth_time[-1]))
    test_start = int(np.searchsorted(test_imr_time_all, test_common_start, side='left'))
    if test_start >= len(test_imr_time_all):
        raise ValueError('Test common time span starts after the IMR file')
    test_usable_imu_start = float(test_imr_time_all[test_start])
    test_gnss_epochs = tuple(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=test_usable_imu_start, end_time_gpst_s=test_common_end, max_epochs=MAX_TEST_FUSION_EPOCHS, require_measurements=True))
    if not test_gnss_epochs:
        raise ValueError('Test RINEX file contains no usable GPS/BDS pseudorange epochs')
    test_fusion_time = validate_strict_time_axis('test fusion', np.asarray([epoch.time_gpst_s for epoch in test_gnss_epochs], dtype=float))
    if len(test_fusion_time) < 1:
        raise ValueError('No synchronized test GNSS fusion epochs remain')
    test_common_stop = min(int(np.searchsorted(test_imr_time_all, test_common_end, side='left')) + 1, len(test_imr_time_all))
    test_requested_stop = min(int(np.searchsorted(test_imr_time_all, test_fusion_time[-1], side='left')) + 1, test_common_stop)
    test_imr = read_imr(test_files.imr, scaling_mode='cpp_exact', start_record=test_start, stop_record=test_requested_stop)
    test_imr_time = test_imr_time_all[test_start:test_requested_stop].copy()
    del test_imr_tow_all, test_imr_time_all
    if len(test_imr.tow_s) < 2:
        raise ValueError('Selected test IMR window contains fewer than two samples')
    test_query_time = np.concatenate(([test_imr_time[0]], test_fusion_time))
    test_antenna_position = interpolate_ground_truth(test_antenna_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    test_imu_position, test_imu_velocity, test_imu_heading, test_imu_pitch, test_imu_roll = interpolate_ground_truth(test_imu_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    test_fusion_antenna_truth_position = test_antenna_position[1:]
    test_C_b_e = body_to_ecef_from_ie_hpr(test_imu_position[0], test_imu_heading[0], test_imu_pitch[0], test_imu_roll[0], test_rover.mounting_xyz_deg)
    test_lever_arm_b_m = transform_lever_arm_vehicle_to_body(test_rover.lever_arm_vehicle_m, *test_rover.mounting_xyz_deg)
    test_initial_antenna_from_imu = test_imu_position[0] + test_C_b_e @ test_lever_arm_b_m
    test_truth_reference_error_m = np.linalg.norm(test_antenna_position[0] - test_initial_antenna_from_imu)
    if test_truth_reference_error_m > 0.05:
        raise ValueError(f'Test ROVE/IMU ground-truth reference points are inconsistent with the lever arm: {test_truth_reference_error_m:.3f} m')
    test_initial_nav = NavigationState(test_imu_position[0].copy(), test_imu_velocity[0].copy(), test_C_b_e)
    test_P0 = initial_covariance_from_imu_model(test_imu_noise)
    test_Qc = continuous_process_covariance_from_imu_model(test_imu_noise)
    test_ionosphere_coefficients = None
    if USE_IONOSPHERE:
        test_iono_header = read_rinex_navigation_header(test_files.nav)
        test_ionosphere_coefficients = {'G': (np.asarray(test_iono_header['GPSA']), np.asarray(test_iono_header['GPSB'])), 'C': (np.asarray(test_iono_header['BDSA']), np.asarray(test_iono_header['BDSB']))}
    test_gnss_preprocessor = GNSSPreprocessor(SP3Orbit(test_files.sp3), RINEXClock(test_files.clk), min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=test_ionosphere_coefficients)
    test_leo_klobuchar = None if test_ionosphere_coefficients is None else test_ionosphere_coefficients['G']
    test_leo_simulator = LEODownlinkSimulator(tle_provider, test_leo_klobuchar, seed=TEST_LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    print('\n=== DATA02 INDEPENDENT TEST ===')
    model = model.to(DEVICE)
    model.eval()
    nav = test_initial_nav.copy()
    P = test_P0.copy()
    online_rows = []
    zero_error_state_np = np.zeros(INS_STATE_DIM)
    previous_online = None
    test_recurrent_capacity = int(network_nmax)
    test_warm_start = None
    test_recurrent_state = None
    test_fde_stats = FDE_DIA.new_stats() if FDE_DIA_ON else {'enabled': False}
    last_gyro, last_accel = compensate_imu(test_imr.angular_rate_body_radps[0], test_imr.acceleration_body_mps2[0])
    last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
    test_timeline = build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True)
    for event in test_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(test_imr.angular_rate_body_radps[imu_index], test_imr.acceleration_body_mps2[imu_index])
            last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
            nav = mechanize_ecef(nav, last_gyro, last_accel, dt)
            F_error = build_error_state_dynamics(nav, last_accel)
            Phi, Qd = discretize_process_noise_van_loan(F_error, test_Qc, dt)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue
        fusion_index = event.fusion_index
        t = event.time_gpst_s
        epoch = test_gnss_epochs[fusion_index]
        if abs(epoch.time_gpst_s - t) > 1e-09:
            raise RuntimeError('test fusion timeline/epoch timestamp mismatch')
        gnss_measurements = test_gnss_preprocessor.prepare_epoch(epoch, nav, test_lever_arm_b_m)
        leo_measurements = test_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])
        measurements = retain_clock_observable_measurements(tuple(gnss_measurements) + tuple(leo_measurements))
        if not measurements:
            continue
        truth_position_now = test_fusion_antenna_truth_position[fusion_index]
        prior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        prior_error_3d_m = float(np.linalg.norm(prior_position - truth_position_now))
        if previous_online is None:
            warm_measurement_model = build_measurement_model(nav, measurements, test_lever_arm_b_m)
            warm_correction, P, _ = kalman_measurement_update(P, warm_measurement_model.innovation, warm_measurement_model.H, warm_measurement_model.R)
            nav = inject_error_state(nav, warm_correction)
            warm_posterior_residual = build_innovation_only(nav, measurements, test_lever_arm_b_m)
            previous_online = _make_online_context(warm_measurement_model.sat_ids, warm_posterior_residual, zero_error_state_np, warm_correction, last_feature_accel, last_feature_gyro, previous_context=None)
            test_warm_start = {'method': 'ordinary_TC_KF_context_only', 'time_gpst_s': float(t), 'excluded_from_learned_metrics': True, 'neural_state_advanced': False}
            continue
        if FDE_DIA_ON:
            fde_result = FDE_DIA.decision(nav, P, measurements, test_lever_arm_b_m, FDE_SIGNIFICANCE_ALPHA)
            FDE_DIA.accumulate_stats(test_fde_stats, fde_result)
            prefde_measurement_model = fde_result.tested_measurement_model
            measurements = fde_result.measurements
            measurement_model = fde_result.measurement_model
        else:
            fde_result = None
            measurement_model = build_measurement_model(nav, measurements, test_lever_arm_b_m)
            prefde_measurement_model = measurement_model
        if FDE_DIA_ON and (not measurements):
            if previous_online is not None:
                posterior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
                hpl_m, vpl_m = INTEGRITY.protection_levels(P, posterior_position) if INTEGRITY_ON else (float('nan'), float('nan'))
                online_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': test_fusion_antenna_truth_position[fusion_index].copy(), 'hpl_m': hpl_m, 'vpl_m': vpl_m, 'n_input': len(fde_result.tested_measurements), 'n_total': 0, 'n_gnss': 0, 'n_leo': 0, 'sat_ids': (), 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': prior_error_3d_m, 'prefde_innovation_rms_m': float(np.sqrt(np.mean(prefde_measurement_model.innovation ** 2))) if len(prefde_measurement_model.innovation) else 0.0, 'prefde_innovation_max_abs_m': float(np.max(np.abs(prefde_measurement_model.innovation))) if len(prefde_measurement_model.innovation) else 0.0, 'knet_innovation_rms_m': float('nan'), 'knet_innovation_max_abs_m': float('nan'), 'gain_fro_norm': float('nan'), 'gain_max_abs': float('nan'), 'gain_navigation_rows_fro_norm': float('nan'), 'correction_position_norm_m': 0.0, 'correction_velocity_norm_mps': 0.0, 'correction_attitude_norm_rad': 0.0, 'fixed_ood_fraction': float('nan'), 'fixed_max_abs_train_ratio': float('nan'), 'obs_ood_fraction': float('nan'), 'obs_max_abs_train_ratio': float('nan'), 'fde_detected': bool(fde_result.detected), 'fde_identified_sat_ids': tuple(fde_result.identified_sat_ids), 'fde_excluded_sat_ids': tuple(fde_result.excluded_sat_ids), 'fde_ambiguous_sat_ids': tuple(fde_result.ambiguous_sat_ids), 'fde_statistic': float(fde_result.statistic), 'fde_threshold': float(fde_result.threshold), 'fde_dof': int(fde_result.dof), 'fde_post_exclusion_statistic': float(fde_result.post_exclusion_statistic), 'fde_post_exclusion_threshold': float(fde_result.post_exclusion_threshold), 'fde_post_exclusion_dof': int(fde_result.post_exclusion_dof), 'fde_post_exclusion_consistent': bool(fde_result.post_exclusion_consistent), 'fde_local_identification_score': float(fde_result.local_identification_score), 'fde_estimated_fault_m': float(fde_result.estimated_fault_m), 'fde_hard_exclusion_applied': bool(fde_result.excluded_sat_ids), 'fde_state_correction_norm': float(np.linalg.norm(fde_result.dia_state_correction)), 'fde_unresolved': bool(fde_result.unresolved), 'fusion_mode': 'INS_only_after_empty_FDE'})
                previous_online = _make_online_context((), np.empty(0), zero_error_state_np, zero_error_state_np, last_feature_accel, last_feature_gyro, previous_context=previous_online)
            continue
        n = len(measurements)
        current_sat_ids = measurement_model.sat_ids
        innovation_now = measurement_model.innovation
        fixed_k, obs_k, current_mask, channel_k, innovation_k, fixed_k_raw, obs_k_raw = _online_feature_arrays(previous_online, current_sat_ids, innovation_now, last_feature_accel, last_feature_gyro)
        feature_shift = _feature_shift_diagnostics(fixed_k_raw, obs_k_raw, channel_k)
        prefde_innovation = np.asarray(prefde_measurement_model.innovation, dtype=float)
        prefde_innovation_rms_m = float(np.sqrt(np.mean(prefde_innovation ** 2))) if prefde_innovation.size else 0.0
        prefde_innovation_max_abs_m = float(np.max(np.abs(prefde_innovation))) if prefde_innovation.size else 0.0
        knet_innovation_rms_m = float(np.sqrt(np.mean(innovation_k ** 2))) if innovation_k.size else 0.0
        knet_innovation_max_abs_m = float(np.max(np.abs(innovation_k))) if innovation_k.size else 0.0
        if n > test_recurrent_capacity:
            raise RuntimeError(f'Data02 measurement selection exceeded the Data01-defined Yan Eq.(16) network Nmax: count={n}, network_Nmax={test_recurrent_capacity}.')
        fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _pad_neural_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, test_recurrent_capacity)
        with torch.inference_mode():
            output = model(torch.tensor(fixed_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(obs_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(mask_nn[None], dtype=torch.bool, device=DEVICE), torch.tensor(channel_nn[None], dtype=torch.bool, device=DEVICE), recurrent_state=test_recurrent_state)
            test_recurrent_state = output.recurrent_state
            innovation_tensor = torch.tensor(innovation_nn[None], dtype=torch.float32, device=DEVICE)
            correction = fig8_state_update(output, innovation_tensor)[0]
        gain = output.kalman_gain[0].cpu().numpy().astype(float)
        correction = correction.cpu().numpy().astype(float)
        learned_error_state_pred = zero_error_state_np
        learned_error_state_post = correction.copy()
        active_gain = gain[:, :n]
        gain_fro_norm = float(np.linalg.norm(active_gain))
        gain_max_abs = float(np.max(np.abs(active_gain))) if active_gain.size else 0.0
        gain_navigation_rows_fro_norm = float(np.linalg.norm(active_gain[:SUPERVISED_STATE_DIM, :]))
        correction_position_norm_m = float(np.linalg.norm(correction[0:3]))
        correction_velocity_norm_mps = float(np.linalg.norm(correction[3:6]))
        correction_attitude_norm_rad = float(np.linalg.norm(correction[6:9]))
        if not np.all(np.isfinite(active_gain)) or not np.all(np.isfinite(correction)):
            raise FloatingPointError('non-finite learned gain/correction during online fusion')
        P = learned_gain_covariance_update(P, active_gain, measurement_model.H, measurement_model.R, correction)
        nav = inject_error_state(nav, correction)
        posterior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        posterior_error_3d_m = float(np.linalg.norm(posterior_position - truth_position_now))
        hpl_m, vpl_m = INTEGRITY.protection_levels(P, posterior_position) if INTEGRITY_ON else (float('nan'), float('nan'))
        posterior_residual = build_innovation_only(nav, measurements, test_lever_arm_b_m)
        previous_online = _make_online_context(current_sat_ids, posterior_residual, learned_error_state_pred, learned_error_state_post, last_feature_accel, last_feature_gyro, previous_context=previous_online)
        counts = Counter((m.constellation for m in measurements))
        online_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': test_fusion_antenna_truth_position[fusion_index].copy(), 'hpl_m': hpl_m, 'vpl_m': vpl_m, 'n_input': len(fde_result.tested_measurements) if FDE_DIA_ON else len(measurements), 'n_total': n, 'n_gnss': counts['G'] + counts['C'], 'n_leo': counts['L'], 'sat_ids': current_sat_ids, 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': posterior_error_3d_m, 'prefde_innovation_rms_m': prefde_innovation_rms_m, 'prefde_innovation_max_abs_m': prefde_innovation_max_abs_m, 'knet_innovation_rms_m': knet_innovation_rms_m, 'knet_innovation_max_abs_m': knet_innovation_max_abs_m, 'gain_fro_norm': gain_fro_norm, 'gain_max_abs': gain_max_abs, 'gain_navigation_rows_fro_norm': gain_navigation_rows_fro_norm, 'correction_position_norm_m': correction_position_norm_m, 'correction_velocity_norm_mps': correction_velocity_norm_mps, 'correction_attitude_norm_rad': correction_attitude_norm_rad, **feature_shift, **(FDE_DIA.export_fields(fde_result) if FDE_DIA_ON else NO_FDE_EXPORT_FIELDS), 'fusion_mode': 'MaskedCLA_post_FDE_DIA' if FDE_DIA_ON else 'MaskedCLA_no_FDE_DIA'})
    if not online_rows:
        raise ValueError('Online Masked KalmanNet pass produced no usable test epochs; cannot compute test navigation metrics')
    # --- DATA02 EVALUATION, INTEGRITY DIAGNOSTICS, EXPORT ---
    online_time = np.asarray([row['time'] for row in online_rows])
    estimate = np.stack([row['position'] for row in online_rows])
    truth_aligned = np.stack([row['truth'] for row in online_rows])
    ned_error = np.empty_like(estimate)
    for i, (est, truth_i) in enumerate(zip(estimate, truth_aligned)):
        lat, lon, _ = ecef_to_llh(truth_i)
        ned_error[i] = c_ecef_to_ned(lat, lon) @ (est - truth_i)
    error_3d = np.linalg.norm(ned_error, axis=1)
    horizontal_error = np.linalg.norm(ned_error[:, :2], axis=1)
    vertical_error = np.abs(ned_error[:, 2])
    hpl = np.asarray([row['hpl_m'] for row in online_rows], dtype=float)
    vpl = np.asarray([row['vpl_m'] for row in online_rows], dtype=float)
    rmse_ned = np.sqrt(np.mean(ned_error ** 2, axis=0))
    rmse_3d = float(np.sqrt(np.mean(error_3d ** 2)))
    rmse = np.append(rmse_ned, rmse_3d)
    cdf_probability = np.arange(1, len(error_3d) + 1) / len(error_3d)
    diagnostic_fields = ('prior_error_3d_m', 'posterior_error_3d_m', 'prefde_innovation_rms_m', 'prefde_innovation_max_abs_m', 'knet_innovation_rms_m', 'knet_innovation_max_abs_m', 'gain_fro_norm', 'gain_max_abs', 'gain_navigation_rows_fro_norm', 'correction_position_norm_m', 'correction_velocity_norm_mps', 'correction_attitude_norm_rad', 'fixed_ood_fraction', 'fixed_max_abs_train_ratio', 'obs_ood_fraction', 'obs_max_abs_train_ratio')
    for row_index, row in enumerate(online_rows):
        missing = [name for name in diagnostic_fields if name not in row]
        if missing:
            raise RuntimeError(f'online diagnostic row schema mismatch at row {row_index}: missing {missing}')
    diagnostic_arrays = {name: np.asarray([row[name] for row in online_rows], dtype=float) for name in diagnostic_fields}
    def _first_true_event(mask):
        index = np.flatnonzero(np.asarray(mask, dtype=bool))
        if index.size == 0:
            return None
        i = int(index[0])
        return {'row_index': i, 'time_gpst_s': float(online_time[i]), 'posterior_error_3d_m': float(error_3d[i])}
    growth_ratio = np.full(len(error_3d), np.nan, dtype=float)
    if len(error_3d) > 1:
        growth_ratio[1:] = error_3d[1:] / np.maximum(error_3d[:-1], 1e-09)
    divergence_diagnostics = {'first_error_gt_100m': _first_true_event(error_3d > 100.0), 'first_error_gt_1km': _first_true_event(error_3d > 1000.0), 'first_error_gt_100km': _first_true_event(error_3d > 100000.0), 'first_error_growth_gt_10x': _first_true_event(growth_ratio > 10.0), 'first_obs_feature_gt_10x_training_absmax': _first_true_event(diagnostic_arrays['obs_max_abs_train_ratio'] > 10.0), 'max_prefde_innovation_abs_m': float(np.nanmax(diagnostic_arrays['prefde_innovation_max_abs_m'])), 'max_knet_innovation_abs_m': float(np.nanmax(diagnostic_arrays['knet_innovation_max_abs_m'])), 'max_gain_fro_norm': float(np.nanmax(diagnostic_arrays['gain_fro_norm'])), 'max_position_correction_norm_m': float(np.nanmax(diagnostic_arrays['correction_position_norm_m'])), 'max_fixed_feature_abs_training_ratio': float(np.nanmax(diagnostic_arrays['fixed_max_abs_train_ratio'])), 'max_observation_feature_abs_training_ratio': float(np.nanmax(diagnostic_arrays['obs_max_abs_train_ratio']))}
    np.savez(OUTPUT_DIR / 'test_evaluation.npz', dataset_dir=np.asarray(str(test_dataset_dir)), time_gpst_s=online_time, estimate_ecef_m=estimate, truth_ecef_m=truth_aligned, ned_error_m=ned_error, error_3d_m=error_3d, horizontal_error_m=horizontal_error, vertical_error_m=vertical_error, hpl_m=hpl, vpl_m=vpl, **diagnostic_arrays, error_growth_ratio=growth_ratio, rmse_north_east_down_3d_m=rmse, cdf_probability=cdf_probability, north_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 0])), east_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 1])), down_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 2])), three_d_cdf_error_m=np.sort(error_3d))
    with (OUTPUT_DIR / 'test_trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_gpst_s', 'n_input_before_fde', 'n_k_used_after_fde', 'n_gnss', 'n_leo', 'sat_ids_used', 'fde_detected', 'fde_identified_sat_ids', 'fde_excluded_sat_ids', 'fde_ambiguous_sat_ids', 'fde_statistic', 'fde_threshold', 'fde_dof', 'fde_post_exclusion_statistic', 'fde_post_exclusion_threshold', 'fde_post_exclusion_dof', 'fde_post_exclusion_consistent', 'fde_local_identification_score', 'fde_estimated_fault_m', 'fde_hard_exclusion_applied', 'fde_state_correction_norm', 'fde_unresolved', 'fusion_mode', 'hpl_m', 'vpl_m', 'x_ecef_m', 'y_ecef_m', 'z_ecef_m'])
        for row in online_rows:
            writer.writerow([row['time'], row['n_input'], row['n_total'], row['n_gnss'], row['n_leo'], ';'.join(row['sat_ids']), int(row['fde_detected']), ';'.join(row['fde_identified_sat_ids']), ';'.join(row['fde_excluded_sat_ids']), ';'.join(row['fde_ambiguous_sat_ids']), row['fde_statistic'], row['fde_threshold'], row['fde_dof'], row['fde_post_exclusion_statistic'], row['fde_post_exclusion_threshold'], row['fde_post_exclusion_dof'], int(row['fde_post_exclusion_consistent']), row['fde_local_identification_score'], row['fde_estimated_fault_m'], int(row['fde_hard_exclusion_applied']), row['fde_state_correction_norm'], int(row['fde_unresolved']), row['fusion_mode'], row['hpl_m'], row['vpl_m'], *row['position'].tolist()])
    with (OUTPUT_DIR / 'test_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_gpst_s', *diagnostic_fields, 'fde_detected', 'fde_excluded_sat_ids'])
        for row in online_rows:
            writer.writerow([row['time'], *[row[name] for name in diagnostic_fields], int(row['fde_detected']), ';'.join(row['fde_excluded_sat_ids'])])
    proposed_van_loan_stats = dict(_VAN_LOAN_STATS)
    classical_baseline_rmse = None
    if RUN_CLASSICAL_TEST_BASELINE:
        baseline_leo_simulator = LEODownlinkSimulator(tle_provider, test_leo_klobuchar, seed=TEST_LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
        baseline_nav = test_initial_nav.copy()
        baseline_P = test_P0.copy()
        baseline_rows = []
        baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr.angular_rate_body_radps[0], test_imr.acceleration_body_mps2[0])
        for baseline_event in build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
            if isinstance(baseline_event, PropagationSegment):
                imu_index = baseline_event.imu_index
                dt = baseline_event.end_time_gpst_s - baseline_event.start_time_gpst_s
                baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr.angular_rate_body_radps[imu_index], test_imr.acceleration_body_mps2[imu_index])
                baseline_nav = mechanize_ecef(baseline_nav, baseline_last_gyro, baseline_last_accel, dt)
                F_baseline = build_error_state_dynamics(baseline_nav, baseline_last_accel)
                Phi_baseline, Qd_baseline = discretize_process_noise_van_loan(F_baseline, test_Qc, dt)
                baseline_P = Phi_baseline @ baseline_P @ Phi_baseline.T + Qd_baseline
                baseline_P = 0.5 * (baseline_P + baseline_P.T)
                continue
            fusion_index = baseline_event.fusion_index
            t = baseline_event.time_gpst_s
            epoch = test_gnss_epochs[fusion_index]
            baseline_gnss = test_gnss_preprocessor.prepare_epoch(epoch, baseline_nav, test_lever_arm_b_m)
            baseline_leo = baseline_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])
            baseline_measurements = retain_clock_observable_measurements(tuple(baseline_gnss) + tuple(baseline_leo))
            if baseline_measurements:
                if FDE_DIA_ON:
                    baseline_fde = FDE_DIA.decision(baseline_nav, baseline_P, baseline_measurements, test_lever_arm_b_m, FDE_SIGNIFICANCE_ALPHA)
                    baseline_measurements = baseline_fde.measurements
                    baseline_model = baseline_fde.measurement_model
                else:
                    baseline_model = build_measurement_model(baseline_nav, baseline_measurements, test_lever_arm_b_m)
                if baseline_measurements:
                    baseline_dx, baseline_P, _ = kalman_measurement_update(baseline_P, baseline_model.innovation, baseline_model.H, baseline_model.R)
                    baseline_nav = inject_error_state(baseline_nav, baseline_dx)
            baseline_position = gnss_antenna_position(baseline_nav, test_lever_arm_b_m)
            baseline_rows.append((t, baseline_position.copy(), test_fusion_antenna_truth_position[fusion_index].copy()))
        if baseline_rows:
            baseline_time = np.asarray([row[0] for row in baseline_rows], dtype=float)
            baseline_estimate = np.stack([row[1] for row in baseline_rows])
            baseline_truth = np.stack([row[2] for row in baseline_rows])
            baseline_ned_error = np.empty_like(baseline_estimate)
            for i, (est, truth_i) in enumerate(zip(baseline_estimate, baseline_truth)):
                lat, lon, _ = ecef_to_llh(truth_i)
                baseline_ned_error[i] = c_ecef_to_ned(lat, lon) @ (est - truth_i)
            baseline_error_3d = np.linalg.norm(baseline_ned_error, axis=1)
            baseline_rmse_ned = np.sqrt(np.mean(baseline_ned_error ** 2, axis=0))
            baseline_rmse_3d = float(np.sqrt(np.mean(baseline_error_3d ** 2)))
            classical_baseline_rmse = np.append(baseline_rmse_ned, baseline_rmse_3d)
            np.savez(OUTPUT_DIR / 'classical_test_baseline.npz', time_gpst_s=baseline_time, estimate_ecef_m=baseline_estimate, truth_ecef_m=baseline_truth, ned_error_m=baseline_ned_error, error_3d_m=baseline_error_3d, rmse_north_east_down_3d_m=classical_baseline_rmse)
    with (OUTPUT_DIR / 'stanford_data.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_gpst_s', 'horizontal_error_m', 'hpl_m', 'vertical_error_m', 'vpl_m'])
        writer.writerows(zip(online_time, horizontal_error, hpl, vertical_error, vpl))
    summary = {'paper_exact': False, 'intentional_project_differences': {'measurement_mode': 'pseudorange_only', 'leo_orbit': 'TLE/SGP4_instead_of_STK_HPOP', 'leo_clock': 'ideal_zero_unpublished_simulated_clock_generator', 'navigation_error_state': '9_state_[delta_p,delta_v,delta_theta]_instead_of_Yan_15_state', 'imu_bias_states': 'removed_by_user_request', 'Fig8_eta': 'removed_by_user_request'}, 'paper_components': {'leo_Eq1_to_Eq5': {'deterministic_iono_tropo_and_light_time': True, 'stochastic_variance_policy': 'unchanged_from_v6_by_user_request', 'variance_terms_used': ['ionosphere', 'troposphere', 'MP_NLOS'], 'MP_NLOS': 'Ref35_Eq18_elevation_only_same_as_v6', 'sampling': 'independent_zero_mean_Gaussian_same_as_v6', 'not_added_to_LEO_variance': ['URA', 'receiver_noise'], 'paper_difference': 'Yan_Eq4_CN0_dependent_MP_NLOS_not_used_in_LEO_branch'}, 'INS_Eq6_to_Eq9': 'user_requested_9_state_ECEF_error_model_[delta_p,delta_v,delta_theta];IMU_bias_states_and_eta_removed;lever_arm_retained', 'Eq7_Fig8_state_update': {'state_order': '[delta_p,delta_v,delta_theta]', 'classical_TC_gain_rows': INS_STATE_DIM, 'masked_CLA_gain_rows': INS_STATE_DIM, 'direct_truth_label_rows': SUPERVISED_STATE_DIM, 'paper_status': 'explicit_user_requested_reduction_from_Yan_15_state_to_9_state'}, 'MaskedCLA_Eq10_to_Eq29': 'implemented_with_v29_mask_semantics_CNN_position_axis_preserved_Eq23_mask_gates_each_LSTM_position_and_current_epoch_attention_cross_epoch_hc_only', 'training_Eq30_to_Eq32': 'Yan_state_MSE_plus_actual_Fig7_Fig8_closed_loop_forward_plus_Latent_KalmanNet_alternating', 'Fig8_eta': {'enabled': False, 'paper_status': 'Yan_Fig8_includes_eta_but_user_requested_removal_in_this_branch'}, 'FDE_Eq33_to_Eq34': {'enabled': bool(FDE_DIA_ON), 'single_code_switch': 'FDE_DIA_ON = True/False', 'raw_innovation_before_KG': bool(FDE_DIA_ON), 'detection_identification': 'Ref33_DIA', 'hard_exclusion_before_KNet': True, 'Eq34_DIA_state_covariance': 'computed_for_identified_fault_mode', 'Eq34_applied_to_reported_navigation_state': False, 'old_nonpaper_post_exclusion_rejection_loop': False, 'alpha': FDE_SIGNIFICANCE_ALPHA}, 'integrity': {'enabled': bool(INTEGRITY_ON), 'single_code_switch': 'INTEGRITY_ON = True/False', 'module': 'fde_dia_v2.py', 'HPL_VPL': 'computed_from_covariance_paired_with_reported_navigation_state_in_NED', 'Eq34_DIA_covariance_used_for_PL': False, 'horizontal_major_axis_evaluation': 'Ref35_equation_overflow_safe_hypot_algebra', 'nonfinite_covariance_policy': 'report_infinite_PL_instead_of_runtime_overflow', 'stanford_data_csv': True, 'alert_limits': 'not_invented_paper_does_not_publish_numeric_HAL_VAL'}, 'stability': {'gradient_clipping': {'enabled': True, 'norm_type': 2.0, 'max_norm': float(GRADIENT_CLIP_NORM), 'implementation': 'overflow_safe_float64_global_norm_scaling', 'source': 'SongJgit_KalmanNet4SensorFusion', 'paper_status': 'Yan_does_not_report_gradient_clipping'}, 'tbptt': {'V1_detach_step': int(V1_TBPTT_DETACH_STEP), 'forward_state_continuity': True, 'gradient_history_truncated': True, 'source': 'SongJgit_KalmanNet4SensorFusion_detach_step_pattern', 'paper_status': 'Yan_does_not_publish_TBPTT_schedule'}, 'learning_rate': {'active': float(LEARNING_RATE), 'Yan_reported': float(YAN_REPORTED_LEARNING_RATE), 'default_source': 'LatentKalmanNet_and_KalmanNet4SensorFusion_public_configs'}, 'checkpoint_selection': {'untrained_model_eligible': False, 'zero_gain_baseline_eligible': False, 'criterion': 'best_finite_Data01_recursive_Eq30_after_optimizer_update', 'source': 'KalmanNet_and_LatentKalmanNet_validation_checkpoint_pattern'}, 'Fig10_stability_diagnostic': 'omitted_by_user_request', 'recursive_Data01_diagnostic_before_Data02': 'same_epoch_learned_vs_classical_3D_RMSE_nonblocking', 'exact_75_25_stress_generator': 'not_reconstructed_distribution_parameters_unpublished'}}, 'unavoidable_unpublished_completions': ['offline_Eq10_to_Eq14_history_generator_conventional_TC_reference_pass', 'causal_boundary_timing_for_state_innovation_and_state_residual', 'exact_CNN_tensorization_pooling_and_FC_dimensions', 'online_groupwise_L2_feature_normalization_adapted_from_KalmanNet_family_Yan_does_not_publish_input_scaling', 'full_INS_navigation_BPTT_is_KalmanNet_guided_completion_not_published_by_Yan', 'KalmanNet_V2_short_trajectory_training_with_ground_truth_chunk_initial_state', 'KalmanNet_paper_recommended_V2_warmup_then_V1_full_trajectory_tuning', 'V2_and_V1_epoch_counts_are_explicit_overridable_completion_not_published_by_KalmanNet_or_Yan', 'V1_continues_same_network_and_Adam_states_after_V2_no_reinitialization', 'V1_full_sequence_initial_state_is_ground_truth_navigation_state_not_classical_KF_posterior', 'V1_TBPTT_detach_step_2_from_KalmanNet4SensorFusion_stability_pattern_not_published_by_Yan', 'stability_default_lr_1e-3_from_public_LatentKalmanNet_and_KalmanNet4SensorFusion_configs_Yan_reports_1e-2', 'V2_chunk_boundary_Yan_lagged_feature_context_is_project_adaptation_no_classical_KF_state_reset', 'PyTorch_default_nonzero_KG_output_initialization_Yan_does_not_publish_FC_initialization', 'global_L2_gradient_clipping_max_norm_1_from_KalmanNet4SensorFusion_Yan_unpublished', 'checkpoint_selection_excludes_untrained_and_zero_gain_baseline_following_KalmanNet_family_pattern', 'V2_random_partition_all_Q_batches_reused_theta_then_psi_per_Latent_KalmanNet_Algorithm2', 'exact_Fig8_pooling_operator_and_FC_tensorization_unpublished', 'LEO_variance_intentionally_kept_as_v6_instead_of_Yan_Eq4', 'random_LEO_masking_angle_distribution_from_Ref43', 'FDE_false_alarm_probability_and_exact_fault_mode_matrices', 'learned_gain_covariance_bookkeeping'], 'dataset_protocol': {'training_dataset': str(TRAIN_DATASET_DIR), 'test_dataset': str(test_dataset_dir), 'test_used_for_training': False, 'training_fusion_epoch_cap': None if MAX_FUSION_EPOCHS is None else int(MAX_FUSION_EPOCHS), 'test_fusion_epoch_cap': None if MAX_TEST_FUSION_EPOCHS is None else int(MAX_TEST_FUSION_EPOCHS), 'shortened_diagnostic_run': bool(MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None)}, 'training': {'samples': int(training_sample_count), 'Nmax': int(network_nmax), 'training_observed_Nmax': int(nmax), 'Data01_sequence_Nmax': int(data01_sequence_nmax), 'Data02_sequence_Nmax': int(data02_sequence_nmax), 'Nmax_source': 'automatic_max_observed_sequence_length', 'v2_warmup_epochs': int(KNET_V2_WARMUP_EPOCHS), 'v1_finetune_epochs': int(KNET_V1_FINE_TUNE_EPOCHS), 'total_alternating_epochs': int(epochs_ran), 'recursive_Data01_acceptance_gate': data01_acceptance_gate, 'gamma_l2_completion': float(GAMMA_L2), 'feature_normalization': {'mode': FEATURE_NORMALIZATION_MODE, 'fit_scope': 'none_online_per_epoch', 'groupwise': ['Delta_alpha', 'Delta_omega', 'Delta_xhat', 'Delta_xtilde', 'Delta_y_previous', 'Delta_ytilde_current'], 'mask_aware_observations': True, 'targets_normalized': False, 'innovation_for_state_update_normalized': False, 'source': 'KalmanNet_KalmanNet4SensorFusion_LatentKalmanNet_family_pattern', 'paper_status': 'Yan_does_not_publish_input_scaling'}, 'lstm_axis_semantics': {'paper_index_k': 'fusion_epoch_temporal_axis', 'paper_index_t': 'padded_feature_position_axis', 'recurrent_state_shape': '[num_layers,B,D,hidden_size]', 'mask_rule': 'Yan_Eq25_retain_previous_position_state_when_M_k_t_is_zero'}, 'alternating_partition': {'psi_representation': 'masked_conv_feature_extractor', 'theta_filter': 'masked_lstm_masked_attention_gain_head', 'source': 'Latent_KalmanNet_Algorithm2_encoder_vs_recurrent_filter_decomposition', 'batch_schedule': 'random_partition_D_into_Q_batches_once_per_V2_epoch;theta_all_Q_then_psi_same_all_Q', 'full_dataset_coverage_per_phase': True, 'paper_status': 'batch_schedule_matches_Latent_KalmanNet_Algorithm2;Yan_requires_alternating_but_does_not_publish_exact_module_partition', 'frozen_counterpart_mode': 'eval_to_disable_dropout_stochasticity', 'warm_start': 'not_applied_because_Yan_has_no_supervised_target_for_intermediate_CNN_features'}, 'sequence_training': {'scope': 'Data01_only', 'method': 'Yan_Fig7_Fig8_actual_closed_loop_forward', 'raw_measurements_fixed': True, 'estimator_dependent_features_recomputed': True, 'navigation_propagation': '9_state_no_bias_no_eta_ECEF_INS_mechanization', 'linearized_Phi_H_training_surrogate': False, 'cross_fusion_navigation_gradient': 'KalmanNet_guided_BPTT_through_9_state_correction_ECEF_INS_and_innovation', 'cross_fusion_neural_gradient': 'LSTM_h_c_BPTT_within_each_V2_short_trajectory_and_within_optional_V1_full_sequence', 'training_sequence_length': int(training_sample_count), 'training_sequence_count': int(trajectory_count), 'sequence_batch_size': int(TRAJECTORY_BATCH_SIZE), 'sequence_order': 'chronological_within_each_short_trajectory;each_V2_epoch_randomly_partitions_all_trajectories_once;same_ordered_batches_reused_for_theta_then_psi', 'sequence_boundary_context': 'stored_causal_predecessor_used_once_at_sequence_start', 'internal_classical_KF_reset': False, 'optimizer_step_inside_sequence': False, 'optimizer_step_after_sequence': 'one_step_per_Algorithm2_batch_after_all_trajectories_in_that_batch', 'external_filter_recurrence': 'learned_posterior_physically_propagated_to_next_prior_and_features_across_complete_Data01_training_sequence', 'lagged_state_feature_timing': 'completed_previous_fusion_context_as_in_online_Data02', 'observation_residual_feature_timing': 'Yan_Eq11_Delta_y_k_minus_1;posterior_Delta_y_k_is_stored_after_epoch_k_and_used_at_k_plus_1;same_epoch_use_rejected_as_circular', 'masked_CLA_recurrent_state': 'Yan_Eq25_per_position_h_c_carried_across_fusion_epochs_k;attention_is_current_epoch_position_fusion', 'input_tensorization': 'Yan_Eq15_structure_adapted_to_9_state;fixed24_once;only_residual_and_innovation_blocks_zero_padded_to_fixed_Nmax;no_satellite_token_broadcast', 'synthetic_prior_augmentation': False, 'Data02_used': False}, 'final_actual_closed_loop_Eq30': float(final_training_metrics['eq30']), 'final_actual_closed_loop_total_objective': float(final_training_metrics['objective']), 'final_actual_closed_loop_position_rmse_m': float(final_training_metrics['position_rmse_m']), 'final_clean_teacher_forced_Eq30': float(final_teacher_forced_metrics['eq30']), 'final_clean_teacher_forced_position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m']), 'final_selected_training_stage': selected_training_stage, 'final_selected_training_eq30': float(best_val_state_loss), 'final_selected_training_position_rmse_m': float(best_val_position_rmse_m)}, 'test': {'epochs': int(len(online_rows)), 'causal_warm_start': test_warm_start, 'neural_recurrent_state': 'carried_across_online_fusion_epochs_after_causal_warm_start', 'rmse_ned3d_m': [float(v) for v in rmse], 'imu_bias_states_enabled': False, 'eta_enabled': False, 'divergence_diagnostics': divergence_diagnostics, 'classical_baseline_rmse_ned3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None, 'fde_stats': test_fde_stats}, 'runtime_checks': {'van_loan_fast_calls': int(proposed_van_loan_stats['taylor_calls']), 'van_loan_exact_calls': int(proposed_van_loan_stats['exact_fallback_calls']), 'van_loan_max_validation_phi_abs': float(proposed_van_loan_stats['max_validation_phi_abs']), 'van_loan_max_validation_qd_abs': float(proposed_van_loan_stats['max_validation_qd_abs'])}}
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    test_measurement_counts = np.asarray([row['n_total'] for row in online_rows], dtype=float)
    test_gnss_counts = np.asarray([row['n_gnss'] for row in online_rows], dtype=float)
    test_leo_counts = np.asarray([row['n_leo'] for row in online_rows], dtype=float)
    test_performance_summary = {'epochs': int(len(online_rows)), 'rmse_n_e_d_3d_m': [float(v) for v in rmse], 'error_3d_m': {'median': float(np.median(error_3d)), 'p95': float(np.percentile(error_3d, 95.0)), 'p99': float(np.percentile(error_3d, 99.0)), 'max': float(np.max(error_3d))}, 'measurements_used': {'min': int(np.min(test_measurement_counts)), 'median': float(np.median(test_measurement_counts)), 'max': int(np.max(test_measurement_counts)), 'gnss_median': float(np.median(test_gnss_counts)), 'leo_median': float(np.median(test_leo_counts))}, 'divergence': divergence_diagnostics, 'fde': test_fde_stats if FDE_DIA_ON else {'enabled': False}, 'integrity_median_hpl_vpl_m': [float(np.nanmedian(hpl)), float(np.nanmedian(vpl))] if INTEGRITY_ON else None, 'classical_baseline_rmse_n_e_d_3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None}
    print('\n=== DATA02 INDEPENDENT PERFORMANCE ===')
    print(json.dumps(test_performance_summary, indent=2))
