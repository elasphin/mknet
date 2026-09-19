'GNSS/LEO/INS + Masked KalmanNet simulation.\n\nUses Yan et al. where details are published. Project deviations are\npseudorange-only, TLE/SGP4 LEO, and a 9-state no-bias/no-eta filter.\nData01 trains the network; Data02 is an independent, non-blocking evaluation.'
from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple
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
import matplotlib.pyplot as plt
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
GPS_UTC_LEAP_SECONDS = 18.0  # fixed for the 2022/2023 project epochs
IN_KAGGLE = Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
KAGGLE_PROJECT_ROOT = Path(os.environ.get('MKNET_PROJECT_ROOT', '/kaggle/input/datasets/dlrmrsj/mknet-project-d'))
DATASET_ROOT = KAGGLE_PROJECT_ROOT
TRAIN_DATASET_DIR = DATASET_ROOT / 'Data01_20230102_ISA-100C_Vehicle_Complex'
TEST_DATASET_DIR: Path | None = DATASET_ROOT / 'Data02_20220309_ISA-100C_Vehicle_Complex'
README_XML_PATH = TRAIN_DATASET_DIR / "README.xml"
IMU_ERROR_MODEL_PATH = KAGGLE_PROJECT_ROOT / "IMUErrorModel.txt"
ROVE_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ROVE_GroundTruth.txt"
IMU_GROUND_TRUTH_PATH = TRAIN_DATASET_DIR / "ISA-100C_GroundTruth.txt"
RINEX_OBS_PATH = TRAIN_DATASET_DIR / "ROVE.23O"
IMR_PATH = TRAIN_DATASET_DIR / "ISA-100C.imr"
SP3_PATH = TRAIN_DATASET_DIR / "WUM0MGXFIN_20230020000_01D_05M_ORB.SP3"
CLK_PATH = TRAIN_DATASET_DIR / "WUM0MGXFIN_20230020000_01D_30S_CLK.CLK"
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
        return float(base + GPS_UTC_LEAP_SECONDS)
    raise ValueError(f'Unsupported time system: {time_system}')
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
                yield ('imu', segment_start, t, imu_index)
            yield ('fusion', t, t, fusion_index)
            segment_start = t
            fusion_index += 1
        if interval_end > segment_start + 1e-12:
            yield ('imu', segment_start, interval_end, imu_index)
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
# --- DATASET READERS ---
# --- RINEX observation: one pseudorange signal per GPS/BDS satellite ---
_FREQUENCY_HZ = {('G', '1'): 1575420000.0, ('G', '2'): 1227600000.0, ('G', '5'): 1176450000.0, ('C', '1'): 1575420000.0, ('C', '2'): 1561098000.0, ('C', '5'): 1176450000.0, ('C', '7'): 1207140000.0, ('C', '8'): 1191795000.0, ('C', '6'): 1268520000.0}
_SIGNAL_PREFS = {'G': ('1C', '1W', '1P', '2W', '2L', '2X', '5Q', '5X', '5I'), 'C': ('2I', '1I', '2X', '1X', '1P', '1D', '5X', '5P', '5D', '7I', '7X', '6I', '6X')}
class SatelliteMeasurement(NamedTuple):
    sat_id: str
    constellation: str
    signal_suffix: str
    frequency_hz: float
    pseudorange_m: float
    cn0_dbhz: float | None = None
class ObservationEpoch(NamedTuple):
    time_gpst_s: float
    measurements: tuple[SatelliteMeasurement, ...]
class RINEXObservationFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.time_scale = 'GPS'
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
                    field_count = self._observation_count.get(constellation, 0)
                    payload = sat_line[3:].rstrip('\n')
                    fields = [payload[i:i + 16] for i in range(0, len(payload), 16)]
                    while len(fields) < field_count:
                        continuation = stream.readline()
                        if not continuation:
                            break
                        payload = continuation[3:].rstrip('\n')
                        fields.extend(payload[i:i + 16] for i in range(0, len(payload), 16))
                    fields = fields[:field_count]
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
# --- GNSS precise products are parsed once inside GNSSPreprocessor below. ---
# --- SmartPNT IMR ---
IMR_HEADER_FORMAT_BODY = "8scdiidddiid32s?BBB32s6h?iii354s"
IMR_RECORD_FORMAT_BODY = "d6i"
@lru_cache(maxsize=8)
def _read_imr_layout(path: str | Path):
    """Return only the IMR layout/scales actually used by this simulation."""
    path = Path(path)
    with path.open('rb') as stream:
        buffer = stream.read(512)
    endian = '<' if buffer[8] == 0 else '>'
    values = struct.unpack(endian + IMR_HEADER_FORMAT_BODY, buffer)
    data_rate_hz = float(values[5])
    gyro_scale = float(values[6])
    accel_scale = float(values[7])
    time_tag_bias_ms = float(values[10])
    record_bytes = struct.calcsize(endian + IMR_RECORD_FORMAT_BODY)
    payload_bytes = max(0, path.stat().st_size - 512)
    record_count = payload_bytes // record_bytes
    record_dtype = np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))], align=False)
    if record_dtype.itemsize != record_bytes:
        raise RuntimeError('NumPy IMR dtype does not match the documented record size')
    return path, data_rate_hz, gyro_scale, accel_scale, time_tag_bias_ms, record_dtype, record_count, record_bytes
# --- SmartPNT README.xml ---
def _unique_file(directory: Path, patterns: tuple[str, ...], label: str) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) != 1:
        names = ', '.join(path.name for path in matches) or 'none'
        raise FileNotFoundError(f'Expected exactly one {label} file in {directory}, found {len(matches)}: {names}')
    return matches[0]
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
    r_e = np.asarray(nav.position_ecef_m, dtype=float).reshape(3)
    acceleration = Cmid @ np.asarray(specific_force_body_mps2) + _gravitation_j2_ecef(r_e) - _earth_rate_cross(_earth_rate_cross(r_e)) - 2.0 * _earth_rate_cross(nav.velocity_ecef_mps)
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavigationState(position, velocity, C1)
def gnss_antenna_position(nav: NavigationState, lever_arm_b_m: Array) -> Array:
    """Paper Eq. (9): IMU reference position -> GNSS antenna position."""
    return nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
# --- Pseudorange preparation ---
class PseudorangeMeasurement(NamedTuple):
    sat_id: str
    constellation: str
    pseudorange_m: float
    satellite_position_reception_ecef_m: Array
    satellite_clock_bias_s: float
    ionosphere_delay_m: float
    troposphere_delay_m: float
    sigma_code_m: float
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
    raise RuntimeError(f'Transmit-time iteration did not converge for {sat_id}: iterations={max_iterations}, last_satellite_position_delta_m={last_position_delta_m:.6g}, last_light_time_residual_m={last_light_time_residual_m:.6g}, last_transit_s={transit_s:.12g}')
class GNSSPreprocessor:
    """Parse precise GNSS products once and prepare corrected pseudoranges epoch by epoch."""
    def __init__(self, sp3_path: str | Path, clock_path: str | Path, interpolation_points: int=9, min_elevation_deg: float=5.0, use_troposphere: bool=True, use_ionosphere: bool=True, broadcast_ionosphere_coefficients: dict[str, tuple[Array, Array]] | None=None):
        self.interpolation_points = int(interpolation_points)
        temporary = defaultdict(lambda: [[], [], []])
        current_time = None
        time_scale = 'GPS'
        with Path(sp3_path).open('r', encoding='ascii', errors='replace') as stream:
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
        self.orbit_series = {}
        for sat_id, (times, positions, clocks) in temporary.items():
            times = np.asarray(times, dtype=float)
            order = np.argsort(times)
            times = times[order]
            clock = np.asarray(clocks, dtype=float)[order]
            finite_clock = np.isfinite(clock)
            self.orbit_series[sat_id] = {'time': times, 'position': np.asarray(positions, dtype=float)[order], 'clock_time': times[finite_clock], 'clock_value': clock[finite_clock]}

        clock_records = defaultdict(lambda: [[], []])
        clock_time_scale = 'GPS'
        with Path(clock_path).open('r', encoding='ascii', errors='replace') as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'TIME SYSTEM ID':
                    fields = line[:10].split()
                    if fields:
                        clock_time_scale = fields[0]
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
                t = calendar_to_gpst_seconds(year, month, day, hour, minute, second, clock_time_scale)
                clock_records[sat_id][0].append(t)
                clock_records[sat_id][1].append(values[0])
        self.clock_series = {}
        for sat_id, (times, bias) in clock_records.items():
            order = np.argsort(times)
            self.clock_series[sat_id] = (np.asarray(times)[order], np.asarray(bias)[order])
        self.clock_satellites = frozenset(self.clock_series)
        self.min_elevation_rad = np.deg2rad(min_elevation_deg)
        self.use_troposphere = bool(use_troposphere)
        self.use_ionosphere = bool(use_ionosphere)
        self.iono = broadcast_ionosphere_coefficients or {}

    def _position(self, sat_id: str, time_gpst_s: float) -> np.ndarray:
        s = self.orbit_series[sat_id]
        times = s['time']
        if time_gpst_s < times[0] or time_gpst_s > times[-1]:
            raise ValueError(f'SP3 time outside product span for {sat_id}')
        count = min(self.interpolation_points, len(times))
        i = int(np.searchsorted(times, time_gpst_s))
        start = max(0, min(len(times) - count, i - count // 2))
        window_times = times[start:start + count]
        window_positions = s['position'][start:start + count]
        scale = max(float(np.max(np.abs(window_times - time_gpst_s))), 1.0)
        x = (window_times - time_gpst_s) / scale
        position = []
        for column in range(window_positions.shape[1]):
            interpolator = BarycentricInterpolator(x, window_positions[:, column], rng=0)
            position.append(float(interpolator(0.0)))
        return np.asarray(position)

    def prepare_epoch(self, epoch: ObservationEpoch, nav: NavigationState, lever_arm_b_m: Array) -> tuple[PseudorangeMeasurement, ...]:
        antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
        receiver_llh = ecef_to_llh(antenna_position)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch.measurements:
            try:
                initial_position = self._position(raw.sat_id, epoch.time_gpst_s)
                receiver = np.asarray(antenna_position, dtype=float).reshape(3)
                sat_rx = np.asarray(initial_position, dtype=float).reshape(3)
                reception_time = float(epoch.time_gpst_s)
                rho0, _, _ = geometric_range(receiver, sat_rx, 0.0)
                tau0 = float(rho0 / SPEED_OF_LIGHT_MPS)
                tx0 = reception_time - tau0
                sat_tx0 = np.asarray(self._position(raw.sat_id, tx0), dtype=float).reshape(3)
                rho1, _, _ = geometric_range(receiver, sat_tx0, tau0)
                transit_s = float(rho1 / SPEED_OF_LIGHT_MPS)
                transmit_time = reception_time - transit_s
                state_tx_position = np.asarray(self._position(raw.sat_id, transmit_time), dtype=float).reshape(3)
            except (KeyError, ValueError):
                continue

            if raw.sat_id in self.clock_satellites:
                clock_times, clock_bias = self.clock_series[raw.sat_id]
                if transmit_time < clock_times[0] or transmit_time > clock_times[-1]:
                    raise ValueError(f'CLK time outside product span for {raw.sat_id}')
                satellite_clock_bias_s = float(np.interp(transmit_time, clock_times, clock_bias))
            else:
                orbit = self.orbit_series[raw.sat_id]
                orbit_times = orbit['time']
                if transmit_time < orbit_times[0] or transmit_time > orbit_times[-1]:
                    raise ValueError(f'SP3 time outside product span for {raw.sat_id}')
                if orbit['clock_time'].size == 0:
                    raise KeyError(raw.sat_id)
                satellite_clock_bias_s = float(np.interp(transmit_time, orbit['clock_time'], orbit['clock_value']))

            _, los, satellite_position_rx = geometric_range(antenna_position, state_tx_position, transit_s)
            elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
            if elevation < self.min_elevation_rad:
                continue
            ionosphere = 0.0
            height_m = None
            if self.use_ionosphere:
                alpha, beta = self.iono[raw.constellation]
                reception_week = int(math.floor(float(epoch.time_gpst_s) / GPS_WEEK_S))
                tow = float(epoch.time_gpst_s) - reception_week * GPS_WEEK_S
                if raw.constellation == 'G':
                    reference_frequency_hz = 1575420000.0
                else:
                    if raw.signal_suffix not in {'2I', '2X', '1I', '1X'}:
                        continue
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
            tropo_sigma = float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) if self.use_troposphere else 0.0
            mp_sigma = yan_goGPS_mp_nlos_sigma_m(elevation, cn0_dbhz)
            receiver_sigma = ref35_receiver_noise_sigma_m(cn0_dbhz)
            sigma_code_m = math.sqrt(LEO_URA_SIGMA_M ** 2 + iono_sigma ** 2 + tropo_sigma ** 2 + mp_sigma ** 2 + receiver_sigma ** 2)
            out.append(PseudorangeMeasurement(raw.sat_id, raw.constellation, float(raw.pseudorange_m), satellite_position_rx, satellite_clock_bias_s, float(ionosphere), float(troposphere), float(sigma_code_m)))
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
_VAN_LOAN_STATS = {'taylor_calls': 0, 'exact_fallback_calls': 0, 'validation_calls': 0, 'max_norm_1': 0.0, 'max_taylor_remainder_bound': 0.0, 'max_validation_phi_abs': 0.0, 'max_validation_qd_abs': 0.0}
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
        E = np.eye(B.shape[0], dtype=float)
        term = np.eye(B.shape[0], dtype=float)
        for k in range(1, VAN_LOAN_TAYLOR_ORDER + 1):
            term = (term @ B) / float(k)
            E += term
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
class TCMeasurementModel(NamedTuple):
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]
    clock_projector: Array
def retain_clock_observable_measurements(measurements) -> tuple[PseudorangeMeasurement, ...]:
    """Remove lone GPS/BDS codes that carry no position information after clock projection."""
    measurements = tuple(measurements)
    counts = Counter((m.constellation for m in measurements if m.constellation in {'G', 'C'}))
    return tuple((m for m in measurements if m.constellation not in {'G', 'C'} or counts[m.constellation] >= 2))
def select_measurements_to_network_capacity(
    measurements,
    capacity: int,
) -> tuple[PseudorangeMeasurement, ...]:
    """Select at most ``capacity`` observations without using test truth.

    Nmax is fixed from Data01 before the neural network is built.  If a later
    Data02 epoch contains more observations than that fixed capacity, keep the
    observations with the smallest modeled code standard deviation
    ``sigma_code_m``.  ``sat_id`` is used only as a deterministic tie-breaker.

    The final subset is returned in the *original measurement order* so the
    existing slot/order convention is not changed merely by the capacity
    selection.  For GPS/BDS, a constellation is admitted only when at least
    two measurements from that constellation can be retained; this preserves
    the receiver-clock projection requirement used by this reproduction.

    Yan et al. specify zero padding to a fixed Nmax but do not publish a rule
    for test epochs with Nk > training Nmax.  This is therefore an explicit
    project completion, not a claimed paper detail.
    """
    if capacity <= 0:
        raise ValueError('network observation capacity must be positive')

    measurements = tuple(retain_clock_observable_measurements(measurements))
    if len(measurements) <= capacity:
        return measurements

    ranked = sorted(
        enumerate(measurements),
        key=lambda item: (
            float(item[1].sigma_code_m),
            str(item[1].sat_id),
            int(item[0]),
        ),
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    constellation_counts: Counter = Counter()

    # Greedy quality ordering with the clock-observability constraint.
    for rank_pos, (original_index, measurement) in enumerate(ranked):
        if len(selected_indices) >= capacity:
            break
        if original_index in selected_set:
            continue

        constellation = measurement.constellation
        remaining_slots = capacity - len(selected_indices)

        if constellation in {'G', 'C'} and constellation_counts[constellation] == 0:
            # Starting a GPS/BDS constellation requires a pair so that the
            # receiver-clock projection remains observable.
            if remaining_slots < 2:
                continue
            partner = None
            for partner_index, partner_measurement in ranked[rank_pos + 1:]:
                if (
                    partner_index not in selected_set
                    and partner_measurement.constellation == constellation
                ):
                    partner = partner_index
                    break
            if partner is None:
                continue
            selected_indices.extend([original_index, partner])
            selected_set.update([original_index, partner])
            constellation_counts[constellation] += 2
            continue

        selected_indices.append(original_index)
        selected_set.add(original_index)
        constellation_counts[constellation] += 1

    selected_indices = sorted(selected_indices[:capacity])
    selected = tuple(measurements[index] for index in selected_indices)
    selected = tuple(retain_clock_observable_measurements(selected))

    if len(selected) > capacity:
        raise RuntimeError('capacity selector returned too many measurements')
    if not selected:
        raise RuntimeError(
            'capacity selector removed all observations; check training Nmax '
            'and constellation availability'
        )
    return selected

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
        range_vector = np.asarray(antenna_position) - m.satellite_position_reception_ecef_m
        geometric_range_m = float(np.linalg.norm(range_vector))
        los = range_vector / geometric_range_m
        predicted = geometric_range_m + 0.0 - SPEED_OF_LIGHT_MPS * m.satellite_clock_bias_s + m.ionosphere_delay_m + m.troposphere_delay_m
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
        range_vector = np.asarray(antenna_position) - m.satellite_position_reception_ecef_m
        geometric_range_m = float(np.linalg.norm(range_vector))
        predicted = geometric_range_m + 0.0 - SPEED_OF_LIGHT_MPS * m.satellite_clock_bias_s + m.ionosphere_delay_m + m.troposphere_delay_m
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
def reset_error_state_covariance(posterior_covariance: Array, injected_error_state: Array) -> Array:
    """Map covariance into the post-feedback 9-state error coordinates."""
    P = np.asarray(posterior_covariance, dtype=float)
    dx = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    if P.shape != (INS_STATE_DIM, INS_STATE_DIM):
        raise ValueError('posterior covariance has wrong shape for error-state reset')
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(dx)):
        raise ValueError('non-finite covariance/error-state in feedback reset')
    phi = np.asarray(ATTITUDE_FEEDBACK_SIGN * dx[6:9], dtype=float).reshape(3)
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
    reset_jacobian = np.eye(INS_STATE_DIM)
    reset_jacobian[6:9, 6:9] = np.eye(3) + a * K + b * K2
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
class TorchNavigationState(NamedTuple):
    position_ecef_m: torch.Tensor
    velocity_ecef_mps: torch.Tensor
    body_to_ecef_dcm: torch.Tensor
def _torch_so3_exponential(rotation_vector_rad: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues exponential matching _so3_exponential()."""
    v = rotation_vector_rad.reshape(3)
    theta2 = torch.dot(v, v)
    theta2_safe = theta2.clamp_min(1e-30)
    theta = torch.sqrt(theta2_safe)
    x, y, z = v.unbind()
    zero = v.new_zeros(())
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
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
def _torch_mechanize_ecef(nav: TorchNavigationState, angular_rate_body_radps: torch.Tensor, specific_force_body_mps2: torch.Tensor, dt_s: float) -> TorchNavigationState:
    """Differentiable ECEF mechanization used only for Data01 BPTT."""
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    earth_rotvec = C0.new_tensor(-OMEGA_IE_E * dt)
    C1 = _torch_so3_exponential(earth_rotvec) @ C0 @ _torch_so3_exponential(angular_rate_body_radps.reshape(3) * dt)
    Cmid = 0.5 * (C0 + C1)
    omega = C0.new_tensor(OMEGA_IE_E)
    r_e = nav.position_ecef_m.reshape(3)
    x, y, z = r_e.unbind()
    radius = torch.linalg.vector_norm(r_e).clamp_min(1.0)
    z2_r2 = z * z / (radius * radius)
    j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / radius) ** 2
    xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / radius ** 3
    gravity = scale * torch.stack((x * xy_factor, y * xy_factor, z * z_factor))
    acceleration = Cmid @ specific_force_body_mps2.reshape(3) + gravity - torch.linalg.cross(omega, torch.linalg.cross(omega, r_e)) - 2.0 * torch.linalg.cross(omega, nav.velocity_ecef_mps)
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return TorchNavigationState(position, velocity, C1)
# --- TLE/SGP4 LEO orbit and pseudorange-only simulation ---
class TLESGP4Provider:
    """LEO orbit source using TLE/SGP4 (WGS-72) transformed from TEME to ECEF."""
    def __init__(self, path: str | Path, max_tle_age_days: float, allow_degraded_eop: bool=False, allow_non_tle_files: bool=True):
        self.max_tle_age_s = float(max_tle_age_days) * 86400.0
        self.allow_degraded_eop = allow_degraded_eop
        by_id = {}
        for source in sorted(Path(path).iterdir()):
            if not source.is_file():
                continue
            lines = source.read_text(encoding='ascii', errors='replace').splitlines()
            found = False
            for i in range(len(lines) - 1):
                line1, line2 = lines[i].strip(), lines[i + 1].strip()
                if not (line1.startswith('1 ') and line2.startswith('2 ')):
                    continue
                try:
                    verify_checksum(line1, line2)
                except ValueError:
                    if allow_non_tle_files:
                        continue
                    raise
                satrec = Satrec.twoline2rv(line1, line2, WGS72)
                norad = str(satrec.satnum_str).strip()
                epoch_utc = Time(satrec.jdsatepoch, satrec.jdsatepochF, format='jd', scale='utc')
                epoch_datetime_utc = epoch_utc.to_datetime(timezone=timezone.utc)
                epoch_gpst_s = (epoch_datetime_utc - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP_SECONDS
                by_id.setdefault(f'NORAD-{norad}', []).append((float(epoch_gpst_s), satrec))
                found = True
            if not found and not allow_non_tle_files:
                raise ValueError(f'No valid TLE in {source}')
        self._elements = {sat_id: tuple(sorted(elements, key=lambda e: e[0])) for sat_id, elements in by_id.items()}
        self._element_times = {sat_id: tuple(element[0] for element in elements) for sat_id, elements in self._elements.items()}
        self.satellite_ids = tuple(sorted(self._elements, key=lambda s: int(s.split('-')[1])))
    def state_at(self, time_gpst_s: float, sat_id: str) -> tuple[Array, Array]:
        elements = self._elements[sat_id]
        index = bisect_right(self._element_times[sat_id], time_gpst_s) - 1
        if index < 0:
            raise ValueError(f'No prior TLE for {sat_id}')
        epoch_gpst_s, satrec = elements[index]
        if time_gpst_s - epoch_gpst_s > self.max_tle_age_s:
            raise ValueError(f'TLE too old for {sat_id}')
        utc_datetime = GPS_EPOCH + timedelta(seconds=float(time_gpst_s) - GPS_UTC_LEAP_SECONDS)
        utc = Time(utc_datetime, scale='utc')
        error, position_km, velocity_km_s = satrec.sgp4(float(utc.jd1), float(utc.jd2))
        if error:
            raise ValueError(SGP4_ERRORS.get(error, f'SGP4 error {error}'))
        position = CartesianRepresentation(np.asarray(position_km) * u.km)
        velocity = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        teme = TEME(position.with_differentials(velocity), obstime=utc)
        degraded = 'warn' if self.allow_degraded_eop else 'error'
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        return (np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3), np.asarray(itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s)).reshape(3))
def ref35_ionosphere_sigma_m(elevation_rad: float, receiver_latitude_rad: float) -> float:
    """Ref. [35] ionospheric residual sigma [m]."""
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else (4.5 if latitude_deg <= 55.0 else 6.0)
    R = 6_378_140.0
    h_i = 350_000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))
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
        receive_week = int(math.floor(float(receive_time_gpst_s) / GPS_WEEK_S))
        tow = float(receive_time_gpst_s) - receive_week * GPS_WEEK_S
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
        initial_position, _ = self.provider.state_at(receive_time_gpst_s, sat_id)
        initial_range, initial_los, _ = geometric_range(receiver_position_ecef_m, initial_position, 0.0)
        initial_elevation, _ = elevation_azimuth_from_ned_matrix(c_ecef_ned, initial_los)
        self.prefilter_checked += 1
        if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
            self.prefilter_rejected += 1
            return None
        self.prefilter_passed += 1
        transmit_time, transit_s, state_tx_position = _iterate_transmit_time(lambda sid, t: self.provider.state_at(t, sid)[0], sat_id, receive_time_gpst_s, receiver_position_ecef_m, initial_position, initial_range / SPEED_OF_LIGHT_MPS, self.tx_epsilon_position_m, self.tx_max_iterations)
        rho, los, sat_rx = geometric_range(receiver_position_ecef_m, state_tx_position, transit_s)
        elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
        if elevation < self.minimum_elevation_rad:
            return None
        receiver_lat, _, receiver_height_m = receiver_llh
        ionosphere_m, ionosphere_path_scale = self._paper_ionosphere_delay_m(receive_time_gpst_s, receiver_position_ecef_m, sat_rx, elevation, azimuth, receiver_llh=receiver_llh)
        troposphere_m = saastamoinen_delay_m(receiver_height_m, elevation) if self.use_troposphere else 0.0
        iono_sigma_m = ionosphere_path_scale * ref35_ionosphere_sigma_m(elevation, receiver_lat) if self.use_ionosphere else 0.0
        tropo_sigma_m = float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) if self.use_troposphere else 0.0
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
# --- MASKED CLA NETWORK --- Yan Eqs. (10)-(29); 9-state/pseudorange-only. Pooling and [9,Nmax] FC reshape are project completions.
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 24 in the user-requested 9-state branch
OBSERVATION_FEATURE_DIM = 2                 # [Delta y_{k-1}, Delta ytilde_k]
# Exact Yan Eq. (15)/(16) network-input order:
# [Delta alpha, Delta omega, Delta xhat, Delta xtilde,
#  Delta y_{k-1}(1:Nk), Delta ytilde_k(1:Nk), zero-padding]
SUPERVISED_STATE_DIM = DIRECT_STATE_LABEL_DIM
MASK_NORMALIZATION_EPS = 1e-6
FIG8_POOL_KERNEL_SIZE = 3
class MaskedCLAOutput(NamedTuple):
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
        # Yan et al. do not publish the FC/KG initialization.  For this direct
        # K @ raw-pseudorange-innovation implementation, an exactly zero gain
        # is the only initialization whose closed-loop behavior is guaranteed
        # to equal the finite no-neural-correction baseline.  The first backward
        # pass gives the gain head a nonzero gradient; deeper CLA gradients begin
        # flowing after the head has moved away from zero.
        nn.init.zeros_(self.gain_head.weight)
        if self.gain_head.bias is not None:
            nn.init.zeros_(self.gain_head.bias)
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
    output_override = os.environ.get('MKNET_OUTPUT_DIR')
    OUTPUT_DIR = Path(output_override).expanduser() if output_override else (Path('/kaggle/working/direct_run') if IN_KAGGLE else Path('direct_run'))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Runtime defaults targeting ~1600 neural training/validation samples.
    # The lagged feature construction discards the first usable fusion row, so
    # a 1601-fusion-epoch cap typically yields 1600 neural samples; the
    # chronological 80/20 split then gives about 1280 train and 320 validation samples. Environment variables still override all values.
    MAX_FUSION_EPOCHS = _env_optional_int('MKNET_MAX_FUSION_EPOCHS', 50)
    MAX_TEST_FUSION_EPOCHS = _env_optional_int('MKNET_MAX_TEST_FUSION_EPOCHS', 50)

    # Training follows the common KalmanNet pattern: one initialization per
    # physical trajectory, recursive state propagation, validation-based model
    # selection, and truncated BPTT only to bound graph length.  Yan reports
    # Adam and an initial LR of 0.01; the public KalmanNet repositories commonly
    # use 1e-3.  This direct raw-pseudorange gain path was numerically unstable
    # at 1e-3, so 1e-4 is the conservative default and remains overridable.
    TRAINING_EPOCHS = _env_int('MKNET_TRAINING_EPOCHS', 5)
    YAN_REPORTED_LEARNING_RATE = 0.01
    LEARNING_RATE = _env_positive_float('MKNET_LEARNING_RATE', 1e-4)
    FILTER_BLOCK_LEARNING_RATE = LEARNING_RATE
    REPRESENTATION_BLOCK_LEARNING_RATE = LEARNING_RATE
    OPTIMIZER_WINDOW_SIZE = _env_int('MKNET_OPTIMIZER_WINDOW_SIZE', 4)
    if OPTIMIZER_WINDOW_SIZE <= 0:
        raise ValueError('MKNET_OPTIMIZER_WINDOW_SIZE must be positive')
    GRADIENT_CLIP_NORM = _env_positive_float('MKNET_GRADIENT_CLIP_NORM', 1.0)
    EARLY_STOPPING_PATIENCE = _env_int('MKNET_EARLY_STOPPING_PATIENCE', 6)
    # Chronological Data01 tail is validation; Data02 remains untouched test data.
    VALIDATION_FRACTION = float(os.environ.get('MKNET_VALIDATION_FRACTION', '0.20'))
    if not 0.0 < VALIDATION_FRACTION < 1.0:
        raise ValueError('MKNET_VALIDATION_FRACTION must be strictly between 0 and 1')
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
    readme_root = ET.fromstring(README_XML_PATH.read_text(encoding='utf-8', errors='replace'))
    rover_xml = next((item for item in readme_root.findall('ROVE') if (item.findtext('ID') or '').strip() == '01'), None)
    if rover_xml is None:
        raise KeyError('Rover 01 not found')
    rover_imu_type = (rover_xml.findtext('SINS_IMUType') or '').strip()
    rover_lever_arm_vehicle_m = np.fromstring(rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ')
    rover_mounting_xyz_deg = np.fromstring(rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    antenna_truth = load_ie_ground_truth(ROVE_GROUND_TRUTH_PATH)
    imu_truth = load_ie_ground_truth(IMU_GROUND_TRUTH_PATH)
    imu_model_text = IMU_ERROR_MODEL_PATH.read_text(encoding='utf-8', errors='replace')
    imu_model_keys = ('ISDV_Pos', 'ISDV_Vel', 'ISDV_Att', 'ISDV_AccelBias', 'ISDV_GyrosBias', 'PNSD_Pos', 'PNSD_Vel', 'PNSD_Att', 'PNSD_AccelBias', 'PNSD_GyrosBias')
    imu_model_values = None
    for match in re.finditer(r'IMU\s*\{(.*?)\}', imu_model_text, flags=re.S):
        block = match.group(1)
        type_match = re.search(r'IMU_Type\s*=\s*"([^"]+)"', block)
        if not type_match or type_match.group(1) != rover_imu_type:
            continue
        values = {}
        for key in imu_model_keys:
            value_match = re.search(rf'{key}\s*=\s*([^\r\n]+)', block)
            if value_match:
                values[key] = np.fromstring(value_match.group(1), sep=' ', dtype=float)
        if len(values) == len(imu_model_keys):
            imu_model_values = values
            break
    if imu_model_values is None:
        raise KeyError(f'No complete IMU noise model for training IMU type {rover_imu_type!r}')
    rinex = RINEXObservationFile(RINEX_OBS_PATH)
    first_rinex_epoch = next(rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    if first_rinex_epoch is None:
        raise ValueError('RINEX file contains no GPS/BDS epochs')
    imr_path, imr_data_rate_hz, imr_gyro_scale, imr_accel_scale, imr_time_tag_bias_ms, imr_record_dtype, imr_record_count, imr_record_bytes = _read_imr_layout(IMR_PATH)
    if imr_record_count < 2:
        raise ValueError('IMR file contains fewer than two usable samples')
    imr_records = np.memmap(imr_path, dtype=imr_record_dtype, mode='r', offset=512, shape=(imr_record_count,))
    imr_tow_all = np.asarray(imr_records['tow'], dtype=np.float64).copy()
    del imr_records
    imr_tow_all[imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    imr_tow_all -= imr_time_tag_bias_ms * 1e-3
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
    imr_count = requested_stop - start
    imr_records = np.fromfile(imr_path, dtype=imr_record_dtype, count=imr_count, offset=512 + start * imr_record_bytes)
    imr_tow = imr_records['tow'].astype(np.float64, copy=True)
    imr_tow[imr_tow > GPS_WEEK_S] -= GPS_WEEK_S
    imr_tow -= imr_time_tag_bias_ms * 1e-3
    imr_angular_rate_body_radps = np.deg2rad(imr_records['counts'][:, :3].astype(float) * imr_gyro_scale * imr_data_rate_hz)
    imr_acceleration_body_mps2 = imr_records['counts'][:, 3:6].astype(float) * imr_accel_scale * imr_data_rate_hz
    imr_time = imr_time_all[start:requested_stop].copy()
    del imr_tow_all, imr_time_all
    if len(imr_tow) < 2:
        raise ValueError('Selected training IMR window contains fewer than two samples')
    query_time = np.concatenate(([imr_time[0]], fusion_time))
    antenna_position = interpolate_ground_truth(antenna_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    imu_position, imu_velocity, imu_heading, imu_pitch, imu_roll = interpolate_ground_truth(imu_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    initial_imu_truth_position = imu_position[0]
    initial_imu_truth_velocity = imu_velocity[0]
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]
    C_b_e = body_to_ecef_from_ie_hpr(initial_imu_truth_position, imu_heading[0], imu_pitch[0], imu_roll[0], rover_mounting_xyz_deg)
    lever_arm_b_m = c_vehicle_to_body_zxy(*rover_mounting_xyz_deg) @ np.asarray(rover_lever_arm_vehicle_m, dtype=float).reshape(3)
    initial_antenna_from_imu = initial_imu_truth_position + C_b_e @ lever_arm_b_m
    truth_reference_error_m = np.linalg.norm(antenna_position[0] - initial_antenna_from_imu)
    if truth_reference_error_m > 0.05:
        raise ValueError(f'ROVE/ISA-100C ground-truth reference points are inconsistent with the lever arm: {truth_reference_error_m:.3f} m')
    fusion_truth_body_to_ecef = np.stack([body_to_ecef_from_ie_hpr(p, h, pt, r, rover_mounting_xyz_deg) for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])])
    initial_nav = NavigationState(initial_imu_truth_position.copy(), initial_imu_truth_velocity.copy(), C_b_e)
    d2r = np.pi / 180.0
    initial_sigma = np.concatenate((imu_model_values['ISDV_Pos'], imu_model_values['ISDV_Vel'], imu_model_values['ISDV_Att'] * d2r))
    P0 = np.diag(initial_sigma ** 2)
    process_density = np.concatenate((imu_model_values['PNSD_Pos'], imu_model_values['PNSD_Vel'], imu_model_values['PNSD_Att'] * d2r))
    Qc = np.diag(process_density ** 2)
    ionosphere_coefficients = None
    if USE_IONOSPHERE:
        iono_header = {}
        with NAV_PATH.open('r', encoding='ascii', errors='replace') as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'IONOSPHERIC CORR':
                    fields = line[:60].split()
                    if len(fields) >= 5:
                        iono_header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
        ionosphere_coefficients = {'G': (np.asarray(iono_header['GPSA']), np.asarray(iono_header['GPSB'])), 'C': (np.asarray(iono_header['BDSA']), np.asarray(iono_header['BDSB']))}
    gnss_preprocessor = GNSSPreprocessor(SP3_PATH, CLK_PATH, min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=ionosphere_coefficients)
    tle_provider = TLESGP4Provider(LEO_TLE_DIR, max_tle_age_days=TLE_MAX_AGE_DAYS, allow_degraded_eop=TLE_ALLOW_DEGRADED_EOP, allow_non_tle_files=TLE_ALLOW_NON_TLE_FILES)
    leo_klobuchar = None if ionosphere_coefficients is None else ionosphere_coefficients['G']
    leo_simulator = LEODownlinkSimulator(tle_provider, leo_klobuchar, seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    nav = initial_nav.copy()
    P = P0.copy()
    history_rows = []
    data01_sequence_nmax = 0
    interval_segments_since_previous_usable_fusion = []
    last_gyro = np.asarray(imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(imr_acceleration_body_mps2[0], dtype=float).reshape(3)
    # --- DATA01 CLASSICAL INS/TC HISTORY ---
    training_timeline = build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True)
    for event_kind, event_start, event_end, event_index in training_timeline:
        if event_kind == 'imu':
            imu_index = event_index
            dt = event_end - event_start
            last_gyro = np.asarray(imr_angular_rate_body_radps[imu_index], dtype=float).reshape(3)
            last_accel = np.asarray(imr_acceleration_body_mps2[imu_index], dtype=float).reshape(3)
            nav = mechanize_ecef(nav, last_gyro, last_accel, dt)
            interval_segments_since_previous_usable_fusion.append((imu_index, dt))
            F_error = build_error_state_dynamics(nav, last_accel)
            Phi, Qd = discretize_process_noise_van_loan(F_error, Qc, dt)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue
        fusion_index = event_index
        t = event_start
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
        relative_attitude = fusion_truth_body_to_ecef[fusion_index] @ prior_nav.body_to_ecef_dcm.T
        attitude_target = SpatialRotation.from_matrix(relative_attitude).as_rotvec() / ATTITUDE_FEEDBACK_SIGN
        target_state_9 = np.concatenate([fusion_imu_truth_position[fusion_index] - prior_nav.position_ecef_m, fusion_imu_truth_velocity[fusion_index] - prior_nav.velocity_ecef_mps, attitude_target])
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
    # KalmanNet-family discipline: network observation dimensionality is fixed
    # before test data are evaluated.  For this Yan-style variable-observation
    # adaptation, Nmax is therefore determined strictly from the actual Data01
    # neural inputs, never from Data02.
    nmax = max((len(row['sat_ids']) for row in history_rows[1:]))
    network_nmax = int(nmax)
    if TEST_DATASET_DIR is not None:
        test_dataset_dir = Path(TEST_DATASET_DIR)
    else:
        test_dataset_candidates = sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir() and path.resolve() != TRAIN_DATASET_DIR.resolve())
        if len(test_dataset_candidates) != 1:
            names = ', '.join(path.name for path in test_dataset_candidates) or 'none'
            raise RuntimeError(f'TEST_DATASET_DIR is not set and automatic selection is ambiguous. Found {len(test_dataset_candidates)} non-training dataset directories: {names}. Set TEST_DATASET_DIR to the exact second SmartPNT-Pos dataset.')
        test_dataset_dir = test_dataset_candidates[0]
    if test_dataset_dir.resolve() == TRAIN_DATASET_DIR.resolve():
        raise ValueError('The online test dataset must be distinct from the training dataset')
    data02_sequence_nmax = None  # deliberately unknown until independent test
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
    data01_total_sample_count = int(len(feature_time))
    if data01_total_sample_count <= 1:
        raise ValueError('Data01 produced too few KalmanNet samples for train/validation separation')
    validation_sample_count = max(1, int(round(data01_total_sample_count * VALIDATION_FRACTION)))
    training_sample_count = data01_total_sample_count - validation_sample_count
    validation_start_sample = training_sample_count
    if training_sample_count <= 0 or validation_sample_count <= 0:
        raise ValueError('invalid chronological Data01 train/validation split')

    print(
        f"training setup: Data01_total={data01_total_sample_count}, "
        f"train={training_sample_count}, validation={validation_sample_count}, "
        f"split=chronological_{1.0-VALIDATION_FRACTION:.2f}/{VALIDATION_FRACTION:.2f}, "
        f"Nmax={network_nmax} (Data01 only), "
        f"mode=continuous_recursive_TBPTT, one_initialization_per_phase, "
        f"epochs={TRAINING_EPOCHS}, lr_theta={FILTER_BLOCK_LEARNING_RATE:g}, "
        f"lr_psi={REPRESENTATION_BLOCK_LEARNING_RATE:g}, "
        f"window={OPTIMIZER_WINDOW_SIZE}, grad_clip={GRADIENT_CLIP_NORM:g}, "
        f"gain_head_init=exact_zero, checkpoint=best_causal_recursive_validation_Eq30, "
        f"FDE={('ON' if FDE_DIA_ON else 'OFF')}"
    )

    # --- MASKED-CLA MODEL + YAN/LATENT ALTERNATING PARTITION ---
    # Architecture remains Yan Eqs. (22)-(29). Yan states that alternating
    # optimization is used but does not publish a module-level split. Following
    # the encoder/filter decomposition of Latent-KalmanNet, this reproduction
    # treats the masked CNN as representation block psi and the recurrent
    # LSTM+attention+KG head as filter block theta. No artificial encoder target
    # is invented, so both phases use Yan's final state loss.
    model = MaskedCLA(nmax=network_nmax, dropout=0.2).to(DEVICE)
    LSTM_CONFIGURED_DROPOUT = float(model.lstm.lstm.dropout)
    representation_modules = (model.conv,)
    filter_modules = (model.lstm, model.attention, model.gain_head)
    representation_parameters = [p for module in representation_modules for p in module.parameters()]
    filter_parameters = [p for module in filter_modules for p in module.parameters()]
    if {id(p) for p in representation_parameters} & {id(p) for p in filter_parameters}:
        raise RuntimeError('alternating parameter blocks overlap')
    if {id(p) for p in representation_parameters + filter_parameters} != {id(p) for p in model.parameters()}:
        raise RuntimeError('alternating blocks must cover the entire MaskedCLA network')
    filter_optimizer = torch.optim.Adam(filter_parameters, lr=FILTER_BLOCK_LEARNING_RATE)
    representation_optimizer = torch.optim.Adam(representation_parameters, lr=REPRESENTATION_BLOCK_LEARNING_RATE)
    all_network_parameters = representation_parameters + filter_parameters
    training_history = []

    training_imr_gyro_t = torch.as_tensor(imr_angular_rate_body_radps, dtype=torch.float64, device=DEVICE)
    training_imr_accel_t = torch.as_tensor(imr_acceleration_body_mps2, dtype=torch.float64, device=DEVICE)
    lever_arm_train_t = torch.as_tensor(lever_arm_b_m, dtype=torch.float64, device=DEVICE).reshape(3)
    fusion_truth_position_t = torch.as_tensor(fusion_imu_truth_position, dtype=torch.float64, device=DEVICE)
    fusion_truth_velocity_t = torch.as_tensor(fusion_imu_truth_velocity, dtype=torch.float64, device=DEVICE)
    fusion_truth_dcm_t = torch.as_tensor(fusion_truth_body_to_ecef, dtype=torch.float64, device=DEVICE)
    zero_error_state_t = torch.zeros(INS_STATE_DIM, dtype=torch.float64, device=DEVICE)
    target_base_t = torch.tensor(target_states_9, dtype=torch.float64, device=DEVICE)

    def _measurement_innovation(nav_state: TorchNavigationState, prepared_measurements):
        sat_ids, satellite_position, observed, satellite_clock, ionosphere, troposphere, projector_t = prepared_measurements
        if not sat_ids:
            return nav_state.position_ecef_m.new_empty(0), ()
        antenna_position = nav_state.position_ecef_m + nav_state.body_to_ecef_dcm @ lever_arm_train_t
        geometric = torch.linalg.vector_norm(antenna_position.unsqueeze(0) - satellite_position, dim=1)
        predicted = geometric - SPEED_OF_LIGHT_MPS * satellite_clock + ionosphere + troposphere
        return projector_t @ (observed - predicted), sat_ids

    def _network_inputs(previous_context, current_sat_ids, innovation_now, current_accel, current_gyro):
        """Build Yan Eqs. (10)-(17), normalize causally, and pad to Data01 Nmax."""
        n = len(current_sat_ids)
        innovation_now = innovation_now.reshape(n)
        fixed_raw = torch.cat((
            current_accel.reshape(3) - previous_context['accel'],
            current_gyro.reshape(3) - previous_context['gyro'],
            previous_context['state_residual'],
            previous_context['state_innovation'],
        ))
        previous_index = {sat_id: i for i, sat_id in enumerate(previous_context['sat_ids'])}
        zero = innovation_now.new_zeros(())
        residual_values = []
        residual_valid = []
        for sat_id in current_sat_ids:
            index = previous_index.get(sat_id)
            residual_values.append(zero if index is None else previous_context['residual'][index])
            residual_valid.append(index is not None)
        residual_k = torch.stack(residual_values)
        residual_mask = torch.as_tensor(residual_valid, dtype=torch.bool, device=DEVICE)
        current_mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
        obs_raw = torch.stack((residual_k, innovation_now), dim=1)
        channel_mask = torch.stack((residual_mask, current_mask), dim=1)

        def l2_group(values: torch.Tensor) -> torch.Tensor:
            return values / torch.clamp(torch.linalg.vector_norm(values), min=FEATURE_L2_EPS)

        fixed_nn = torch.cat((
            l2_group(fixed_raw[0:3]),
            l2_group(fixed_raw[3:6]),
            l2_group(fixed_raw[6:15]),
            l2_group(fixed_raw[15:24]),
        ))
        obs_columns = []
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            values = torch.where(valid, obs_raw[:, channel], torch.zeros_like(obs_raw[:, channel]))
            obs_columns.append(l2_group(values))
        obs_nn = torch.stack(obs_columns, dim=1)
        obs_nn = torch.where(channel_mask, obs_nn, torch.zeros_like(obs_nn))

        if n > network_nmax:
            raise RuntimeError(f'Data01 observation count {n} exceeds fixed Nmax={network_nmax}')
        pad = network_nmax - n
        if pad:
            obs_nn = torch.cat((obs_nn, torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=obs_nn.dtype, device=DEVICE)))
            current_mask = torch.cat((current_mask, torch.zeros(pad, dtype=torch.bool, device=DEVICE)))
            channel_mask = torch.cat((channel_mask, torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=torch.bool, device=DEVICE)))
            innovation_now = torch.cat((innovation_now, torch.zeros(pad, dtype=innovation_now.dtype, device=DEVICE)))
        if not bool((torch.all(torch.isfinite(fixed_nn)) & torch.all(torch.isfinite(obs_nn))).detach().cpu()):
            raise FloatingPointError('non-finite normalized KalmanNet feature')
        return fixed_nn, obs_nn, current_mask, channel_mask, innovation_now

    def _propagate_interval(nav_state: TorchNavigationState, segments, *, training: bool):
        if not segments:
            return nav_state, training_imr_gyro_t[0], training_imr_accel_t[0]
        segment_tuple = tuple((int(i), float(dt)) for i, dt in segments)
        def core(position, velocity, dcm):
            state = TorchNavigationState(position, velocity, dcm)
            feature_gyro = training_imr_gyro_t[segment_tuple[0][0]]
            feature_accel = training_imr_accel_t[segment_tuple[0][0]]
            for imu_index, dt in segment_tuple:
                feature_gyro = training_imr_gyro_t[imu_index].reshape(3)
                feature_accel = training_imr_accel_t[imu_index].reshape(3)
                state = _torch_mechanize_ecef(state, feature_gyro, feature_accel, dt)
            return state.position_ecef_m, state.velocity_ecef_mps, state.body_to_ecef_dcm, feature_gyro, feature_accel
        inputs = (nav_state.position_ecef_m, nav_state.velocity_ecef_mps, nav_state.body_to_ecef_dcm)
        if training and any(value.requires_grad for value in inputs):
            position, velocity, dcm, feature_gyro, feature_accel = checkpoint(core, *inputs, use_reentrant=False)
        else:
            position, velocity, dcm, feature_gyro, feature_accel = core(*inputs)
        return TorchNavigationState(position, velocity, dcm), feature_gyro, feature_accel

    def _rollout(start_sample: int, stop_sample: int, *, phase: str | None=None, initial_state=None, loss_start_sample: int | None=None):
        """Recursive Data01 rollout with one causal initialization and optional TBPTT carry.

        For training, each call is one TBPTT optimizer window. Numerical INS,
        causal feature context, and LSTM memory are carried from the previous
        window and detached only at the window boundary. For validation, the
        rollout begins at sample 0 and reaches the chronological holdout without
        any truth reset; loss starts only at ``loss_start_sample``.
        """
        if not 0 <= start_sample < stop_sample <= data01_total_sample_count:
            raise ValueError('invalid Data01 rollout bounds')
        training = phase is not None
        metric_start = start_sample if loss_start_sample is None else int(loss_start_sample)
        if not start_sample <= metric_start < stop_sample:
            raise ValueError('loss_start_sample must lie inside rollout bounds')

        if initial_state is None:
            # Feature sample k uses history_rows[k] as the predecessor context.
            # This is a one-time causal classical warm start, not a truth reset.
            start_nav = history_rows[start_sample]['posterior_nav']
            nav_state = TorchNavigationState(
                torch.as_tensor(start_nav.position_ecef_m, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.velocity_ecef_mps, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.body_to_ecef_dcm, dtype=torch.float64, device=DEVICE).clone(),
            )
            context_row = history_rows[start_sample]
            x_pred = np.asarray(context_row['x_pred'], dtype=float).reshape(INS_STATE_DIM)
            x_post = np.asarray(context_row['x_post'], dtype=float).reshape(INS_STATE_DIM)
            previous_context = {
                'sat_ids': tuple(context_row['sat_ids']),
                'residual': torch.as_tensor(np.asarray(context_row['residual'], dtype=float), dtype=torch.float64, device=DEVICE),
                'x_pred': torch.as_tensor(x_pred, dtype=torch.float64, device=DEVICE),
                'x_post': torch.as_tensor(x_post, dtype=torch.float64, device=DEVICE),
                'state_innovation': torch.as_tensor(x_post - x_pred, dtype=torch.float64, device=DEVICE),
                'state_residual': torch.as_tensor(np.zeros(INS_STATE_DIM) if start_sample == 0 else x_post - np.asarray(history_rows[start_sample - 1]['x_post'], dtype=float).reshape(INS_STATE_DIM), dtype=torch.float64, device=DEVICE),
                'accel': torch.as_tensor(np.asarray(context_row['accel'], dtype=float), dtype=torch.float64, device=DEVICE),
                'gyro': torch.as_tensor(np.asarray(context_row['gyro'], dtype=float), dtype=torch.float64, device=DEVICE),
            }
            if len(previous_context['sat_ids']) != int(previous_context['residual'].numel()):
                raise RuntimeError('stored Data01 context satellite/residual mismatch')
            feature_accel = previous_context['accel']
            feature_gyro = previous_context['gyro']
            recurrent_state = None
        else:
            nav_state = initial_state['nav']
            previous_context = initial_state['context']
            feature_accel = initial_state['feature_accel']
            feature_gyro = initial_state['feature_gyro']
            recurrent_state = initial_state['recurrent_state']

        state_losses = []
        state_sum = 0.0
        position_sum = 0.0
        metric_count = 0
        skipped_no_measurements = 0
        max_prior_position_error_m = 0.0
        max_innovation_abs_m = 0.0
        max_gain_fro_norm = 0.0
        max_position_correction_norm_m = 0.0
        max_velocity_correction_norm_mps = 0.0
        max_attitude_correction_norm_rad = 0.0

        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context:
            for sample_index in range(start_sample, stop_sample):
                row = history_rows[sample_index + 1]
                if row['preceding_interval_segments']:
                    nav_state, feature_gyro, feature_accel = _propagate_interval(
                        nav_state, row['preceding_interval_segments'], training=training
                    )
                nav_finite = (
                    torch.all(torch.isfinite(nav_state.position_ecef_m))
                    & torch.all(torch.isfinite(nav_state.velocity_ecef_mps))
                    & torch.all(torch.isfinite(nav_state.body_to_ecef_dcm))
                )
                if not bool(nav_finite.detach().cpu()):
                    raise FloatingPointError(f'non-finite INS state at Data01 sample {sample_index}')

                fusion_index = int(row['fusion_index'])
                prior_pos_error = torch.linalg.vector_norm(
                    fusion_truth_position_t[fusion_index] - nav_state.position_ecef_m
                )
                max_prior_position_error_m = max(max_prior_position_error_m, float(prior_pos_error.detach().cpu()))

                measurements = row['fixed_measurements']
                if not measurements:
                    skipped_no_measurements += 1
                    continue
                if tuple(m.sat_id for m in measurements) != tuple(row['sat_ids']):
                    raise RuntimeError('fixed Data01 observation ordering changed during rollout')
                if not np.array_equal(
                    np.asarray([m.pseudorange_m for m in measurements], dtype=float),
                    np.asarray(row['fixed_y_m'], dtype=float),
                ):
                    raise RuntimeError('fixed Data01 pseudorange vector changed during rollout')

                measurements = tuple(retain_clock_observable_measurements(measurements))
                variances = np.asarray([float(m.sigma_code_m) ** 2 for m in measurements], dtype=float)
                if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
                    raise ValueError('training pseudorange variances must be finite and positive')
                prepared = (
                    tuple(m.sat_id for m in measurements),
                    torch.as_tensor(np.stack([m.satellite_position_reception_ecef_m for m in measurements]), dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.pseudorange_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.satellite_clock_bias_s for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.ionosphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.troposphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor(_clock_projector(measurements, variances), dtype=torch.float64, device=DEVICE),
                )
                innovation_now, current_sat_ids = _measurement_innovation(nav_state, prepared)
                if innovation_now.numel():
                    max_innovation_abs_m = max(max_innovation_abs_m, float(torch.max(torch.abs(innovation_now)).detach().cpu()))
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _network_inputs(
                    previous_context, current_sat_ids, innovation_now, feature_accel, feature_gyro
                )
                output = model(
                    fixed_nn.float().unsqueeze(0),
                    obs_nn.float().unsqueeze(0),
                    mask_nn.unsqueeze(0),
                    channel_nn.unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
                recurrent_state = output.recurrent_state
                correction = fig8_state_update(output, innovation_nn.float().unsqueeze(0))[0].double()
                valid_gain = output.kalman_gain[0, :, :len(current_sat_ids)]
                max_gain_fro_norm = max(max_gain_fro_norm, float(torch.linalg.matrix_norm(valid_gain).detach().cpu()))
                max_position_correction_norm_m = max(max_position_correction_norm_m, float(torch.linalg.vector_norm(correction[0:3]).detach().cpu()))
                max_velocity_correction_norm_mps = max(max_velocity_correction_norm_mps, float(torch.linalg.vector_norm(correction[3:6]).detach().cpu()))
                max_attitude_correction_norm_rad = max(max_attitude_correction_norm_rad, float(torch.linalg.vector_norm(correction[6:9]).detach().cpu()))
                if not bool(torch.all(torch.isfinite(correction)).detach().cpu()):
                    raise FloatingPointError(f'non-finite learned correction at Data01 sample {sample_index}')

                if sample_index >= metric_start:
                    relative_attitude = fusion_truth_dcm_t[fusion_index].reshape(3, 3) @ nav_state.body_to_ecef_dcm.reshape(3, 3).T
                    target_state = torch.cat((
                        fusion_truth_position_t[fusion_index] - nav_state.position_ecef_m,
                        fusion_truth_velocity_t[fusion_index] - nav_state.velocity_ecef_mps,
                        _torch_so3_log(relative_attitude) / ATTITUDE_FEEDBACK_SIGN,
                    ))
                    state_error = target_state - correction[:SUPERVISED_STATE_DIM]
                    state_loss = torch.sum(state_error ** 2)
                    if not bool(torch.isfinite(state_loss).detach().cpu()):
                        raise FloatingPointError(
                            f'non-finite Yan Eq.(30) loss at sample {sample_index}; '
                            f'maxInnovation={max_innovation_abs_m:.6g}, maxGain={max_gain_fro_norm:.6g}'
                        )
                    state_losses.append(state_loss)
                    state_sum += float(state_loss.detach().cpu())
                    position_sum += float(torch.sum(state_error[:3] ** 2).detach().cpu())
                    metric_count += 1

                correction = correction.reshape(INS_STATE_DIM)
                nav_state = TorchNavigationState(
                    nav_state.position_ecef_m + correction[0:3],
                    nav_state.velocity_ecef_mps + correction[3:6],
                    _torch_so3_exponential(ATTITUDE_FEEDBACK_SIGN * correction[6:9]) @ nav_state.body_to_ecef_dcm,
                )
                posterior_residual, posterior_sat_ids = _measurement_innovation(nav_state, prepared)
                if posterior_sat_ids != current_sat_ids:
                    raise RuntimeError('satellite ordering changed inside one fusion update')
                x_post = correction
                state_residual = x_post - previous_context['x_post']
                previous_context = {
                    'sat_ids': tuple(current_sat_ids),
                    'residual': posterior_residual,
                    'x_pred': zero_error_state_t,
                    'x_post': x_post,
                    'state_innovation': x_post,
                    'state_residual': state_residual,
                    'accel': feature_accel.reshape(3),
                    'gyro': feature_gyro.reshape(3),
                }

        if metric_count <= 0:
            raise RuntimeError('Data01 rollout produced no supervised samples')
        mean_loss = torch.stack(state_losses).mean()
        if training:
            if not bool(torch.isfinite(mean_loss).detach().cpu()):
                raise FloatingPointError('non-finite training loss before backward')
            mean_loss.backward()
        if training:
            carried = {
                'nav': TorchNavigationState(nav_state.position_ecef_m.detach(), nav_state.velocity_ecef_mps.detach(), nav_state.body_to_ecef_dcm.detach()),
                'context': {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in previous_context.items()},
                'feature_accel': feature_accel.detach(),
                'feature_gyro': feature_gyro.detach(),
                'recurrent_state': None if recurrent_state is None else tuple(value.detach() for value in recurrent_state),
            }
        else:
            carried = None
        return {
            'eq30': float(state_sum / metric_count),
            'trajectory_eq30': float(mean_loss.detach().cpu()),
            'position_rmse_m': math.sqrt(position_sum / metric_count),
            'state_sum': float(state_sum),
            'position_sum': float(position_sum),
            'learned_updates': int(metric_count),
            'skipped_no_measurements': int(skipped_no_measurements),
            'max_prior_position_error_m': float(max_prior_position_error_m),
            'max_innovation_abs_m': float(max_innovation_abs_m),
            'max_gain_fro_norm': float(max_gain_fro_norm),
            'max_position_correction_norm_m': float(max_position_correction_norm_m),
            'max_velocity_correction_norm_mps': float(max_velocity_correction_norm_mps),
            'max_attitude_correction_norm_rad': float(max_attitude_correction_norm_rad),
            'rollout_state': carried,
        }

    def _train_phase(phase: str, optimizer: torch.optim.Optimizer, epoch: int):
        """One chronological pass; alternate blocks and detach only at TBPTT window boundaries."""
        if phase not in {'filter', 'representation'}:
            raise ValueError("phase must be 'filter' or 'representation'")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if phase == 'filter':
            model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
            for module in filter_modules:
                module.train()
            active_parameters = filter_parameters
        else:
            # The frozen LSTM stays in train mode so cuDNN can backpropagate to
            # its input; its dropout is disabled to keep the frozen map deterministic.
            for module in representation_modules:
                module.train()
            model.lstm.train()
            model.lstm.lstm.dropout = 0.0
            active_parameters = representation_parameters
        for parameter in active_parameters:
            parameter.requires_grad_(True)
        carried = None
        total_state_sum = 0.0
        total_position_sum = 0.0
        total_count = 0
        total_skipped = 0
        objective_values = []
        gradient_before = []
        gradient_after = []
        clipped_steps = 0
        optimizer_steps = 0
        maxima = {
            'max_prior_position_error_m': 0.0,
            'max_innovation_abs_m': 0.0,
            'max_gain_fro_norm': 0.0,
            'max_position_correction_norm_m': 0.0,
            'max_velocity_correction_norm_mps': 0.0,
            'max_attitude_correction_norm_rad': 0.0,
        }
        with torch.no_grad():
            gain_l2_before = math.sqrt(sum(float(torch.sum(p.detach().double() ** 2).cpu()) for p in model.gain_head.parameters()))

        for window_start in range(0, training_sample_count, OPTIMIZER_WINDOW_SIZE):
            window_stop = min(window_start + OPTIMIZER_WINDOW_SIZE, training_sample_count)
            optimizer.zero_grad(set_to_none=True)
            metrics = _rollout(
                window_start,
                window_stop,
                phase=phase,
                initial_state=carried,
            )
            carried = metrics['rollout_state']
            regularization = (GAMMA_L2 * sum(torch.sum(p.double() * p.double()) for p in all_network_parameters) if GAMMA_L2 else torch.zeros((), dtype=torch.float64, device=DEVICE))
            if not bool(torch.isfinite(regularization).detach().cpu()):
                raise FloatingPointError(f'non-finite Eq.(32) regularization in {phase} phase')
            if GAMMA_L2:
                regularization.backward()
            active_grads = [p for p in active_parameters if p.grad is not None]
            if active_grads:
                total_sq = 0.0
                for parameter in active_grads:
                    if not bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu()):
                        raise FloatingPointError('non-finite gradient before clipping')
                    grad64 = parameter.grad.detach().double()
                    total_sq += float(torch.sum(grad64 * grad64).cpu())
                before = math.sqrt(total_sq)
                if not math.isfinite(before):
                    raise FloatingPointError('non-finite global gradient norm')
                if before > GRADIENT_CLIP_NORM:
                    scale = GRADIENT_CLIP_NORM / max(before, 1e-300)
                    with torch.no_grad():
                        for parameter in active_grads:
                            parameter.grad.mul_(scale)
                after = math.sqrt(sum(float(torch.sum(p.grad.detach().double() ** 2).cpu()) for p in active_grads))
                if not math.isfinite(after):
                    raise FloatingPointError('non-finite clipped gradient norm')
            else:
                before = after = 0.0
            gradient_before.append(before)
            gradient_after.append(after)
            if before > GRADIENT_CLIP_NORM:
                clipped_steps += 1
            objective = metrics['trajectory_eq30'] + float(regularization.detach().cpu())
            if not math.isfinite(objective):
                raise FloatingPointError(f'non-finite {phase} objective')
            optimizer.step()
            if not all(bool(torch.all(torch.isfinite(p)).detach().cpu()) for p in model.parameters()):
                raise FloatingPointError(f'non-finite model parameter after {phase} optimizer step')
            optimizer_steps += 1
            objective_values.append(objective)
            total_state_sum += metrics['state_sum']
            total_position_sum += metrics['position_sum']
            total_count += metrics['learned_updates']
            total_skipped += metrics['skipped_no_measurements']
            for key in maxima:
                maxima[key] = max(maxima[key], metrics[key])

        if total_count <= 0 or optimizer_steps <= 0:
            raise RuntimeError(f'{phase} phase produced no training updates')
        with torch.no_grad():
            gain_l2_after = math.sqrt(sum(float(torch.sum(p.detach().double() ** 2).cpu()) for p in model.gain_head.parameters()))
        out = {
            'epoch': int(epoch),
            'eq30': float(total_state_sum / total_count),
            'trajectory_mean_eq30': float(total_state_sum / total_count),
            'position_rmse_m': math.sqrt(total_position_sum / total_count),
            'objective': float(np.mean(objective_values)),
            'learned_updates': int(total_count),
            'skipped_no_measurements': int(total_skipped),
            'optimizer_steps': int(optimizer_steps),
            'gradient_l2_norm': float(np.mean(gradient_before)),
            'gradient_l2_norm_max': float(np.max(gradient_before)),
            'gradient_l2_norm_after_clip': float(np.mean(gradient_after)),
            'gradient_l2_norm_after_clip_max': float(np.max(gradient_after)),
            'gradient_clip_max_norm': float(GRADIENT_CLIP_NORM),
            'gradient_clipped_optimizer_steps': int(clipped_steps),
            'gain_head_parameter_l2_before': float(gain_l2_before),
            'gain_head_parameter_l2_after': float(gain_l2_after),
            'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE),
            'state_initializations': 1,
            'truth_restarts': 0,
            'detach_policy': 'navigation_context_and_LSTM_memory_only_at_optimizer_window_boundary',
        }
        out.update(maxima)
        return out

    def _evaluate_recursive(metric_start: int, metric_stop: int):
        """Causal recursive evaluation; validation is reached through its Data01 prefix."""
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
        metrics = _rollout(0, metric_stop, phase=None, loss_start_sample=metric_start)
        metrics['objective'] = metrics['eq30']
        return metrics

    def _safe_monitor(scope: str, metric_start: int, metric_stop: int):
        try:
            metrics = _evaluate_recursive(metric_start, metric_stop)
        except FloatingPointError as error:
            message = str(error)
            print(f'WARNING: {scope} diverged: {message}')
            return {
                'eq30': float('inf'),
                'position_rmse_m': float('inf'),
                'objective': float('inf'),
                'learned_updates': 0,
                'diverged': True,
                'divergence_reason': message,
            }
        metrics['diverged'] = False
        metrics['divergence_reason'] = None
        return metrics

    def _evaluate_teacher_forced_eq30():
        """Yan X/M/Y sequence metric retained only as a nonrecursive diagnostic."""
        model.eval()
        model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
        state_sum = 0.0
        position_sum = 0.0
        recurrent_state = None
        with torch.inference_mode():
            for sample_index in range(training_sample_count):
                output = model(
                    torch.as_tensor(fixed_normalized[sample_index], dtype=torch.float32, device=DEVICE).unsqueeze(0),
                    torch.as_tensor(observations_normalized[sample_index], dtype=torch.float32, device=DEVICE).unsqueeze(0),
                    torch.as_tensor(satellite_masks[sample_index], dtype=torch.bool, device=DEVICE).unsqueeze(0),
                    torch.as_tensor(channel_masks[sample_index], dtype=torch.bool, device=DEVICE).unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
                recurrent_state = output.recurrent_state
                correction = fig8_state_update(
                    output,
                    torch.as_tensor(innovation_padded[sample_index], dtype=torch.float32, device=DEVICE).unsqueeze(0),
                )[0].double()
                residual = target_base_t[sample_index] - correction[:SUPERVISED_STATE_DIM]
                state_sum += float(torch.sum(residual ** 2).cpu())
                position_sum += float(torch.sum(residual[:3] ** 2).cpu())
        return {
            'eq30': state_sum / training_sample_count,
            'position_rmse_m': math.sqrt(position_sum / training_sample_count),
        }

    # Exact-zero KG initialization is itself the deterministic fallback model.
    if not torch.count_nonzero(model.gain_head.weight).item() == 0:
        raise RuntimeError('gain head must be exactly zero at initialization')
    if model.gain_head.bias is not None and torch.count_nonzero(model.gain_head.bias).item() != 0:
        raise RuntimeError('gain-head bias must be exactly zero at initialization')

    pretraining_zero_gain_metrics = _safe_monitor(
        'pre-training zero-gain Data01 training rollout', 0, training_sample_count
    )
    pretraining_validation_metrics = _safe_monitor(
        'pre-training zero-gain Data01 validation rollout', validation_start_sample, data01_total_sample_count
    )
    if pretraining_zero_gain_metrics['diverged'] or pretraining_validation_metrics['diverged']:
        raise RuntimeError('zero-gain recursive baseline must be finite before training')
    if abs(float(pretraining_zero_gain_metrics['max_gain_fro_norm'])) > 1e-12:
        raise RuntimeError('zero-gain initialization produced a nonzero Kalman gain')
    if abs(float(pretraining_zero_gain_metrics['max_position_correction_norm_m'])) > 1e-9:
        raise RuntimeError('zero-gain initialization produced a nonzero position correction')

    print(
        f"pre-training zero-K train: Eq30={pretraining_zero_gain_metrics['eq30']:.6g}, "
        f"posRMSE={pretraining_zero_gain_metrics['position_rmse_m']:.3f} m; "
        f"causal validation: Eq30={pretraining_validation_metrics['eq30']:.6g}, "
        f"posRMSE={pretraining_validation_metrics['position_rmse_m']:.3f} m"
    )

    # Initial stable model is checkpoint-eligible. This prevents a finite but
    # catastrophically worse learned rollout from replacing the safe baseline.
    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_selection_eq30 = float(pretraining_validation_metrics['eq30'])
    best_selection_rmse_m = float(pretraining_validation_metrics['position_rmse_m'])
    best_selection_stage = 'initial_zero_gain_causal_validation_baseline'
    best_selection_epoch = 0
    training_history.append({
        'stage': best_selection_stage,
        'global_training_epoch': 0,
        'eligible_for_checkpoint_selection': True,
        'gain_head_initialization': 'exact_zero_weight_and_bias',
        'Data01_train_recursive_eq30': float(pretraining_zero_gain_metrics['eq30']),
        'Data01_train_recursive_position_rmse_m': float(pretraining_zero_gain_metrics['position_rmse_m']),
        'Data01_validation_recursive_eq30': float(pretraining_validation_metrics['eq30']),
        'Data01_validation_recursive_position_rmse_m': float(pretraining_validation_metrics['position_rmse_m']),
        'truth_restarts': 0,
    })

    print('\n=== DATA01 CONTINUOUS RECURSIVE TRAINING ===')
    final_training_metrics = pretraining_zero_gain_metrics
    epochs_ran = 0
    epochs_without_improvement = 0
    for epoch in range(1, TRAINING_EPOCHS + 1):
        try:
            filter_phase = _train_phase('filter', filter_optimizer, epoch)
            representation_phase = _train_phase('representation', representation_optimizer, epoch)
        except FloatingPointError as error:
            print(f'WARNING: training stopped at epoch {epoch} after numerical divergence: {error}')
            model.load_state_dict({key: value.to(DEVICE) for key, value in best_model_state.items()})
            break

        epochs_ran = epoch
        train_recursive = _safe_monitor(
            f'epoch {epoch} Data01 training monitor', 0, training_sample_count
        )
        validation_recursive = _safe_monitor(
            f'epoch {epoch} Data01 validation monitor', validation_start_sample, data01_total_sample_count
        )
        validation_eq30 = float(validation_recursive['eq30'])
        validation_rmse = float(validation_recursive['position_rmse_m'])
        improved = (
            math.isfinite(validation_eq30)
            and math.isfinite(validation_rmse)
            and validation_eq30 < best_selection_eq30
        )
        if improved:
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_selection_eq30 = validation_eq30
            best_selection_rmse_m = validation_rmse
            best_selection_stage = 'continuous_recursive_tbptt_alternating'
            best_selection_epoch = int(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        final_training_metrics = representation_phase
        training_history.append({
            'stage': 'continuous_recursive_tbptt_alternating',
            'epoch': int(epoch),
            'global_training_epoch': int(epoch),
            'training_sequence_length': int(training_sample_count),
            'training_sequence_count': 1,
            'state_initialization': 'single_causal_classical_posterior_context_at_sequence_start',
            'truth_restarts': 0,
            'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE),
            'filter_phase_eq30': float(filter_phase['eq30']),
            'filter_phase_position_rmse_m': float(filter_phase['position_rmse_m']),
            'filter_phase_optimizer_steps': int(filter_phase['optimizer_steps']),
            'filter_phase_gradient_l2_norm': float(filter_phase['gradient_l2_norm']),
            'filter_phase_gradient_l2_norm_after_clip': float(filter_phase['gradient_l2_norm_after_clip']),
            'filter_phase_gain_head_l2_before': float(filter_phase['gain_head_parameter_l2_before']),
            'filter_phase_gain_head_l2_after': float(filter_phase['gain_head_parameter_l2_after']),
            'filter_phase_max_gain_fro_norm': float(filter_phase['max_gain_fro_norm']),
            'representation_phase_eq30': float(representation_phase['eq30']),
            'representation_phase_position_rmse_m': float(representation_phase['position_rmse_m']),
            'representation_phase_optimizer_steps': int(representation_phase['optimizer_steps']),
            'representation_phase_gradient_l2_norm': float(representation_phase['gradient_l2_norm']),
            'representation_phase_gradient_l2_norm_after_clip': float(representation_phase['gradient_l2_norm_after_clip']),
            'Data01_train_recursive_eq30': float(train_recursive['eq30']),
            'Data01_train_recursive_position_rmse_m': float(train_recursive['position_rmse_m']),
            'Data01_validation_recursive_eq30': float(validation_recursive['eq30']),
            'Data01_validation_recursive_position_rmse_m': float(validation_recursive['position_rmse_m']),
            'Data01_train_recursive_diverged': bool(train_recursive['diverged']),
            'Data01_validation_recursive_diverged': bool(validation_recursive['diverged']),
            'Data01_validation_divergence_reason': validation_recursive['divergence_reason'],
            'best_Data01_validation_recursive_eq30_so_far': float(best_selection_eq30),
            'best_Data01_validation_recursive_position_rmse_m_so_far': float(best_selection_rmse_m),
            'filter_block_learning_rate': float(filter_optimizer.param_groups[0]['lr']),
            'representation_block_learning_rate': float(representation_optimizer.param_groups[0]['lr']),
            'checkpoint_improved': bool(improved),
        })
        print(
            f"epoch {epoch:03d}/{TRAINING_EPOCHS}: "
            f"theta_RMSE={filter_phase['position_rmse_m']:.3f}m, "
            f"theta_grad={filter_phase['gradient_l2_norm']:.3g}->{filter_phase['gradient_l2_norm_after_clip']:.3g}, "
            f"theta_Kparam={filter_phase['gain_head_parameter_l2_before']:.3g}->{filter_phase['gain_head_parameter_l2_after']:.3g}; "
            f"psi_RMSE={representation_phase['position_rmse_m']:.3f}m; "
            f"train_recursive_RMSE={train_recursive['position_rmse_m']:.3f}m; "
            f"val_recursive_RMSE={validation_recursive['position_rmse_m']:.3f}m; "
            f"best_val_RMSE={best_selection_rmse_m:.3f}m"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'early stopping: no causal validation improvement for {EARLY_STOPPING_PATIENCE} epochs')
            break

    model.load_state_dict({key: value.to(DEVICE) for key, value in best_model_state.items()})
    print(
        f"restored best causal Data01 checkpoint: stage={best_selection_stage}, "
        f"epoch={best_selection_epoch}, val_Eq30={best_selection_eq30:.6g}, "
        f"val_posRMSE={best_selection_rmse_m:.3f} m"
    )
    final_training_metrics = _evaluate_recursive(0, training_sample_count)
    final_validation_metrics = _evaluate_recursive(validation_start_sample, data01_total_sample_count)
    final_teacher_forced_metrics = _evaluate_teacher_forced_eq30()
    zero_gain_rmse = float(pretraining_zero_gain_metrics['position_rmse_m'])
    trained_rmse = float(final_training_metrics['position_rmse_m'])
    validation_rmse = float(final_validation_metrics['position_rmse_m'])

    training_performance_summary = {
        'split': {
            'mode': 'chronological_holdout_causal_continuation',
            'Data01_total_samples': int(data01_total_sample_count),
            'training_samples': int(training_sample_count),
            'validation_samples': int(validation_sample_count),
            'validation_fraction': float(VALIDATION_FRACTION),
            'Data02_used_for_validation': False,
        },
        'initial_zero_gain': {
            'train_eq30': float(pretraining_zero_gain_metrics['eq30']),
            'train_position_rmse_m': zero_gain_rmse,
            'validation_eq30': float(pretraining_validation_metrics['eq30']),
            'validation_position_rmse_m': float(pretraining_validation_metrics['position_rmse_m']),
        },
        'trained_closed_loop_train': {
            'eq30': float(final_training_metrics['eq30']),
            'position_rmse_m': trained_rmse,
            'rmse_ratio_vs_zero_gain': trained_rmse / max(zero_gain_rmse, 1e-12),
            'max_innovation_abs_m': float(final_training_metrics['max_innovation_abs_m']),
            'max_gain_fro_norm': float(final_training_metrics['max_gain_fro_norm']),
            'max_position_correction_norm_m': float(final_training_metrics['max_position_correction_norm_m']),
        },
        'validation_closed_loop': {
            'eq30': float(final_validation_metrics['eq30']),
            'position_rmse_m': validation_rmse,
            'validation_to_train_rmse_ratio': validation_rmse / max(trained_rmse, 1e-12),
            'max_innovation_abs_m': float(final_validation_metrics['max_innovation_abs_m']),
            'max_gain_fro_norm': float(final_validation_metrics['max_gain_fro_norm']),
            'max_position_correction_norm_m': float(final_validation_metrics['max_position_correction_norm_m']),
        },
        'teacher_forced_train_diagnostic': final_teacher_forced_metrics,
        'selected_checkpoint': {
            'stage': best_selection_stage,
            'epoch': int(best_selection_epoch),
            'selection_metric': 'causal_recursive_Data01_validation_Eq30',
            'Data01_validation_recursive_eq30': float(best_selection_eq30),
            'Data01_validation_recursive_position_rmse_m': float(best_selection_rmse_m),
        },
    }
    print('Data01 train/validation performance:', json.dumps(training_performance_summary, indent=2))
    model.eval()
    model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
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
    torch.save({
        'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        'checkpoint_schema': 'v103_9_state_masked_cla_continuous_tbptt',
        'nmax': int(network_nmax),
        'training_observed_nmax': int(nmax),
        'data01_sequence_nmax': int(data01_sequence_nmax),
        'data02_sequence_nmax': None,
        'nmax_source': 'Data01_actual_neural_observation_max_only',
        'state_order': '[delta_p,delta_v,delta_theta]',
        'kalman_gain_state_dimension': INS_STATE_DIM,
        'direct_state_label_dimension': SUPERVISED_STATE_DIM,
        'feature_normalization_mode': FEATURE_NORMALIZATION_MODE,
        'feature_l2_eps': float(FEATURE_L2_EPS),
        'normalizer_source': normalizer_source,
        'architecture': 'Yan_masked_CNN24_k3_LSTM5x64_attention_FC_KG',
        'alternating_partition': 'psi=masked_conv;theta=masked_lstm+attention+gain_head',
        'alternating_partition_status': 'project_completion_Yan_cites_alternating_optimization_but_does_not_publish_module_split',
        'latent_kalmannet_warm_start': 'not_applied_no_published_supervised_target_for_intermediate_Yan_CNN_features',
        'training_stage_selected': selected_training_stage,
        'training_epochs_ran': int(epochs_ran),
        'selected_epoch': int(best_epoch),
        'training_sequence_length': int(training_sample_count),
        'validation_sequence_length': int(validation_sample_count),
        'data01_total_sequence_length': int(data01_total_sample_count),
        'validation_fraction': float(VALIDATION_FRACTION),
        'validation_protocol': 'chronological_tail_reached_by_causal_full_prefix_rollout_no_truth_reset',
        'training_sequence_count': 1,
        'state_initialization': 'single_causal_classical_posterior_context_at_sequence_start',
        'periodic_truth_restarts': 0,
        'optimizer_step_inside_sequence': True,
        'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE),
        'tbptt_detach_policy': 'detach_navigation_context_and_LSTM_memory_only_at_optimizer_window_boundary',
        'cross_fusion_navigation_gradient': 'retained_inside_each_optimizer_window',
        'cross_fusion_neural_gradient': 'retained_inside_each_optimizer_window',
        'gain_head_initialization': 'exact_zero_weight_and_bias',
        'initial_checkpoint_eligible': True,
        'learning_rate': float(LEARNING_RATE),
        'gradient_clip_norm': float(GRADIENT_CLIP_NORM),
        'early_stopping_patience': int(EARLY_STOPPING_PATIENCE),
        'synthetic_prior_augmentation': False,
        'lstm_sequence_state': 'Yan_Eq25_per_position_hc_carried_across_fusion_epochs_k',
        'final_train_recursive_eq30_state_loss': float(final_training_metrics['eq30']),
        'final_train_recursive_position_rmse_m': float(final_training_metrics['position_rmse_m']),
        'final_validation_recursive_eq30_state_loss': float(final_validation_metrics['eq30']),
        'final_validation_recursive_position_rmse_m': float(final_validation_metrics['position_rmse_m']),
        'selected_validation_recursive_eq30': best_val_state_loss,
        'selected_validation_recursive_position_rmse_m': best_val_position_rmse_m,
        'final_clean_teacher_forced_eq30': float(final_teacher_forced_metrics['eq30']),
        'final_clean_teacher_forced_position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m']),
        'state_target_scaling': 'none_raw_physical_units_per_Yan_Eq30',
        'measurement_mode': 'pseudorange_only',
    }, OUTPUT_DIR / 'best_model.pt')
    (OUTPUT_DIR / 'history.json').write_text(json.dumps(training_history, indent=2), encoding='utf-8')

    # --- TRAIN / VALIDATION EPOCH-LOSS CURVE ---
    # Use the same recursive Eq. (30) state-MSE metric for both Data01 subsets.
    # This is the validation metric already used for checkpoint selection, so
    # train and validation curves are directly comparable across continuous-training epochs.
    epoch_loss_rows = [
        row for row in training_history
        if 'global_training_epoch' in row
        and 'Data01_train_recursive_eq30' in row
        and 'Data01_validation_recursive_eq30' in row
    ]
    if epoch_loss_rows:
        epoch_numbers = np.asarray(
            [row['global_training_epoch'] for row in epoch_loss_rows], dtype=int
        )
        train_epoch_loss = np.asarray(
            [row['Data01_train_recursive_eq30'] for row in epoch_loss_rows], dtype=float
        )
        validation_epoch_loss = np.asarray(
            [row['Data01_validation_recursive_eq30'] for row in epoch_loss_rows], dtype=float
        )

        # Matplotlib cannot meaningfully draw inf values. Keep them in history.json,
        # but leave gaps in the figure at non-finite epochs.
        train_plot_loss = np.where(np.isfinite(train_epoch_loss), train_epoch_loss, np.nan)
        validation_plot_loss = np.where(np.isfinite(validation_epoch_loss), validation_epoch_loss, np.nan)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(epoch_numbers, train_plot_loss, marker='o', linewidth=1.8, label='Train recursive Eq. (30) loss')
        ax.plot(epoch_numbers, validation_plot_loss, marker='o', linewidth=1.8, label='Validation recursive Eq. (30) loss')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel('State MSE loss (Eq. 30)')
        ax.set_title('Data01 Train vs Validation Epoch Loss')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        finite_positive = np.concatenate([
            train_plot_loss[np.isfinite(train_plot_loss) & (train_plot_loss > 0.0)],
            validation_plot_loss[np.isfinite(validation_plot_loss) & (validation_plot_loss > 0.0)],
        ])
        if finite_positive.size:
            dynamic_range = float(np.max(finite_positive) / max(np.min(finite_positive), 1e-300))
            if dynamic_range >= 100.0:
                ax.set_yscale('log')

        fig.tight_layout()
        epoch_loss_path = OUTPUT_DIR / 'epoch_loss_train_validation.png'
        fig.savefig(epoch_loss_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'Train/validation epoch-loss plot saved to: {epoch_loss_path}')
    else:
        print('WARNING: no per-epoch recursive train/validation losses were found; epoch-loss plot was not created.')
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
        diag_last_gyro = np.asarray(imr_angular_rate_body_radps[0], dtype=float).reshape(3)
        diag_last_accel = np.asarray(imr_acceleration_body_mps2[0], dtype=float).reshape(3)
        diag_feature_gyro, diag_feature_accel = (diag_last_gyro, diag_last_accel)
        van_loan_stats_before_data01_diag = dict(_VAN_LOAN_STATS)
        try:
            for event_kind, event_start, event_end, event_index in build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True):
                if event_kind == 'imu':
                    imu_index = event_index
                    dt = event_end - event_start
                    diag_last_gyro = np.asarray(imr_angular_rate_body_radps[imu_index], dtype=float).reshape(3)
                    diag_last_accel = np.asarray(imr_acceleration_body_mps2[imu_index], dtype=float).reshape(3)
                    diag_feature_gyro, diag_feature_accel = (diag_last_gyro, diag_last_accel)
                    diag_nav = mechanize_ecef(diag_nav, diag_last_gyro, diag_last_accel, dt)
                    diag_F = build_error_state_dynamics(diag_nav, diag_last_accel)
                    diag_Phi, diag_Qd = discretize_process_noise_van_loan(diag_F, Qc, dt)
                    diag_P = diag_Phi @ diag_P @ diag_Phi.T + diag_Qd
                    diag_P = 0.5 * (diag_P + diag_P.T)
                    continue
                fusion_index = event_index
                t = event_start
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
        learned_gate = np.asarray(diag_error_3d, dtype=float).reshape(-1)
        classical_gate = np.asarray(diag_classical_error_3d, dtype=float).reshape(-1)
        if learned_gate.shape != classical_gate.shape or learned_gate.size == 0:
            data01_acceptance_gate = {'passed': False, 'reason': 'missing_or_misaligned_same_epoch_errors', 'epochs_compared': 0, 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
        else:
            finite_gate = np.isfinite(learned_gate) & np.isfinite(classical_gate)
            if not np.all(finite_gate):
                data01_acceptance_gate = {'passed': False, 'reason': 'nonfinite_or_missing_same_epoch_error', 'epochs_compared': int(np.count_nonzero(finite_gate)), 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
            else:
                learned_gate_rmse = float(np.sqrt(np.mean(learned_gate ** 2)))
                classical_gate_rmse = float(np.sqrt(np.mean(classical_gate ** 2)))
                gate_passed = bool(learned_gate_rmse <= classical_gate_rmse)
                data01_acceptance_gate = {'passed': gate_passed, 'reason': 'learned_rmse_not_worse_than_classical' if gate_passed else 'learned_rmse_worse_than_classical', 'epochs_compared': int(learned_gate.size), 'learned_rmse_3d_m': learned_gate_rmse, 'classical_rmse_3d_m': classical_gate_rmse, 'criterion': 'learned_same_epoch_rmse_3d_m <= classical_same_epoch_rmse_3d_m', 'paper_basis': 'Yan_Fig9a_qualitative_learned_error_below_traditional_KF', 'absolute_threshold_used': False}
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
    test_root = Path(test_dataset_dir)
    test_readme_path = test_root / 'README.xml'
    if not test_readme_path.exists():
        raise FileNotFoundError(f'Missing README.xml in dataset: {test_root}')
    test_readme_root = ET.fromstring(test_readme_path.read_text(encoding='utf-8', errors='replace'))
    test_rover_xml = next((item for item in test_readme_root.findall('ROVE') if (item.findtext('ID') or '').strip() == '01'), None)
    if test_rover_xml is None:
        raise KeyError('Rover 01 not found in test README.xml')
    test_rover_imu_type = (test_rover_xml.findtext('SINS_IMUType') or '').strip()
    test_rover_lever_arm_vehicle_m = np.fromstring(test_rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ')
    test_rover_mounting_xyz_deg = np.fromstring(test_rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    test_rover_ground_truth_path = _unique_file(test_root, ('ROVE_GroundTruth.txt', 'ROVE_01_GroundTruth.txt', 'Rove_01_GroundTruth.txt'), 'rover 01 ground truth')
    test_imu_ground_truth_path = test_root / f'{test_rover_imu_type}_GroundTruth.txt'
    test_rinex_obs_path = _unique_file(test_root, ('ROVE.*O', 'ROVE.*o', 'ROVE_01.*O', 'ROVE_01.*o'), 'rover 01 RINEX observation')
    test_imr_path = test_root / f'{test_rover_imu_type}.imr'
    test_sp3_path = _unique_file(test_root, ('*.SP3', '*.sp3'), 'SP3 precise-orbit')
    test_clk_path = _unique_file(test_root, ('*.CLK', '*.clk'), 'RINEX clock')
    test_nav_path = _unique_file(test_root, ('brdm*.*p', 'brdm*.*P', 'brdm*.rnx', 'BRDM*.RNX'), 'broadcast navigation')
    test_required = [IMU_ERROR_MODEL_PATH, LEO_TLE_DIR, test_imu_ground_truth_path, test_imr_path]
    test_missing = [str(path) for path in test_required if not Path(path).exists()]
    if test_missing:
        raise FileNotFoundError('Missing test inputs:\n' + '\n'.join(test_missing))
    test_antenna_truth = load_ie_ground_truth(test_rover_ground_truth_path)
    test_imu_truth = load_ie_ground_truth(test_imu_ground_truth_path)
    test_imu_model_values = imu_model_values if test_rover_imu_type == rover_imu_type else None
    if test_imu_model_values is None:
        for match in re.finditer(r'IMU\s*\{(.*?)\}', imu_model_text, flags=re.S):
            block = match.group(1)
            type_match = re.search(r'IMU_Type\s*=\s*"([^"]+)"', block)
            if not type_match or type_match.group(1) != test_rover_imu_type:
                continue
            values = {}
            for key in imu_model_keys:
                value_match = re.search(rf'{key}\s*=\s*([^\r\n]+)', block)
                if value_match:
                    values[key] = np.fromstring(value_match.group(1), sep=' ', dtype=float)
            if len(values) == len(imu_model_keys):
                test_imu_model_values = values
                break
    if test_imu_model_values is None:
        raise KeyError(f'No complete IMU noise model for test IMU type {test_rover_imu_type!r}')
    test_rinex = RINEXObservationFile(test_rinex_obs_path)
    first_test_rinex_epoch = next(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    if first_test_rinex_epoch is None:
        raise ValueError('Test RINEX file contains no GPS/BDS epochs')
    test_imr_path, test_imr_data_rate_hz, test_imr_gyro_scale, test_imr_accel_scale, test_imr_time_tag_bias_ms, test_imr_record_dtype, test_imr_record_count, test_imr_record_bytes = _read_imr_layout(test_imr_path)
    if test_imr_record_count < 2:
        raise ValueError('Test IMR file contains fewer than two usable samples')
    test_imr_records = np.memmap(test_imr_path, dtype=test_imr_record_dtype, mode='r', offset=512, shape=(test_imr_record_count,))
    test_imr_tow_all = np.asarray(test_imr_records['tow'], dtype=np.float64).copy()
    del test_imr_records
    test_imr_tow_all[test_imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    test_imr_tow_all -= test_imr_time_tag_bias_ms * 1e-3
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
    test_imr_count = test_requested_stop - test_start
    test_imr_records = np.fromfile(test_imr_path, dtype=test_imr_record_dtype, count=test_imr_count, offset=512 + test_start * test_imr_record_bytes)
    test_imr_tow = test_imr_records['tow'].astype(np.float64, copy=True)
    test_imr_tow[test_imr_tow > GPS_WEEK_S] -= GPS_WEEK_S
    test_imr_tow -= test_imr_time_tag_bias_ms * 1e-3
    test_imr_angular_rate_body_radps = np.deg2rad(test_imr_records['counts'][:, :3].astype(float) * test_imr_gyro_scale * test_imr_data_rate_hz)
    test_imr_acceleration_body_mps2 = test_imr_records['counts'][:, 3:6].astype(float) * test_imr_accel_scale * test_imr_data_rate_hz
    test_imr_time = test_imr_time_all[test_start:test_requested_stop].copy()
    del test_imr_tow_all, test_imr_time_all
    if len(test_imr_tow) < 2:
        raise ValueError('Selected test IMR window contains fewer than two samples')
    test_query_time = np.concatenate(([test_imr_time[0]], test_fusion_time))
    test_antenna_position = interpolate_ground_truth(test_antenna_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    test_imu_position, test_imu_velocity, test_imu_heading, test_imu_pitch, test_imu_roll = interpolate_ground_truth(test_imu_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    test_fusion_antenna_truth_position = test_antenna_position[1:]
    test_C_b_e = body_to_ecef_from_ie_hpr(test_imu_position[0], test_imu_heading[0], test_imu_pitch[0], test_imu_roll[0], test_rover_mounting_xyz_deg)
    test_lever_arm_b_m = c_vehicle_to_body_zxy(*test_rover_mounting_xyz_deg) @ np.asarray(test_rover_lever_arm_vehicle_m, dtype=float).reshape(3)
    test_initial_antenna_from_imu = test_imu_position[0] + test_C_b_e @ test_lever_arm_b_m
    test_truth_reference_error_m = np.linalg.norm(test_antenna_position[0] - test_initial_antenna_from_imu)
    if test_truth_reference_error_m > 0.05:
        raise ValueError(f'Test ROVE/IMU ground-truth reference points are inconsistent with the lever arm: {test_truth_reference_error_m:.3f} m')
    test_initial_nav = NavigationState(test_imu_position[0].copy(), test_imu_velocity[0].copy(), test_C_b_e)
    test_initial_sigma = np.concatenate((test_imu_model_values['ISDV_Pos'], test_imu_model_values['ISDV_Vel'], test_imu_model_values['ISDV_Att'] * d2r))
    test_P0 = np.diag(test_initial_sigma ** 2)
    test_process_density = np.concatenate((test_imu_model_values['PNSD_Pos'], test_imu_model_values['PNSD_Vel'], test_imu_model_values['PNSD_Att'] * d2r))
    test_Qc = np.diag(test_process_density ** 2)
    test_ionosphere_coefficients = None
    if USE_IONOSPHERE:
        test_iono_header = {}
        with test_nav_path.open('r', encoding='ascii', errors='replace') as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'IONOSPHERIC CORR':
                    fields = line[:60].split()
                    if len(fields) >= 5:
                        test_iono_header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
        test_ionosphere_coefficients = {'G': (np.asarray(test_iono_header['GPSA']), np.asarray(test_iono_header['GPSB'])), 'C': (np.asarray(test_iono_header['BDSA']), np.asarray(test_iono_header['BDSB']))}
    test_gnss_preprocessor = GNSSPreprocessor(test_sp3_path, test_clk_path, min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=test_ionosphere_coefficients)
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
    data02_observed_nmax = 0
    data02_postfde_nmax_before_capacity = 0
    data02_capacity_selection_epochs = 0
    data02_capacity_excluded_measurements = 0
    last_gyro = np.asarray(test_imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(test_imr_acceleration_body_mps2[0], dtype=float).reshape(3)
    last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
    test_timeline = build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True)
    for event_kind, event_start, event_end, event_index in test_timeline:
        if event_kind == 'imu':
            imu_index = event_index
            dt = event_end - event_start
            last_gyro = np.asarray(test_imr_angular_rate_body_radps[imu_index], dtype=float).reshape(3)
            last_accel = np.asarray(test_imr_acceleration_body_mps2[imu_index], dtype=float).reshape(3)
            last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
            nav = mechanize_ecef(nav, last_gyro, last_accel, dt)
            F_error = build_error_state_dynamics(nav, last_accel)
            Phi, Qd = discretize_process_noise_van_loan(F_error, test_Qc, dt)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue
        fusion_index = event_index
        t = event_start
        epoch = test_gnss_epochs[fusion_index]
        if abs(epoch.time_gpst_s - t) > 1e-09:
            raise RuntimeError('test fusion timeline/epoch timestamp mismatch')
        gnss_measurements = test_gnss_preprocessor.prepare_epoch(epoch, nav, test_lever_arm_b_m)
        leo_measurements = test_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])
        measurements = retain_clock_observable_measurements(tuple(gnss_measurements) + tuple(leo_measurements))
        data02_observed_nmax = max(data02_observed_nmax, len(measurements))
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
            fde_result = FDE_DIA.decision(
                nav,
                P,
                measurements,
                test_lever_arm_b_m,
                FDE_SIGNIFICANCE_ALPHA,
            )
            FDE_DIA.accumulate_stats(test_fde_stats, fde_result)
            prefde_measurement_model = fde_result.tested_measurement_model
            measurements = fde_result.measurements
        else:
            fde_result = None
            prefde_measurement_model = build_measurement_model(
                nav,
                measurements,
                test_lever_arm_b_m,
            )

        # The neural architecture is frozen at the Data01-derived Nmax.
        # FDE sees the full independent-test measurement set first; only the
        # post-FDE neural input is capped if Nk exceeds the trained capacity.
        data02_postfde_nmax_before_capacity = max(
            data02_postfde_nmax_before_capacity,
            len(measurements),
        )
        pre_capacity_count = len(measurements)
        if pre_capacity_count > test_recurrent_capacity:
            measurements = select_measurements_to_network_capacity(
                measurements,
                test_recurrent_capacity,
            )
            data02_capacity_selection_epochs += 1
            data02_capacity_excluded_measurements += (
                pre_capacity_count - len(measurements)
            )

        measurement_model = build_measurement_model(
            nav,
            measurements,
            test_lever_arm_b_m,
        )
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
            raise RuntimeError(
                f'Data02 capacity-selection invariant failed: '
                f'count={n}, Data01_network_Nmax={test_recurrent_capacity}'
            )
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
        baseline_last_gyro = np.asarray(test_imr_angular_rate_body_radps[0], dtype=float).reshape(3)
        baseline_last_accel = np.asarray(test_imr_acceleration_body_mps2[0], dtype=float).reshape(3)
        for event_kind, event_start, event_end, event_index in build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
            if event_kind == 'imu':
                imu_index = event_index
                dt = event_end - event_start
                baseline_last_gyro = np.asarray(test_imr_angular_rate_body_radps[imu_index], dtype=float).reshape(3)
                baseline_last_accel = np.asarray(test_imr_acceleration_body_mps2[imu_index], dtype=float).reshape(3)
                baseline_nav = mechanize_ecef(baseline_nav, baseline_last_gyro, baseline_last_accel, dt)
                F_baseline = build_error_state_dynamics(baseline_nav, baseline_last_accel)
                Phi_baseline, Qd_baseline = discretize_process_noise_van_loan(F_baseline, test_Qc, dt)
                baseline_P = Phi_baseline @ baseline_P @ Phi_baseline.T + Qd_baseline
                baseline_P = 0.5 * (baseline_P + baseline_P.T)
                continue
            fusion_index = event_index
            t = event_start
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
    summary = {
        'paper_exact': False,
        'reference': 'Yan_et_al_IEEE_IoT_Journal_2026_Masked_KalmanNet',
        'intentional_project_differences': {
            'measurement_mode': 'pseudorange_only',
            'leo_orbit': 'TLE_SGP4_instead_of_STK_HPOP',
            'leo_clock': 'ideal_zero',
            'navigation_error_state': '9_state_[delta_p,delta_v,delta_theta]_instead_of_Yan_15_state',
            'imu_bias_states': False,
            'Fig8_eta': False,
        },
        'masked_cla': {
            'architecture': 'Yan_Eqs_10_to_29_masked_CNN_24_k3_LSTM_5x64_dropout_0.2_attention_KG_head',
            'Nmax': int(network_nmax),
            'Nmax_source': 'Data01_only',
            'lstm_recurrence_axis': 'fusion_epoch_k_with_parallel_padded_feature_positions_t',
            'feature_normalization': FEATURE_NORMALIZATION_MODE,
            'gain_head_initialization': 'exact_zero_weight_and_bias',
            'gain_head_initialization_reason': 'closed_loop_safe_zero_correction_baseline_for_raw_pseudorange_innovation;Yan_does_not_publish_FC_initialization',
        },
        'training': {
            'Data01_total_samples': int(data01_total_sample_count),
            'training_samples': int(training_sample_count),
            'validation_samples': int(validation_sample_count),
            'validation_fraction': float(VALIDATION_FRACTION),
            'sequence_protocol': 'one_continuous_chronological_Data01_trajectory_per_phase_no_periodic_truth_reset',
            'state_initializations_per_phase': 1,
            'truth_restarts_inside_phase': 0,
            'tbptt': {
                'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE),
                'detach_policy': 'detach_navigation_context_and_LSTM_memory_only_at_optimizer_window_boundary',
                'numerical_state_continuity_across_windows': True,
            },
            'alternating_optimization': {
                'enabled': True,
                'psi_representation': 'masked_CNN',
                'theta_filter': 'masked_LSTM_attention_gain_head',
                'order_each_epoch': 'theta_full_chronological_pass_then_psi_full_chronological_pass',
                'warm_start': 'not_used_no_published_intermediate_CNN_supervision_in_Yan',
                'paper_status': 'Yan_states_alternating_optimization_but_does_not_publish_this_module_split',
            },
            'optimizer': 'Adam',
            'learning_rate': float(LEARNING_RATE),
            'Yan_reported_initial_learning_rate': float(YAN_REPORTED_LEARNING_RATE),
            'gradient_clip_l2': float(GRADIENT_CLIP_NORM),
            'early_stopping_patience': int(EARLY_STOPPING_PATIENCE),
            'Eq32_gamma_l2': float(GAMMA_L2),
            'epochs_configured': int(TRAINING_EPOCHS),
            'epochs_ran': int(epochs_ran),
            'loss': 'Yan_Eq30_state_MSE_plus_Eq32_L2_regularization',
            'checkpoint': {
                'criterion': 'best_finite_causal_recursive_Data01_validation_Eq30',
                'initial_zero_gain_model_is_eligible_fallback': True,
                'selected_stage': selected_training_stage,
                'selected_epoch': int(best_epoch),
                'selected_validation_eq30': float(best_val_state_loss),
                'selected_validation_position_rmse_m': float(best_val_position_rmse_m),
            },
            'validation_protocol': 'chronological_tail_loss_after_causal_prefix_rollout_no_validation_boundary_reset',
            'final_train_recursive_eq30': float(final_training_metrics['eq30']),
            'final_train_recursive_position_rmse_m': float(final_training_metrics['position_rmse_m']),
            'final_validation_recursive_eq30': float(final_validation_metrics['eq30']),
            'final_validation_recursive_position_rmse_m': float(final_validation_metrics['position_rmse_m']),
            'teacher_forced_train_diagnostic': final_teacher_forced_metrics,
            'recursive_Data01_acceptance_gate': data01_acceptance_gate,
        },
        'test': {
            'dataset': str(test_dataset_dir),
            'independent_from_training': True,
            'epochs': int(len(online_rows)),
            'causal_warm_start': test_warm_start,
            'recursive_state_carried_without_periodic_truth_reset': True,
            'rmse_ned3d_m': [float(v) for v in rmse],
            'divergence_diagnostics': divergence_diagnostics,
            'classical_baseline_rmse_ned3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None,
            'fde_stats': test_fde_stats,
            'Data02_observed_Nmax': int(data02_observed_nmax),
            'Data02_postFDE_Nmax_before_capacity': int(data02_postfde_nmax_before_capacity),
            'capacity_selection_epochs': int(data02_capacity_selection_epochs),
            'capacity_excluded_measurements': int(data02_capacity_excluded_measurements),
        },
        'paper_components': {
            'FDE_enabled': bool(FDE_DIA_ON),
            'integrity_enabled': bool(INTEGRITY_ON),
            'FDE_uses_raw_innovation_before_learned_KG': bool(FDE_DIA_ON),
            'Data02_used_for_training_or_architecture': False,
        },
        'runtime_checks': {
            'van_loan_fast_calls': int(proposed_van_loan_stats['taylor_calls']),
            'van_loan_exact_calls': int(proposed_van_loan_stats['exact_fallback_calls']),
            'van_loan_max_validation_phi_abs': float(proposed_van_loan_stats['max_validation_phi_abs']),
            'van_loan_max_validation_qd_abs': float(proposed_van_loan_stats['max_validation_qd_abs']),
        },
    }
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    test_measurement_counts = np.asarray([row['n_total'] for row in online_rows], dtype=float)
    test_gnss_counts = np.asarray([row['n_gnss'] for row in online_rows], dtype=float)
    test_leo_counts = np.asarray([row['n_leo'] for row in online_rows], dtype=float)
    test_performance_summary = {'epochs': int(len(online_rows)), 'rmse_n_e_d_3d_m': [float(v) for v in rmse], 'error_3d_m': {'median': float(np.median(error_3d)), 'p95': float(np.percentile(error_3d, 95.0)), 'p99': float(np.percentile(error_3d, 99.0)), 'max': float(np.max(error_3d))}, 'measurements_used': {'min': int(np.min(test_measurement_counts)), 'median': float(np.median(test_measurement_counts)), 'max': int(np.max(test_measurement_counts)), 'gnss_median': float(np.median(test_gnss_counts)), 'leo_median': float(np.median(test_leo_counts))}, 'divergence': divergence_diagnostics, 'fde': test_fde_stats if FDE_DIA_ON else {'enabled': False}, 'integrity_median_hpl_vpl_m': [float(np.nanmedian(hpl)), float(np.nanmedian(vpl))] if INTEGRITY_ON else None, 'classical_baseline_rmse_n_e_d_3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None, 'network_capacity': {'Nmax_Data01_only': int(network_nmax), 'Data02_observed_Nmax': int(data02_observed_nmax), 'Data02_postFDE_Nmax_before_capacity': int(data02_postfde_nmax_before_capacity), 'epochs_capped': int(data02_capacity_selection_epochs), 'measurements_excluded': int(data02_capacity_excluded_measurements), 'selection_policy': 'smallest_sigma_code_m_then_sat_id_tie_break_preserve_original_order'}}
    print('\n=== DATA02 INDEPENDENT PERFORMANCE ===')
    print(json.dumps(test_performance_summary, indent=2))
