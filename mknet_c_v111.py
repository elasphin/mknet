from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import json
import math
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
from sgp4.api import SGP4_ERRORS, Satrec, WGS72
from sgp4.io import verify_checksum


EARTH_SEMI_MAJOR_AXIS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SEMI_MINOR_AXIS_M = EARTH_SEMI_MAJOR_AXIS_M * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_ROTATION_RATE_RADPS = 7.292115e-5
EARTH_GRAVITATIONAL_PARAMETER_M3PS2 = 3.986004418e14
SPEED_OF_LIGHT_MPS = 299792458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0
GPS_UTC_LEAP_SECONDS = 18.0 
KAGGLE_PROJECT_ROOT = Path('/kaggle/input/datasets/elasphin/mknet-project')
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
MAX_TRUTH_INTERPOLATION_GAP_S = 2.0
MIN_GNSS_ELEVATION_DEG = 5.0
LEO_MIN_ELEVATION_DEG = 10.0
LEO_SEED = 0
TLE_MAX_AGE_DAYS = 1.0
TLE_ALLOW_DEGRADED_EOP = False
TLE_ALLOW_NON_TLE_FILES = True
LEO_TX_EPSILON_POSITION_M = 1e-3
LEO_TX_MAX_ITERATIONS = 20
FEATURE_NORMALIZATION_MODE = 'online_group_l2'
FEATURE_L2_EPS = 1e-12
LEO_PREFILTER_GUARD_DEG = 0.5
LEO_URA_SIGMA_M = 1.5

try:
    from fde_dia_v2 import IntegrityMonitor
except ModuleNotFoundError:
    IntegrityMonitor = None



# Calendar to GPST.
def to_gpst(year: int, month: int, day: int, hour: int, minute: int, second: float, time_system: str='GPS') -> float:
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


# Anchor IMR TOW to GPS week.
def anchor_imr_gpst(tow_s: np.ndarray, anchor_time_gpst_s: float) -> np.ndarray:
    tow_s = np.asarray(tow_s, dtype=float).reshape(-1)
    if tow_s.size == 0:
        return tow_s.copy()
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
    return time_gpst_s


# Validate GPST axis.
def check_time_axis(name: str, time_gpst_s: np.ndarray) -> np.ndarray:
    time_gpst_s = np.asarray(time_gpst_s, dtype=float).reshape(-1)
    if time_gpst_s.size == 0 or not np.all(np.isfinite(time_gpst_s)) or (time_gpst_s.size > 1 and np.any(np.diff(time_gpst_s) <= 0.0)):
        raise ValueError(f'{name} time axis is empty, non-finite, or not strictly increasing')
    return time_gpst_s


# Ordered IMU/fusion timeline.
def fusion_timeline(imu_time_gpst_s: np.ndarray, fusion_time_gpst_s: np.ndarray, *, through_last_fusion: bool=True):
    imu_time = check_time_axis('IMU', imu_time_gpst_s)
    fusion_time = np.asarray(fusion_time_gpst_s, dtype=float).reshape(-1)
    if fusion_time.size == 0:
        return
    check_time_axis('fusion', fusion_time)
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


# ECEF to WGS-84 LLH.
def ecef_llh(position_ecef_m: np.ndarray) -> tuple[float, float, float]:
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


def ecef_to_ned(lat_rad: float, lon_rad: float) -> np.ndarray:
    slat, clat = (np.sin(lat_rad), np.cos(lat_rad))
    slon, clon = (np.sin(lon_rad), np.cos(lon_rad))
    return np.array([[-slat * clon, -slat * slon, clat], [-slon, clon, 0.0], [-clat * clon, -clat * slon, -slat]])


# SmartPNT vehicle-to-body rotation.
@lru_cache(maxsize=8)
def vehicle_to_body(x_rot_deg: float, y_rot_deg: float, z_rot_deg: float) -> np.ndarray:
    gamma, beta, alpha = np.deg2rad([x_rot_deg, y_rot_deg, z_rot_deg])
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Ry = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    Rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


# Saastamoinen delay.
def tropo_delay(height_m: float, elevation_rad: float) -> float:
    if elevation_rad <= 0.0:
        return float('inf')
    h = max(-100.0, min(float(height_m), 10000.0))
    temperature_k = 15.0 - 0.0065 * h + 273.15
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-05 * h) ** 5.2568
    water_vapor_hpa = 6.108 * 0.7 * math.exp((17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45))
    z = math.pi / 2.0 - elevation_rad
    return 0.002277 / math.cos(z) * (pressure_hpa + (1255.0 / temperature_k + 0.05) * water_vapor_hpa - 1.16 * math.tan(z) ** 2)


# Klobuchar delay.
def iono_delay(time_gps_tow_s: float, latitude_rad: float, longitude_rad: float, elevation_rad: float, azimuth_rad: float, alpha_s, beta_s) -> float:
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


# Sagnac-corrected range/LOS.
def range_los(receiver_position_ecef_m: np.ndarray, satellite_position_tx_ecef_m: np.ndarray, transit_s: float) -> tuple[float, np.ndarray, np.ndarray]:
    angle = EARTH_ROTATION_RATE_RADPS * float(transit_s)
    c, s = (np.cos(angle), np.sin(angle))
    C_rx_tx = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sat_rx = C_rx_tx @ np.asarray(satellite_position_tx_ecef_m, dtype=float).reshape(3)
    los = np.asarray(receiver_position_ecef_m, dtype=float).reshape(3) - sat_rx
    rho = float(np.linalg.norm(los))
    return (rho, los / rho, sat_rx)


# Elevation/azimuth.
def elev_az(c_ecef_ned: np.ndarray, los_satellite_to_receiver_ecef: np.ndarray) -> tuple[float, float]:
    receiver_to_satellite_ned = np.asarray(c_ecef_ned, dtype=float) @ -np.asarray(los_satellite_to_receiver_ecef, dtype=float)
    elevation = float(np.arcsin(np.clip(-receiver_to_satellite_ned[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(receiver_to_satellite_ned[1], receiver_to_satellite_ned[0]) % (2.0 * np.pi))
    return (elevation, azimuth)


class SatMeasurement(NamedTuple):
    sat_id: str
    constellation: str
    signal_suffix: str
    frequency_hz: float
    pseudorange_m: float
    cn0_dbhz: float | None = None


class ObsEpoch(NamedTuple):
    time_gpst_s: float
    measurements: tuple[SatMeasurement, ...]


class RINEXObs:

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

    def _decode_measurement(self, sat_id: str, fields: list[str]) -> SatMeasurement | None:
        constellation = sat_id[0]
        index_by_type = self._index_by_type.get(constellation, {})
        SIGNAL_PREFS = {'G': ('1C', '1W', '1P', '2W', '2L', '2X', '5Q', '5X', '5I'), 'C': ('2I', '1I', '2X', '1X', '1P', '1D', '5X', '5P', '5D', '7I', '7X', '6I', '6X')}
        for suffix in SIGNAL_PREFS.get(constellation, ()):
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
                FREQUENCY_HZ = {('G', '1'): 1575420000.0, ('G', '2'): 1227600000.0, ('G', '5'): 1176450000.0, ('C', '1'): 1575420000.0, ('C', '2'): 1561098000.0, ('C', '5'): 1176450000.0, ('C', '7'): 1207140000.0, ('C', '8'): 1191795000.0, ('C', '6'): 1268520000.0}
                frequency = FREQUENCY_HZ.get((constellation, suffix[0]))
            if frequency is None:
                continue
            cn0 = None
            snr_index = index_by_type.get('S' + suffix)
            if snr_index is not None and snr_index < len(fields):
                raw_snr = fields[snr_index].ljust(16)[:14].strip()
                if raw_snr:
                    value = float(raw_snr.replace('D', 'E'))
                    cn0 = value if math.isfinite(value) else None
            return SatMeasurement(sat_id, constellation, suffix, float(frequency), float(pseudorange), cn0)
        return None
    
    # Usable RINEX epochs.
    def iter_epochs(self, allowed_constellations: set[str] | None=None, *, start_time_gpst_s: float | None=None, end_time_gpst_s: float | None=None, max_epochs: int | None=None, require_measurements: bool=False):
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
                time_gpst_s = to_gpst(year, month, day, hour, minute, second, self.time_scale)
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
                yield ObsEpoch(time_gpst_s, tuple(measurements))
                yielded += 1
                if max_epochs is not None and yielded >= int(max_epochs):
                    break


@lru_cache(maxsize=8)
def read_imr_layout(path: str | Path):
    path = Path(path)
    with path.open('rb') as stream:
        buffer = stream.read(512)
    endian = '<' if buffer[8] == 0 else '>'
    values = struct.unpack(endian + "8scdiidddiid32s?BBB32s6h?iii354s", buffer)
    data_rate_hz = float(values[5])
    gyro_scale = float(values[6])
    accel_scale = float(values[7])
    time_tag_bias_ms = float(values[10])
    record_bytes = struct.calcsize(endian + "d6i")
    payload_bytes = max(0, path.stat().st_size - 512)
    record_count = payload_bytes // record_bytes
    record_dtype = np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))], align=False)
    return path, data_rate_hz, gyro_scale, accel_scale, time_tag_bias_ms, record_dtype, record_count, record_bytes


# SmartPNT metadata.
def unique_file(directory: Path, patterns: tuple[str, ...], label: str) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) != 1:
        names = ', '.join(path.name for path in matches) or 'none'
        raise FileNotFoundError(f'Expected exactly one {label} file in {directory}, found {len(matches)}: {names}')
    return matches[0]


# Inertial Explorer truth
@dataclass
class Truth:
    time_gpst_s: np.ndarray
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray
    heading_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray


def load_truth(path: str | Path) -> Truth:
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
    return Truth(time_gpst_s, a[:, 2:5], a[:, 5:8], a[:, 8], a[:, 9], a[:, 10])


# Interpolate truth.
def interp_truth(truth: Truth, query_time_gpst_s: np.ndarray, max_gap_s: float, *, position_only: bool=False):
    truth_time = truth.time_gpst_s
    query = np.asarray(query_time_gpst_s, dtype=float)
    upper = np.searchsorted(truth_time, query, side='left')
    upper = np.clip(upper, 0, len(truth_time) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = truth_time[upper] == query
    lower[exact] = upper[exact]
    gap = truth_time[upper] - truth_time[lower]
    if np.any(gap > float(max_gap_s)) or np.any(query < truth_time[0]) or np.any(query > truth_time[-1]):
        raise ValueError('Invalid ground-truth interpolation span')
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


# Body-to-ECEF DCM.
def body_to_ecef(position_ecef_m: np.ndarray, heading_deg: float, pitch_deg: float, roll_deg: float, mounting_xyz_deg: np.ndarray) -> np.ndarray:
    lat, lon, _ = ecef_llh(position_ecef_m)
    C_e_n = ecef_to_ned(lat, lon).T
    heading, pitch, roll = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh = (np.cos(heading), np.sin(heading))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cr, sr = (np.cos(roll), np.sin(roll))
    C_n_f = np.array([[cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh], [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch], [-sp, sr * cp, cr * cp]])
    C_b_v = vehicle_to_body(*mounting_xyz_deg)
    return _project_rotation(C_e_n @ (C_n_f @ C_F_V) @ C_b_v.T)


Array = np.ndarray
# uses 9 states Eq. (7)
INS_STATE_DIM = 9
DIRECT_STATE_LABEL_DIM = 9
ATTITUDE_FEEDBACK_SIGN = -1.0
J2_UNITLESS = 1.08262668e-3
OMEGA_IE_E = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RADPS])
IDENTITY_3 = np.eye(3)
IDENTITY_STATE = np.eye(INS_STATE_DIM)
C_F_V = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


# SO(3) exponential.
def _so3_exp(rotation_vector_rad: Array) -> Array:
    v = np.asarray(rotation_vector_rad, dtype=float).reshape(3)
    theta2 = float(v @ v)
    x, y, z = v
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
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
OMEGA_IE_SKEW = np.array([[0.0, -EARTH_ROTATION_RATE_RADPS, 0.0], [EARTH_ROTATION_RATE_RADPS, 0.0, 0.0], [0.0, 0.0, 0.0]])


# Re-orthogonalize DCM.
def _project_rotation(matrix: Array) -> Array:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float).reshape(3, 3))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


@dataclass
class NavState:
    position_ecef_m: Array
    velocity_ecef_mps: Array
    body_to_ecef_dcm: Array


def _gravity_j2(position_ecef_m: Array) -> Array:
    x, y, z = np.asarray(position_ecef_m, dtype=float).reshape(3)
    r = float(np.linalg.norm([x, y, z]))
    z2_r2 = z * z / (r * r)
    j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / r) ** 2
    xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / r**3
    return scale * np.array([x * xy_factor, y * xy_factor, z * z_factor])


# Cached Earth rotation.
@lru_cache(maxsize=128)
def _earth_rotation(dt_s: float) -> Array:
    matrix = _so3_exp(-OMEGA_IE_E * float(dt_s))
    matrix.setflags(write=False)
    return matrix


# ECEF INS mechanization.
def mechanize(nav: NavState, angular_rate_body_radps: Array, specific_force_body_mps2: Array, dt_s: float) -> NavState:
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    C1 = _project_rotation(_earth_rotation(dt) @ C0 @ _so3_exp(np.asarray(angular_rate_body_radps, dtype=float) * dt))
    Cmid = 0.5 * (C0 + C1)
    r_e = np.asarray(nav.position_ecef_m, dtype=float).reshape(3)
    acceleration = Cmid @ np.asarray(specific_force_body_mps2) + _gravity_j2(r_e) - np.array([-EARTH_ROTATION_RATE_RADPS * (EARTH_ROTATION_RATE_RADPS * r_e[0]), EARTH_ROTATION_RATE_RADPS * (-EARTH_ROTATION_RATE_RADPS * r_e[1]), 0.0]) - 2.0 * np.array([-EARTH_ROTATION_RATE_RADPS * nav.velocity_ecef_mps[1], EARTH_ROTATION_RATE_RADPS * nav.velocity_ecef_mps[0], 0.0])
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavState(position, velocity, C1)


# Pseudorange preparation.
class PseudoObs(NamedTuple):
    sat_id: str
    constellation: str
    pseudorange_m: float
    satellite_position_reception_ecef_m: Array
    satellite_clock_bias_s: float
    ionosphere_delay_m: float
    troposphere_delay_m: float
    variance_m2: float


def _transmit_time(position_at, sat_id: str, reception_time_gpst_s: float, receiver_position_ecef_m: Array, initial_position_ecef_m: Array, initial_transit_s: float, epsilon_position_m: float, max_iterations: int):
    previous_position = np.asarray(initial_position_ecef_m, dtype=float).reshape(3)
    transit_s = float(initial_transit_s)
    reception_time = float(reception_time_gpst_s)
    epsilon_position_m = float(epsilon_position_m)
    last_position_delta_m = float('inf')
    last_light_time_residual_m = float('inf')
    for _ in range(int(max_iterations)):
        transmit_time = reception_time - transit_s
        position = np.asarray(position_at(sat_id, transmit_time), dtype=float).reshape(3)
        rho, _, _ = range_los(receiver_position_ecef_m, position, transit_s)
        next_transit_s = float(rho / SPEED_OF_LIGHT_MPS)
        last_position_delta_m = float(np.linalg.norm(position - previous_position))
        last_light_time_residual_m = float(abs(next_transit_s - transit_s) * SPEED_OF_LIGHT_MPS)
        if last_position_delta_m < epsilon_position_m:
            return (transmit_time, transit_s, position)
        previous_position = position
        transit_s = next_transit_s
    raise RuntimeError(f'Transmit-time iteration did not converge for {sat_id}: iterations={max_iterations}, last_satellite_position_delta_m={last_position_delta_m:.6g}, last_light_time_residual_m={last_light_time_residual_m:.6g}, last_transit_s={transit_s:.12g}')


# GNSS pseudorange preparation.
class GNSSProcessor:

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
                    current_time = to_gpst(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[5]), time_scale)
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
                t = to_gpst(year, month, day, hour, minute, second, clock_time_scale)
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

    def prepare_epoch(self, epoch: ObsEpoch, nav: NavState, lever_arm_b_m: Array) -> tuple[PseudoObs, ...]:
        antenna_position = nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
        receiver_llh = ecef_llh(antenna_position)
        c_ecef_ned = ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch.measurements:
            try:
                initial_position = self._position(raw.sat_id, epoch.time_gpst_s)
                receiver = np.asarray(antenna_position, dtype=float).reshape(3)
                sat_rx = np.asarray(initial_position, dtype=float).reshape(3)
                reception_time = float(epoch.time_gpst_s)
                rho0, _, _ = range_los(receiver, sat_rx, 0.0)
                tau0 = float(rho0 / SPEED_OF_LIGHT_MPS)
                tx0 = reception_time - tau0
                sat_tx0 = np.asarray(self._position(raw.sat_id, tx0), dtype=float).reshape(3)
                rho1, _, _ = range_los(receiver, sat_tx0, tau0)
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
                satellite_clock_bias_s = float(np.interp(transmit_time, orbit['clock_time'], orbit['clock_value']))

            _, los, satellite_position_rx = range_los(antenna_position, state_tx_position, transit_s)
            elevation, azimuth = elev_az(c_ecef_ned, los)
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
                ionosphere = iono_delay(tow, lat, lon, elevation, azimuth, alpha, beta) * (reference_frequency_hz / raw.frequency_hz) ** 2
            troposphere = 0.0
            if self.use_troposphere:
                if height_m is None:
                    height_m = receiver_llh[2]
                troposphere = tropo_delay(height_m, elevation)
            cn0_dbhz = float(raw.cn0_dbhz) if raw.cn0_dbhz is not None else 45.0
            out.append(PseudoObs(
                raw.sat_id,
                raw.constellation,
                float(raw.pseudorange_m),
                satellite_position_rx,
                satellite_clock_bias_s,
                float(ionosphere),
                float(troposphere),
                float(
                    LEO_URA_SIGMA_M ** 2
                    + ((iono_noise(elevation, receiver_llh[0]) * ((1575420000.0 if raw.constellation == 'G' else 1561098000.0) / raw.frequency_hz) ** 2) ** 2 if self.use_ionosphere else 0.0)
                    + ((1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) ** 2 if self.use_troposphere else 0.0)
                    + mp_nlos_sigma(elevation, cn0_dbhz) ** 2
                    + receiver_noise(cn0_dbhz) ** 2
                ),
            ))
        return tuple(out)

# 9-state error dynamics/KF.


# ECEF error dynamics.  # Yan Eq. (7)
def error_dynamics(nav: NavState, specific_force_body_mps2: Array) -> Array:
    F = np.zeros((INS_STATE_DIM, INS_STATE_DIM))
    r_e = nav.position_ecef_m
    radius = float(np.linalg.norm(r_e))
    gravity = _gravity_j2(r_e)
    radial = r_e / radius
    C = nav.body_to_ecef_dcm
    F[0:3, 3:6] = IDENTITY_3
    F[3:6, 0:3] = -(2.0 / radius) * np.outer(gravity, radial)
    F[3:6, 3:6] = -2.0 * OMEGA_IE_SKEW
    force_e = C @ np.asarray(specific_force_body_mps2)
    x, y, z = force_e
    F[3:6, 6:9] = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    return F

# Van Loan discretization.
VAN_LOAN_STATS = {'taylor_calls': 0, 'exact_fallback_calls': 0, 'validation_calls': 0, 'max_norm_1': 0.0, 'max_taylor_remainder_bound': 0.0, 'max_validation_phi_abs': 0.0, 'max_validation_qd_abs': 0.0}
VAN_LOAN_TAYLOR_ORDER = 10
VAN_LOAN_TAYLOR_MAX_NORM_1 = 0.20
VAN_LOAN_VALIDATE_CALLS = 8
def van_loan(F: Array, Qc: Array, dt_s: float) -> tuple[Array, Array]:
    n = INS_STATE_DIM
    A = np.zeros((2 * n, 2 * n), dtype=float)
    A[:n, :n] = F
    A[:n, n:] = Qc
    A[n:, n:] = -F.T
    B = A * float(dt_s)
    norm_1 = float(np.linalg.norm(B, 1))
    VAN_LOAN_STATS['max_norm_1'] = max(VAN_LOAN_STATS['max_norm_1'], norm_1)
    if norm_1 <= VAN_LOAN_TAYLOR_MAX_NORM_1:
        E = np.eye(B.shape[0], dtype=float)
        term = np.eye(B.shape[0], dtype=float)
        for k in range(1, VAN_LOAN_TAYLOR_ORDER + 1):
            term = (term @ B) / float(k)
            E += term
        VAN_LOAN_STATS['taylor_calls'] += 1
        remainder_bound = math.exp(norm_1) * norm_1 ** (VAN_LOAN_TAYLOR_ORDER + 1) / math.factorial(VAN_LOAN_TAYLOR_ORDER + 1)
        VAN_LOAN_STATS['max_taylor_remainder_bound'] = max(VAN_LOAN_STATS['max_taylor_remainder_bound'], float(remainder_bound))
        if VAN_LOAN_STATS['validation_calls'] < VAN_LOAN_VALIDATE_CALLS:
            E_ref = expm(B)
            Phi_fast = E[:n, :n]
            Qd_fast = E[:n, n:] @ Phi_fast.T
            Phi_ref = E_ref[:n, :n]
            Qd_ref = E_ref[:n, n:] @ Phi_ref.T
            VAN_LOAN_STATS['max_validation_phi_abs'] = max(VAN_LOAN_STATS['max_validation_phi_abs'], float(np.max(np.abs(Phi_fast - Phi_ref))))
            VAN_LOAN_STATS['max_validation_qd_abs'] = max(VAN_LOAN_STATS['max_validation_qd_abs'], float(np.max(np.abs(Qd_fast - Qd_ref))))
            VAN_LOAN_STATS['validation_calls'] += 1
    else:
        E = expm(B)
        VAN_LOAN_STATS['exact_fallback_calls'] += 1
    Phi = E[:n, :n]
    Qd = E[:n, n:] @ Phi.T
    return (Phi, 0.5 * (Qd + Qd.T))


class TCModel(NamedTuple):
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]


# Preserve clock observability.
def clock_observable(measurements) -> tuple[PseudoObs, ...]:
    measurements = tuple(measurements)
    counts = Counter((m.constellation for m in measurements if m.constellation in {'G', 'C'}))
    return tuple((m for m in measurements if m.constellation not in {'G', 'C'} or counts[m.constellation] >= 2))


# Limit to Data01 Nmax.
def select_measurements(
    measurements,
    capacity: int,
) -> tuple[PseudoObs, ...]:
    measurements = tuple(clock_observable(measurements))
    if len(measurements) <= capacity:
        return measurements

    ranked = sorted(
        enumerate(measurements),
        key=lambda item: (
            float(item[1].variance_m2),
            str(item[1].sat_id),
            int(item[0]),
        ),
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    constellation_counts: Counter = Counter()

    # Quality order with clock observability.
    for rank_pos, (original_index, measurement) in enumerate(ranked):
        if len(selected_indices) >= capacity:
            break
        if original_index in selected_set:
            continue

        constellation = measurement.constellation
        remaining_slots = capacity - len(selected_indices)

        if constellation in {'G', 'C'} and constellation_counts[constellation] == 0:
            # GPS/BDS needs a pair for clock projection.

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
    selected = tuple(clock_observable(selected))

    return selected


def _clock_projection(measurements, variances: Array) -> Array:
    n = len(measurements)
    projector = np.eye(n)
    for system in ('G', 'C'):
        index = np.asarray([i for i, m in enumerate(measurements) if m.constellation == system], dtype=int)
        if index.size == 0:
            continue
        weights = 1.0 / variances[index]
        normalized_weight = weights / np.sum(weights)
        block = np.eye(index.size) - np.ones((index.size, 1)) @ normalized_weight[None, :]
        projector[np.ix_(index, index)] = block
    return projector


# TC measurement model.
def measurement_model(nav: NavState, measurements, lever_arm_b_m: Array) -> TCModel:
    measurements = clock_observable(measurements)
    n = len(measurements)
    if n == 0:
        return TCModel(np.empty(0), np.zeros((0, INS_STATE_DIM)), np.zeros((0, 0)), ())
    raw_innovation = np.empty(n)
    H_raw = np.zeros((n, INS_STATE_DIM))
    variances = np.empty(n)
    lever_e = nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    antenna_position = nav.position_ecef_m + lever_e
    x, y, z = lever_e
    lever_skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    sat_ids: list[str] = []
    for i, m in enumerate(measurements):
        range_vector = np.asarray(antenna_position) - m.satellite_position_reception_ecef_m
        geometric_range_m = float(np.linalg.norm(range_vector))
        los = range_vector / geometric_range_m
        predicted = geometric_range_m + 0.0 - SPEED_OF_LIGHT_MPS * m.satellite_clock_bias_s + m.ionosphere_delay_m + m.troposphere_delay_m
        raw_innovation[i] = m.pseudorange_m - predicted
        H_raw[i, 0:3] = los
        H_raw[i, 6:9] = -ATTITUDE_FEEDBACK_SIGN * (los @ lever_skew)
        variances[i] = float(m.variance_m2)
        sat_ids.append(m.sat_id)
    projector = _clock_projection(measurements, variances)
    innovation = projector @ raw_innovation
    H = projector @ H_raw
    R_raw = np.diag(variances)
    R = projector @ R_raw @ projector.T
    R = 0.5 * (R + R.T)
    return TCModel(innovation, H, R, tuple(sat_ids))


# Posterior innovation.
def innovation_only(nav: NavState, measurements, lever_arm_b_m: Array) -> Array:
    measurements = clock_observable(measurements)
    n = len(measurements)
    if n == 0:
        return np.empty(0)
    antenna_position = nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    raw_innovation = np.empty(n)
    variances = np.empty(n)
    for i, m in enumerate(measurements):
        range_vector = np.asarray(antenna_position) - m.satellite_position_reception_ecef_m
        geometric_range_m = float(np.linalg.norm(range_vector))
        predicted = geometric_range_m + 0.0 - SPEED_OF_LIGHT_MPS * m.satellite_clock_bias_s + m.ionosphere_delay_m + m.troposphere_delay_m
        raw_innovation[i] = m.pseudorange_m - predicted
        variances[i] = float(m.variance_m2)
    return _clock_projection(measurements, variances) @ raw_innovation
# Optional integrity monitoring.
INTEGRITY = IntegrityMonitor(ecef_llh, ecef_to_ned) if IntegrityMonitor is not None else None


# Classical TC/KF update.
def kf_update(P: Array, innovation: Array, H: Array, R: Array):
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)
    if not np.all(np.isfinite(K)):
        raise FloatingPointError('non-finite 9-state classical Kalman gain')
    dx = K @ innovation
    I_KH = IDENTITY_STATE - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_reset = reset_covariance(P_post, dx)
    return (dx, P_reset, K)


# Learned-gain Joseph covariance.
def learned_covariance(prior_covariance: Array, learned_gain: Array, measurement_jacobian: Array, measurement_covariance: Array, injected_error_state: Array) -> Array:
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    learned_gain = np.asarray(learned_gain, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    injected_error_state = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    update_matrix = IDENTITY_STATE - learned_gain @ measurement_jacobian
    posterior_covariance = update_matrix @ prior_covariance @ update_matrix.T + learned_gain @ measurement_covariance @ learned_gain.T
    return reset_covariance(posterior_covariance, injected_error_state)


# Covariance reset.
def reset_covariance(posterior_covariance: Array, injected_error_state: Array) -> Array:
    P = np.asarray(posterior_covariance, dtype=float)
    dx = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    phi = np.asarray(ATTITUDE_FEEDBACK_SIGN * dx[6:9], dtype=float).reshape(3)
    theta_squared = float(phi @ phi)
    x, y, z = phi
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
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


def inject_error(nav: NavState, dx: Array) -> NavState:
    dx = np.asarray(dx, dtype=float).reshape(INS_STATE_DIM)
    out = NavState(nav.position_ecef_m.copy(), nav.velocity_ecef_mps.copy(), nav.body_to_ecef_dcm.copy())
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _project_rotation(_so3_exp(ATTITUDE_FEEDBACK_SIGN * dx[6:9]) @ out.body_to_ecef_dcm)
    return out


# Differentiable Data01 state.
class TorchNavState(NamedTuple):
    position_ecef_m: torch.Tensor
    velocity_ecef_mps: torch.Tensor
    body_to_ecef_dcm: torch.Tensor


# Torch SO(3) exponential.
def _torch_so3_exp(rotation_vector_rad: torch.Tensor) -> torch.Tensor:
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


# Torch SO(3) logarithm.
def _torch_so3_log(rotation_matrix: torch.Tensor) -> torch.Tensor:
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


# Torch ECEF mechanization.
def _torch_mechanize(nav: TorchNavState, angular_rate_body_radps: torch.Tensor, specific_force_body_mps2: torch.Tensor, dt_s: float) -> TorchNavState:
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    earth_rotvec = C0.new_tensor(-OMEGA_IE_E * dt)
    C1 = _torch_so3_exp(earth_rotvec) @ C0 @ _torch_so3_exp(angular_rate_body_radps.reshape(3) * dt)
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
    return TorchNavState(position, velocity, C1)


# TLE/SGP4 TEME-to-ECEF.
class TLEProvider:

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


# Ionosphere sigma  Eq. (3)
def iono_noise(elevation_rad: float, receiver_latitude_rad: float) -> float:
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else (4.5 if latitude_deg <= 55.0 else 6.0)
    R = 6_378_140.0
    h_i = 350_000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))


# Multipath/NLOS sigma Eq. (4)
GOGPS_A = 30.0
GOGPS_S0_DBHZ = 10.0
GOGPS_S1_DBHZ = 50.0
GOGPS_A_DB = 20.0
def mp_nlos_sigma(elevation_rad: float, cn0_dbhz: float) -> float:
    e = max(float(elevation_rad), 1e-06)
    cn0 = float(cn0_dbhz)
    if cn0 >= GOGPS_S1_DBHZ:
        variance = 1.0
    else:
        ratio = (cn0 - GOGPS_S1_DBHZ) / (GOGPS_S0_DBHZ - GOGPS_S1_DBHZ)
        term = 10.0 ** (-(cn0 - GOGPS_S1_DBHZ) / GOGPS_A_DB) * ((GOGPS_A / 10.0 ** (-(GOGPS_S0_DBHZ - GOGPS_S1_DBHZ) / GOGPS_A_DB) - 1.0) * ratio + 1.0)
        variance = 1.0 / max(math.sin(e) ** 2, 1e-06) * term
    return float(math.sqrt(max(variance, 1e-12)))


# receiver sigmaEq. (3)
LEO_CODE_LOOP_BANDWIDTH_HZ = 2.0
LEO_CORRELATOR_SPACING_CHIPS = 0.1
LEO_CORRELATOR_ACCUMULATION_S = 0.02
LEO_CODE_CHIPPING_RATE_HZ = 1.023e6
def receiver_noise(cn0_dbhz: float) -> float:
    cn0_linear = 10.0 ** (float(cn0_dbhz) / 10.0)
    d = LEO_CORRELATOR_SPACING_CHIPS
    bl = LEO_CODE_LOOP_BANDWIDTH_HZ
    tau = LEO_CORRELATOR_ACCUMULATION_S
    sigma_chips = math.sqrt(bl * d / (2.0 * cn0_linear) * (1.0 + 2.0 / ((2.0 - d) * cn0_linear * tau)))
    return float(SPEED_OF_LIGHT_MPS / LEO_CODE_CHIPPING_RATE_HZ * sigma_chips)


# LEO pseudorange simulation.
LEO_IONOSPHERE_LOWER_HEIGHT_M = 100_000.0
LEO_IONOSPHERE_UPPER_HEIGHT_M = 1_000_000.0
class LEOSimulator:

    def __init__(self, provider: TLEProvider, klobuchar_coefficients: tuple[Array, Array] | None, seed: int=0, tx_epsilon_position_m: float=0.001, tx_max_iterations: int=20, minimum_elevation_deg: float=10.0, prefilter_guard_deg: float=LEO_PREFILTER_GUARD_DEG, use_ionosphere: bool=True, use_troposphere: bool=True):
        self.provider = provider
        self.klobuchar_coefficients = klobuchar_coefficients
        self.tx_epsilon_position_m = float(tx_epsilon_position_m)
        self.tx_max_iterations = int(tx_max_iterations)
        self.minimum_elevation_rad = np.deg2rad(minimum_elevation_deg)
        self.prefilter_guard_rad = np.deg2rad(float(prefilter_guard_deg))
        self.use_ionosphere = bool(use_ionosphere)
        self.use_troposphere = bool(use_troposphere)
        self.prefilter_checked = 0
        self.prefilter_rejected = 0
        self.prefilter_passed = 0
        self.rng = np.random.default_rng(seed)

    def _leo_iono_delay(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array, satellite_position_reception_ecef_m: Array, elevation_rad: float, azimuth_rad: float, receiver_llh: tuple[float, float, float] | None=None) -> tuple[float, float]:
        if not self.use_ionosphere:
            return (0.0, 0.0)
        alpha, beta = self.klobuchar_coefficients
        if receiver_llh is None:
            receiver_llh = ecef_llh(receiver_position_ecef_m)
        lat, lon, _ = receiver_llh
        receive_week = int(math.floor(float(receive_time_gpst_s) / GPS_WEEK_S))
        tow = float(receive_time_gpst_s) - receive_week * GPS_WEEK_S
        iklo_m = iono_delay(tow, lat, lon, elevation_rad, azimuth_rad, alpha, beta)
        _, _, satellite_height_m = ecef_llh(satellite_position_reception_ecef_m)
        if satellite_height_m >= LEO_IONOSPHERE_UPPER_HEIGHT_M:
            scale = 1.0
        elif satellite_height_m <= LEO_IONOSPHERE_LOWER_HEIGHT_M:
            scale = 0.0
        else:
            scale = (satellite_height_m - LEO_IONOSPHERE_LOWER_HEIGHT_M) / (LEO_IONOSPHERE_UPPER_HEIGHT_M - LEO_IONOSPHERE_LOWER_HEIGHT_M)
        return (float(scale * iklo_m), float(scale))

    def simulate_one(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array, sat_id: str, receiver_llh: tuple[float, float, float] | None=None, c_ecef_ned: Array | None=None):
        if receiver_llh is None:
            receiver_llh = ecef_llh(receiver_position_ecef_m)
        if c_ecef_ned is None:
            c_ecef_ned = ecef_to_ned(receiver_llh[0], receiver_llh[1])
        initial_position, _ = self.provider.state_at(receive_time_gpst_s, sat_id)
        initial_range, initial_los, _ = range_los(receiver_position_ecef_m, initial_position, 0.0)
        initial_elevation, _ = elev_az(c_ecef_ned, initial_los)
        self.prefilter_checked += 1
        if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
            self.prefilter_rejected += 1
            return None
        self.prefilter_passed += 1
        transmit_time, transit_s, state_tx_position = _transmit_time(lambda sid, t: self.provider.state_at(t, sid)[0], sat_id, receive_time_gpst_s, receiver_position_ecef_m, initial_position, initial_range / SPEED_OF_LIGHT_MPS, self.tx_epsilon_position_m, self.tx_max_iterations)
        rho, los, sat_rx = range_los(receiver_position_ecef_m, state_tx_position, transit_s)
        elevation, azimuth = elev_az(c_ecef_ned, los)
        if elevation < self.minimum_elevation_rad:
            return None
        receiver_lat, _, receiver_height_m = receiver_llh
        ionosphere_m, ionosphere_path_scale = self._leo_iono_delay(receive_time_gpst_s, receiver_position_ecef_m, sat_rx, elevation, azimuth, receiver_llh=receiver_llh)
        troposphere_m = tropo_delay(receiver_height_m, elevation) if self.use_troposphere else 0.0
        iono_sigma_m = ionosphere_path_scale * iono_noise(elevation, receiver_lat) if self.use_ionosphere else 0.0
        tropo_sigma_m = float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) if self.use_troposphere else 0.0
        mp_sigma_m = float(0.13 + 0.53 * math.exp(-float(np.rad2deg(elevation)) / 10.0))
        ionosphere_residual_m = float(self.rng.normal(0.0, iono_sigma_m)) if iono_sigma_m > 0.0 else 0.0
        troposphere_residual_m = float(self.rng.normal(0.0, tropo_sigma_m)) if tropo_sigma_m > 0.0 else 0.0
        mp_nlos_error_m = float(self.rng.normal(0.0, mp_sigma_m)) if mp_sigma_m > 0.0 else 0.0
        pseudorange_m = rho + ionosphere_m + troposphere_m + ionosphere_residual_m + troposphere_residual_m + mp_nlos_error_m
        return PseudoObs(
            sat_id,
            'L',
            float(pseudorange_m),
            sat_rx,
            0.0,
            float(ionosphere_m),
            float(troposphere_m),
            float(max(iono_sigma_m ** 2 + tropo_sigma_m ** 2 + mp_sigma_m ** 2, 1e-12)),
        )

    def simulate_epoch(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array):
        receiver_llh = ecef_llh(receiver_position_ecef_m)
        c_ecef_ned = ecef_to_ned(receiver_llh[0], receiver_llh[1])
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


# Masked CLA 
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 24 in the user-requested 9-state branch
OBSERVATION_FEATURE_DIM = 2                 # [Delta y_{k-1}, Delta ytilde_k]
# [Dalpha, Domega, Dxhat, Dxtilde, Dy_prev, Dy_innov, zero-pad] Eqs. (15)-(16)
SUPERVISED_STATE_DIM = DIRECT_STATE_LABEL_DIM
MASK_NORMALIZATION_EPS = 1e-6
FIG8_POOL_KERNEL_SIZE = 3


# CLA output.
class CLAOutput(NamedTuple):
    kalman_gain: torch.Tensor
    attention: torch.Tensor
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None


# Masked convolution. Eqs. (22)-(23)
class MaskedConv(nn.Module):

    def __init__(self, out_channels: int=24, kernel_size: int=3, pool_kernel_size: int=FIG8_POOL_KERNEL_SIZE) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.pool_kernel_size = int(pool_kernel_size)
        self.pool_padding = self.pool_kernel_size // 2
        self.register_buffer('_mask_kernel', torch.ones(1, 1, self.kernel_size), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


# Masked LSTM.  Eqs. (24)-(25)
class MaskedLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
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
        batch_size, position_count, _ = x.shape
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
            previous_hidden, previous_cell = recurrent_state

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

        #  Eq. (25)
        valid = mask.bool().unsqueeze(0).unsqueeze(-1)
        next_hidden = torch.where(valid, candidate_hidden, previous_hidden)
        next_cell = torch.where(valid, candidate_cell, previous_cell)

        output = next_hidden[-1]  # [B,D,H]
        return output, (next_hidden, next_cell)


# Masked attention - Eqs. 26-29
class MaskedAttention(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


# Masked CLA - Fig. 8
class MaskedCLA(nn.Module):

    def __init__(self, nmax: int, dropout: float=0.2) -> None:
        super().__init__()
        self.nmax = int(nmax)
        self.dropout = float(dropout)
        self.eq15_padded_dim = FIXED_FEATURE_DIM + 2 * self.nmax
        self.conv = MaskedConv(out_channels=24, kernel_size=3, pool_kernel_size=FIG8_POOL_KERNEL_SIZE)
        self.lstm = MaskedLSTM(input_size=24, hidden_size=64, num_layers=5, dropout=self.dropout)
        self.attention = MaskedAttention(64)
        self.gain_head = nn.Linear(64, INS_STATE_DIM * self.nmax)
        # Zero-gain initialization is a project choice.
        nn.init.zeros_(self.gain_head.weight)
        if self.gain_head.bias is not None:
            nn.init.zeros_(self.gain_head.bias)

    def forward(self, fixed: torch.Tensor, observations: torch.Tensor, mask: torch.Tensor, channel_mask: torch.Tensor, recurrent_state: tuple[torch.Tensor, torch.Tensor] | None=None) -> CLAOutput:
        batch_size = fixed.shape[0]
        satellite_valid = mask.bool()
        channel_bool = channel_mask.bool()
        satellite_count = satellite_valid.sum(dim=1)
        if not torch.equal(satellite_valid, torch.arange(self.nmax, device=mask.device).unsqueeze(0) < satellite_count.unsqueeze(1)):
            raise ValueError('mask must contain a contiguous valid prefix for Eq. (16)')
        if not torch.equal(channel_bool[:, :, 1], satellite_valid):
            raise ValueError('innovation channel_mask must match mask for Eq. (16)')
        observations = observations * channel_bool.to(dtype=observations.dtype)
        fixed_valid = torch.ones((batch_size, FIXED_FEATURE_DIM), dtype=torch.bool, device=fixed.device)
        packed_values = []
        packed_masks = []
        for batch_index, count in enumerate(satellite_count.detach().cpu().tolist()):
            suffix_length = 2 * (self.nmax - count)

            #  Eqs. (15)-(17)
            packed_values.append(torch.cat((
                fixed[batch_index],  
                observations[batch_index, :count, 0], 
                observations[batch_index, :count, 1], 
                observations.new_zeros(suffix_length), 
            ), dim=0))
            packed_masks.append(torch.cat((
                fixed_valid[batch_index],
                channel_bool[batch_index, :count, 0],
                channel_bool[batch_index, :count, 1],
                torch.zeros(suffix_length, dtype=torch.bool, device=channel_mask.device),
            ), dim=0))
        x_bar = torch.stack(packed_values, dim=0)
        feature_mask = torch.stack(packed_masks, dim=0)
        # CNN -> LSTM -> attention. Eqs. (22)-(29)
        conv_features = self.conv(x_bar, feature_mask)  # [B,D,24]
        lstm_positions, next_lstm_state = self.lstm(
            conv_features,
            feature_mask,
            recurrent_state,
        )  # [B,D,64], state [L,B,D,64]
        context, attention = self.attention(lstm_positions, feature_mask)
        gain = self.gain_head(context).view(batch_size, INS_STATE_DIM, self.nmax)
        gain = gain * satellite_valid.to(dtype=gain.dtype).unsqueeze(1)
        return CLAOutput(kalman_gain=gain, attention=attention, recurrent_state=next_lstm_state)


# State correction with learned gain.
def state_update(network_output: CLAOutput, innovation: torch.Tensor) -> torch.Tensor:
    return torch.bmm(network_output.kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)


# Pipeline
if __name__ == '__main__':
    OUTPUT_DIR = Path('/kaggle/working/direct_run')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_FUSION_EPOCHS = 500
    MAX_TEST_FUSION_EPOCHS = 100
    TRAINING_EPOCHS = 30
    LEARNING_RATE = 1e-4
    OPTIMIZER_WINDOW_SIZE = 4
    GRADIENT_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 6
    VALIDATION_FRACTION = 0.20
    GAMMA_L2 = 1e-6
    SEED = 0
    TEST_LEO_SEED = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    rover_xml = next(item for item in ET.fromstring(README_XML_PATH.read_text(encoding='utf-8', errors='replace')).findall('ROVE') if (item.findtext('ID') or '').strip() == '01')
    rover_imu_type = (rover_xml.findtext('SINS_IMUType') or '').strip()
    rover_mounting_xyz_deg = np.fromstring(rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    antenna_truth = load_truth(ROVE_GROUND_TRUTH_PATH)
    imu_truth = load_truth(IMU_GROUND_TRUTH_PATH)
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
    rinex = RINEXObs(RINEX_OBS_PATH)
    first_rinex_epoch = next(rinex.iter_epochs(allowed_constellations={'G', 'C'}))
    imr_path, imr_data_rate_hz, imr_gyro_scale, imr_accel_scale, imr_time_tag_bias_ms, imr_record_dtype, imr_record_count, imr_record_bytes = read_imr_layout(IMR_PATH)
    imr_records = np.memmap(imr_path, dtype=imr_record_dtype, mode='r', offset=512, shape=(imr_record_count,))
    imr_tow_all = np.asarray(imr_records['tow'], dtype=np.float64).copy()
    del imr_records
    imr_tow_all[imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    imr_tow_all -= imr_time_tag_bias_ms * 1e-3
    imr_time_all = anchor_imr_gpst(imr_tow_all, float(first_rinex_epoch.time_gpst_s))
    antenna_truth_time = check_time_axis('training antenna truth', antenna_truth.time_gpst_s)
    imu_truth_time = check_time_axis('training IMU truth', imu_truth.time_gpst_s)
    common_end = min(float(imr_time_all[-1]), float(antenna_truth_time[-1]), float(imu_truth_time[-1]))
    start = int(np.searchsorted(imr_time_all, max(float(imr_time_all[0]), float(antenna_truth_time[0]), float(imu_truth_time[0])), side='left'))
    gnss_epochs = tuple(rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=float(imr_time_all[start]), end_time_gpst_s=common_end, max_epochs=MAX_FUSION_EPOCHS, require_measurements=True))
    fusion_time = check_time_axis('training fusion', np.asarray([epoch.time_gpst_s for epoch in gnss_epochs], dtype=float))
    if len(fusion_time) < 3:
        raise ValueError('Fewer than three synchronized GNSS fusion epochs remain; cannot build lagged Masked KalmanNet features')
    requested_stop = min(int(np.searchsorted(imr_time_all, fusion_time[-1], side='left')) + 1, min(int(np.searchsorted(imr_time_all, common_end, side='left')) + 1, len(imr_time_all)))
    imr_records = np.fromfile(imr_path, dtype=imr_record_dtype, count=requested_stop - start, offset=512 + start * imr_record_bytes)
    imr_angular_rate_body_radps = np.deg2rad(imr_records['counts'][:, :3].astype(float) * imr_gyro_scale * imr_data_rate_hz)
    imr_acceleration_body_mps2 = imr_records['counts'][:, 3:6].astype(float) * imr_accel_scale * imr_data_rate_hz
    imr_time = imr_time_all[start:requested_stop].copy()
    del imr_tow_all, imr_time_all
    query_time = np.concatenate(([imr_time[0]], fusion_time))
    antenna_position = interp_truth(antenna_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    imu_position, imu_velocity, imu_heading, imu_pitch, imu_roll = interp_truth(imu_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    initial_imu_truth_position = imu_position[0]
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]
    C_b_e = body_to_ecef(initial_imu_truth_position, imu_heading[0], imu_pitch[0], imu_roll[0], rover_mounting_xyz_deg)
    lever_arm_b_m = vehicle_to_body(*rover_mounting_xyz_deg) @ np.fromstring(rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ').reshape(3)
    fusion_truth_body_to_ecef = np.stack([body_to_ecef(p, h, pt, r, rover_mounting_xyz_deg) for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])])
    initial_nav = NavState(initial_imu_truth_position.copy(), imu_velocity[0].copy(), C_b_e)
    P0 = np.diag(np.concatenate((imu_model_values['ISDV_Pos'], imu_model_values['ISDV_Vel'], imu_model_values['ISDV_Att'] * (np.pi / 180.0))) ** 2)
    Qc = np.diag(np.concatenate((imu_model_values['PNSD_Pos'], imu_model_values['PNSD_Vel'], imu_model_values['PNSD_Att'] * (np.pi / 180.0))) ** 2)
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
    gnss_preprocessor = GNSSProcessor(SP3_PATH, CLK_PATH, min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=True, use_troposphere=True, broadcast_ionosphere_coefficients=ionosphere_coefficients)
    tle_provider = TLEProvider(LEO_TLE_DIR, max_tle_age_days=TLE_MAX_AGE_DAYS, allow_degraded_eop=TLE_ALLOW_DEGRADED_EOP, allow_non_tle_files=TLE_ALLOW_NON_TLE_FILES)
    leo_simulator = LEOSimulator(tle_provider, ionosphere_coefficients['G'], seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=True, use_troposphere=True)
    nav = NavState(initial_nav.position_ecef_m.copy(), initial_nav.velocity_ecef_mps.copy(), initial_nav.body_to_ecef_dcm.copy())
    P = P0.copy()
    history_rows = []
    interval_segments_since_previous_usable_fusion = []
    last_gyro = np.asarray(imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(imr_acceleration_body_mps2[0], dtype=float).reshape(3)
    # Data01 history.
    for event_kind, event_start, event_end, event_index in fusion_timeline(imr_time, fusion_time, through_last_fusion=True):
        if event_kind == 'imu':
            imu_index = event_index
            dt = event_end - event_start
            last_gyro = np.asarray(imr_angular_rate_body_radps[imu_index], dtype=float).reshape(3)
            last_accel = np.asarray(imr_acceleration_body_mps2[imu_index], dtype=float).reshape(3)
            nav = mechanize(nav, last_gyro, last_accel, dt)
            interval_segments_since_previous_usable_fusion.append((imu_index, dt))
            F_error = error_dynamics(nav, last_accel)
            Phi, Qd = van_loan(F_error, Qc, dt)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue
        fusion_index = event_index
        t = event_start
        epoch = gnss_epochs[fusion_index]
        gnss_measurements = gnss_preprocessor.prepare_epoch(epoch, nav, lever_arm_b_m)
        leo_measurements = leo_simulator.simulate_epoch(t, fusion_antenna_truth_position[fusion_index])
        measurements = clock_observable(tuple(gnss_measurements) + tuple(leo_measurements))
        if not measurements:
            continue
        meas = measurement_model(nav, measurements, lever_arm_b_m)
        correction, P, _ = kf_update(P, meas.innovation, meas.H, meas.R)
        nav = inject_error(nav, correction)
        posterior_residual = innovation_only(nav, measurements, lever_arm_b_m)
        # Keep observations fixed during recursive training.
        fixed_measurements = tuple(measurements)
        history_rows.append({
            'time': t,
            'sat_ids': meas.sat_ids,
            'innovation': meas.innovation.copy(),
            'residual': posterior_residual.copy(),
            'x_pred': np.zeros(INS_STATE_DIM),
            'x_post': correction.copy(),
            'accel': last_accel.copy(),
            'gyro': last_gyro.copy(),
            'posterior_nav': NavState(nav.position_ecef_m.copy(), nav.velocity_ecef_mps.copy(), nav.body_to_ecef_dcm.copy()),
            'fusion_index': int(fusion_index),
            'fixed_measurements': fixed_measurements,
            'preceding_interval_segments': tuple(interval_segments_since_previous_usable_fusion),
        })
        interval_segments_since_previous_usable_fusion = []
    if len(history_rows) < 4:
        raise ValueError('Classical TC pass produced fewer than four usable measurement rows; check GNSS products, TLE coverage, masks, and time synchronization')
    
    network_nmax = max(len(row['sat_ids']) for row in history_rows[1:])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Project L2 normalization.
    def _l2_norm(values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(values))
        return np.zeros_like(values) if norm <= FEATURE_L2_EPS else values / norm

    def _normalize_fixed(fixed_raw: Array) -> Array:
        fixed_raw = np.asarray(fixed_raw, dtype=float).reshape(FIXED_FEATURE_DIM)
        return np.concatenate([
            _l2_norm(fixed_raw[0:3]),
            _l2_norm(fixed_raw[3:6]),
            _l2_norm(fixed_raw[6:15]),
            _l2_norm(fixed_raw[15:24]),
        ])

    def _normalize_obs(obs_raw: Array, channel_mask: Array) -> Array:
        obs_raw = np.asarray(obs_raw, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        channel_mask = np.asarray(channel_mask, dtype=bool).reshape(-1, OBSERVATION_FEATURE_DIM)
        normalized = np.zeros_like(obs_raw)
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            if np.any(valid):
                normalized[valid, channel] = _l2_norm(obs_raw[valid, channel])
        return normalized

    # Train/validation split.
    data01_total_sample_count = len(history_rows) - 1
    validation_sample_count = max(1, int(round(data01_total_sample_count * VALIDATION_FRACTION)))
    training_sample_count = data01_total_sample_count - validation_sample_count
    validation_start_sample = training_sample_count

    # Training - Alternating blocks: CNN / LSTM+attention+KG
    model = MaskedCLA(nmax=network_nmax, dropout=0.2).to(DEVICE)
    LSTM_CONFIGURED_DROPOUT = float(model.lstm.lstm.dropout)
    representation_parameters = list(model.conv.parameters())
    filter_parameters = [
        *model.lstm.parameters(),
        *model.attention.parameters(),
        *model.gain_head.parameters(),
    ]
    filter_optimizer = torch.optim.Adam(filter_parameters, lr=LEARNING_RATE)
    representation_optimizer = torch.optim.Adam(representation_parameters, lr=LEARNING_RATE)
    all_network_parameters = representation_parameters + filter_parameters
    training_imr_gyro_t = torch.as_tensor(imr_angular_rate_body_radps, dtype=torch.float64, device=DEVICE)
    training_imr_accel_t = torch.as_tensor(imr_acceleration_body_mps2, dtype=torch.float64, device=DEVICE)
    lever_arm_train_t = torch.as_tensor(lever_arm_b_m, dtype=torch.float64, device=DEVICE).reshape(3)
    fusion_truth_position_t = torch.as_tensor(fusion_imu_truth_position, dtype=torch.float64, device=DEVICE)
    fusion_truth_velocity_t = torch.as_tensor(fusion_imu_truth_velocity, dtype=torch.float64, device=DEVICE)
    fusion_truth_dcm_t = torch.as_tensor(fusion_truth_body_to_ecef, dtype=torch.float64, device=DEVICE)

    def _torch_innovation(nav_state: TorchNavState, prepared_measurements):
        sat_ids, satellite_position, observed, satellite_clock, ionosphere, troposphere, projector_t = prepared_measurements
        if not sat_ids:
            return nav_state.position_ecef_m.new_empty(0), ()
        antenna_position = nav_state.position_ecef_m + nav_state.body_to_ecef_dcm @ lever_arm_train_t
        geometric = torch.linalg.vector_norm(antenna_position.unsqueeze(0) - satellite_position, dim=1)
        predicted = geometric - SPEED_OF_LIGHT_MPS * satellite_clock + ionosphere + troposphere
        return projector_t @ (observed - predicted), sat_ids

    # Build network input.  # Yan Eqs. (10)-(17)
    def _network_input(previous_context, current_sat_ids, innovation_now, current_accel, current_gyro):
        n = len(current_sat_ids)
        innovation_now = innovation_now.reshape(n)
        fixed_raw = torch.cat((
            current_accel.reshape(3) - previous_context['accel'],  # Eq. (14)
            current_gyro.reshape(3) - previous_context['gyro'],  #Eq. (14)
            previous_context['state_residual'],  #Eq. (13)
            previous_context['state_innovation'],  # Eq. (12)
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
        obs_raw = torch.stack((residual_k, innovation_now), dim=1)  # Yan Eqs. (11), (10)
        channel_mask = torch.stack((residual_mask, current_mask), dim=1)

        fixed_nn = torch.cat((
            fixed_raw[0:3] / torch.clamp(torch.linalg.vector_norm(fixed_raw[0:3]), min=FEATURE_L2_EPS),
            fixed_raw[3:6] / torch.clamp(torch.linalg.vector_norm(fixed_raw[3:6]), min=FEATURE_L2_EPS),
            fixed_raw[6:15] / torch.clamp(torch.linalg.vector_norm(fixed_raw[6:15]), min=FEATURE_L2_EPS),
            fixed_raw[15:24] / torch.clamp(torch.linalg.vector_norm(fixed_raw[15:24]), min=FEATURE_L2_EPS),
        ))
        obs_columns = []
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_mask[:, channel]
            values = torch.where(valid, obs_raw[:, channel], torch.zeros_like(obs_raw[:, channel]))
            obs_columns.append(values / torch.clamp(torch.linalg.vector_norm(values), min=FEATURE_L2_EPS))
        obs_nn = torch.stack(obs_columns, dim=1)
        obs_nn = torch.where(channel_mask, obs_nn, torch.zeros_like(obs_nn))

        pad = network_nmax - n
        if pad:
            obs_nn = torch.cat((obs_nn, torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=obs_nn.dtype, device=DEVICE)))
            current_mask = torch.cat((current_mask, torch.zeros(pad, dtype=torch.bool, device=DEVICE)))
            channel_mask = torch.cat((channel_mask, torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=torch.bool, device=DEVICE)))
            innovation_now = torch.cat((innovation_now, torch.zeros(pad, dtype=innovation_now.dtype, device=DEVICE)))
        return fixed_nn, obs_nn, current_mask, channel_mask, innovation_now

    def _propagate(nav_state: TorchNavState, segments, *, training: bool):
        if not segments:
            return nav_state, training_imr_gyro_t[0], training_imr_accel_t[0]
        segment_tuple = tuple((int(i), float(dt)) for i, dt in segments)
        def core(position, velocity, dcm):
            state = TorchNavState(position, velocity, dcm)
            feature_gyro = training_imr_gyro_t[segment_tuple[0][0]]
            feature_accel = training_imr_accel_t[segment_tuple[0][0]]
            for imu_index, dt in segment_tuple:
                feature_gyro = training_imr_gyro_t[imu_index].reshape(3)
                feature_accel = training_imr_accel_t[imu_index].reshape(3)
                state = _torch_mechanize(state, feature_gyro, feature_accel, dt)
            return state.position_ecef_m, state.velocity_ecef_mps, state.body_to_ecef_dcm, feature_gyro, feature_accel
        inputs = (nav_state.position_ecef_m, nav_state.velocity_ecef_mps, nav_state.body_to_ecef_dcm)
        if training and any(value.requires_grad for value in inputs):
            position, velocity, dcm, feature_gyro, feature_accel = checkpoint(core, *inputs, use_reentrant=False)
        else:
            position, velocity, dcm, feature_gyro, feature_accel = core(*inputs)
        return TorchNavState(position, velocity, dcm), feature_gyro, feature_accel

    # Recursive TBPTT
    def _rollout(start_sample: int, stop_sample: int, *, phase: str | None=None, initial_state=None, loss_start_sample: int | None=None):
        training = phase is not None
        metric_start = start_sample if loss_start_sample is None else int(loss_start_sample)

        if initial_state is None:
            start_nav = history_rows[start_sample]['posterior_nav']
            nav_state = TorchNavState(
                torch.as_tensor(start_nav.position_ecef_m, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.velocity_ecef_mps, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.body_to_ecef_dcm, dtype=torch.float64, device=DEVICE).clone(),
            )
            context_row = history_rows[start_sample]
            x_pred = np.asarray(context_row['x_pred'], dtype=float).reshape(INS_STATE_DIM)
            x_post = np.asarray(context_row['x_post'], dtype=float).reshape(INS_STATE_DIM)
            previous_context = {
                'sat_ids': tuple(context_row['sat_ids']),
                'residual': torch.as_tensor(context_row['residual'], dtype=torch.float64, device=DEVICE),
                'x_post': torch.as_tensor(x_post, dtype=torch.float64, device=DEVICE),
                'state_innovation': torch.as_tensor(x_post - x_pred, dtype=torch.float64, device=DEVICE),
                'state_residual': torch.as_tensor(np.zeros(INS_STATE_DIM) if start_sample == 0 else x_post - np.asarray(history_rows[start_sample - 1]['x_post'], dtype=float), dtype=torch.float64, device=DEVICE),
                'accel': torch.as_tensor(context_row['accel'], dtype=torch.float64, device=DEVICE),
                'gyro': torch.as_tensor(context_row['gyro'], dtype=torch.float64, device=DEVICE),
            }
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

        with (torch.enable_grad() if training else torch.inference_mode()):
            for sample_index in range(start_sample, stop_sample):
                row = history_rows[sample_index + 1]
                if row['preceding_interval_segments']:
                    nav_state, feature_gyro, feature_accel = _propagate(nav_state, row['preceding_interval_segments'], training=training)

                fusion_index = int(row['fusion_index'])
                measurements = tuple(clock_observable(row['fixed_measurements']))
                variances = np.asarray([m.variance_m2 for m in measurements], dtype=float)
                prepared = (
                    tuple(m.sat_id for m in measurements),
                    torch.as_tensor(np.stack([m.satellite_position_reception_ecef_m for m in measurements]), dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.pseudorange_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.satellite_clock_bias_s for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.ionosphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor([m.troposphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE),
                    torch.as_tensor(_clock_projection(measurements, variances), dtype=torch.float64, device=DEVICE),
                )
                innovation_now, current_sat_ids = _torch_innovation(nav_state, prepared)
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _network_input(previous_context, current_sat_ids, innovation_now, feature_accel, feature_gyro)
                output = model(
                    fixed_nn.float().unsqueeze(0),
                    obs_nn.float().unsqueeze(0),
                    mask_nn.unsqueeze(0),
                    channel_nn.unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
                recurrent_state = output.recurrent_state
                correction = state_update(output, innovation_nn.float().unsqueeze(0))[0].double()

                if sample_index >= metric_start:
                    state_error = torch.cat((
                        fusion_truth_position_t[fusion_index] - nav_state.position_ecef_m,
                        fusion_truth_velocity_t[fusion_index] - nav_state.velocity_ecef_mps,
                        _torch_so3_log(fusion_truth_dcm_t[fusion_index] @ nav_state.body_to_ecef_dcm.T) / ATTITUDE_FEEDBACK_SIGN,
                    )) - correction[:SUPERVISED_STATE_DIM]
                    state_loss = torch.sum(state_error ** 2)  # Yan Eq. (30)
                    if not bool(torch.isfinite(state_loss).detach().cpu()):
                        raise FloatingPointError(f'non-finite Yan Eq.(30) loss at sample {sample_index}')
                    state_losses.append(state_loss)
                    state_sum += float(state_loss.detach().cpu())
                    position_sum += float(torch.sum(state_error[:3] ** 2).detach().cpu())
                    metric_count += 1

                correction = correction.reshape(INS_STATE_DIM)
                nav_state = TorchNavState(
                    nav_state.position_ecef_m + correction[0:3],
                    nav_state.velocity_ecef_mps + correction[3:6],
                    _torch_so3_exp(ATTITUDE_FEEDBACK_SIGN * correction[6:9]) @ nav_state.body_to_ecef_dcm,
                )
                posterior_residual, _ = _torch_innovation(nav_state, prepared)
                previous_context = {
                    'sat_ids': tuple(current_sat_ids),
                    'residual': posterior_residual,
                    'x_post': correction,
                    'state_innovation': correction,
                    'state_residual': correction - previous_context['x_post'],
                    'accel': feature_accel.reshape(3),
                    'gyro': feature_gyro.reshape(3),
                }

        mean_loss = torch.stack(state_losses).mean()
        if training:
            mean_loss.backward()
            carried = {
                'nav': TorchNavState(nav_state.position_ecef_m.detach(), nav_state.velocity_ecef_mps.detach(), nav_state.body_to_ecef_dcm.detach()),
                'context': {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in previous_context.items()},
                'feature_accel': feature_accel.detach(),
                'feature_gyro': feature_gyro.detach(),
                'recurrent_state': None if recurrent_state is None else tuple(value.detach() for value in recurrent_state),
            }
        else:
            carried = None
        return {
            'eq30': state_sum / metric_count,
            'position_rmse_m': math.sqrt(position_sum / metric_count),
            'state_sum': state_sum,
            'position_sum': position_sum,
            'count': metric_count,
            'rollout_state': carried,
        }

    def _train_block(phase: str, optimizer: torch.optim.Optimizer):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if phase == 'filter':
            model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
            model.lstm.train()
            model.attention.train()
            model.gain_head.train()
            active_parameters = filter_parameters
        else:
            model.conv.train()
            model.lstm.train()
            model.lstm.lstm.dropout = 0.0
            active_parameters = representation_parameters
        for parameter in active_parameters:
            parameter.requires_grad_(True)

        carried = None
        state_sum = 0.0
        position_sum = 0.0
        count = 0
        losses = []
        gradients = []

        for window_start in range(0, training_sample_count, OPTIMIZER_WINDOW_SIZE):
            optimizer.zero_grad(set_to_none=True)
            metrics = _rollout(
                window_start,
                min(window_start + OPTIMIZER_WINDOW_SIZE, training_sample_count),
                phase=phase,
                initial_state=carried,
            )
            carried = metrics['rollout_state']
            regularization = GAMMA_L2 * sum(
                torch.sum(p.double() * p.double()) for p in all_network_parameters
            )  # Yan Eq. (32)
            regularization.backward()
            active_grads = [p for p in active_parameters if p.grad is not None]
            grad_sq = 0.0
            for parameter in active_grads:
                if not bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu()):
                    raise FloatingPointError('non-finite gradient')
                grad64 = parameter.grad.detach().double()
                grad_sq += float(torch.sum(grad64 * grad64).cpu())
            grad_norm = math.sqrt(grad_sq)
            if grad_norm > GRADIENT_CLIP_NORM:
                scale = GRADIENT_CLIP_NORM / grad_norm
                with torch.no_grad():
                    for parameter in active_grads:
                        parameter.grad.mul_(scale)
            optimizer.step()
            loss = metrics['eq30'] + float(regularization.detach().cpu())
            if not math.isfinite(loss):
                raise FloatingPointError(f'non-finite {phase} loss')
            losses.append(loss)
            gradients.append(grad_norm)
            state_sum += metrics['state_sum']
            position_sum += metrics['position_sum']
            count += metrics['count']

        return {
            'loss': float(np.mean(losses)),
            'position_rmse_m': math.sqrt(position_sum / count),
            'gradient_l2_norm': float(np.mean(gradients)),
        }

    # Causal validation.
    def _evaluate(metric_start: int, metric_stop: int):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
        return _rollout(0, metric_stop, phase=None, loss_start_sample=metric_start)

    # Zero-gain fallback checkpoint.
    initial_validation = _evaluate(validation_start_sample, data01_total_sample_count)
    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_selection_eq30 = float(initial_validation['eq30'])
    best_selection_rmse_m = float(initial_validation['position_rmse_m'])
    best_selection_epoch = 0
    training_history = []
    epochs_without_improvement = 0

    for epoch in range(1, TRAINING_EPOCHS + 1):
        try:
            filter_phase = _train_block('filter', filter_optimizer)
            representation_phase = _train_block('representation', representation_optimizer)
            overall_train = _evaluate(0, training_sample_count)
            validation = _evaluate(validation_start_sample, data01_total_sample_count)
        except FloatingPointError as error:
            print(f'Epoch {epoch:02d}: stopped - {error}')
            break

        overall_train_loss = float(overall_train['eq30'])
        validation_loss = float(validation['eq30'])
        validation_rmse = float(validation['position_rmse_m'])
        if math.isfinite(validation_loss) and validation_loss < best_selection_eq30:
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_selection_eq30 = validation_loss
            best_selection_rmse_m = validation_rmse
            best_selection_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        training_history.append({
            'epoch': epoch,
            'filter_train_loss': filter_phase['loss'],
            'filter_gradient_l2_norm': filter_phase['gradient_l2_norm'],
            'representation_train_loss': representation_phase['loss'],
            'representation_gradient_l2_norm': representation_phase['gradient_l2_norm'],
            'overall_train_loss': overall_train_loss,
            'validation_loss': validation_loss,
            'validation_position_rmse_m': validation_rmse,
        })
        print(
            f"Epoch {epoch:02d}/{TRAINING_EPOCHS} | "
            f"filter train loss={filter_phase['loss']:.6g} | "
            f"filter grad={filter_phase['gradient_l2_norm']:.3g} | "
            f"representation train loss={representation_phase['loss']:.6g} | "
            f"representation grad={representation_phase['gradient_l2_norm']:.3g} | "
            f"overall train loss={overall_train_loss:.6g} | "
            f"validation loss={validation_loss:.6g} | "
            f"validation RMSE={validation_rmse:.3f} m"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict({key: value.to(DEVICE) for key, value in best_model_state.items()})
    model.eval()
    model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
    best_epoch = int(best_selection_epoch)
    best_val_state_loss = float(best_selection_eq30)
    best_val_position_rmse_m = float(best_selection_rmse_m)

    # Test fusion context.
    def _online_context(sat_ids, posterior_residual, x_pred, x_post, accel, gyro, previous_context=None):
        x_pred = np.asarray(x_pred, dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(x_post, dtype=float).reshape(INS_STATE_DIM)
        return {
            'sat_ids': tuple(sat_ids),
            'residual': np.asarray(posterior_residual, dtype=float).copy(),
            'x_post': x_post.copy(),
            'state_innovation': (x_post - x_pred).copy(),
            'state_residual': (np.zeros(INS_STATE_DIM) if previous_context is None else x_post - np.asarray(previous_context['x_post'], dtype=float).reshape(INS_STATE_DIM)).copy(),
            'accel': np.asarray(accel, dtype=float).reshape(3).copy(),
            'gyro': np.asarray(gyro, dtype=float).reshape(3).copy(),
        }

    # Checkpoint.
    torch.save({
        'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        'nmax': int(network_nmax),
        'state_order': '[delta_p,delta_v,delta_theta]',
        'architecture': 'Yan_masked_CNN24_k3_LSTM5x64_attention_FC_KG',
        'feature_normalization': FEATURE_NORMALIZATION_MODE,
        'selected_epoch': best_epoch,
        'validation_loss': best_val_state_loss,
        'validation_position_rmse_m': best_val_position_rmse_m,
        'learning_rate': LEARNING_RATE,
        'optimizer_window_size': OPTIMIZER_WINDOW_SIZE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
    }, OUTPUT_DIR / 'best_model.pt')
    (OUTPUT_DIR / 'history.json').write_text(json.dumps(training_history, indent=2), encoding='utf-8')

    if training_history:
        import matplotlib.pyplot as plt

        epochs = [row['epoch'] for row in training_history]
        overall_train_losses = [row['overall_train_loss'] for row in training_history]
        validation_losses = [row['validation_loss'] for row in training_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, overall_train_losses, marker='o', label='Overall train loss')
        plt.plot(epochs, validation_losses, marker='o', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overall Train Loss vs Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'training_validation_loss.png', dpi=150)
        plt.show()
        plt.close()

    # Test
    test_root = Path(TEST_DATASET_DIR)
    test_rover_xml = next(item for item in ET.fromstring((test_root / 'README.xml').read_text(encoding='utf-8', errors='replace')).findall('ROVE') if (item.findtext('ID') or '').strip() == '01')
    test_rover_imu_type = (test_rover_xml.findtext('SINS_IMUType') or '').strip()
    test_rover_mounting_xyz_deg = np.fromstring(test_rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    test_antenna_truth = load_truth(unique_file(test_root, ('ROVE_GroundTruth.txt', 'ROVE_01_GroundTruth.txt', 'Rove_01_GroundTruth.txt'), 'rover 01 ground truth'))
    test_imu_truth = load_truth(test_root / f'{test_rover_imu_type}_GroundTruth.txt')
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

    test_rinex = RINEXObs(unique_file(test_root, ('ROVE.*O', 'ROVE.*o', 'ROVE_01.*O', 'ROVE_01.*o'), 'rover 01 RINEX observation'))
    first_test_rinex_epoch = next(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}))
    test_imr_path, test_imr_data_rate_hz, test_imr_gyro_scale, test_imr_accel_scale, test_imr_time_tag_bias_ms, test_imr_record_dtype, test_imr_record_count, test_imr_record_bytes = read_imr_layout(test_root / f'{test_rover_imu_type}.imr')
    test_imr_records = np.memmap(test_imr_path, dtype=test_imr_record_dtype, mode='r', offset=512, shape=(test_imr_record_count,))
    test_imr_tow_all = np.asarray(test_imr_records['tow'], dtype=np.float64).copy()
    del test_imr_records
    test_imr_tow_all[test_imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    test_imr_tow_all -= test_imr_time_tag_bias_ms * 1e-3
    test_imr_time_all = anchor_imr_gpst(test_imr_tow_all, float(first_test_rinex_epoch.time_gpst_s))
    test_antenna_truth_time = check_time_axis('test antenna truth', test_antenna_truth.time_gpst_s)
    test_imu_truth_time = check_time_axis('test IMU truth', test_imu_truth.time_gpst_s)
    test_common_end = min(float(test_imr_time_all[-1]), float(test_antenna_truth_time[-1]), float(test_imu_truth_time[-1]))
    test_start = int(np.searchsorted(test_imr_time_all, max(float(test_imr_time_all[0]), float(test_antenna_truth_time[0]), float(test_imu_truth_time[0])), side='left'))
    test_gnss_epochs = tuple(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=float(test_imr_time_all[test_start]), end_time_gpst_s=test_common_end, max_epochs=MAX_TEST_FUSION_EPOCHS, require_measurements=True))
    test_fusion_time = check_time_axis('test fusion', np.asarray([epoch.time_gpst_s for epoch in test_gnss_epochs], dtype=float))
    test_requested_stop = min(int(np.searchsorted(test_imr_time_all, test_fusion_time[-1], side='left')) + 1, int(np.searchsorted(test_imr_time_all, test_common_end, side='left')) + 1, len(test_imr_time_all))
    test_imr_records = np.fromfile(test_imr_path, dtype=test_imr_record_dtype, count=test_requested_stop - test_start, offset=512 + test_start * test_imr_record_bytes)
    test_imr_angular_rate_body_radps = np.deg2rad(test_imr_records['counts'][:, :3].astype(float) * test_imr_gyro_scale * test_imr_data_rate_hz)
    test_imr_acceleration_body_mps2 = test_imr_records['counts'][:, 3:6].astype(float) * test_imr_accel_scale * test_imr_data_rate_hz
    test_imr_time = test_imr_time_all[test_start:test_requested_stop].copy()
    del test_imr_tow_all, test_imr_time_all

    test_query_time = np.concatenate(([test_imr_time[0]], test_fusion_time))
    test_antenna_position = interp_truth(test_antenna_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    test_imu_position, test_imu_velocity, test_imu_heading, test_imu_pitch, test_imu_roll = interp_truth(test_imu_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    test_fusion_antenna_truth_position = test_antenna_position[1:]
    test_C_b_e = body_to_ecef(test_imu_position[0], test_imu_heading[0], test_imu_pitch[0], test_imu_roll[0], test_rover_mounting_xyz_deg)
    test_lever_arm_b_m = vehicle_to_body(*test_rover_mounting_xyz_deg) @ np.fromstring(test_rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ').reshape(3)
    test_Qc = np.diag(np.concatenate((test_imu_model_values['PNSD_Pos'], test_imu_model_values['PNSD_Vel'], test_imu_model_values['PNSD_Att'] * (np.pi / 180.0))) ** 2)

    test_iono_header = {}
    with unique_file(test_root, ('brdm*.*p', 'brdm*.*P', 'brdm*.rnx', 'BRDM*.RNX'), 'broadcast navigation').open('r', encoding='ascii', errors='replace') as stream:
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
    test_gnss_processor = GNSSProcessor(
        unique_file(test_root, ('*.SP3', '*.sp3'), 'SP3 precise-orbit'),
        unique_file(test_root, ('*.CLK', '*.clk'), 'RINEX clock'),
        min_elevation_deg=MIN_GNSS_ELEVATION_DEG,
        use_ionosphere=True,
        use_troposphere=True,
        broadcast_ionosphere_coefficients=test_ionosphere_coefficients,
    )
    test_leo_simulator = LEOSimulator(
        tle_provider,
        test_ionosphere_coefficients['G'],
        seed=TEST_LEO_SEED,
        tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M,
        tx_max_iterations=LEO_TX_MAX_ITERATIONS,
        minimum_elevation_deg=LEO_MIN_ELEVATION_DEG,
        prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG,
        use_ionosphere=True,
        use_troposphere=True,
    )

    nav = NavState(test_imu_position[0].copy(), test_imu_velocity[0].copy(), test_C_b_e)
    P = np.diag(np.concatenate((
        test_imu_model_values['ISDV_Pos'],
        test_imu_model_values['ISDV_Vel'],
        test_imu_model_values['ISDV_Att'] * (np.pi / 180.0),
    )) ** 2)
    online_rows = []
    previous_online = None
    recurrent_state = None
    last_gyro = np.asarray(test_imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(test_imr_acceleration_body_mps2[0], dtype=float).reshape(3)
    last_feature_gyro, last_feature_accel = last_gyro, last_accel

    for event_kind, event_start, event_end, event_index in fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
        if event_kind == 'imu':
            last_gyro = np.asarray(test_imr_angular_rate_body_radps[event_index], dtype=float).reshape(3)
            last_accel = np.asarray(test_imr_acceleration_body_mps2[event_index], dtype=float).reshape(3)
            last_feature_gyro, last_feature_accel = last_gyro, last_accel
            nav = mechanize(nav, last_gyro, last_accel, event_end - event_start)
            Phi, Qd = van_loan(error_dynamics(nav, last_accel), test_Qc, event_end - event_start)
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
            continue

        fusion_index = event_index
        t = event_start
        measurements = clock_observable(tuple(test_gnss_processor.prepare_epoch(test_gnss_epochs[fusion_index], nav, test_lever_arm_b_m)) + tuple(test_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])))
        if not measurements:
            continue

        if previous_online is None:
            warm = measurement_model(nav, measurements, test_lever_arm_b_m)
            correction, P, _ = kf_update(P, warm.innovation, warm.H, warm.R)
            nav = inject_error(nav, correction)
            previous_online = _online_context(warm.sat_ids, innovation_only(nav, measurements, test_lever_arm_b_m), np.zeros(INS_STATE_DIM), correction, last_feature_accel, last_feature_gyro)
            continue

        if len(measurements) > network_nmax:
            measurements = select_measurements(measurements, network_nmax)
        meas = measurement_model(nav, measurements, test_lever_arm_b_m)
        current_sat_ids = meas.sat_ids
        innovation_k = np.asarray(meas.innovation, dtype=float).reshape(len(current_sat_ids))
        residual_k = np.zeros(len(current_sat_ids), dtype=float)
        residual_mask = np.zeros(len(current_sat_ids), dtype=bool)
        previous_residual = dict(zip(previous_online['sat_ids'], previous_online['residual']))
        for slot, sat_id in enumerate(current_sat_ids):
            if sat_id in previous_residual:
                residual_k[slot] = previous_residual[sat_id]
                residual_mask[slot] = True

        fixed_nn = _normalize_fixed(np.concatenate([
            np.asarray(last_feature_accel, dtype=float).reshape(3) - previous_online['accel'],  # Yan Eq. (14)
            np.asarray(last_feature_gyro, dtype=float).reshape(3) - previous_online['gyro'],  # Yan Eq. (14)
            previous_online['state_residual'],  # Yan Eq. (13)
            previous_online['state_innovation'],  # Yan Eq. (12)
        ]))
        mask_nn = np.ones(len(current_sat_ids), dtype=bool)
        channel_nn = np.stack([residual_mask, mask_nn], axis=1)
        obs_nn = _normalize_obs(
            np.stack([residual_k, innovation_k], axis=1),  # Yan Eqs. (11), (10)
            channel_nn,
        )

        count = len(current_sat_ids)
        obs_padded = np.zeros((network_nmax, OBSERVATION_FEATURE_DIM), dtype=float)
        mask_padded = np.zeros(network_nmax, dtype=bool)
        channel_padded = np.zeros((network_nmax, OBSERVATION_FEATURE_DIM), dtype=bool)
        innovation_padded = np.zeros(network_nmax, dtype=float)
        obs_padded[:count] = obs_nn
        mask_padded[:count] = mask_nn
        channel_padded[:count] = channel_nn
        innovation_padded[:count] = innovation_k  # Yan Eq. (16)
        with torch.inference_mode():
            output = model(
                torch.tensor(fixed_nn[None], dtype=torch.float32, device=DEVICE),
                torch.tensor(obs_padded[None], dtype=torch.float32, device=DEVICE),
                torch.tensor(mask_padded[None], dtype=torch.bool, device=DEVICE),
                torch.tensor(channel_padded[None], dtype=torch.bool, device=DEVICE),
                recurrent_state=recurrent_state,
            )
            recurrent_state = output.recurrent_state
            correction = state_update(
                output,
                torch.tensor(innovation_padded[None], dtype=torch.float32, device=DEVICE),
            )[0].cpu().numpy().astype(float)
        active_gain = output.kalman_gain[0].cpu().numpy().astype(float)[:, :len(measurements)]
        if not np.all(np.isfinite(correction)) or not np.all(np.isfinite(active_gain)):
            raise FloatingPointError('non-finite Data02 neural update')
        P = learned_covariance(P, active_gain, meas.H, meas.R, correction)
        nav = inject_error(nav, correction)
        posterior_position = nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(test_lever_arm_b_m).reshape(3)
        previous_online = _online_context(meas.sat_ids, innovation_only(nav, measurements, test_lever_arm_b_m), np.zeros(INS_STATE_DIM), correction, last_feature_accel, last_feature_gyro, previous_context=previous_online)
        hpl_m, vpl_m = INTEGRITY.protection_levels(P, posterior_position) if INTEGRITY is not None else (float('nan'), float('nan'))
        online_rows.append((t, posterior_position.copy(), test_fusion_antenna_truth_position[fusion_index].copy(), hpl_m, vpl_m))

    if not online_rows:
        raise RuntimeError('Data02 test produced no usable neural epochs')

    online_time = np.asarray([row[0] for row in online_rows], dtype=float)
    estimate = np.stack([row[1] for row in online_rows])
    truth_aligned = np.stack([row[2] for row in online_rows])
    ned_error = np.empty_like(estimate)
    for i, (est, truth_i) in enumerate(zip(estimate, truth_aligned)):
        lat, lon, _ = ecef_llh(truth_i)
        ned_error[i] = ecef_to_ned(lat, lon) @ (est - truth_i)
    error_3d = np.linalg.norm(ned_error, axis=1)
    rmse = np.append(np.sqrt(np.mean(ned_error ** 2, axis=0)), math.sqrt(float(np.mean(error_3d ** 2))))

    np.savez(
        OUTPUT_DIR / 'test_evaluation.npz',
        time_gpst_s=online_time,
        estimate_ecef_m=estimate,
        truth_ecef_m=truth_aligned,
        ned_error_m=ned_error,
        error_3d_m=error_3d,
        hpl_m=np.asarray([row[3] for row in online_rows], dtype=float),
        vpl_m=np.asarray([row[4] for row in online_rows], dtype=float),
        rmse_north_east_down_3d_m=rmse,
    )
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps({
        'selected_epoch': best_epoch,
        'validation_loss': best_val_state_loss,
        'validation_position_rmse_m': best_val_position_rmse_m,
        'test_epochs': len(online_rows),
        'test_position_rmse_ned3d_m': [float(v) for v in rmse],
        'measurement_mode': 'pseudorange_only',
        'leo_orbit': 'TLE_SGP4',
        'navigation_state': '9_state_[delta_p,delta_v,delta_theta]',
        'FDE_DIA': 'removed',
    }, indent=2), encoding='utf-8')

    print(f'Data02 test | position RMSE: N={rmse[0]:.3f} m, E={rmse[1]:.3f} m, D={rmse[2]:.3f} m, 3D={rmse[3]:.3f} m')
