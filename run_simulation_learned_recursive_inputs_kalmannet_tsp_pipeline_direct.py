from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import csv
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
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as SpatialRotation
from sgp4.api import Satrec, WGS72
from sgp4.io import verify_checksum

REPRODUCIBILITY_SEED = 12345

def set_reproducibility_seed(seed: int) -> None:
    """Make model initialization, V2 mini-batch sampling, and stochastic code reproducible."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

EARTH_SEMI_MAJOR_AXIS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SEMI_MINOR_AXIS_M = EARTH_SEMI_MAJOR_AXIS_M * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_ROTATION_RATE_RADPS = 7.292115e-05
EARTH_GRAVITATIONAL_PARAMETER_M3PS2 = 398600441800000.0
SPEED_OF_LIGHT_MPS = 299792458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0
IMU_ERROR_MODEL_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/IMUErrorModel.txt')
IMR_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/ISA-100C.imr')
IMU_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/ISA-100C_GroundTruth.txt')
ROVE_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/ROVE_GroundTruth.txt')
RINEX_OBS_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/ROVE.23O')
SP3_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20230020000_01D_05M_ORB.SP3')
CLK_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20230020000_01D_30S_CLK.CLK')
NAV_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/brdm0020.23p')
README_XML_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data01_20230102_ISA-100C_Vehicle_Complex/README.xml')
LEO_TLE_DIR = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/LEO_TLE')

TEST_README_XML_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/README.xml')
TEST_ROVE_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/ROVE_GroundTruth.txt')
TEST_IMU_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/ISA-100C_GroundTruth.txt')
TEST_RINEX_OBS_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/ROVE.22O')
TEST_IMR_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/ISA-100C.imr')
TEST_SP3_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20220680000_01D_15M_ORB.SP3')
TEST_CLK_PATH =  Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20220680000_01D_30S_CLK.CLK')
TEST_NAV_PATH = Path('/kaggle/input/datasets/dlrmrsj/mknet-project-d/Data02_20220309_ISA-100C_Vehicle_Complex/brdm0680.22p')

OUTPUT_DIR = Path('/kaggle/working/direct_run')
MAX_TRUTH_INTERPOLATION_GAP_S = 2.0
MIN_GNSS_ELEVATION_DEG = 5.0
USE_IONOSPHERE = True
USE_TROPOSPHERE = True
LEO_MIN_ELEVATION_DEG = 10.0
LEO_SEED = 0
TLE_MAX_AGE_DAYS = 1.0
TLE_ALLOW_DEGRADED_EOP = False
TLE_ALLOW_NON_TLE_FILES = True
GNSS_TX_EPSILON_POSITION_M = 0.001
GNSS_TX_MAX_ITERATIONS = 20
LEO_TX_EPSILON_POSITION_M = 0.001
LEO_TX_MAX_ITERATIONS = 20
RUN_CLASSICAL_TEST_BASELINE = False
LEO_PREFILTER_GUARD_DEG = 0.5
VAN_LOAN_TAYLOR_ORDER = 10
VAN_LOAN_TAYLOR_MAX_NORM_1 = 0.2
VAN_LOAN_VALIDATE_CALLS = 8
LEO_IONOSPHERE_LOWER_HEIGHT_M = 100000.0
LEO_IONOSPHERE_UPPER_HEIGHT_M = 1000000.0
LEO_URA_SIGMA_M = 1.5
LEO_CODE_LOOP_BANDWIDTH_HZ = 2.0
LEO_CORRELATOR_SPACING_CHIPS = 0.1
LEO_CORRELATOR_ACCUMULATION_S = 0.02
LEO_CODE_CHIPPING_RATE_HZ = 1023000.0
GOGPS_A = 30.0
GOGPS_S0_DBHZ = 10.0
GOGPS_S1_DBHZ = 50.0
GOGPS_A_DB = 20.0

def calendar_to_gpst_seconds(year: int, month: int, day: int, hour: int, minute: int, second: float, time_system: str='GPS') -> float:
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

def anchor_imr_tow_to_gpst_seconds(tow_s: np.ndarray, anchor_time_gpst_s: float) -> np.ndarray:
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

def build_exact_fusion_timeline(imu_time_gpst_s: np.ndarray, fusion_time_gpst_s: np.ndarray, *, through_last_fusion: bool=True):
    imu_time = np.asarray(imu_time_gpst_s, dtype=float).reshape(-1)
    fusion_time = np.asarray(fusion_time_gpst_s, dtype=float).reshape(-1)

    if fusion_time.size == 0:
        return fusion_time

    if through_last_fusion:
        stop = int(np.searchsorted(imu_time, fusion_time[-1], side='left')) + 1
        imu_time = imu_time[:min(max(stop, 2), len(imu_time))]
    
    fusion_index = 0
    segment_start = float(imu_time[0])
    for imu_index in range(1, len(imu_time)):
        interval_end = float(imu_time[imu_index])
        while fusion_index < len(fusion_time) and fusion_time[fusion_index] <= interval_end + 1e-12:
            t = float(fusion_time[fusion_index])

            if t > segment_start + 1e-12:
                yield ('propagation', segment_start, t, imu_index)

            yield ('fusion', t, t, fusion_index)
            segment_start = t
            fusion_index += 1

        if interval_end > segment_start + 1e-12:
            yield ('propagation', segment_start, interval_end, imu_index)

        segment_start = interval_end

def ecef_to_llh(position_ecef_m: np.ndarray) -> tuple[float, float, float]:
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
    gamma, beta, alpha = np.deg2rad([x_rot_deg, y_rot_deg, z_rot_deg])
    cb, sb = (np.cos(beta), np.sin(beta))
    cg, sg = (np.cos(gamma), np.sin(gamma))
    ca, sa = (np.cos(alpha), np.sin(alpha))
    Ry = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    Rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz

def saastamoinen_delay_m(height_m: float, elevation_rad: float) -> float:
    if elevation_rad <= 0.0:
        return float('inf')
    h = max(-100.0, min(float(height_m), 10000.0))
    temperature_k = 15.0 - 0.0065 * h + 273.15
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-05 * h) ** 5.2568
    water_vapor_hpa = 6.108 * 0.7 * math.exp((17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45))
    z = math.pi / 2.0 - elevation_rad
    return 0.002277 / math.cos(z) * (pressure_hpa + (1255.0 / temperature_k + 0.05) * water_vapor_hpa - 1.16 * math.tan(z) ** 2)

def klobuchar_delay_m(time_gps_tow_s: float, latitude_rad: float, longitude_rad: float, elevation_rad: float, azimuth_rad: float, alpha_s, beta_s) -> float:
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
    angle = EARTH_ROTATION_RATE_RADPS * float(transit_s)
    c, s = (np.cos(angle), np.sin(angle))
    C_rx_tx = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sat_rx = C_rx_tx @ np.asarray(satellite_position_tx_ecef_m, dtype=float).reshape(3)
    los = np.asarray(receiver_position_ecef_m, dtype=float).reshape(3) - sat_rx
    rho = float(np.linalg.norm(los))
    return (rho, los / rho, sat_rx)

def elevation_azimuth_from_ned_matrix(c_ecef_ned: np.ndarray, los_satellite_to_receiver_ecef: np.ndarray) -> tuple[float, float]:
    receiver_to_satellite_ned = np.asarray(c_ecef_ned, dtype=float) @ -np.asarray(los_satellite_to_receiver_ecef, dtype=float)
    elevation = float(np.arcsin(np.clip(-receiver_to_satellite_ned[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(receiver_to_satellite_ned[1], receiver_to_satellite_ned[0]) % (2.0 * np.pi))
    return (elevation, azimuth)

_FREQUENCY_HZ = {('G', '1'): 1575420000.0, ('G', '2'): 1227600000.0, ('G', '5'): 1176450000.0, ('C', '1'): 1575420000.0, ('C', '2'): 1561098000.0, ('C', '5'): 1176450000.0, ('C', '7'): 1207140000.0, ('C', '8'): 1191795000.0, ('C', '6'): 1268520000.0}
_SIGNAL_PREFS = {'G': ('1C', '1W', '1P', '2W', '2L', '2X', '5Q', '5X', '5I'), 'C': ('2I', '1I', '2X', '1X', '1P', '1D', '5X', '5P', '5D', '7I', '7X', '6I', '6X')}

class RINEXObservationFile:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.version = 0.0
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
                    count = self._observation_count.get(constellation, 0)
                    payload = sat_line[3:].rstrip('\n')
                    fields = [payload[i:i + 16] for i in range(0, len(payload), 16)]
                    while len(fields) < count:
                        continuation = stream.readline()
                        if not continuation:
                            break
                        payload = continuation[3:].rstrip('\n')
                        fields.extend(payload[i:i + 16] for i in range(0, len(payload), 16))
                    fields = fields[:count]
                    if allowed_constellations and constellation not in allowed_constellations:
                        continue

                    index_by_type = self._index_by_type.get(constellation, {})
                    measurement = None
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
                        measurement = (sat_id, constellation, suffix, float(frequency), float(pseudorange), cn0)
                        break
                    if measurement is not None:
                        measurements.append(measurement)
                if start_time_gpst_s is not None and time_gpst_s < float(start_time_gpst_s):
                    continue
                if require_measurements and not measurements:
                    continue
                yield (time_gpst_s, tuple(measurements))
                yielded += 1
                if max_epochs is not None and yielded >= int(max_epochs):
                    break

class RINEXClock:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.time_scale = 'GPS'
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
        self._series = {}
        for sat_id, (times, bias) in records.items():
            order = np.argsort(times)
            self._series[sat_id] = (np.asarray(times)[order], np.asarray(bias)[order])

class SP3Orbit:

    def __init__(self, path: str | Path, interpolation_points: int=9):
        self.path = Path(path)
        self.interpolation_points = interpolation_points
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
        self.series = {}
        for sat_id, (times, positions, clocks) in temporary.items():
            times = np.asarray(times, dtype=float)
            order = np.argsort(times)
            times = times[order]
            clock = np.asarray(clocks, dtype=float)[order]
            finite_clock = np.isfinite(clock)
            self.series[sat_id] = {
                'time': times,
                'position': np.asarray(positions, dtype=float)[order],
                'clock_time': times[finite_clock],
                'clock_value': clock[finite_clock],
            }

    def position(self, sat_id: str, time_gpst_s: float) -> np.ndarray:
        s = self.series[sat_id]
        times = s['time']
        count = min(self.interpolation_points, len(times))
        i = int(np.searchsorted(times, time_gpst_s))
        start = max(0, min(len(times) - count, i - count // 2))
        w = slice(start, start + count)
        interpolation_times = times[w]
        interpolation_values = s['position'][w]
        scale = max(float(np.max(np.abs(interpolation_times - time_gpst_s))), 1.0)
        x = (interpolation_times - time_gpst_s) / scale
        y = []
        for column in range(interpolation_values.shape[1]):
            polynomial = BarycentricInterpolator(x, interpolation_values[:, column], rng=0)
            y.append(float(polynomial(0.0)))
        return np.asarray(y)

IMR_HEADER_FORMAT_BODY = '8scdiidddiid32s?BBB32s6h?iii354s'
IMR_RECORD_FORMAT_BODY = 'd6i'


def interpolate_ground_truth(truth, query_time_gpst_s: np.ndarray, max_gap_s: float, *, position_only: bool=False):
    truth_time, truth_position, truth_velocity, truth_heading, truth_pitch, truth_roll = truth
    query = np.asarray(query_time_gpst_s, dtype=float)
    upper = np.searchsorted(truth_time, query, side='left')
    upper = np.clip(upper, 0, len(truth_time) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = truth_time[upper] == query
    lower[exact] = upper[exact]
    gap = truth_time[upper] - truth_time[lower]
    weight = np.zeros_like(query)
    nz = gap > 0.0
    weight[nz] = (query[nz] - truth_time[lower[nz]]) / gap[nz]
    w0 = 1.0 - weight
    position = w0[:, None] * truth_position[lower] + weight[:, None] * truth_position[upper]
    if position_only:
        return position
    velocity = w0[:, None] * truth_velocity[lower] + weight[:, None] * truth_velocity[upper]
    heading_unwrapped = np.unwrap(np.deg2rad(truth_heading))
    heading = np.rad2deg(w0 * heading_unwrapped[lower] + weight * heading_unwrapped[upper]) % 360.0
    pitch = w0 * truth_pitch[lower] + weight * truth_pitch[upper]
    roll = w0 * truth_roll[lower] + weight * truth_roll[upper]
    return (position, velocity, heading, pitch, roll)

def body_to_ecef_from_ie_hpr(position_ecef_m: np.ndarray, heading_deg: float, pitch_deg: float, roll_deg: float, mounting_xyz_deg: np.ndarray) -> np.ndarray:
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
    relative = truth_body_to_ecef @ prior_body_to_ecef.T
    rotvec = SpatialRotation.from_matrix(relative).as_rotvec()
    return rotvec / ATTITUDE_FEEDBACK_SIGN

Array = np.ndarray
INS_STATE_DIM = 9
DIRECT_STATE_LABEL_DIM = 9
ATTITUDE_FEEDBACK_SIGN = -1.0
J2_UNITLESS = 0.00108262668
OMEGA_IE_E = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RADPS])
IDENTITY_3 = np.eye(3)
IDENTITY_STATE = np.eye(INS_STATE_DIM)
C_F_V = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

def _so3_exponential(rotation_vector_rad: Array) -> Array:
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
omega_x, omega_y, omega_z = OMEGA_IE_E
OMEGA_IE_SKEW = np.array([[0.0, -omega_z, omega_y], [omega_z, 0.0, -omega_x], [-omega_y, omega_x, 0.0]])

def _rotation(matrix: Array) -> Array:
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
    scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / r ** 3
    return scale * np.array([x * xy_factor, y * xy_factor, z * z_factor])

def mechanize_ecef(nav: NavigationState, angular_rate_body_radps: Array, specific_force_body_mps2: Array, dt_s: float) -> NavigationState:
    dt = float(dt_s)
    C0 = nav.body_to_ecef_dcm
    C1 = _rotation(_so3_exponential(-OMEGA_IE_E * dt) @ C0 @ _so3_exponential(np.asarray(angular_rate_body_radps, dtype=float) * dt))
    Cmid = 0.5 * (C0 + C1)
    r = np.asarray(nav.position_ecef_m, dtype=float).reshape(3)
    wxr = np.array([-EARTH_ROTATION_RATE_RADPS * r[1], EARTH_ROTATION_RATE_RADPS * r[0], 0.0])
    wxwxr = np.array([-EARTH_ROTATION_RATE_RADPS * wxr[1], EARTH_ROTATION_RATE_RADPS * wxr[0], 0.0])
    v = np.asarray(nav.velocity_ecef_mps, dtype=float).reshape(3)
    wxv = np.array([-EARTH_ROTATION_RATE_RADPS * v[1], EARTH_ROTATION_RATE_RADPS * v[0], 0.0])
    acceleration = Cmid @ np.asarray(specific_force_body_mps2) + _gravitation_j2_ecef(r) - wxwxr - 2.0 * wxv
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavigationState(position, velocity, C1)

def gnss_antenna_position(nav: NavigationState, lever_arm_b_m: Array) -> Array:
    return nav.position_ecef_m + nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)

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
    pass

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

class GNSSPreprocessor:

    def __init__(self, orbit: SP3Orbit, clock: RINEXClock, min_elevation_deg: float=5.0, use_troposphere: bool=True, use_ionosphere: bool=True, broadcast_ionosphere_coefficients: dict[str, tuple[Array, Array]] | None=None):
        self.orbit = orbit
        self.clock = clock
        self.clock_satellites = frozenset(clock._series)
        self.min_elevation_rad = np.deg2rad(min_elevation_deg)
        self.use_troposphere = use_troposphere
        self.use_ionosphere = use_ionosphere
        self.iono = broadcast_ionosphere_coefficients or {}

    def prepare_epoch(self, epoch, nav: NavigationState, lever_arm_b_m: Array) -> tuple[PseudorangeMeasurement, ...]:
        epoch_time_gpst_s, epoch_measurements = epoch
        antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
        receiver_llh = ecef_to_llh(antenna_position)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch_measurements:
            sat_id, constellation, signal_suffix, frequency_hz, pseudorange_m, cn0_dbhz_raw = raw
            try:
                initial_position = self.orbit.position(sat_id, epoch_time_gpst_s)
            except (KeyError, ValueError):
                continue
            initial_range, _, _ = geometric_range(antenna_position, initial_position, 0.0)
            try:
                transmit_time, transit_s, state_tx_position = _iterate_transmit_time(
                    self.orbit.position,
                    sat_id,
                    epoch_time_gpst_s,
                    antenna_position,
                    initial_position,
                    initial_range / SPEED_OF_LIGHT_MPS,
                    GNSS_TX_EPSILON_POSITION_M,
                    GNSS_TX_MAX_ITERATIONS,
                )
            except (KeyError, ValueError, TransmitTimeConvergenceError):
                continue

            if sat_id in self.clock_satellites:
                clock_times, clock_bias = self.clock._series[sat_id]
                satellite_clock_bias_s = float(np.interp(transmit_time, clock_times, clock_bias))
            else:
                orbit_clock = self.orbit.series[sat_id]
                if orbit_clock['clock_time'].size == 0:
                    continue
                satellite_clock_bias_s = float(np.interp(transmit_time, orbit_clock['clock_time'], orbit_clock['clock_value']))

            _, los, satellite_position_rx = geometric_range(antenna_position, state_tx_position, transit_s)
            elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
            if elevation < self.min_elevation_rad:
                continue
            ionosphere = 0.0
            height_m = None
            if self.use_ionosphere:
                alpha, beta = self.iono[constellation]
                week = int(math.floor(float(epoch_time_gpst_s) / GPS_WEEK_S))
                tow = float(epoch_time_gpst_s) - week * GPS_WEEK_S
                if constellation == 'G':
                    reference_frequency_hz = 1575420000.0
                else:
                    if signal_suffix not in {'2I', '2X', '1I', '1X'}:
                        continue
                    tow = (tow - 14.0) % 604800.0
                    reference_frequency_hz = 1561098000.0
                lat, lon, height_m = receiver_llh
                ionosphere = klobuchar_delay_m(tow, lat, lon, elevation, azimuth, alpha, beta) * (reference_frequency_hz / frequency_hz) ** 2
            troposphere = 0.0
            if self.use_troposphere:
                if height_m is None:
                    height_m = receiver_llh[2]
                troposphere = saastamoinen_delay_m(height_m, elevation)

            cn0_dbhz = float(cn0_dbhz_raw) if cn0_dbhz_raw is not None else 45.0
            iono_reference_frequency_hz = 1575420000.0 if constellation == 'G' else 1561098000.0
            iono_sigma = ref35_ionosphere_sigma_m(elevation, receiver_llh[0]) * (iono_reference_frequency_hz / frequency_hz) ** 2 if self.use_ionosphere else 0.0
            tropo_sigma = float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) if self.use_troposphere else 0.0
            e = max(float(elevation), 1e-06)
            if cn0_dbhz >= GOGPS_S1_DBHZ:
                mp_variance = 1.0
            else:
                ratio = (cn0_dbhz - GOGPS_S1_DBHZ) / (GOGPS_S0_DBHZ - GOGPS_S1_DBHZ)
                term = 10.0 ** (-(cn0_dbhz - GOGPS_S1_DBHZ) / GOGPS_A_DB) * ((GOGPS_A / 10.0 ** (-(GOGPS_S0_DBHZ - GOGPS_S1_DBHZ) / GOGPS_A_DB) - 1.0) * ratio + 1.0)
                mp_variance = 1.0 / max(math.sin(e) ** 2, 1e-06) * term
            mp_sigma = float(math.sqrt(max(mp_variance, 1e-12)))
            cn0_linear = 10.0 ** (cn0_dbhz / 10.0)
            d = LEO_CORRELATOR_SPACING_CHIPS
            bl = LEO_CODE_LOOP_BANDWIDTH_HZ
            tau = LEO_CORRELATOR_ACCUMULATION_S
            sigma_chips = math.sqrt(bl * d / (2.0 * cn0_linear) * (1.0 + 2.0 / ((2.0 - d) * cn0_linear * tau)))
            receiver_sigma = float(SPEED_OF_LIGHT_MPS / LEO_CODE_CHIPPING_RATE_HZ * sigma_chips)
            sigma_code_m = math.sqrt(LEO_URA_SIGMA_M ** 2 + iono_sigma ** 2 + tropo_sigma ** 2 + mp_sigma ** 2 + receiver_sigma ** 2)
            out.append(PseudorangeMeasurement(sat_id, constellation, float(pseudorange_m), satellite_position_rx, satellite_clock_bias_s, float(ionosphere), float(troposphere), float(sigma_code_m)))
        return tuple(out)

def build_error_state_dynamics(nav: NavigationState, specific_force_body_mps2: Array) -> Array:
    F = np.zeros((INS_STATE_DIM, INS_STATE_DIM))
    r_e = nav.position_ecef_m
    radius = float(np.linalg.norm(r_e))
    gravity = _gravitation_j2_ecef(r_e)
    radial = r_e / radius
    C = nav.body_to_ecef_dcm
    F[0:3, 3:6] = IDENTITY_3
    F[3:6, 0:3] = -(2.0 / radius) * np.outer(gravity, radial)
    F[3:6, 3:6] = -2.0 * OMEGA_IE_SKEW
    force_e = C @ np.asarray(specific_force_body_mps2)
    fx, fy, fz = force_e
    F[3:6, 6:9] = np.array([[0.0, -fz, fy], [fz, 0.0, -fx], [-fy, fx, 0.0]])
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    return F

_VAN_LOAN_STATS = {'taylor_calls': 0, 'exact_fallback_calls': 0, 'validation_calls': 0, 'max_norm_1': 0.0, 'max_taylor_remainder_bound': 0.0, 'max_validation_phi_abs': 0.0, 'max_validation_qd_abs': 0.0}

def discretize_process_noise_van_loan(F: Array, Qc: Array, dt_s: float) -> tuple[Array, Array]:
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
            term = term @ B / float(k)
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

def predict_pseudorange(antenna_position_ecef_m: Array, measurement: PseudorangeMeasurement) -> tuple[float, Array]:
    range_vector = np.asarray(antenna_position_ecef_m) - measurement.satellite_position_reception_ecef_m
    geometric_range = float(np.linalg.norm(range_vector))
    los = range_vector / geometric_range
    predicted = geometric_range + 0.0 - SPEED_OF_LIGHT_MPS * measurement.satellite_clock_bias_s + measurement.ionosphere_delay_m + measurement.troposphere_delay_m
    return (float(predicted), los)

def retain_clock_observable_measurements(measurements) -> tuple[PseudorangeMeasurement, ...]:
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
        weights = 1.0 / variances[index]
        normalized_weight = weights / np.sum(weights)
        block = np.eye(index.size) - np.ones((index.size, 1)) @ normalized_weight[None, :]
        projector[np.ix_(index, index)] = block
    return projector

def build_measurement_model(nav: NavigationState, measurements, lever_arm_b_m: Array):
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return (np.empty(0), np.zeros((0, INS_STATE_DIM)), np.zeros((0, 0)), (), np.zeros((0, 0)))
    raw_innovation = np.empty(n)
    H_raw = np.zeros((n, INS_STATE_DIM))
    variances = np.empty(n)
    lever_e = nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    antenna_position = nav.position_ecef_m + lever_e
    lx, ly, lz = lever_e
    lever_skew = np.array([[0.0, -lz, ly], [lz, 0.0, -lx], [-ly, lx, 0.0]])
    sat_ids: list[str] = []
    for i, m in enumerate(measurements):
        predicted, los = predict_pseudorange(antenna_position, m)
        raw_innovation[i] = m.pseudorange_m - predicted
        H_raw[i, 0:3] = los
        H_raw[i, 6:9] = -ATTITUDE_FEEDBACK_SIGN * (los @ lever_skew)
        variance = float(m.sigma_code_m) ** 2
        variances[i] = variance
        sat_ids.append(m.sat_id)
    projector = _clock_projector(measurements, variances)
    innovation = projector @ raw_innovation
    H = projector @ H_raw
    R_raw = np.diag(variances)
    R = projector @ R_raw @ projector.T
    R = 0.5 * (R + R.T)
    return (innovation, H, R, tuple(sat_ids), projector)

def build_innovation_only(nav: NavigationState, measurements, lever_arm_b_m: Array) -> Array:
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
        variances[i] = variance
    return _clock_projector(measurements, variances) @ raw_innovation

def kalman_measurement_update(P: Array, innovation: Array, H: Array, R: Array):
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)
    dx = K @ innovation
    I_KH = IDENTITY_STATE - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_reset = reset_error_state_covariance(P_post, dx)
    return (dx, P_reset, K)

def learned_gain_covariance_update(prior_covariance: Array, learned_gain: Array, measurement_jacobian: Array, measurement_covariance: Array, injected_error_state: Array) -> Array:
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    learned_gain = np.asarray(learned_gain, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    injected_error_state = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    measurement_count = measurement_jacobian.shape[0]
    update_matrix = IDENTITY_STATE - learned_gain @ measurement_jacobian
    posterior_covariance = update_matrix @ prior_covariance @ update_matrix.T + learned_gain @ measurement_covariance @ learned_gain.T
    return reset_error_state_covariance(posterior_covariance, injected_error_state)

def reset_error_state_covariance(posterior_covariance: Array, injected_error_state: Array) -> Array:
    P = np.asarray(posterior_covariance, dtype=float)
    dx = np.asarray(injected_error_state, dtype=float).reshape(INS_STATE_DIM)
    reset_jacobian = np.eye(INS_STATE_DIM)
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
    reset_jacobian[6:9, 6:9] = np.eye(3) + a * K + b * K2
    reset_covariance = reset_jacobian @ P @ reset_jacobian.T
    reset_covariance = 0.5 * (reset_covariance + reset_covariance.T)
    return reset_covariance

def inject_error_state(nav: NavigationState, dx: Array) -> NavigationState:
    dx = np.asarray(dx, dtype=float).reshape(INS_STATE_DIM)
    out = nav.copy()
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _rotation(_so3_exponential(ATTITUDE_FEEDBACK_SIGN * dx[6:9]) @ out.body_to_ecef_dcm)
    return out

class TLESGP4Provider:

    def __init__(self, path: str | Path, max_tle_age_days: float, allow_degraded_eop: bool=False, allow_non_tle_files: bool=True):
        self.max_tle_age_s = float(max_tle_age_days) * 86400.0
        self.allow_degraded_eop = allow_degraded_eop
        by_id = {}
        tle_pairs = []
        for source in sorted(Path(path).iterdir()):
            if not source.is_file():
                continue
            lines = source.read_text(encoding='ascii', errors='replace').splitlines()
            for i in range(len(lines) - 1):
                line1, line2 = lines[i].strip(), lines[i + 1].strip()
                if line1.startswith('1 ') and line2.startswith('2 '):
                    try:
                        verify_checksum(line1, line2)
                    except ValueError:
                        if allow_non_tle_files:
                            continue
                        raise
                    tle_pairs.append((line1, line2))
        for line1, line2 in tle_pairs:
            satrec = Satrec.twoline2rv(line1, line2, WGS72)
            norad = str(satrec.satnum_str).strip()
            epoch = Time(satrec.jdsatepoch, satrec.jdsatepochF, format='jd', scale='utc')
            by_id.setdefault(f'NORAD-{norad}', []).append((float(epoch.gps), satrec))
        self._elements = {sat_id: tuple(sorted(elements, key=lambda e: e[0])) for sat_id, elements in by_id.items()}
        self._element_times = {sat_id: tuple(element[0] for element in elements) for sat_id, elements in self._elements.items()}
        self.satellite_ids = tuple(sorted(self._elements, key=lambda s: int(s.split('-')[1])))

    def state_at(self, time_gpst_s: float, sat_id: str):
        elements = self._elements[sat_id]
        index = bisect_right(self._element_times[sat_id], time_gpst_s) - 1
        element = elements[index]
        utc = Time(float(time_gpst_s), format='gps').utc
        error, position_km, velocity_km_s = element[1].sgp4(float(utc.jd1), float(utc.jd2))
        position = CartesianRepresentation(np.asarray(position_km) * u.km)
        velocity = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        teme = TEME(position.with_differentials(velocity), obstime=utc)
        degraded = 'warn' if self.allow_degraded_eop else 'error'
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        return (np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3), np.asarray(itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s)).reshape(3))

def ref35_ionosphere_sigma_m(elevation_rad: float, receiver_latitude_rad: float) -> float:
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else 4.5 if latitude_deg <= 55.0 else 6.0
    R = 6378140.0
    h_i = 350000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))


class LEODownlinkSimulator:

    def __init__(self, provider: TLESGP4Provider, klobuchar_coefficients: tuple[Array, Array] | None, seed: int=0, tx_epsilon_position_m: float=0.001, tx_max_iterations: int=20, minimum_elevation_deg: float=10.0, prefilter_guard_deg: float=LEO_PREFILTER_GUARD_DEG, use_ionosphere: bool=True, use_troposphere: bool=True):
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

    def simulate_epoch(self, receive_time_gpst_s: float, receiver_position_ecef_m: Array):
        receiver_llh = ecef_to_llh(receiver_position_ecef_m)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        measurements = []
        for sat_id in self.provider.satellite_ids:
            try:
                initial_state = self.provider.state_at(receive_time_gpst_s, sat_id)
                initial_range, initial_los, _ = geometric_range(receiver_position_ecef_m, initial_state[0], 0.0)
                initial_elevation, _ = elevation_azimuth_from_ned_matrix(c_ecef_ned, initial_los)
                self.prefilter_checked += 1
                if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
                    self.prefilter_rejected += 1
                    continue
                self.prefilter_passed += 1
                _, transit_s, state_tx_position = _iterate_transmit_time(
                    lambda sid, t: self.provider.state_at(t, sid)[0],
                    sat_id,
                    receive_time_gpst_s,
                    receiver_position_ecef_m,
                    initial_state[0],
                    initial_range / SPEED_OF_LIGHT_MPS,
                    self.tx_epsilon_position_m,
                    self.tx_max_iterations,
                )
                rho, los, sat_rx = geometric_range(receiver_position_ecef_m, state_tx_position, transit_s)
                elevation, azimuth = elevation_azimuth_from_ned_matrix(c_ecef_ned, los)
                if elevation < self.minimum_elevation_rad:
                    continue
                receiver_lat, receiver_lon, receiver_height_m = receiver_llh

                if self.use_ionosphere:
                    alpha, beta = self.klobuchar_coefficients
                    week = int(math.floor(float(receive_time_gpst_s) / GPS_WEEK_S))
                    tow = float(receive_time_gpst_s) - week * GPS_WEEK_S
                    iklo_m = klobuchar_delay_m(tow, receiver_lat, receiver_lon, elevation, azimuth, alpha, beta)
                    _, _, satellite_height_m = ecef_to_llh(sat_rx)
                    if satellite_height_m >= LEO_IONOSPHERE_UPPER_HEIGHT_M:
                        ionosphere_path_scale = 1.0
                    elif satellite_height_m <= LEO_IONOSPHERE_LOWER_HEIGHT_M:
                        ionosphere_path_scale = 0.0
                    else:
                        ionosphere_path_scale = (satellite_height_m - LEO_IONOSPHERE_LOWER_HEIGHT_M) / (LEO_IONOSPHERE_UPPER_HEIGHT_M - LEO_IONOSPHERE_LOWER_HEIGHT_M)
                    ionosphere_m = float(ionosphere_path_scale * iklo_m)
                else:
                    ionosphere_m = 0.0
                    ionosphere_path_scale = 0.0

                troposphere_m = saastamoinen_delay_m(receiver_height_m, elevation) if self.use_troposphere else 0.0
                iono_sigma_m = ionosphere_path_scale * ref35_ionosphere_sigma_m(elevation, receiver_lat) if self.use_ionosphere else 0.0
                tropo_sigma_m = float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation)) ** 2)) if self.use_troposphere else 0.0
                elevation_deg = float(np.rad2deg(elevation))
                mp_sigma_m = float(0.13 + 0.53 * math.exp(-elevation_deg / 10.0))
                ionosphere_residual_m = float(self.rng.normal(0.0, iono_sigma_m)) if iono_sigma_m > 0.0 else 0.0
                troposphere_residual_m = float(self.rng.normal(0.0, tropo_sigma_m)) if tropo_sigma_m > 0.0 else 0.0
                mp_nlos_error_m = float(self.rng.normal(0.0, mp_sigma_m)) if mp_sigma_m > 0.0 else 0.0
                total_variance_m2 = iono_sigma_m ** 2 + tropo_sigma_m ** 2 + mp_sigma_m ** 2
                sigma_code_m = math.sqrt(max(total_variance_m2, 1e-12))
                pseudorange_m = rho + ionosphere_m + troposphere_m + ionosphere_residual_m + troposphere_residual_m + mp_nlos_error_m
                measurements.append(PseudorangeMeasurement(sat_id, 'L', float(pseudorange_m), sat_rx, 0.0, float(ionosphere_m), float(troposphere_m), float(sigma_code_m)))
            except ValueError:
                continue
            except RuntimeError as exc:
                if 'Transmit-time iteration did not converge' in str(exc):
                    continue
                raise
        return tuple(measurements)

FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM
OBSERVATION_FEATURE_DIM = 2
SUPERVISED_STATE_DIM = DIRECT_STATE_LABEL_DIM
MASK_EPS = 1e-06

class MaskedCLA(nn.Module):

    def __init__(self, nmax: int, dropout: float=0.2) -> None:
        super().__init__()
        self.nmax = int(nmax)
        self.eq15_padded_dim = FIXED_FEATURE_DIM + 2 * self.nmax

        # Masked 1-D convolution: kept directly in this class.
        self.conv_weight = nn.Parameter(torch.empty(24, 1, 3))
        self.conv_bias = nn.Parameter(torch.zeros(24))
        nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        self.register_buffer(
            '_conv_mask_kernel',
            torch.ones(1, 1, 3),
            persistent=False,
        )

        # Recurrent and attention blocks.
        self.lstm = nn.LSTM(
            input_size=24,
            hidden_size=64,
            num_layers=5,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_proj = nn.Linear(64, 64, bias=True)
        self.attention_v = nn.Linear(64, 1, bias=False)
        self.gain_head = nn.Linear(64, INS_STATE_DIM * self.nmax)

        # Pipeline_EKF compatibility attributes.  The original KalmanNet model
        # stores its recurrent hidden states internally and resets them by calling
        # init_hidden_KNet() once at the start of every train/CV batch.  MaskedCLA
        # passes the LSTM state explicitly, so this method records the same reset
        # event while the rollout helper still carries the explicit state tensor.
        self.batch_size = 1
        self._repo_hidden_reset_counter = 0
        self._pipeline_recurrent_state = None

    def init_hidden_KNet(self) -> None:
        self._repo_hidden_reset_counter += 1
        self._pipeline_recurrent_state = None

    def forward(
        self,
        fixed: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        channel_mask: torch.Tensor,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None=None,
    ):
        batch_size = fixed.shape[0]
        satellite_valid = mask.bool()
        channel_bool = channel_mask.bool()
        satellite_count = satellite_valid.sum(dim=1)
        observations = observations * channel_bool.to(dtype=observations.dtype)

        fixed_valid = torch.ones(
            (batch_size, FIXED_FEATURE_DIM),
            dtype=torch.bool,
            device=fixed.device,
        )
        packed_values = []
        packed_masks = []

        for batch_index, count in enumerate(
            satellite_count.detach().cpu().tolist()
        ):
            suffix_length = 2 * (self.nmax - count)
            observation_residual = observations[batch_index, :count, 0]
            observation_innovation = observations[batch_index, :count, 1]
            zero_padding = observations.new_zeros(suffix_length)

            packed_values.append(torch.cat((
                fixed[batch_index],
                observation_residual,
                observation_innovation,
                zero_padding,
            )))

            packed_masks.append(torch.cat((
                fixed_valid[batch_index],
                channel_bool[batch_index, :count, 0],
                channel_bool[batch_index, :count, 1],
                torch.zeros(
                    suffix_length,
                    dtype=torch.bool,
                    device=channel_mask.device,
                ),
            )))

        x_bar = torch.stack(packed_values, dim=0)
        feature_mask = torch.stack(packed_masks, dim=0)

        # Masked convolution.
        m = feature_mask.to(dtype=x_bar.dtype).unsqueeze(1)
        z = F.conv1d(
            x_bar.unsqueeze(1) * m,
            self.conv_weight,
            bias=None,
            stride=1,
            padding=1,
        )
        local_count = F.conv1d(
            m,
            self._conv_mask_kernel.to(dtype=x_bar.dtype),
            stride=1,
            padding=1,
        )
        valid_window = (local_count > 0).to(dtype=x_bar.dtype)
        conv_features = F.relu(
            z / local_count.clamp_min(MASK_EPS)
            + self.conv_bias.view(1, -1, 1) * valid_window
        )
        conv_features = (conv_features * m).transpose(1, 2)

        # Masked recurrent update.  The LSTM still advances one network epoch
        # at a time while preserving a separate hidden/cell state per position.
        position_count = conv_features.shape[1]
        state_shape = (5, batch_size, position_count, 64)
        if recurrent_state is None:
            previous_hidden = conv_features.new_zeros(state_shape)
            previous_cell = conv_features.new_zeros(state_shape)
        else:
            previous_hidden, previous_cell = recurrent_state

        step_input = conv_features.reshape(
            batch_size * position_count, 1, 24
        )
        h0 = previous_hidden.reshape(
            5, batch_size * position_count, 64
        ).contiguous()
        c0 = previous_cell.reshape(
            5, batch_size * position_count, 64
        ).contiguous()

        _, (candidate_hidden_flat, candidate_cell_flat) = self.lstm(
            step_input, (h0, c0)
        )
        candidate_hidden = candidate_hidden_flat.reshape(
            5, batch_size, position_count, 64
        )
        candidate_cell = candidate_cell_flat.reshape(
            5, batch_size, position_count, 64
        )

        valid = feature_mask.bool().unsqueeze(0).unsqueeze(-1)
        next_hidden = torch.where(
            valid, candidate_hidden, previous_hidden
        )
        next_cell = torch.where(
            valid, candidate_cell, previous_cell
        )
        lstm_positions = next_hidden[-1]

        # Attention.
        score = self.attention_v(
            torch.tanh(self.attention_proj(lstm_positions))
        ).squeeze(-1)
        attention = torch.softmax(
            score.masked_fill(~feature_mask.bool(), -torch.inf),
            dim=1,
        )
        context = torch.sum(
            attention.unsqueeze(-1) * lstm_positions,
            dim=1,
        )

        gain = self.gain_head(context).view(
            batch_size, INS_STATE_DIM, self.nmax
        )
        gain = gain * satellite_valid.to(dtype=gain.dtype).unsqueeze(1)

        return gain, attention, (next_hidden, next_cell)

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducibility_seed(REPRODUCIBILITY_SEED)
    print(f'Reproducibility seed: {REPRODUCIBILITY_SEED}')
    MAX_FUSION_EPOCHS = 501
    MAX_TEST_FUSION_EPOCHS = 101
    # KalmanNet_TSP V2 training controls.  TRAINING_EPOCHS is used as the
    # repository's n_steps: one randomly sampled mini-batch and one optimizer
    # update per step.  V2 splits the long training trajectory into independent
    # short trajectories; BPTT is complete inside each short trajectory only.
    TRAINING_EPOCHS = 50
    V2_BPTT_LENGTH = 10
    V2_BATCH_SIZE = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.001
    VALIDATION_FRACTION = 0.20
    TEST_LEO_SEED = LEO_SEED + 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n=== MASKED KALMANNET PERFORMANCE EVALUATION ===')
    if MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None:
        print('WARNING: fusion-epoch cap active; reported metrics are diagnostic.')
    root = ET.fromstring(README_XML_PATH.read_text(encoding='utf-8', errors='replace'))
    rover_xml = next(rover_xml for rover_xml in root.findall('ROVE') if (rover_xml.findtext('ID') or '').strip() == '01')
    rover_lever_arm_vehicle_m = np.fromstring(rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ')
    rover_mounting_xyz_deg = np.fromstring(rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    antenna_truth_rows = []
    antenna_truth_started = False
    for line in ROVE_GROUND_TRUTH_PATH.read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if len(fields) < 24:
            if antenna_truth_started:
                break
            continue
        try:
            row = (int(fields[0]), float(fields[1]), float(fields[9]), float(fields[10]), float(fields[11]), float(fields[15]), float(fields[16]), float(fields[17]), float(fields[21]), float(fields[22]), float(fields[23]))
        except (ValueError, IndexError):
            if antenna_truth_started:
                break
            continue
        antenna_truth_started = True
        antenna_truth_rows.append(row)
    antenna_truth_array = np.asarray(antenna_truth_rows, dtype=float)
    antenna_truth_week = antenna_truth_array[:, 0].astype(int)
    antenna_truth = (antenna_truth_week.astype(float) * GPS_WEEK_S + antenna_truth_array[:, 1], antenna_truth_array[:, 2:5], antenna_truth_array[:, 5:8], antenna_truth_array[:, 8], antenna_truth_array[:, 9], antenna_truth_array[:, 10])
    imu_truth_rows = []
    imu_truth_started = False
    for line in IMU_GROUND_TRUTH_PATH.read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if len(fields) < 24:
            if imu_truth_started:
                break
            continue
        try:
            row = (int(fields[0]), float(fields[1]), float(fields[9]), float(fields[10]), float(fields[11]), float(fields[15]), float(fields[16]), float(fields[17]), float(fields[21]), float(fields[22]), float(fields[23]))
        except (ValueError, IndexError):
            if imu_truth_started:
                break
            continue
        imu_truth_started = True
        imu_truth_rows.append(row)
    imu_truth_array = np.asarray(imu_truth_rows, dtype=float)
    imu_truth_week = imu_truth_array[:, 0].astype(int)
    imu_truth = (imu_truth_week.astype(float) * GPS_WEEK_S + imu_truth_array[:, 1], imu_truth_array[:, 2:5], imu_truth_array[:, 5:8], imu_truth_array[:, 8], imu_truth_array[:, 9], imu_truth_array[:, 10])
    imu_error_text = IMU_ERROR_MODEL_PATH.read_text(encoding='utf-8', errors='replace')
    imu_models = {}
    d2r = np.pi / 180.0
    needed_imu_keys = ('ISDV_Pos', 'ISDV_Vel', 'ISDV_Att', 'PNSD_Pos', 'PNSD_Vel', 'PNSD_Att')
    for match in re.finditer('IMU\\s*\\{(.*?)\\}', imu_error_text, flags=re.S):
        block = match.group(1)
        type_match = re.search('IMU_Type\\s*=\\s*"([^"]+)"', block)
        if not type_match:
            continue
        values = {}
        for key in needed_imu_keys:
            value_match = re.search(f'{key}\\s*=\\s*([^\\r\\n]+)', block)
            if value_match:
                values[key] = np.fromstring(value_match.group(1), sep=' ', dtype=float)
        if len(values) == len(needed_imu_keys):
            imu_models[type_match.group(1)] = {
                'isdv_pos_m': values['ISDV_Pos'],
                'isdv_vel_mps': values['ISDV_Vel'],
                'isdv_att_rad': values['ISDV_Att'] * d2r,
                'pnsd_pos_m_sqrt_s': values['PNSD_Pos'],
                'pnsd_vel_mps_sqrt_s': values['PNSD_Vel'],
                'pnsd_att_rad_sqrt_s': values['PNSD_Att'] * d2r,
            }
    rinex = RINEXObservationFile(RINEX_OBS_PATH)
    first_rinex_epoch = next(rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    with IMR_PATH.open('rb') as stream:
        imr_header_buffer = stream.read(512)
    imr_endian = '<' if imr_header_buffer[8] == 0 else '>'
    imr_header = struct.unpack(imr_endian + IMR_HEADER_FORMAT_BODY, imr_header_buffer)
    imr_time_tag_bias_s = float(imr_header[10]) * 0.001
    imr_record_bytes = struct.calcsize(imr_endian + IMR_RECORD_FORMAT_BODY)
    imr_record_count = (IMR_PATH.stat().st_size - 512) // imr_record_bytes
    imr_record_dtype = np.dtype([('tow', imr_endian + 'f8'), ('counts', imr_endian + 'i4', (6,))], align=False)
    imr_records = np.memmap(IMR_PATH, dtype=imr_record_dtype, mode='r', offset=512, shape=(imr_record_count,))
    imr_tow_all = np.asarray(imr_records['tow'], dtype=np.float64).copy()
    del imr_records
    imr_tow_all[imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    imr_tow_all -= imr_time_tag_bias_s
    anchor_time = float(first_rinex_epoch[0])
    imr_time_all = anchor_imr_tow_to_gpst_seconds(imr_tow_all, anchor_time)
    antenna_truth_time = np.asarray(antenna_truth[0], dtype=float).reshape(-1)
    imu_truth_time = np.asarray(imu_truth[0], dtype=float).reshape(-1)
    common_start = max(float(imr_time_all[0]), float(antenna_truth_time[0]), float(imu_truth_time[0]))
    common_end = min(float(imr_time_all[-1]), float(antenna_truth_time[-1]), float(imu_truth_time[-1]))
    start = int(np.searchsorted(imr_time_all, common_start, side='left'))
    usable_imu_start = float(imr_time_all[start])
    gnss_epochs = tuple(rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=usable_imu_start, end_time_gpst_s=common_end, max_epochs=MAX_FUSION_EPOCHS, require_measurements=True))
    fusion_time = np.asarray([epoch[0] for epoch in gnss_epochs], dtype=float)
    common_stop = min(int(np.searchsorted(imr_time_all, common_end, side='left')) + 1, len(imr_time_all))
    requested_stop = min(int(np.searchsorted(imr_time_all, fusion_time[-1], side='left')) + 1, common_stop)
    with IMR_PATH.open('rb') as stream:
        imr_sample_header_buffer = stream.read(512)
    imr_sample_endian = '<' if imr_sample_header_buffer[8] == 0 else '>'
    imr_sample_header = struct.unpack(imr_sample_endian + IMR_HEADER_FORMAT_BODY, imr_sample_header_buffer)
    imr_data_rate_hz = float(imr_sample_header[5])
    imr_gyro_scale = float(imr_sample_header[6])
    imr_accel_scale = float(imr_sample_header[7])
    imr_record_bytes = struct.calcsize(imr_sample_endian + IMR_RECORD_FORMAT_BODY)
    imr_record_dtype = np.dtype([('tow', imr_sample_endian + 'f8'), ('counts', imr_sample_endian + 'i4', (6,))], align=False)
    imr_records = np.fromfile(IMR_PATH, dtype=imr_record_dtype, count=requested_stop - start, offset=512 + start * imr_record_bytes)
    imr_gyro = np.deg2rad(imr_records['counts'][:, :3].astype(float) * imr_gyro_scale * imr_data_rate_hz)
    imr_accel = imr_records['counts'][:, 3:6].astype(float) * imr_accel_scale * imr_data_rate_hz
    imr_time = imr_time_all[start:requested_stop].copy()
    del imr_tow_all, imr_time_all
    query_time = np.concatenate(([imr_time[0]], fusion_time))
    antenna_position = interpolate_ground_truth(antenna_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    imu_position, imu_velocity, imu_heading, imu_pitch, imu_roll = interpolate_ground_truth(imu_truth, query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]
    fusion_truth_body_to_ecef = np.stack([body_to_ecef_from_ie_hpr(p, h, pt, r, rover_mounting_xyz_deg) for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])])
    lever_arm_b_m = c_vehicle_to_body_zxy(*rover_mounting_xyz_deg) @ np.asarray(rover_lever_arm_vehicle_m, dtype=float).reshape(3)
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
                    if len(fields) >= 5 and fields[0] in {'GPSA', 'GPSB', 'BDSA', 'BDSB'}:
                        iono_header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
        ionosphere_coefficients = {'G': (np.asarray(iono_header['GPSA']), np.asarray(iono_header['GPSB'])), 'C': (np.asarray(iono_header['BDSA']), np.asarray(iono_header['BDSB']))}
    gnss_preprocessor = GNSSPreprocessor(SP3Orbit(SP3_PATH), RINEXClock(CLK_PATH), min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=ionosphere_coefficients)
    tle_provider = TLESGP4Provider(LEO_TLE_DIR, max_tle_age_days=TLE_MAX_AGE_DAYS, allow_degraded_eop=TLE_ALLOW_DEGRADED_EOP, allow_non_tle_files=TLE_ALLOW_NON_TLE_FILES)
    leo_klobuchar = None if ionosphere_coefficients is None else ionosphere_coefficients['G']
    leo_simulator = LEODownlinkSimulator(tle_provider, leo_klobuchar, seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    network_source_rows = []
    data01_sequence_nmax = 0
    network_interval_segments_since_previous_fusion = []
    last_gyro, last_accel = compensate_imu(imr_gyro[0], imr_accel[0])

    for event_type, event_start, event_end, event_index in build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True):
        if event_type == 'propagation':
            dt = event_end - event_start
            last_gyro, last_accel = compensate_imu(imr_gyro[event_index], imr_accel[event_index])
            network_interval_segments_since_previous_fusion.append((event_index, dt))
            continue

        fusion_index = event_index
        leo_measurements = leo_simulator.simulate_epoch(event_start, fusion_antenna_truth_position[fusion_index])
        data01_sequence_nmax = max(data01_sequence_nmax, len(gnss_epochs[fusion_index][1]) + len(leo_measurements))
        network_source_rows.append({
            'time': float(event_start),
            'fusion_index': int(fusion_index),
            'leo_measurements': tuple(leo_measurements),
            'preceding_interval_segments': tuple(network_interval_segments_since_previous_fusion),
            'accel': last_accel.copy(),
            'gyro': last_gyro.copy(),
        })
        network_interval_segments_since_previous_fusion = []

    # ------------------------------------------------------------------
    # Neural sequence definition
    # ------------------------------------------------------------------
    # Nmax comes from the maximum available raw GNSS + simulated LEO observation
    # count, not from the set surviving a classical Kalman-filter trajectory.
    network_nmax = int(data01_sequence_nmax)

    # Eq. (10)-(15) features are generated causally inside the learned rollout.
    # The first fusion epoch provides context; samples begin at the next epoch.
    data01_total_sample_count = int(max(len(network_source_rows) - 1, 0))
    if data01_total_sample_count < 2:
        raise RuntimeError(
            f'Not enough Data01 sequential samples: {data01_total_sample_count}'
        )

    validation_sample_count = max(
        1, int(round(data01_total_sample_count * VALIDATION_FRACTION))
    )
    training_sample_count = data01_total_sample_count - validation_sample_count
    validation_start_sample = training_sample_count

    if training_sample_count < 2:
        raise RuntimeError(
            f'Not enough Data01 samples for chronological train/validation split: '
            f'total={data01_total_sample_count}, train={training_sample_count}, '
            f'validation={validation_sample_count}'
        )

    # KalmanNet_TSP/Simulations/utils.py::Short_Traj_Split semantics, ported
    # literally to this trajectory representation:
    #   1) split the long source trajectory into blocks of T+1,
    #   2) unconditionally pop the last split,
    #   3) use element 0 of every surviving block as train_init,
    #   4) supervise the following T epochs.
    # The source length is training_sample_count + 1 because sample k uses source
    # rows k (boundary/init) and k+1 (the first supervised fusion epoch).
    v2_stride = V2_BPTT_LENGTH + 1
    v2_source_length = training_sample_count + 1
    # Equivalent to repository Short_Traj_Split: split by T+1 and unconditionally
    # discard the last split, even when it is complete.
    v2_chunk_starts = list(range(0, max(v2_source_length - v2_stride, 0), v2_stride))
    v2_sequence_count = len(v2_chunk_starts)
    v2_supervised_sample_count = v2_sequence_count * V2_BPTT_LENGTH

    if v2_sequence_count == 0:
        raise RuntimeError(
            f'No complete V2 short trajectories: train_samples={training_sample_count}, '
            f'V2_BPTT_LENGTH={V2_BPTT_LENGTH}'
        )
    if V2_BATCH_SIZE > v2_sequence_count:
        raise RuntimeError(
            f'V2_BATCH_SIZE={V2_BATCH_SIZE} exceeds available short trajectories '
            f'N_E={v2_sequence_count}. Reduce V2_BATCH_SIZE or V2_BPTT_LENGTH.'
        )

    print(
        f'training setup: Data01_samples={data01_total_sample_count}, '
        f'train_samples={training_sample_count}, validation_samples={validation_sample_count}, '
        f'split=chronological_first_{100.0 * (1.0 - VALIDATION_FRACTION):.0f}%_'
        f'last_{100.0 * VALIDATION_FRACTION:.0f}%, '
        f'Nmax={network_nmax}, training_steps={TRAINING_EPOCHS}, '
        f'V2_BPTT_length={V2_BPTT_LENGTH}, V2_sequences={v2_sequence_count}, '
        f'V2_batch_size={V2_BATCH_SIZE}, '
        f'optimizer=Adam, lr={LEARNING_RATE:g}, weight_decay={WEIGHT_DECAY:g}, '
        f'loss=MSE, V2_random_short_trajectory_minibatch=True, '
        f'network_inputs=learned_recursive_state_not_classical_KF'
    )

    model = MaskedCLA(nmax=network_nmax, dropout=0.2).to(DEVICE)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print('\n=== NEURAL NETWORK TRAINING CONFIGURATION ===')
    print(f'Device: {DEVICE}')
    print(f'Reproducibility seed: {REPRODUCIBILITY_SEED}')
    print('Training algorithm: KalmanNet_TSP Pipeline_EKF structural port + V2 short-trajectory BPTT')
    print('Optimizer: Adam')
    print('Loss function: Mean Squared Error (MSELoss, reduction=mean)')
    print(f'Total training steps / epochs: {TRAINING_EPOCHS}')
    print(f'V2 BPTT sequence length: {V2_BPTT_LENGTH} fusion epochs')
    print(f'V2 mini-batch size: {V2_BATCH_SIZE} short trajectories')
    print(f'Learning rate: {LEARNING_RATE:.6g}')
    print(f'Weight decay (L2 regularization): {WEIGHT_DECAY:.6g}')
    print(f'Training samples before V2 split: {training_sample_count}')
    print(f'Validation samples: {validation_sample_count}')
    print(f'Validation fraction: {VALIDATION_FRACTION:.3f}')
    print(f'Available V2 short trajectories: {v2_sequence_count}')
    print(f'Supervised samples used by V2: {v2_supervised_sample_count}')
    print(f'Network maximum measurement slots (Nmax): {network_nmax}')
    print(f'Supervised state dimension: {SUPERVISED_STATE_DIM}')
    print(f'Total model parameters: {total_parameter_count:,}')
    print(f'Trainable model parameters: {trainable_parameter_count:,}')
    print('Repository Pipeline_EKF train/CV operation order: exact structural port')
    print('Repository end-to-end KalmanNet autograd recursion: NOT exact in this project')
    print('Reason: INS mechanization, GNSS preprocessing, pseudorange model, and state injection are NumPy operations; learned corrections must be detached before that numerical navigation loop.')

    # KalmanNet_TSP/Pipeline_EKF.py uses one optimizer over all trainable model
    # parameters.  No alternating theta/psi freezing is used in this V2 path.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss(reduction='mean')

    def _evaluate_recursive_sequence(start_sample, stop_sample, collect_features=False):
        """
        Closed-loop sequence:
          INS propagation -> current-state measurements -> Yan features ->
          Masked KalmanNet -> correction -> corrected state -> next epoch.

        No classical Kalman estimate, gain, correction, innovation, residual,
        x_pred or x_post is used as a network input.
        """
        boundary_row = network_source_rows[start_sample]
        boundary_fusion_index = int(boundary_row['fusion_index'])
        nav_state = NavigationState(
            fusion_imu_truth_position[boundary_fusion_index].copy(),
            fusion_imu_truth_velocity[boundary_fusion_index].copy(),
            fusion_truth_body_to_ecef[boundary_fusion_index].copy(),
        )

        boundary_gnss = gnss_preprocessor.prepare_epoch(
            gnss_epochs[boundary_fusion_index],
            nav_state,
            lever_arm_b_m,
        )
        boundary_measurements = retain_clock_observable_measurements(
            tuple(boundary_gnss) + tuple(boundary_row['leo_measurements'])
        )
        if not boundary_measurements:
            raise RuntimeError(
                f'No measurements available at sequence boundary sample={start_sample}'
            )

        _, _, _, boundary_sat_ids, _ = build_measurement_model(
            nav_state,
            boundary_measurements,
            lever_arm_b_m,
        )
        zero_state = np.zeros(INS_STATE_DIM, dtype=float)
        previous_context = {
            'sat_ids': tuple(boundary_sat_ids),
            'residual': build_innovation_only(
                nav_state,
                boundary_measurements,
                lever_arm_b_m,
            ),
            'x_post': zero_state.copy(),
            'state_innovation': zero_state.copy(),
            'state_residual': zero_state.copy(),
            'accel': np.asarray(
                boundary_row['accel'], dtype=float
            ).reshape(3).copy(),
            'gyro': np.asarray(
                boundary_row['gyro'], dtype=float
            ).reshape(3).copy(),
        }

        recurrent_state = model._pipeline_recurrent_state
        correction_sequence = []
        target_sequence = []
        feature_fixed_rows = []
        feature_obs_rows = []
        feature_channel_rows = []
        for sample_index in range(start_sample, stop_sample):
            source_row = network_source_rows[sample_index + 1]

            current_accel = previous_context['accel']
            current_gyro = previous_context['gyro']
            for imu_index, dt in source_row['preceding_interval_segments']:
                current_gyro, current_accel = compensate_imu(
                    imr_gyro[int(imu_index)],
                    imr_accel[int(imu_index)],
                )
                nav_state = mechanize_ecef(
                    nav_state,
                    current_gyro,
                    current_accel,
                    float(dt),
                )

            fusion_index = int(source_row['fusion_index'])
            gnss_now = gnss_preprocessor.prepare_epoch(
                gnss_epochs[fusion_index],
                nav_state,
                lever_arm_b_m,
            )
            measurements = retain_clock_observable_measurements(
                tuple(gnss_now) + tuple(source_row['leo_measurements'])
            )
            if not measurements:
                continue

            innovation_now, _, _, current_sat_ids, _ = build_measurement_model(
                nav_state, measurements, lever_arm_b_m
            )

            current_sat_ids = tuple(current_sat_ids)
            innovation_k = np.asarray(
                innovation_now, dtype=float
            ).reshape(-1)
            measurement_count = len(current_sat_ids)

            delta_accel = (
                np.asarray(current_accel, dtype=float).reshape(3)
                - previous_context['accel']
            )
            delta_gyro = (
                np.asarray(current_gyro, dtype=float).reshape(3)
                - previous_context['gyro']
            )
            fixed_k = np.concatenate((
                delta_accel,
                delta_gyro,
                previous_context['state_residual'],
                previous_context['state_innovation'],
            ))

            previous_residual_by_sat = dict(zip(
                previous_context['sat_ids'],
                previous_context['residual'],
            ))
            residual_k = np.zeros(measurement_count, dtype=float)
            residual_mask = np.zeros(measurement_count, dtype=bool)
            for slot, sat_id in enumerate(current_sat_ids):
                if sat_id in previous_residual_by_sat:
                    residual_k[slot] = previous_residual_by_sat[sat_id]
                    residual_mask[slot] = True

            current_mask = np.ones(measurement_count, dtype=bool)
            obs_k = np.stack(
                (residual_k, innovation_k),
                axis=1,
            )
            channel_k = np.stack(
                (residual_mask, current_mask),
                axis=1,
            )
            obs_k = np.where(channel_k, obs_k, 0.0)

            obs_nn = np.zeros(
                (network_nmax, OBSERVATION_FEATURE_DIM),
                dtype=float,
            )
            mask_nn = np.zeros(network_nmax, dtype=bool)
            channel_nn = np.zeros(
                (network_nmax, OBSERVATION_FEATURE_DIM),
                dtype=bool,
            )
            innovation_nn = np.zeros(network_nmax, dtype=float)

            obs_nn[:measurement_count] = obs_k
            mask_nn[:measurement_count] = current_mask
            channel_nn[:measurement_count] = channel_k
            innovation_nn[:measurement_count] = innovation_k
            fixed_nn = np.asarray(fixed_k, dtype=float)

            fixed_t = torch.as_tensor(
                fixed_nn[None], dtype=torch.float32, device=DEVICE
            )
            obs_t = torch.as_tensor(
                obs_nn[None], dtype=torch.float32, device=DEVICE
            )
            mask_t = torch.as_tensor(
                mask_nn[None], dtype=torch.bool, device=DEVICE
            )
            channel_t = torch.as_tensor(
                channel_nn[None], dtype=torch.bool, device=DEVICE
            )
            innovation_t = torch.as_tensor(
                innovation_nn[None], dtype=torch.float32, device=DEVICE
            )

            kalman_gain, _, recurrent_state = model(
                fixed_t,
                obs_t,
                mask_t,
                channel_t,
                recurrent_state=recurrent_state,
            )
            model._pipeline_recurrent_state = recurrent_state

            correction = torch.bmm(
                kalman_gain,
                innovation_t.unsqueeze(-1),
            ).squeeze(-1)[0]

            target_state = np.concatenate([
                fusion_imu_truth_position[fusion_index] - nav_state.position_ecef_m,
                fusion_imu_truth_velocity[fusion_index] - nav_state.velocity_ecef_mps,
                attitude_error_state_target(
                    nav_state.body_to_ecef_dcm,
                    fusion_truth_body_to_ecef[fusion_index],
                ),
            ])
            target_t = torch.as_tensor(
                target_state, dtype=torch.float32, device=DEVICE
            )

            # Keep every network output attached to the same recurrent graph.
            # No hidden-state detach is performed between time steps.
            correction_sequence.append(correction)
            target_sequence.append(target_t)

            if collect_features:
                feature_fixed_rows.append(np.asarray(fixed_k, dtype=float).copy())
                feature_obs_rows.append(np.asarray(obs_k, dtype=float).copy())
                feature_channel_rows.append(np.asarray(channel_k, dtype=bool).copy())

            # The learned correction, not a classical correction, drives the next
            # navigation state and therefore the next network input.
            correction_np = correction.detach().cpu().numpy().astype(float)
            nav_state = inject_error_state(nav_state, correction_np)

            posterior_residual = build_innovation_only(
                nav_state, measurements, lever_arm_b_m
            )
            previous_x_post = previous_context['x_post']
            x_post = correction_np.copy()
            previous_context = {
                'sat_ids': tuple(current_sat_ids),
                'residual': np.asarray(posterior_residual, dtype=float).copy(),
                'x_post': x_post,
                'state_innovation': x_post.copy(),
                'state_residual': x_post - previous_x_post,
                'accel': np.asarray(current_accel, dtype=float).reshape(3).copy(),
                'gyro': np.asarray(current_gyro, dtype=float).reshape(3).copy(),
            }

        if not correction_sequence:
            raise RuntimeError(
                f'No learned updates in sequence [{start_sample}, {stop_sample})'
            )

        # Sequence loss for validation/diagnostics.  V2 training itself is
        # performed by _run_v2_training_batch() below on short trajectories.
        correction_sequence_t = torch.stack(correction_sequence, dim=0)
        target_sequence_t = torch.stack(target_sequence, dim=0)
        sequence_loss = loss_fn(
            correction_sequence_t[:, :SUPERVISED_STATE_DIM],
            target_sequence_t,
        )
        position_rmse = torch.sqrt(
            torch.mean(
                (correction_sequence_t[:, :3] - target_sequence_t[:, :3]) ** 2
            )
        )

        return sequence_loss, position_rmse, feature_fixed_rows, feature_obs_rows, feature_channel_rows


    def _run_v2_training_batch(short_trajectory_starts):
        """Forward one KalmanNet-TSP-style V2 mini-batch.

        Batch semantics match Pipeline_EKF.NNTrain:
          * the batch is a random set of short trajectories;
          * hidden/cell state is initialized once at the batch boundary;
          * all T time steps are forwarded before forming one batch MSE;
          * no recurrent-state detach occurs inside a short trajectory;
          * the caller performs one backward() and one optimizer.step().
        """
        starts = [int(value) for value in short_trajectory_starts]
        trajectories = []
        for start_sample in starts:
            boundary_row = network_source_rows[start_sample]
            boundary_fusion_index = int(boundary_row['fusion_index'])
            nav_state = NavigationState(
                fusion_imu_truth_position[boundary_fusion_index].copy(),
                fusion_imu_truth_velocity[boundary_fusion_index].copy(),
                fusion_truth_body_to_ecef[boundary_fusion_index].copy(),
            )
            boundary_gnss = gnss_preprocessor.prepare_epoch(
                gnss_epochs[boundary_fusion_index], nav_state, lever_arm_b_m
            )
            boundary_measurements = retain_clock_observable_measurements(
                tuple(boundary_gnss) + tuple(boundary_row['leo_measurements'])
            )
            if not boundary_measurements:
                raise RuntimeError(f'No measurements available at V2 boundary sample={start_sample}')
            _, _, _, boundary_sat_ids, _ = build_measurement_model(
                nav_state, boundary_measurements, lever_arm_b_m
            )
            zero_state = np.zeros(INS_STATE_DIM, dtype=float)
            trajectories.append({
                'start_sample': start_sample,
                'nav_state': nav_state,
                'previous_context': {
                    'sat_ids': tuple(boundary_sat_ids),
                    'residual': build_innovation_only(nav_state, boundary_measurements, lever_arm_b_m),
                    'x_post': zero_state.copy(),
                    'state_innovation': zero_state.copy(),
                    'state_residual': zero_state.copy(),
                    'accel': np.asarray(boundary_row['accel'], dtype=float).reshape(3).copy(),
                    'gyro': np.asarray(boundary_row['gyro'], dtype=float).reshape(3).copy(),
                },
            })

        # MaskedCLA has LSTM hidden/cell state rather than KalmanNetNN's covariance
        # GRU states.  The outer Pipeline_EKF-style loop calls init_hidden_KNet(),
        # which sets this explicit state to None (the MaskedCLA zero-state reset).
        recurrent_state = model._pipeline_recurrent_state
        correction_steps = []
        target_steps = []

        for local_t in range(V2_BPTT_LENGTH):
            fixed_batch = []
            observation_batch = []
            mask_batch = []
            channel_batch = []
            innovation_batch = []
            target_batch = []
            step_metadata = []

            for trajectory in trajectories:
                sample_index = trajectory['start_sample'] + local_t
                source_row = network_source_rows[sample_index + 1]
                nav_state = trajectory['nav_state']
                previous_context = trajectory['previous_context']

                current_accel = previous_context['accel']
                current_gyro = previous_context['gyro']
                for imu_index, dt in source_row['preceding_interval_segments']:
                    current_gyro, current_accel = compensate_imu(
                        imr_gyro[int(imu_index)],
                        imr_accel[int(imu_index)],
                    )
                    nav_state = mechanize_ecef(
                        nav_state,
                        current_gyro,
                        current_accel,
                        float(dt),
                    )

                fusion_index = int(source_row['fusion_index'])
                gnss_now = gnss_preprocessor.prepare_epoch(
                    gnss_epochs[fusion_index],
                    nav_state,
                    lever_arm_b_m,
                )
                measurements = retain_clock_observable_measurements(
                    tuple(gnss_now) + tuple(source_row['leo_measurements'])
                )
                if not measurements:
                    raise RuntimeError(
                        'V2 requires a fixed-length trajectory, but no measurements '
                        f'were available at chunk_start={trajectory["start_sample"]}, '
                        f'local_t={local_t}, fusion_index={fusion_index}'
                    )

                innovation_now, _, _, current_sat_ids, _ = build_measurement_model(
                    nav_state,
                    measurements,
                    lever_arm_b_m,
                )
                current_sat_ids = tuple(current_sat_ids)
                innovation_k = np.asarray(
                    innovation_now, dtype=float
                ).reshape(-1)
                measurement_count = len(current_sat_ids)

                delta_accel = (
                    np.asarray(current_accel, dtype=float).reshape(3)
                    - previous_context['accel']
                )
                delta_gyro = (
                    np.asarray(current_gyro, dtype=float).reshape(3)
                    - previous_context['gyro']
                )
                fixed_k = np.concatenate((
                    delta_accel,
                    delta_gyro,
                    previous_context['state_residual'],
                    previous_context['state_innovation'],
                ))

                previous_residual_by_sat = dict(zip(
                    previous_context['sat_ids'],
                    previous_context['residual'],
                ))
                residual_k = np.zeros(measurement_count, dtype=float)
                residual_mask = np.zeros(measurement_count, dtype=bool)
                for slot, sat_id in enumerate(current_sat_ids):
                    if sat_id in previous_residual_by_sat:
                        residual_k[slot] = previous_residual_by_sat[sat_id]
                        residual_mask[slot] = True

                current_mask = np.ones(measurement_count, dtype=bool)
                obs_k = np.stack((residual_k, innovation_k), axis=1)
                channel_k = np.stack((residual_mask, current_mask), axis=1)
                obs_k = np.where(channel_k, obs_k, 0.0)

                obs_nn = np.zeros(
                    (network_nmax, OBSERVATION_FEATURE_DIM), dtype=float
                )
                mask_nn = np.zeros(network_nmax, dtype=bool)
                channel_nn = np.zeros(
                    (network_nmax, OBSERVATION_FEATURE_DIM), dtype=bool
                )
                innovation_nn = np.zeros(network_nmax, dtype=float)

                obs_nn[:measurement_count] = obs_k
                mask_nn[:measurement_count] = current_mask
                channel_nn[:measurement_count] = channel_k
                innovation_nn[:measurement_count] = innovation_k

                target_state = np.concatenate([
                    fusion_imu_truth_position[fusion_index] - nav_state.position_ecef_m,
                    fusion_imu_truth_velocity[fusion_index] - nav_state.velocity_ecef_mps,
                    attitude_error_state_target(
                        nav_state.body_to_ecef_dcm,
                        fusion_truth_body_to_ecef[fusion_index],
                    ),
                ])

                trajectory['nav_state'] = nav_state
                fixed_batch.append(np.asarray(fixed_k, dtype=np.float32))
                observation_batch.append(np.asarray(obs_nn, dtype=np.float32))
                mask_batch.append(mask_nn)
                channel_batch.append(channel_nn)
                innovation_batch.append(np.asarray(innovation_nn, dtype=np.float32))
                target_batch.append(np.asarray(target_state, dtype=np.float32))
                step_metadata.append((
                    measurements,
                    current_sat_ids,
                    current_accel,
                    current_gyro,
                ))

            fixed_t = torch.as_tensor(
                np.stack(fixed_batch), dtype=torch.float32, device=DEVICE
            )
            obs_t = torch.as_tensor(
                np.stack(observation_batch), dtype=torch.float32, device=DEVICE
            )
            mask_t = torch.as_tensor(
                np.stack(mask_batch), dtype=torch.bool, device=DEVICE
            )
            channel_t = torch.as_tensor(
                np.stack(channel_batch), dtype=torch.bool, device=DEVICE
            )
            innovation_t = torch.as_tensor(
                np.stack(innovation_batch), dtype=torch.float32, device=DEVICE
            )
            target_t = torch.as_tensor(
                np.stack(target_batch), dtype=torch.float32, device=DEVICE
            )

            kalman_gain, _, recurrent_state = model(
                fixed_t,
                obs_t,
                mask_t,
                channel_t,
                recurrent_state=recurrent_state,
            )
            model._pipeline_recurrent_state = recurrent_state
            correction_t = torch.bmm(
                kalman_gain,
                innovation_t.unsqueeze(-1),
            ).squeeze(-1)

            correction_steps.append(correction_t)
            target_steps.append(target_t)

            # The navigation state and Yan recursive feature context are numerical
            # environment states.  As in the audited project path, the neural BPTT
            # graph is carried by the LSTM recurrent state; navigation injection is
            # detached before the next physical propagation.
            correction_np = correction_t.detach().cpu().numpy().astype(float)
            for batch_index, trajectory in enumerate(trajectories):
                measurements, current_sat_ids, current_accel, current_gyro = step_metadata[batch_index]
                nav_state = inject_error_state(
                    trajectory['nav_state'], correction_np[batch_index]
                )
                posterior_residual = build_innovation_only(
                    nav_state,
                    measurements,
                    lever_arm_b_m,
                )
                previous_context = trajectory['previous_context']
                previous_x_post = previous_context['x_post']
                x_post = correction_np[batch_index].copy()
                trajectory['nav_state'] = nav_state
                trajectory['previous_context'] = {
                    'sat_ids': tuple(current_sat_ids),
                    'residual': np.asarray(posterior_residual, dtype=float).copy(),
                    'x_post': x_post,
                    'state_innovation': x_post.copy(),
                    'state_residual': x_post - previous_x_post,
                    'accel': np.asarray(current_accel, dtype=float).reshape(3).copy(),
                    'gyro': np.asarray(current_gyro, dtype=float).reshape(3).copy(),
                }

        # Match Pipeline_EKF tensor semantics: one MSE over the complete mini-batch
        # and all time steps, followed by exactly one backward() in the caller.
        correction_batch_t = torch.stack(correction_steps, dim=2)
        target_batch_t = torch.stack(target_steps, dim=2)
        batch_loss = loss_fn(
            correction_batch_t[:, :SUPERVISED_STATE_DIM, :],
            target_batch_t,
        )
        position_rmse = torch.sqrt(torch.mean(
            (correction_batch_t[:, :3, :] - target_batch_t[:, :3, :]) ** 2
        ))
        return batch_loss, position_rmse

    # ------------------------------------------------------------------
    # KalmanNet_TSP Pipeline_EKF.NNTrain structural port
    # ------------------------------------------------------------------
    # These names intentionally mirror the repository.  The outer training/CV
    # operation order and tensor-loss/optimizer operations below are kept in the
    # same order as Pipeline_EKF.py.  Architecture-specific KalmanNetNN internals
    # (GRU_Q/GRU_Sigma/GRU_S and differentiable f/h state recursion) cannot be
    # copied without replacing MaskedCLA and the NumPy navigation environment.
    N_steps = int(TRAINING_EPOCHS)
    N_B = int(V2_BATCH_SIZE)
    N_E = int(v2_sequence_count)
    N_CV = 1  # this project has one chronological CV trajectory

    MSE_cv_linear_epoch = torch.zeros([N_steps])
    MSE_cv_dB_epoch = torch.zeros([N_steps])
    MSE_train_linear_epoch = torch.zeros([N_steps])
    MSE_train_dB_epoch = torch.zeros([N_steps])

    MSE_cv_dB_opt = torch.tensor(1000.0)
    MSE_cv_idx_opt = 0
    best_model_path = OUTPUT_DIR / 'best-model.pt'

    training_history = []
    best_validation_mse = float('inf')
    best_epoch = None

    print('\n=== DATA01 KALMANNET_TSP PIPELINE_EKF ORDER + V2 SHORT TRAJECTORIES ===')
    print('NN input source: learned recursive trajectory (no classical KF features)')
    print(
        'Exact outer order: optimizer.zero_grad -> model.train -> batch_size -> '
        'init_hidden_KNet -> random.sample -> InitSequence analogue -> T-step forward -> '
        'MSE -> dB -> backward(retain_graph=True) -> optimizer.step -> model.eval -> '
        'CV batch_size -> init_hidden_KNet -> no_grad CV forward -> CV MSE/dB -> save best'
    )

    for ti in range(0, N_steps):

        ###############################
        ### Training Sequence Batch ###
        ###############################
        optimizer.zero_grad()
        model.train()
        model.batch_size = N_B
        model.init_hidden_KNet()

        # Pipeline_EKF uses random.sample without replacement inside each step.
        n_e = random.sample(range(N_E), k=N_B)
        selected_v2_starts = [v2_chunk_starts[index] for index in n_e]

        # _run_v2_training_batch performs this project's InitSequence analogue
        # (truth state at each reserved T+1 boundary), then T sequential forwards.
        MSE_trainbatch_linear_LOSS, train_position_rmse = _run_v2_training_batch(selected_v2_starts)

        # dB loss is recorded before backward(), matching Pipeline_EKF.py.
        MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
        MSE_train_dB_epoch[ti] = 10 * torch.log10(MSE_train_linear_epoch[ti])

        ##################
        ### Optimizing ###
        ##################
        MSE_trainbatch_linear_LOSS.backward(retain_graph=True)
        optimizer.step()

        #################################
        ### Validation Sequence Batch ###
        #################################
        model.eval()
        model.batch_size = N_CV
        model.init_hidden_KNet()
        with torch.no_grad():
            validation_loss, validation_position_rmse, _, _, _ = _evaluate_recursive_sequence(
                validation_start_sample, data01_total_sample_count
            )
            MSE_cvbatch_linear_LOSS = validation_loss
            MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
            MSE_cv_dB_epoch[ti] = 10 * torch.log10(MSE_cv_linear_epoch[ti])

            if MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt:
                MSE_cv_dB_opt = MSE_cv_dB_epoch[ti].clone()
                MSE_cv_idx_opt = int(ti)
                torch.save(model, best_model_path)

        train_mse = float(MSE_train_linear_epoch[ti])
        validation_mse = float(MSE_cv_linear_epoch[ti])
        train_mse_db = float(MSE_train_dB_epoch[ti])
        validation_mse_db = float(MSE_cv_dB_epoch[ti])
        train_position_rmse_m = float(train_position_rmse.detach().cpu())
        validation_position_rmse_m = float(validation_position_rmse.detach().cpu())

        best_epoch = int(MSE_cv_idx_opt + 1)
        best_validation_mse = float(10.0 ** (float(MSE_cv_dB_opt) / 10.0))

        training_history.append({
            'epoch': int(ti + 1),
            'optimization_mse': train_mse,
            'train_mse': train_mse,
            'validation_mse': validation_mse,
            'train_mse_db': train_mse_db,
            'validation_mse_db': validation_mse_db,
            'train_position_component_rmse_m': train_position_rmse_m,
            'validation_position_component_rmse_m': validation_position_rmse_m,
            'learning_rate': float(LEARNING_RATE),
            'weight_decay': float(WEIGHT_DECAY),
            'training_algorithm': 'KalmanNet_TSP_Pipeline_EKF_structural_port_V2',
            'bptt_length': int(V2_BPTT_LENGTH),
            'batch_size': int(V2_BATCH_SIZE),
            'batch_short_trajectory_indices': [int(v) for v in n_e],
            'batch_short_trajectory_starts': [int(v) for v in selected_v2_starts],
            'split': 'chronological_train_validation_then_repository_Short_Traj_Split_semantics',
            'validation_available': True,
            'network_input_source': 'learned_recursive_state_no_classical_KF',
        })

        ########################
        ### Training Summary ###
        ########################
        print(
            ti,
            'MSE Training :', MSE_train_dB_epoch[ti], '[dB]',
            'MSE Validation :', MSE_cv_dB_epoch[ti], '[dB]'
        )
        print(
            f'  Train mini-batch MSE: {train_mse:.6g} | '
            f'Validation MSE: {validation_mse:.6g} | '
            f'Train position RMSE: {train_position_rmse_m:.3f} m | '
            f'Validation position RMSE: {validation_position_rmse_m:.3f} m'
        )

        if ti > 1:
            d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
            d_cv = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
            print('diff MSE Training :', d_train, '[dB]', 'diff MSE Validation :', d_cv, '[dB]')

        print('Optimal idx:', MSE_cv_idx_opt, 'Optimal :', MSE_cv_dB_opt, '[dB]')

    if not best_model_path.is_file():
        raise RuntimeError('Pipeline_EKF-style training did not produce a validation checkpoint')

    # Pipeline_EKF.NNTest later reloads best-model.pt.  Reload it here before this
    # project's post-training diagnostics so all subsequent metrics use the same
    # selected checkpoint.
    try:
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        model = torch.load(best_model_path, map_location=DEVICE)
    model = model.to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    training_losses = []
    training_position_mse = []
    learned_fixed_rows = []
    learned_obs_rows = []
    learned_channel_rows = []
    with torch.no_grad():
        for start_sample in v2_chunk_starts:
            model.init_hidden_KNet()
            loss, position_rmse, fixed_rows, obs_rows, channel_rows = _evaluate_recursive_sequence(
                start_sample, start_sample + V2_BPTT_LENGTH, collect_features=True
            )
            training_losses.append(float(loss.detach().cpu()))
            training_position_mse.append(float(position_rmse.detach().cpu()) ** 2)
            learned_fixed_rows.extend(fixed_rows)
            learned_obs_rows.extend(obs_rows)
            learned_channel_rows.extend(channel_rows)

        model.init_hidden_KNet()
        validation_loss, validation_position_rmse, _, _, _ = _evaluate_recursive_sequence(
            validation_start_sample, data01_total_sample_count
        )

    final_training_mse = float(np.mean(training_losses))
    final_training_position_rmse_m = float(math.sqrt(np.mean(training_position_mse)))
    final_validation_mse = float(validation_loss.detach().cpu())
    final_validation_position_rmse_m = float(validation_position_rmse.detach().cpu())

    print('\n=== SELECTED MODEL: TRAINING AND VALIDATION METRICS ===')
    print(f'Best validation checkpoint step: {best_epoch}')
    print(f'Minimum validation MSE during training: {best_validation_mse:.6g}')
    print(f'Minimum validation MSE during training [dB]: {10.0 * math.log10(max(float(best_validation_mse), 1e-30)):.3f}')
    print(f'Full V2 training-set MSE at selected checkpoint: {final_training_mse:.6g}')
    print(f'Full V2 training-set MSE at selected checkpoint [dB]: {10.0 * math.log10(max(float(final_training_mse), 1e-30)):.3f}')
    print(f'Validation MSE at selected checkpoint: {final_validation_mse:.6g}')
    print(f'Validation MSE at selected checkpoint [dB]: {10.0 * math.log10(max(float(final_validation_mse), 1e-30)):.3f}')
    print(f'Full V2 training-set position RMSE: {final_training_position_rmse_m:.3f} m')
    print(f'Validation position RMSE: {final_validation_position_rmse_m:.3f} m')

    # Training-distribution diagnostics are now computed from the selected model's
    # learned closed-loop training trajectory, not from classical KF features.
    fixed_for_stats = np.stack(learned_fixed_rows)
    train_fixed_min = np.min(fixed_for_stats, axis=0)
    train_fixed_max = np.max(fixed_for_stats, axis=0)
    train_fixed_abs_max = np.maximum(
        np.max(np.abs(fixed_for_stats), axis=0), 1e-12
    )

    train_obs_min = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    train_obs_abs_max = np.empty(OBSERVATION_FEATURE_DIM, dtype=float)
    for diagnostic_channel in range(OBSERVATION_FEATURE_DIM):
        valid_values = []
        for obs_row, channel_row in zip(
            learned_obs_rows, learned_channel_rows
        ):
            valid = channel_row[:, diagnostic_channel]
            if np.any(valid):
                valid_values.extend(
                    obs_row[valid, diagnostic_channel].tolist()
                )
        valid_values = np.asarray(valid_values, dtype=float)
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

    # Loss-vs-training-step outputs (the repository variable n_steps).
    history_epoch = np.asarray(
        [row['epoch'] for row in training_history], dtype=int
    )
    history_train_mse = np.asarray(
        [row['train_mse'] for row in training_history], dtype=float
    )
    history_validation_mse = np.asarray(
        [row['validation_mse'] for row in training_history], dtype=float
    )

    with (OUTPUT_DIR / 'loss_epoch_train_validation.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_mse', 'validation_mse'])
        writer.writerows(
            zip(
                history_epoch.tolist(),
                history_train_mse.tolist(),
                history_validation_mse.tolist(),
            )
        )

    np.savez(
        OUTPUT_DIR / 'loss_epoch_train_validation.npz',
        epoch=history_epoch,
        train_mse=history_train_mse,
        validation_mse=history_validation_mse,
        best_epoch=np.asarray(best_epoch),
        best_validation_mse=np.asarray(best_validation_mse),
    )

    fig = plt.figure(figsize=(8, 5))
    plt.plot(history_epoch, history_train_mse, label='Train mini-batch MSE')
    plt.plot(history_epoch, history_validation_mse, label='Validation MSE')
    plt.xlabel('Epoch / Training step')
    plt.ylabel('MSE Loss')
    plt.title('Masked KalmanNet: Train and Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'loss_epoch_train_validation.png', dpi=180
    )
    plt.close(fig)

    if np.all(history_train_mse > 0.0) and np.all(
        history_validation_mse > 0.0
    ):
        fig = plt.figure(figsize=(8, 5))
        plt.semilogy(
            history_epoch, history_train_mse, label='Train mini-batch MSE'
        )
        plt.semilogy(
            history_epoch,
            history_validation_mse,
            label='Validation MSE',
        )
        plt.xlabel('Training step')
        plt.ylabel('MSE Loss (log scale)')
        plt.title('Masked KalmanNet: Train and Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(
            OUTPUT_DIR / 'loss_epoch_train_validation_log.png', dpi=180
        )
        plt.close(fig)

    model.eval()
    torch.save({
        'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        'checkpoint_schema': 'masked_cla_training_validation_repo_audited',
        'nmax': int(network_nmax),
        'training_observed_nmax': int(network_nmax),
        'data01_sequence_nmax': int(data01_sequence_nmax),
        'state_order': '[delta_p,delta_v,delta_theta]',
        'kalman_gain_state_dimension': INS_STATE_DIM,
        'direct_state_label_dimension': SUPERVISED_STATE_DIM,
        'measurement_mode': 'pseudorange_only',
        'feature_normalization': 'none_raw_inputs',
        'sequential_input_generation_changed': True,
        'network_input_source': 'learned_recursive_state_no_classical_KF',
        'training_algorithm': 'KalmanNet_TSP_Pipeline_EKF_structural_port_V2',
        'training_sequence_length': int(V2_BPTT_LENGTH),
        'training_sequence_count': int(v2_sequence_count),
        'training_sequence_sampling': 'repository_random.sample_without_replacement_per_optimizer_step',
        'training_batch_size': int(V2_BATCH_SIZE),
        'training_supervised_sample_count': int(v2_supervised_sample_count),
        'training_source_sample_count_before_V2_split': int(training_sample_count),
        'v2_reserved_boundary_state_count': int(v2_sequence_count),
        'v2_unused_or_boundary_transition_count': int(training_sample_count - v2_supervised_sample_count),
        'v2_chunk_initialization': 'truth_navigation_state_at_reserved_boundary_analogous_to_train_init',
        'validation_sequence_length': int(validation_sample_count),
        'validation_sequence_count': 1,
        'validation_sequence_order': 'chronological_preserved',
        'validation_start_sample': int(validation_start_sample),
        'validation_fraction': float(validation_sample_count / data01_total_sample_count),
        'optimizer': 'Adam',
        'reproducibility_seed': int(REPRODUCIBILITY_SEED),
        'loss': 'MSELoss(reduction=mean)',
        'learning_rate': float(LEARNING_RATE),
        'weight_decay': float(WEIGHT_DECAY),
        'alternating_optimization': False,
        'all_model_parameters_updated_together': True,
        'validation_available': True,
        'validation_reason': 'chronological_tail_holdout_from_Data01',
        'checkpoint_selection': 'minimum_chronological_validation_MSE',
        'best_epoch': int(best_epoch),
        'best_validation_mse': float(best_validation_mse),
        'final_training_mse': float(final_training_mse),
        'final_training_position_component_rmse_m': float(final_training_position_rmse_m),
        'final_validation_mse': float(final_validation_mse),
        'final_validation_position_component_rmse_m': float(final_validation_position_rmse_m),
        'v2_short_trajectory_chunking': True,
        'v2_recurrent_state_reset_at_batch_start': True,
        'repository_pipeline_operation_order_ported': True,
        'repository_exact_end_to_end_autograd_recursion': False,
        'repository_exact_end_to_end_autograd_limitation': 'navigation_mechanization_measurement_and_state_injection_are_numpy_and_learned_correction_is_detached_before_physical_recursion',
        'v2_recurrent_detach_inside_short_trajectory': False,
        'v2_one_loss_one_backward_one_step_per_minibatch': True,
        'removed_gradient_clipping': True,
        'removed_optimizer_windows': True,
        'removed_v1_extra_finetune_stage': True,
        'removed_custom_gain_head_scaling': True,
        'chronological_Data01_validation_split': True,
        'closed_loop_feature_regeneration_during_training': True,
        'bptt_structure': 'V2_independent_short_trajectories_LSTM_BPTT_with_navigation_feedback_detached',
        'classical_KF_features_used_for_NN_training': False,
    }, OUTPUT_DIR / 'best_model.pt')
    (OUTPUT_DIR / 'history.json').write_text(json.dumps(training_history, indent=2), encoding='utf-8')
    # Data01 is used only for training/validation and for training-distribution
    # statistics required by the independent Data02 evaluation.  The separate
    # full recursive Data01 post-training diagnostic is intentionally omitted.
    missing_test_files = [
        path for path in (
            TEST_README_XML_PATH,
            TEST_ROVE_GROUND_TRUTH_PATH,
            TEST_IMU_GROUND_TRUTH_PATH,
            TEST_RINEX_OBS_PATH,
            TEST_IMR_PATH,
            TEST_SP3_PATH,
            TEST_CLK_PATH,
            TEST_NAV_PATH,
        )
        if not path.is_file()
    ]
    if missing_test_files:
        raise FileNotFoundError(
            'Missing Data02 input file(s):\n' + '\n'.join(str(path) for path in missing_test_files)
        )

    test_root = ET.fromstring(TEST_README_XML_PATH.read_text(encoding='utf-8', errors='replace'))
    test_rover_xml = next(rover_xml for rover_xml in test_root.findall('ROVE') if (rover_xml.findtext('ID') or '').strip() == '01')
    test_rover_imu_type = (test_rover_xml.findtext('SINS_IMUType') or '').strip()
    test_rover_lever_arm_vehicle_m = np.fromstring(test_rover_xml.findtext('SINS_LeverArm_GNSS') or '', sep=' ')
    test_rover_mounting_xyz_deg = np.fromstring(test_rover_xml.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    test_antenna_truth_rows = []
    test_antenna_truth_started = False
    for line in TEST_ROVE_GROUND_TRUTH_PATH.read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if len(fields) < 24:
            if test_antenna_truth_started:
                break
            continue
        try:
            row = (int(fields[0]), float(fields[1]), float(fields[9]), float(fields[10]), float(fields[11]), float(fields[15]), float(fields[16]), float(fields[17]), float(fields[21]), float(fields[22]), float(fields[23]))
        except (ValueError, IndexError):
            if test_antenna_truth_started:
                break
            continue
        test_antenna_truth_started = True
        test_antenna_truth_rows.append(row)
    test_antenna_truth_array = np.asarray(test_antenna_truth_rows, dtype=float)
    test_antenna_truth_week = test_antenna_truth_array[:, 0].astype(int)
    test_antenna_truth = (test_antenna_truth_week.astype(float) * GPS_WEEK_S + test_antenna_truth_array[:, 1], test_antenna_truth_array[:, 2:5], test_antenna_truth_array[:, 5:8], test_antenna_truth_array[:, 8], test_antenna_truth_array[:, 9], test_antenna_truth_array[:, 10])
    test_imu_truth_rows = []
    test_imu_truth_started = False
    for line in TEST_IMU_GROUND_TRUTH_PATH.read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if len(fields) < 24:
            if test_imu_truth_started:
                break
            continue
        try:
            row = (int(fields[0]), float(fields[1]), float(fields[9]), float(fields[10]), float(fields[11]), float(fields[15]), float(fields[16]), float(fields[17]), float(fields[21]), float(fields[22]), float(fields[23]))
        except (ValueError, IndexError):
            if test_imu_truth_started:
                break
            continue
        test_imu_truth_started = True
        test_imu_truth_rows.append(row)
    test_imu_truth_array = np.asarray(test_imu_truth_rows, dtype=float)
    test_imu_truth_week = test_imu_truth_array[:, 0].astype(int)
    test_imu_truth = (test_imu_truth_week.astype(float) * GPS_WEEK_S + test_imu_truth_array[:, 1], test_imu_truth_array[:, 2:5], test_imu_truth_array[:, 5:8], test_imu_truth_array[:, 8], test_imu_truth_array[:, 9], test_imu_truth_array[:, 10])
    test_imu_noise = imu_models[test_rover_imu_type]
    test_rinex = RINEXObservationFile(TEST_RINEX_OBS_PATH)
    first_test_rinex_epoch = next(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}), None)
    with TEST_IMR_PATH.open('rb') as stream:
        test_imr_header_buffer = stream.read(512)
    test_imr_endian = '<' if test_imr_header_buffer[8] == 0 else '>'
    test_imr_header = struct.unpack(test_imr_endian + IMR_HEADER_FORMAT_BODY, test_imr_header_buffer)
    test_imr_time_tag_bias_s = float(test_imr_header[10]) * 0.001
    test_imr_record_bytes = struct.calcsize(test_imr_endian + IMR_RECORD_FORMAT_BODY)
    test_imr_record_count = (TEST_IMR_PATH.stat().st_size - 512) // test_imr_record_bytes
    test_imr_record_dtype = np.dtype([('tow', test_imr_endian + 'f8'), ('counts', test_imr_endian + 'i4', (6,))], align=False)
    test_imr_records = np.memmap(TEST_IMR_PATH, dtype=test_imr_record_dtype, mode='r', offset=512, shape=(test_imr_record_count,))
    test_imr_tow_all = np.asarray(test_imr_records['tow'], dtype=np.float64).copy()
    del test_imr_records
    test_imr_tow_all[test_imr_tow_all > GPS_WEEK_S] -= GPS_WEEK_S
    test_imr_tow_all -= test_imr_time_tag_bias_s
    test_anchor_time = float(first_test_rinex_epoch[0])
    test_imr_time_all = anchor_imr_tow_to_gpst_seconds(test_imr_tow_all, test_anchor_time)
    test_antenna_truth_time = np.asarray(test_antenna_truth[0], dtype=float).reshape(-1)
    test_imu_truth_time = np.asarray(test_imu_truth[0], dtype=float).reshape(-1)
    test_common_start = max(float(test_imr_time_all[0]), float(test_antenna_truth_time[0]), float(test_imu_truth_time[0]))
    test_common_end = min(float(test_imr_time_all[-1]), float(test_antenna_truth_time[-1]), float(test_imu_truth_time[-1]))
    test_start = int(np.searchsorted(test_imr_time_all, test_common_start, side='left'))
    test_usable_imu_start = float(test_imr_time_all[test_start])
    test_gnss_epochs = tuple(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=test_usable_imu_start, end_time_gpst_s=test_common_end, max_epochs=MAX_TEST_FUSION_EPOCHS, require_measurements=True))
    test_fusion_time = np.asarray([epoch[0] for epoch in test_gnss_epochs], dtype=float)
    test_common_stop = min(int(np.searchsorted(test_imr_time_all, test_common_end, side='left')) + 1, len(test_imr_time_all))
    test_requested_stop = min(int(np.searchsorted(test_imr_time_all, test_fusion_time[-1], side='left')) + 1, test_common_stop)
    with TEST_IMR_PATH.open('rb') as stream:
        test_imr_sample_header_buffer = stream.read(512)
    test_imr_sample_endian = '<' if test_imr_sample_header_buffer[8] == 0 else '>'
    test_imr_sample_header = struct.unpack(test_imr_sample_endian + IMR_HEADER_FORMAT_BODY, test_imr_sample_header_buffer)
    test_imr_data_rate_hz = float(test_imr_sample_header[5])
    test_imr_gyro_scale = float(test_imr_sample_header[6])
    test_imr_accel_scale = float(test_imr_sample_header[7])
    test_imr_record_bytes = struct.calcsize(test_imr_sample_endian + IMR_RECORD_FORMAT_BODY)
    test_imr_record_dtype = np.dtype([('tow', test_imr_sample_endian + 'f8'), ('counts', test_imr_sample_endian + 'i4', (6,))], align=False)
    test_imr_records = np.fromfile(TEST_IMR_PATH, dtype=test_imr_record_dtype, count=test_requested_stop - test_start, offset=512 + test_start * test_imr_record_bytes)
    test_imr_gyro = np.deg2rad(test_imr_records['counts'][:, :3].astype(float) * test_imr_gyro_scale * test_imr_data_rate_hz)
    test_imr_accel = test_imr_records['counts'][:, 3:6].astype(float) * test_imr_accel_scale * test_imr_data_rate_hz
    test_imr_time = test_imr_time_all[test_start:test_requested_stop].copy()
    del test_imr_tow_all, test_imr_time_all
    test_query_time = np.concatenate(([test_imr_time[0]], test_fusion_time))
    test_antenna_position = interpolate_ground_truth(test_antenna_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S, position_only=True)
    test_imu_position, test_imu_velocity, test_imu_heading, test_imu_pitch, test_imu_roll = interpolate_ground_truth(test_imu_truth, test_query_time, MAX_TRUTH_INTERPOLATION_GAP_S)
    test_fusion_antenna_truth_position = test_antenna_position[1:]
    test_C_b_e = body_to_ecef_from_ie_hpr(test_imu_position[0], test_imu_heading[0], test_imu_pitch[0], test_imu_roll[0], test_rover_mounting_xyz_deg)
    test_lever_arm_b_m = c_vehicle_to_body_zxy(*test_rover_mounting_xyz_deg) @ np.asarray(test_rover_lever_arm_vehicle_m, dtype=float).reshape(3)
    test_initial_nav = NavigationState(test_imu_position[0].copy(), test_imu_velocity[0].copy(), test_C_b_e)
    test_P0_sigma = np.concatenate([test_imu_noise['isdv_pos_m'], test_imu_noise['isdv_vel_mps'], test_imu_noise['isdv_att_rad']])
    test_P0 = np.diag(test_P0_sigma ** 2)
    test_Qc_density = np.concatenate([test_imu_noise['pnsd_pos_m_sqrt_s'], test_imu_noise['pnsd_vel_mps_sqrt_s'], test_imu_noise['pnsd_att_rad_sqrt_s']])
    test_Qc = np.diag(test_Qc_density ** 2)
    test_ionosphere_coefficients = None
    if USE_IONOSPHERE:
        test_iono_header = {}
        with TEST_NAV_PATH.open('r', encoding='ascii', errors='replace') as stream:
            stream.readline()
            for line in stream:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'IONOSPHERIC CORR':
                    fields = line[:60].split()
                    if len(fields) >= 5 and fields[0] in {'GPSA', 'GPSB', 'BDSA', 'BDSB'}:
                        test_iono_header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
        test_ionosphere_coefficients = {'G': (np.asarray(test_iono_header['GPSA']), np.asarray(test_iono_header['GPSB'])), 'C': (np.asarray(test_iono_header['BDSA']), np.asarray(test_iono_header['BDSB']))}
    test_gnss_preprocessor = GNSSPreprocessor(SP3Orbit(TEST_SP3_PATH), RINEXClock(TEST_CLK_PATH), min_elevation_deg=MIN_GNSS_ELEVATION_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE, broadcast_ionosphere_coefficients=test_ionosphere_coefficients)
    test_leo_klobuchar = None if test_ionosphere_coefficients is None else test_ionosphere_coefficients['G']
    test_leo_simulator = LEODownlinkSimulator(tle_provider, test_leo_klobuchar, seed=TEST_LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
    print('\n=== DATA02 INDEPENDENT TEST ===')
    model = model.to(DEVICE)
    model.eval()
    nav = test_initial_nav.copy()
    P = test_P0.copy()
    online_rows = []
    previous_online = None
    test_recurrent_capacity = int(network_nmax)
    test_recurrent_state = None
    data02_capacity_selection_epochs = 0
    data02_capacity_excluded_measurements = 0
    last_gyro, last_accel = compensate_imu(test_imr_gyro[0], test_imr_accel[0])
    last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
    for event_type, event_start, event_end, event_index in build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
        if event_type == 'propagation':
            imu_index = event_index
            dt = event_end - event_start
            last_gyro, last_accel = compensate_imu(test_imr_gyro[imu_index], test_imr_accel[imu_index])
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
        gnss_measurements = test_gnss_preprocessor.prepare_epoch(epoch, nav, test_lever_arm_b_m)
        leo_measurements = test_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])
        measurements = retain_clock_observable_measurements(tuple(gnss_measurements) + tuple(leo_measurements))
        if not measurements:
            continue
        truth_position_now = test_fusion_antenna_truth_position[fusion_index]
        prior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        prior_error_3d_m = float(np.linalg.norm(prior_position - truth_position_now))
        if previous_online is None:
            _, _, _, warm_sat_ids, _ = build_measurement_model(
                nav, measurements, test_lever_arm_b_m
            )
            # First epoch initializes Yan's previous-epoch features only.
            # No classical Kalman update is applied.
            warm_posterior_residual = build_innovation_only(
                nav, measurements, test_lever_arm_b_m
            )
            zero_state = np.zeros(INS_STATE_DIM, dtype=float)
            previous_online = {
                'sat_ids': tuple(warm_sat_ids),
                'residual': np.asarray(warm_posterior_residual, dtype=float).copy(),
                'x_post': zero_state.copy(),
                'state_innovation': zero_state.copy(),
                'state_residual': zero_state.copy(),
                'accel': np.asarray(last_feature_accel, dtype=float).reshape(3).copy(),
                'gyro': np.asarray(last_feature_gyro, dtype=float).reshape(3).copy(),
            }
            continue
        input_innovation, _, _, _, _ = build_measurement_model(nav, measurements, test_lever_arm_b_m)
        pre_capacity_count = len(measurements)
        if pre_capacity_count > test_recurrent_capacity:
            capacity_measurements = tuple(retain_clock_observable_measurements(measurements))
            if len(capacity_measurements) > test_recurrent_capacity:
                ranked = sorted(enumerate(capacity_measurements), key=lambda item: (float(item[1].sigma_code_m), str(item[1].sat_id), int(item[0])))
                selected_indices = []
                selected_set = set()
                constellation_counts = Counter()
                for rank_pos, (original_index, measurement) in enumerate(ranked):
                    if len(selected_indices) >= test_recurrent_capacity:
                        break
                    if original_index in selected_set:
                        continue
                    constellation = measurement.constellation
                    remaining_slots = test_recurrent_capacity - len(selected_indices)
                    if constellation in {'G', 'C'} and constellation_counts[constellation] == 0:
                        if remaining_slots < 2:
                            continue
                        partner = None
                        for partner_index, partner_measurement in ranked[rank_pos + 1:]:
                            if partner_index not in selected_set and partner_measurement.constellation == constellation:
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
                selected_indices = sorted(selected_indices[:test_recurrent_capacity])
                measurements = tuple(capacity_measurements[index] for index in selected_indices)
                measurements = tuple(retain_clock_observable_measurements(measurements))
            else:
                measurements = capacity_measurements
            data02_capacity_selection_epochs += 1
            data02_capacity_excluded_measurements += pre_capacity_count - len(measurements)
        innovation_now, measurement_H, measurement_R, current_sat_ids, _ = build_measurement_model(nav, measurements, test_lever_arm_b_m)
        n = len(measurements)
        innovation_k = np.asarray(innovation_now, dtype=float).copy()
        current_mask = np.ones(n, dtype=bool)
        residual_k = np.zeros(n, dtype=float)
        residual_mask = np.zeros(n, dtype=bool)
        previous_residual = dict(zip(previous_online['sat_ids'], previous_online['residual']))
        for slot, sat_id in enumerate(current_sat_ids):
            if sat_id in previous_residual:
                residual_k[slot] = previous_residual[sat_id]
                residual_mask[slot] = True

        fixed_k = np.concatenate([
            np.asarray(last_feature_accel, dtype=float).reshape(3) - previous_online['accel'],
            np.asarray(last_feature_gyro, dtype=float).reshape(3) - previous_online['gyro'],
            previous_online['state_residual'],
            previous_online['state_innovation'],
        ])
        obs_k = np.stack([residual_k, innovation_k], axis=1)
        channel_k = np.stack([residual_mask, current_mask], axis=1)
        obs_k = np.where(channel_k, obs_k, 0.0)

        fixed_outside = (fixed_k < train_fixed_min) | (fixed_k > train_fixed_max)
        fixed_ratio = float(np.max(np.abs(fixed_k) / train_fixed_abs_max))
        obs_outside_count = 0
        obs_valid_count = 0
        obs_ratio = 0.0
        for channel in range(OBSERVATION_FEATURE_DIM):
            valid = channel_k[:, channel]
            if np.any(valid):
                values = obs_k[valid, channel]
                obs_valid_count += int(values.size)
                obs_outside_count += int(np.count_nonzero((values < train_obs_min[channel]) | (values > train_obs_max[channel])))
                obs_ratio = max(obs_ratio, float(np.max(np.abs(values) / train_obs_abs_max[channel])))
        feature_shift = {
            'fixed_ood_fraction': float(np.mean(fixed_outside)),
            'fixed_max_abs_train_ratio': fixed_ratio,
            'obs_ood_fraction': float(obs_outside_count / obs_valid_count) if obs_valid_count else 0.0,
            'obs_max_abs_train_ratio': float(obs_ratio),
        }

        input_innovation = np.asarray(input_innovation, dtype=float)
        input_innovation_rms_m = float(np.sqrt(np.mean(input_innovation ** 2))) if input_innovation.size else 0.0
        input_innovation_max_abs_m = float(np.max(np.abs(input_innovation))) if input_innovation.size else 0.0
        knet_innovation_rms_m = float(np.sqrt(np.mean(innovation_k ** 2))) if innovation_k.size else 0.0
        knet_innovation_max_abs_m = float(np.max(np.abs(innovation_k))) if innovation_k.size else 0.0

        fixed_nn = np.asarray(fixed_k, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_nn = np.zeros((test_recurrent_capacity, OBSERVATION_FEATURE_DIM), dtype=float)
        mask_nn = np.zeros(test_recurrent_capacity, dtype=bool)
        channel_nn = np.zeros((test_recurrent_capacity, OBSERVATION_FEATURE_DIM), dtype=bool)
        innovation_nn = np.zeros(test_recurrent_capacity, dtype=float)
        obs_nn[:n] = obs_k
        mask_nn[:n] = current_mask
        channel_nn[:n] = channel_k
        innovation_nn[:n] = innovation_k
        with torch.inference_mode():
            gain_tensor, _, test_recurrent_state = model(torch.tensor(fixed_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(obs_nn[None], dtype=torch.float32, device=DEVICE), torch.tensor(mask_nn[None], dtype=torch.bool, device=DEVICE), torch.tensor(channel_nn[None], dtype=torch.bool, device=DEVICE), recurrent_state=test_recurrent_state)
            innovation_tensor = torch.tensor(innovation_nn[None], dtype=torch.float32, device=DEVICE)
            correction = torch.bmm(gain_tensor, innovation_tensor.unsqueeze(-1)).squeeze(-1)[0]
        gain = gain_tensor[0].cpu().numpy().astype(float)
        correction = correction.cpu().numpy().astype(float)
        active_gain = gain[:, :n]
        gain_fro_norm = float(np.linalg.norm(active_gain))
        gain_max_abs = float(np.max(np.abs(active_gain))) if active_gain.size else 0.0
        gain_navigation_rows_fro_norm = float(np.linalg.norm(active_gain[:SUPERVISED_STATE_DIM, :]))
        correction_position_norm_m = float(np.linalg.norm(correction[0:3]))
        correction_velocity_norm_mps = float(np.linalg.norm(correction[3:6]))
        correction_attitude_norm_rad = float(np.linalg.norm(correction[6:9]))
        P = learned_gain_covariance_update(P, active_gain, measurement_H, measurement_R, correction)
        nav = inject_error_state(nav, correction)
        posterior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        posterior_error_3d_m = float(np.linalg.norm(posterior_position - truth_position_now))
        posterior_residual = build_innovation_only(nav, measurements, test_lever_arm_b_m)
        previous_x_post = previous_online['x_post']
        previous_online = {
            'sat_ids': tuple(current_sat_ids),
            'residual': np.asarray(posterior_residual, dtype=float).copy(),
            'x_post': correction.copy(),
            'state_innovation': correction.copy(),
            'state_residual': correction - previous_x_post,
            'accel': np.asarray(last_feature_accel, dtype=float).reshape(3).copy(),
            'gyro': np.asarray(last_feature_gyro, dtype=float).reshape(3).copy(),
        }
        counts = Counter((m.constellation for m in measurements))
        online_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': test_fusion_antenna_truth_position[fusion_index].copy(), 'n_input': len(measurements), 'n_total': n, 'n_gnss': counts['G'] + counts['C'], 'n_leo': counts['L'], 'sat_ids': current_sat_ids, 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': posterior_error_3d_m, 'input_innovation_rms_m': input_innovation_rms_m, 'input_innovation_max_abs_m': input_innovation_max_abs_m, 'knet_innovation_rms_m': knet_innovation_rms_m, 'knet_innovation_max_abs_m': knet_innovation_max_abs_m, 'gain_fro_norm': gain_fro_norm, 'gain_max_abs': gain_max_abs, 'gain_navigation_rows_fro_norm': gain_navigation_rows_fro_norm, 'correction_position_norm_m': correction_position_norm_m, 'correction_velocity_norm_mps': correction_velocity_norm_mps, 'correction_attitude_norm_rad': correction_attitude_norm_rad, **feature_shift, 'fusion_mode': 'MaskedCLA'})
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
    rmse_ned = np.sqrt(np.mean(ned_error ** 2, axis=0))
    rmse_3d = float(np.sqrt(np.mean(error_3d ** 2)))
    rmse = np.append(rmse_ned, rmse_3d)
    cdf_probability = np.arange(1, len(error_3d) + 1) / len(error_3d)
    diagnostic_fields = ('prior_error_3d_m', 'posterior_error_3d_m', 'input_innovation_rms_m', 'input_innovation_max_abs_m', 'knet_innovation_rms_m', 'knet_innovation_max_abs_m', 'gain_fro_norm', 'gain_max_abs', 'gain_navigation_rows_fro_norm', 'correction_position_norm_m', 'correction_velocity_norm_mps', 'correction_attitude_norm_rad', 'fixed_ood_fraction', 'fixed_max_abs_train_ratio', 'obs_ood_fraction', 'obs_max_abs_train_ratio')
    diagnostic_arrays = {name: np.asarray([row[name] for row in online_rows], dtype=float) for name in diagnostic_fields}

    growth_ratio = np.full(len(error_3d), np.nan, dtype=float)
    if len(error_3d) > 1:
        growth_ratio[1:] = error_3d[1:] / np.maximum(error_3d[:-1], 1e-09)
    np.savez(OUTPUT_DIR / 'test_evaluation.npz', dataset_dir=np.asarray(str(TEST_ROVE_GROUND_TRUTH_PATH.parent)), time_gpst_s=online_time, estimate_ecef_m=estimate, truth_ecef_m=truth_aligned, ned_error_m=ned_error, error_3d_m=error_3d, horizontal_error_m=horizontal_error, vertical_error_m=vertical_error, **diagnostic_arrays, error_growth_ratio=growth_ratio, rmse_north_east_down_3d_m=rmse, cdf_probability=cdf_probability, north_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 0])), east_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 1])), down_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 2])), three_d_cdf_error_m=np.sort(error_3d))
    with (OUTPUT_DIR / 'test_trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_gpst_s', 'n_input', 'n_used', 'n_gnss', 'n_leo', 'sat_ids_used', 'fusion_mode', 'x_ecef_m', 'y_ecef_m', 'z_ecef_m'])
        for row in online_rows:
            writer.writerow([row['time'], row['n_input'], row['n_total'], row['n_gnss'], row['n_leo'], ';'.join(row['sat_ids']), row['fusion_mode'], *row['position'].tolist()])
    with (OUTPUT_DIR / 'test_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_gpst_s', *diagnostic_fields])
        for row in online_rows:
            writer.writerow([row['time'], *[row[name] for name in diagnostic_fields]])
    classical_baseline_rmse = None
    if RUN_CLASSICAL_TEST_BASELINE:
        baseline_leo_simulator = LEODownlinkSimulator(tle_provider, test_leo_klobuchar, seed=TEST_LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
        baseline_nav = test_initial_nav.copy()
        baseline_P = test_P0.copy()
        baseline_rows = []
        baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr_gyro[0], test_imr_accel[0])
        for baseline_event in build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
            baseline_event_type, baseline_start, baseline_end, baseline_index = baseline_event
            if baseline_event_type == 'propagation':
                imu_index = baseline_index
                dt = baseline_end - baseline_start
                baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr_gyro[imu_index], test_imr_accel[imu_index])
                baseline_nav = mechanize_ecef(baseline_nav, baseline_last_gyro, baseline_last_accel, dt)
                F_baseline = build_error_state_dynamics(baseline_nav, baseline_last_accel)
                Phi_baseline, Qd_baseline = discretize_process_noise_van_loan(F_baseline, test_Qc, dt)
                baseline_P = Phi_baseline @ baseline_P @ Phi_baseline.T + Qd_baseline
                baseline_P = 0.5 * (baseline_P + baseline_P.T)
                continue
            fusion_index = baseline_index
            t = baseline_start
            epoch = test_gnss_epochs[fusion_index]
            baseline_gnss = test_gnss_preprocessor.prepare_epoch(epoch, baseline_nav, test_lever_arm_b_m)
            baseline_leo = baseline_leo_simulator.simulate_epoch(t, test_fusion_antenna_truth_position[fusion_index])
            baseline_measurements = retain_clock_observable_measurements(tuple(baseline_gnss) + tuple(baseline_leo))
            if baseline_measurements:
                baseline_innovation, baseline_H, baseline_R, _, _ = build_measurement_model(baseline_nav, baseline_measurements, test_lever_arm_b_m)
                baseline_dx, baseline_P, _ = kalman_measurement_update(baseline_P, baseline_innovation, baseline_H, baseline_R)
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
    test_measurement_counts = np.asarray([row['n_total'] for row in online_rows], dtype=float)
    test_gnss_counts = np.asarray([row['n_gnss'] for row in online_rows], dtype=float)
    test_leo_counts = np.asarray([row['n_leo'] for row in online_rows], dtype=float)
    print('\n=== DATA02 INDEPENDENT TEST METRICS ===')
    print(f'Test fusion epochs evaluated: {len(online_rows)}')
    print(f'Test RMSE - North: {rmse[0]:.3f} m')
    print(f'Test RMSE - East: {rmse[1]:.3f} m')
    print(f'Test RMSE - Down: {rmse[2]:.3f} m')
    print(f'Test RMSE - 3D position: {rmse[3]:.3f} m')
    print(f'Test 3D error - Mean: {float(np.mean(error_3d)):.3f} m')
    print(f'Test 3D error - Median: {float(np.median(error_3d)):.3f} m')
    print(f'Test 3D error - 95th percentile: {float(np.percentile(error_3d, 95.0)):.3f} m')
    print(f'Test 3D error - 99th percentile: {float(np.percentile(error_3d, 99.0)):.3f} m')
    print(f'Test 3D error - Maximum: {float(np.max(error_3d)):.3f} m')
    print(f'Test horizontal RMSE: {float(np.sqrt(np.mean(horizontal_error ** 2))):.3f} m')
    print(f'Test vertical RMSE: {float(np.sqrt(np.mean(vertical_error ** 2))):.3f} m')
    print(f'Measurements per test epoch - Minimum: {int(np.min(test_measurement_counts))}')
    print(f'Measurements per test epoch - Median: {float(np.median(test_measurement_counts)):.1f}')
    print(f'Measurements per test epoch - Maximum: {int(np.max(test_measurement_counts))}')
    print(f'GNSS measurements per test epoch - Median: {float(np.median(test_gnss_counts)):.1f}')
    print(f'LEO measurements per test epoch - Median: {float(np.median(test_leo_counts)):.1f}')
    print(f'Network measurement capacity Nmax: {network_nmax}')
    print(f'Test epochs requiring measurement-capacity selection: {data02_capacity_selection_epochs}')
    print(f'Total test measurements excluded by Nmax capacity: {data02_capacity_excluded_measurements}')
    if classical_baseline_rmse is not None:
        print(f'Classical baseline RMSE - North: {classical_baseline_rmse[0]:.3f} m')
        print(f'Classical baseline RMSE - East: {classical_baseline_rmse[1]:.3f} m')
        print(f'Classical baseline RMSE - Down: {classical_baseline_rmse[2]:.3f} m')
        print(f'Classical baseline RMSE - 3D position: {classical_baseline_rmse[3]:.3f} m')

    # Final presentation plot: repository-style training mini-batch loss and
    # validation loss versus optimizer step/epoch.  The plot is intentionally
    # produced at the end of the full run so it is the final displayed figure.
    final_loss_plot_path = OUTPUT_DIR / 'epoch_loss_train_validation_final.png'
    fig = plt.figure(figsize=(9, 5.5))
    plt.plot(history_epoch, history_train_mse, label='Train mini-batch MSE')
    plt.plot(history_epoch, history_validation_mse, label='Validation MSE')
    plt.xlabel('Epoch / Training step')
    plt.ylabel('MSE Loss')
    plt.title('Masked KalmanNet V2: Train and Validation Epoch-Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(final_loss_plot_path, dpi=180)
    print(f'Final train/validation epoch-loss plot saved to: {final_loss_plot_path}')
    plt.show()
    plt.close(fig)

