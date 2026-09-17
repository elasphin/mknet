from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
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
from torch.utils.checkpoint import checkpoint
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as SpatialRotation
from sgp4.api import SGP4_ERRORS, Satrec, WGS72
from sgp4.io import verify_checksum

EARTH_SEMI_MAJOR_AXIS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_SEMI_MINOR_AXIS_M = EARTH_SEMI_MAJOR_AXIS_M * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_ROTATION_RATE_RADPS = 7.292115e-05
EARTH_GRAVITATIONAL_PARAMETER_M3PS2 = 398600441800000.0
SPEED_OF_LIGHT_MPS = 299792458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0
README_XML_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/README.xml')
IMU_ERROR_MODEL_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/IMUErrorModel.txt')
ROVE_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/ROVE_GroundTruth.txt')
IMU_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/ISA-100C_GroundTruth.txt')
RINEX_OBS_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/ROVE.23O')
IMR_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/ISA-100C.imr')
SP3_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/WUM0MGXFIN_20230020000_01D_05M_ORB.SP3')
CLK_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/WUM0MGXFIN_20230020000_01D_30S_CLK.CLK')
NAV_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data01_20230102_ISA-100C_Vehicle_Complex/brdm0020.23p')
LEO_TLE_DIR = Path('/kaggle/input/datasets/elasphin/mknet-project/LEO_TLE')

TEST_README_XML_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/README.xml')
TEST_ROVE_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/ROVE_GroundTruth.txt')
TEST_IMU_GROUND_TRUTH_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/ISA-100C_GroundTruth.txt')
TEST_RINEX_OBS_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/ROVE.22O')
TEST_IMR_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/ISA-100C.imr')
TEST_SP3_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20220680000_01D_05M_ORB.SP3')
TEST_CLK_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/WUM0MGXFIN_20220680000_01D_30S_CLK.CLK')
TEST_NAV_PATH = Path('/kaggle/input/datasets/elasphin/mknet-project/Data02_20220309_ISA-100C_Vehicle_Complex/brdm0680.22p')

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
RUN_RECURSIVE_DATA01_DIAGNOSTIC = True
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
                yield PropagationSegment(segment_start, t, imu_index)

            yield FusionMarker(t, fusion_index)
            segment_start = t
            fusion_index += 1

        if interval_end > segment_start + 1e-12:
            yield PropagationSegment(segment_start, interval_end, imu_index)

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
                        measurement = SatelliteMeasurement(sat_id, constellation, suffix, float(frequency), float(pseudorange), cn0)
                        break
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


@dataclass
class GroundTruth:
    time_gpst_s: np.ndarray
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray
    heading_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray


def interpolate_ground_truth(truth: GroundTruth, query_time_gpst_s: np.ndarray, max_gap_s: float, *, position_only: bool=False):
    truth_time = truth.time_gpst_s
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

    def prepare_epoch(self, epoch: ObservationEpoch, nav: NavigationState, lever_arm_b_m: Array) -> tuple[PseudorangeMeasurement, ...]:
        antenna_position = gnss_antenna_position(nav, lever_arm_b_m)
        receiver_llh = ecef_to_llh(antenna_position)
        c_ecef_ned = c_ecef_to_ned(receiver_llh[0], receiver_llh[1])
        out = []
        for raw in epoch.measurements:
            try:
                initial_position = self.orbit.position(raw.sat_id, epoch.time_gpst_s)
            except (KeyError, ValueError):
                continue
            initial_range, _, _ = geometric_range(antenna_position, initial_position, 0.0)
            try:
                transmit_time, transit_s, state_tx_position = _iterate_transmit_time(
                    self.orbit.position,
                    raw.sat_id,
                    epoch.time_gpst_s,
                    antenna_position,
                    initial_position,
                    initial_range / SPEED_OF_LIGHT_MPS,
                    GNSS_TX_EPSILON_POSITION_M,
                    GNSS_TX_MAX_ITERATIONS,
                )
            except (KeyError, ValueError, TransmitTimeConvergenceError):
                continue

            if raw.sat_id in self.clock_satellites:
                clock_times, clock_bias = self.clock._series[raw.sat_id]
                satellite_clock_bias_s = float(np.interp(transmit_time, clock_times, clock_bias))
            else:
                orbit_clock = self.orbit.series[raw.sat_id]
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
                alpha, beta = self.iono[raw.constellation]
                week = int(math.floor(float(epoch.time_gpst_s) / GPS_WEEK_S))
                tow = float(epoch.time_gpst_s) - week * GPS_WEEK_S
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
            tropo_sigma = ref35_troposphere_sigma_m(elevation) if self.use_troposphere else 0.0
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
            out.append(PseudorangeMeasurement(raw.sat_id, raw.constellation, float(raw.pseudorange_m), satellite_position_rx, satellite_clock_bias_s, float(ionosphere), float(troposphere), float(sigma_code_m)))
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

def build_measurement_model(nav: NavigationState, measurements, lever_arm_b_m: Array) -> TCMeasurementModel:
    measurements = retain_clock_observable_measurements(measurements)
    n = len(measurements)
    if n == 0:
        return TCMeasurementModel(np.empty(0), np.zeros((0, INS_STATE_DIM)), np.zeros((0, 0)), (), np.zeros((0, 0)))
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
    return TCMeasurementModel(innovation, H, R, tuple(sat_ids), projector)

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

@dataclass
class TorchNavigationState:
    position_ecef_m: torch.Tensor
    velocity_ecef_mps: torch.Tensor
    body_to_ecef_dcm: torch.Tensor

def _torch_so3_exponential(rotation_vector_rad: torch.Tensor) -> torch.Tensor:
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

@dataclass(frozen=True)
class _TLEElement:
    epoch_gpst_s: float
    satrec: Satrec

@dataclass(frozen=True)
class LEOSatelliteState:
    position_ecef_m: Array
    velocity_ecef_mps: Array

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
            by_id.setdefault(f'NORAD-{norad}', []).append(_TLEElement(float(epoch.gps), satrec))
        self._elements = {sat_id: tuple(sorted(elements, key=lambda e: e.epoch_gpst_s)) for sat_id, elements in by_id.items()}
        self._element_times = {sat_id: tuple((element.epoch_gpst_s for element in elements)) for sat_id, elements in self._elements.items()}
        self.satellite_ids = tuple(sorted(self._elements, key=lambda s: int(s.split('-')[1])))

    def state_at(self, time_gpst_s: float, sat_id: str) -> LEOSatelliteState:
        elements = self._elements[sat_id]
        index = bisect_right(self._element_times[sat_id], time_gpst_s) - 1
        element = elements[index]
        utc = Time(float(time_gpst_s), format='gps').utc
        error, position_km, velocity_km_s = element.satrec.sgp4(float(utc.jd1), float(utc.jd2))
        position = CartesianRepresentation(np.asarray(position_km) * u.km)
        velocity = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        teme = TEME(position.with_differentials(velocity), obstime=utc)
        degraded = 'warn' if self.allow_degraded_eop else 'error'
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        return LEOSatelliteState(np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3), np.asarray(itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s)).reshape(3))

def ref35_ionosphere_sigma_m(elevation_rad: float, receiver_latitude_rad: float) -> float:
    latitude_deg = abs(float(np.rad2deg(receiver_latitude_rad)))
    sigma_vertical_m = 9.0 if latitude_deg <= 20.0 else 4.5 if latitude_deg <= 55.0 else 6.0
    R = 6378140.0
    h_i = 350000.0
    denominator = 1.0 - (R * math.cos(float(elevation_rad)) / (R + h_i)) ** 2
    return float(sigma_vertical_m / math.sqrt(max(denominator, 1e-15)))

def ref35_troposphere_sigma_m(elevation_rad: float) -> float:
    return float(1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elevation_rad)) ** 2))

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
                initial_range, initial_los, _ = geometric_range(receiver_position_ecef_m, initial_state.position_ecef_m, 0.0)
                initial_elevation, _ = elevation_azimuth_from_ned_matrix(c_ecef_ned, initial_los)
                self.prefilter_checked += 1
                if initial_elevation < self.minimum_elevation_rad - self.prefilter_guard_rad:
                    self.prefilter_rejected += 1
                    continue
                self.prefilter_passed += 1
                _, transit_s, state_tx_position = _iterate_transmit_time(
                    lambda sid, t: self.provider.state_at(t, sid).position_ecef_m,
                    sat_id,
                    receive_time_gpst_s,
                    receiver_position_ecef_m,
                    initial_state.position_ecef_m,
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
                tropo_sigma_m = ref35_troposphere_sigma_m(elevation) if self.use_troposphere else 0.0
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

@dataclass(frozen=True)
class MaskedCLAOutput:
    kalman_gain: torch.Tensor
    attention: torch.Tensor
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None

class MaskedConv1d(nn.Module):

    def __init__(self, out_channels: int=24, kernel_size: int=3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.register_buffer('_mask_kernel', torch.ones(1, 1, self.kernel_size), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(1)
        x_masked = x.unsqueeze(1) * m
        z = F.conv1d(x_masked, self.weight, bias=None, stride=1, padding=self.padding)
        local_count = F.conv1d(m, self._mask_kernel.to(dtype=x.dtype), stride=1, padding=self.padding)
        valid_window = (local_count > 0).to(dtype=x.dtype)
        feature_maps = F.relu(z / local_count.clamp_min(MASK_EPS) + self.bias.view(1, -1, 1) * valid_window)
        feature_maps = feature_maps * m
        return feature_maps.transpose(1, 2)

class MaskedStackedLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int=64, num_layers: int=5, dropout: float=0.2) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=float(dropout) if self.num_layers > 1 else 0.0, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, recurrent_state: tuple[torch.Tensor, torch.Tensor] | None=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, position_count, _ = x.shape
        state_shape = (self.num_layers, batch_size, position_count, self.hidden_size)
        if recurrent_state is None:
            previous_hidden = x.new_zeros(state_shape)
            previous_cell = x.new_zeros(state_shape)
        else:
            previous_hidden, previous_cell = recurrent_state
            for name, state in (('hidden', previous_hidden), ('cell', previous_cell)):
                pass
        step_input = x.reshape(batch_size * position_count, 1, self.input_size)
        h0 = previous_hidden.reshape(self.num_layers, batch_size * position_count, self.hidden_size).contiguous()
        c0 = previous_cell.reshape(self.num_layers, batch_size * position_count, self.hidden_size).contiguous()
        _, (candidate_hidden_flat, candidate_cell_flat) = self.lstm(step_input, (h0, c0))
        candidate_hidden = candidate_hidden_flat.reshape(self.num_layers, batch_size, position_count, self.hidden_size)
        candidate_cell = candidate_cell_flat.reshape(self.num_layers, batch_size, position_count, self.hidden_size)
        valid = mask.bool().unsqueeze(0).unsqueeze(-1)
        next_hidden = torch.where(valid, candidate_hidden, previous_hidden)
        next_cell = torch.where(valid, candidate_cell, previous_cell)
        output = next_hidden[-1]
        return (output, (next_hidden, next_cell))

class MaskedAttention(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        alpha = torch.softmax(score.masked_fill(~mask.bool(), -torch.inf), dim=1)
        return (torch.sum(alpha.unsqueeze(-1) * h, dim=1), alpha)

class MaskedCLA(nn.Module):

    def __init__(self, nmax: int, dropout: float=0.2, gain_init_scale: float=0.001) -> None:
        super().__init__()
        self.nmax = int(nmax)
        self.dropout = float(dropout)
        self.gain_init_scale = float(gain_init_scale)
        self.eq15_padded_dim = FIXED_FEATURE_DIM + 2 * self.nmax
        self.conv = MaskedConv1d(out_channels=24, kernel_size=3)
        self.lstm = MaskedStackedLSTM(input_size=24, hidden_size=64, num_layers=5, dropout=self.dropout)
        self.attention = MaskedAttention(64)
        self.gain_head = nn.Linear(64, INS_STATE_DIM * self.nmax)
        nn.init.kaiming_uniform_(self.gain_head.weight, nonlinearity='linear')
        with torch.no_grad():
            self.gain_head.weight.mul_(self.gain_init_scale)
        if self.gain_head.bias is not None:
            nn.init.zeros_(self.gain_head.bias)

    def forward(self, fixed: torch.Tensor, observations: torch.Tensor, mask: torch.Tensor, channel_mask: torch.Tensor, recurrent_state: tuple[torch.Tensor, torch.Tensor] | None=None) -> MaskedCLAOutput:
        batch_size = fixed.shape[0]
        satellite_valid = mask.bool()
        channel_bool = channel_mask.bool()
        satellite_count = satellite_valid.sum(dim=1)
        observations = observations * channel_bool.to(dtype=observations.dtype)
        fixed_valid = torch.ones((batch_size, FIXED_FEATURE_DIM), dtype=torch.bool, device=fixed.device)
        packed_values = []
        packed_masks = []
        satellite_count_cpu = satellite_count.detach().cpu().tolist()
        for batch_index, count in enumerate(satellite_count_cpu):
            suffix_length = 2 * (self.nmax - count)
            observation_residual = observations[batch_index, :count, 0]
            observation_innovation = observations[batch_index, :count, 1]
            zero_padding = observations.new_zeros(suffix_length)
            variable_values = torch.cat((observation_residual, observation_innovation, zero_padding), dim=0)
            variable_mask = torch.cat((channel_bool[batch_index, :count, 0], channel_bool[batch_index, :count, 1], torch.zeros(suffix_length, dtype=torch.bool, device=channel_mask.device)), dim=0)
            packed_values.append(torch.cat((fixed[batch_index], variable_values), dim=0))
            packed_masks.append(torch.cat((fixed_valid[batch_index], variable_mask), dim=0))
        x_bar = torch.stack(packed_values, dim=0)
        feature_mask = torch.stack(packed_masks, dim=0)
        conv_features = self.conv(x_bar, feature_mask)
        lstm_positions, next_lstm_state = self.lstm(conv_features, feature_mask, recurrent_state)
        context, attention = self.attention(lstm_positions, feature_mask)
        gain = self.gain_head(context).view(batch_size, INS_STATE_DIM, self.nmax)
        gain = gain * satellite_valid.to(dtype=gain.dtype).unsqueeze(1)
        return MaskedCLAOutput(kalman_gain=gain, attention=attention, recurrent_state=next_lstm_state)

def fig8_state_update(network_output: MaskedCLAOutput, innovation: torch.Tensor) -> torch.Tensor:
    return torch.bmm(network_output.kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)
if __name__ == '__main__':
    _run_wall_start = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_FUSION_EPOCHS = 1601
    MAX_TEST_FUSION_EPOCHS = 300
    TRAINING_EPOCHS = 12
    KNET_V2_ALTERNATING_EPOCHS = TRAINING_EPOCHS
    KNET_V1_FINE_TUNE_EPOCHS = 3
    YAN_REPORTED_LEARNING_RATE = 0.01
    LEARNING_RATE = 0.001
    FILTER_BLOCK_LEARNING_RATE = LEARNING_RATE
    REPRESENTATION_BLOCK_LEARNING_RATE = LEARNING_RATE
    RECURRENT_TBPTT_DETACH_STEP = 2
    OPTIMIZER_WINDOW_SIZE = 4
    GRADIENT_CLIP_NORM = 1.0
    KNET_V2_TRAJECTORY_LENGTH = 50
    KNET_V2_BATCH_SIZE = 8
    GAIN_HEAD_INITIAL_SCALE = 0.001
    VALIDATION_FRACTION = 0.2
    GAMMA_L2 = 1e-06
    SEED = 0
    TEST_LEO_SEED = LEO_SEED + 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n=== MASKED KALMANNET PERFORMANCE EVALUATION ===')
    if MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None:
        print('WARNING: fusion-epoch cap active; reported metrics are diagnostic.')
    root = ET.fromstring(README_XML_PATH.read_text(encoding='utf-8', errors='replace'))
    rover_xml = next(rover_xml for rover_xml in root.findall('ROVE') if (rover_xml.findtext('ID') or '').strip() == '01')
    rover_imu_type = (rover_xml.findtext('SINS_IMUType') or '').strip()
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
    antenna_truth = GroundTruth(antenna_truth_week.astype(float) * GPS_WEEK_S + antenna_truth_array[:, 1], antenna_truth_array[:, 2:5], antenna_truth_array[:, 5:8], antenna_truth_array[:, 8], antenna_truth_array[:, 9], antenna_truth_array[:, 10])
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
    imu_truth = GroundTruth(imu_truth_week.astype(float) * GPS_WEEK_S + imu_truth_array[:, 1], imu_truth_array[:, 2:5], imu_truth_array[:, 5:8], imu_truth_array[:, 8], imu_truth_array[:, 9], imu_truth_array[:, 10])
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
    imu_noise = imu_models[rover_imu_type]
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
    anchor_time = float(first_rinex_epoch.time_gpst_s)
    imr_time_all = anchor_imr_tow_to_gpst_seconds(imr_tow_all, anchor_time)
    antenna_truth_time = np.asarray(antenna_truth.time_gpst_s, dtype=float).reshape(-1)
    imu_truth_time = np.asarray(imu_truth.time_gpst_s, dtype=float).reshape(-1)
    common_start = max(float(imr_time_all[0]), float(antenna_truth_time[0]), float(imu_truth_time[0]))
    common_end = min(float(imr_time_all[-1]), float(antenna_truth_time[-1]), float(imu_truth_time[-1]))
    start = int(np.searchsorted(imr_time_all, common_start, side='left'))
    usable_imu_start = float(imr_time_all[start])
    gnss_epochs = tuple(rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=usable_imu_start, end_time_gpst_s=common_end, max_epochs=MAX_FUSION_EPOCHS, require_measurements=True))
    fusion_time = np.asarray([epoch.time_gpst_s for epoch in gnss_epochs], dtype=float)
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
    initial_imu_truth_position = imu_position[0]
    initial_imu_truth_velocity = imu_velocity[0]
    fusion_antenna_truth_position = antenna_position[1:]
    fusion_imu_truth_position = imu_position[1:]
    fusion_imu_truth_velocity = imu_velocity[1:]
    C_b_e = body_to_ecef_from_ie_hpr(initial_imu_truth_position, imu_heading[0], imu_pitch[0], imu_roll[0], rover_mounting_xyz_deg)
    lever_arm_b_m = c_vehicle_to_body_zxy(*rover_mounting_xyz_deg) @ np.asarray(rover_lever_arm_vehicle_m, dtype=float).reshape(3)
    initial_antenna_from_imu = initial_imu_truth_position + C_b_e @ lever_arm_b_m
    truth_reference_error_m = np.linalg.norm(antenna_position[0] - initial_antenna_from_imu)
    fusion_truth_body_to_ecef = np.stack([body_to_ecef_from_ie_hpr(p, h, pt, r, rover_mounting_xyz_deg) for p, h, pt, r in zip(imu_position[1:], imu_heading[1:], imu_pitch[1:], imu_roll[1:])])
    initial_nav = NavigationState(initial_imu_truth_position.copy(), initial_imu_truth_velocity.copy(), C_b_e)
    initial_sigma = np.concatenate([imu_noise['isdv_pos_m'], imu_noise['isdv_vel_mps'], imu_noise['isdv_att_rad']])
    P0 = np.diag(initial_sigma ** 2)
    process_density = np.concatenate([imu_noise['pnsd_pos_m_sqrt_s'], imu_noise['pnsd_vel_mps_sqrt_s'], imu_noise['pnsd_att_rad_sqrt_s']])
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
                    if len(fields) >= 5 and fields[0] in {'GPSA', 'GPSB', 'BDSA', 'BDSB'}:
                        iono_header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
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
    last_gyro, last_accel = compensate_imu(imr_gyro[0], imr_accel[0])
    training_timeline = build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True)
    for event in training_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(imr_gyro[imu_index], imr_accel[imu_index])
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
        fixed_measurements = tuple(measurements)
        fixed_sat_ids = tuple((m.sat_id for m in fixed_measurements))
        history_rows.append({'time': t, 'sat_ids': measurement_model.sat_ids, 'innovation': measurement_model.innovation.copy(), 'residual': posterior_residual.copy(), 'x_pred': error_state_pred.copy(), 'x_post': error_state_post.copy(), 'accel': last_accel.copy(), 'gyro': last_gyro.copy(), 'target_state_9': target_state_9.copy(), 'posterior_nav': nav.copy(), 'fusion_index': int(fusion_index), 'fixed_measurements': fixed_measurements, 'fixed_y_m': np.asarray([m.pseudorange_m for m in fixed_measurements], dtype=float), 'leo_measurements': tuple(leo_measurements), 'preceding_interval_segments': tuple(interval_segments_since_previous_usable_fusion)})
        interval_segments_since_previous_usable_fusion = []
    nmax = max((len(row['sat_ids']) for row in history_rows[1:]))
    network_nmax = int(nmax)
    data02_sequence_nmax = None
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
        fixed_k = np.concatenate([delta_accel, delta_gyro, previous_state_residual, previous_state_innovation])
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
    fixed_network = fixed
    observations_network = observations
    data01_total_sample_count = int(len(feature_time))
    validation_sample_count = max(1, int(round(data01_total_sample_count * VALIDATION_FRACTION)))
    training_sample_count = data01_total_sample_count - validation_sample_count
    validation_start_sample = training_sample_count
    training_trajectories = []
    v2_start = 0
    while v2_start + KNET_V2_TRAJECTORY_LENGTH <= training_sample_count:
        training_trajectories.append((v2_start, v2_start + KNET_V2_TRAJECTORY_LENGTH))
        v2_start += KNET_V2_TRAJECTORY_LENGTH + 1
    trajectory_count = len(training_trajectories)
    TRAJECTORY_BATCH_SIZE = min(KNET_V2_BATCH_SIZE, trajectory_count)
    v2_trained_sample_count = int(sum((stop - start for start, stop in training_trajectories)))
    v2_initialization_sample_count = int(trajectory_count)
    v2_unused_sample_count = int(training_sample_count - v2_trained_sample_count)
    print(f'training setup: Data01_total={data01_total_sample_count}, train={training_sample_count}, validation={validation_sample_count}, split=chronological_{1.0 - VALIDATION_FRACTION:.2f}/{VALIDATION_FRACTION:.2f}, Nmax={network_nmax} (source=Data01_only; Data02_not_used_for_architecture), mode=short_trajectory_alternating, Tshort={KNET_V2_TRAJECTORY_LENGTH}, trajectories={trajectory_count}, trajectory_batch={TRAJECTORY_BATCH_SIZE}, batching=LatentKalmanNet_Algorithm2_random_partition_all_Q_batches_same_partition_theta_then_psi, init=ground_truth_navigation_state, V2_alternating_epochs={KNET_V2_ALTERNATING_EPOCHS}, V1_finetune_epochs={KNET_V1_FINE_TUNE_EPOCHS}, lr_theta={FILTER_BLOCK_LEARNING_RATE:g}, lr_psi={REPRESENTATION_BLOCK_LEARNING_RATE:g}, grad_clip_norm={GRADIENT_CLIP_NORM:g}, optimizer_window={OPTIMIZER_WINDOW_SIZE}, recurrent_detach_step={RECURRENT_TBPTT_DETACH_STEP}, navigation_detach_step=1, gain_head_init=scaled_kaiming_{GAIN_HEAD_INITIAL_SCALE:g}, checkpoint=best_recursive_validation_Eq30')
    model = MaskedCLA(nmax=network_nmax, dropout=0.2, gain_init_scale=GAIN_HEAD_INITIAL_SCALE).to(DEVICE)
    LSTM_CONFIGURED_DROPOUT = float(model.lstm.lstm.dropout)
    representation_modules = (model.conv,)
    filter_modules = (model.lstm, model.attention, model.gain_head)
    representation_parameters = [parameter for module in representation_modules for parameter in module.parameters()]
    filter_parameters = [parameter for module in filter_modules for parameter in module.parameters()]
    filter_optimizer = torch.optim.Adam(filter_parameters, lr=FILTER_BLOCK_LEARNING_RATE)
    representation_optimizer = torch.optim.Adam(representation_parameters, lr=REPRESENTATION_BLOCK_LEARNING_RATE)
    trainable_eq32_parameters = representation_parameters + filter_parameters
    training_history = []
    _chunk_starts = [int(start) for start, _ in training_trajectories]
    _periodic_restart_starts = _chunk_starts[1:]
    _interchunk_context_only_samples = int(sum((max(0, int(training_trajectories[i][0] - training_trajectories[i - 1][1])) for i in range(1, trajectory_count))))
    _tail_unused_samples = int(max(0, training_sample_count - training_trajectories[-1][1]))
    _restart_pos_proxy_m = []
    _restart_vel_proxy_mps = []
    _restart_att_proxy_deg = []
    for _sample_index in _periodic_restart_starts:
        _row = history_rows[_sample_index]
        _fusion_index = int(_row['fusion_index'])
        _causal_nav = _row['posterior_nav']
        _restart_pos_proxy_m.append(float(np.linalg.norm(fusion_imu_truth_position[_fusion_index] - _causal_nav.position_ecef_m)))
        _restart_vel_proxy_mps.append(float(np.linalg.norm(fusion_imu_truth_velocity[_fusion_index] - _causal_nav.velocity_ecef_mps)))
        _restart_att_proxy_deg.append(float(np.degrees(np.linalg.norm(attitude_error_state_target(_causal_nav.body_to_ecef_dcm, fusion_truth_body_to_ecef[_fusion_index])))))

    short_trajectory_restart_diagnostic = {'paper_status': 'project_training_adaptation_not_LatentKalmanNet_Algorithm2_and_not_published_by_Yan', 'trajectory_length_samples': int(KNET_V2_TRAJECTORY_LENGTH), 'trajectory_count': int(trajectory_count), 'periodic_truth_restarts_after_initialization': int(max(0, trajectory_count - 1)), 'periodic_restart_fraction_of_training_samples': float(max(0, trajectory_count - 1) / max(training_sample_count, 1)), 'trained_samples_per_phase': int(v2_trained_sample_count), 'interchunk_boundary_context_samples_excluded_from_loss': int(_interchunk_context_only_samples), 'tail_samples_not_used_by_short_stage': int(_tail_unused_samples), 'total_samples_excluded_from_short_stage_loss': int(v2_unused_sample_count), 'deployment_has_periodic_truth_restart': False, 'causal_classical_posterior_proxy_position_error_m': {'median': float(np.median(np.asarray(_restart_pos_proxy_m, dtype=float))) if _restart_pos_proxy_m else 0.0, 'p95': float(np.percentile(np.asarray(_restart_pos_proxy_m, dtype=float), 95.0)) if _restart_pos_proxy_m else 0.0, 'max': float(np.max(np.asarray(_restart_pos_proxy_m, dtype=float))) if _restart_pos_proxy_m else 0.0}, 'causal_classical_posterior_proxy_velocity_error_mps': {'median': float(np.median(np.asarray(_restart_vel_proxy_mps, dtype=float))) if _restart_vel_proxy_mps else 0.0, 'p95': float(np.percentile(np.asarray(_restart_vel_proxy_mps, dtype=float), 95.0)) if _restart_vel_proxy_mps else 0.0, 'max': float(np.max(np.asarray(_restart_vel_proxy_mps, dtype=float))) if _restart_vel_proxy_mps else 0.0}, 'causal_classical_posterior_proxy_attitude_error_deg': {'median': float(np.median(np.asarray(_restart_att_proxy_deg, dtype=float))) if _restart_att_proxy_deg else 0.0, 'p95': float(np.percentile(np.asarray(_restart_att_proxy_deg, dtype=float), 95.0)) if _restart_att_proxy_deg else 0.0, 'max': float(np.max(np.asarray(_restart_att_proxy_deg, dtype=float))) if _restart_att_proxy_deg else 0.0}, 'interpretation': 'proxy_only_not_a_learned_rollout_effect;large_values_indicate_truth_restarts_remove_nontrivial_causal_state_error_at_chunk_boundaries'}
    print(f"short-trajectory truth-restart audit: periodic_restarts={short_trajectory_restart_diagnostic['periodic_truth_restarts_after_initialization']}, context_only={_interchunk_context_only_samples}, unused_tail={_tail_unused_samples}, causal_proxy_pos_median={short_trajectory_restart_diagnostic['causal_classical_posterior_proxy_position_error_m']['median']:.3f}m, causal_proxy_pos_max={short_trajectory_restart_diagnostic['causal_classical_posterior_proxy_position_error_m']['max']:.3f}m")
    training_history.append({'stage': 'short_trajectory_ground_truth_restart_audit', **short_trajectory_restart_diagnostic})
    target_base_t = torch.tensor(target_states_9, dtype=torch.float64, device=DEVICE)

    def _eq32_regularization() -> torch.Tensor:
        if not GAMMA_L2:
            return torch.zeros((), dtype=torch.float64, device=DEVICE)
        l2_all = sum((torch.sum(parameter.double() * parameter.double()) for parameter in trainable_eq32_parameters))
        return GAMMA_L2 * l2_all

    def _torch_training_context_from_stored(history_row_index: int):
        warm = history_rows[history_row_index]
        x_pred = np.asarray(warm['x_pred'], dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(warm['x_post'], dtype=float).reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = np.zeros(INS_STATE_DIM) if history_row_index == 0 else x_post - np.asarray(history_rows[history_row_index - 1]['x_post'], dtype=float).reshape(INS_STATE_DIM)
        residual = np.asarray(warm['residual'], dtype=float).copy()
        accel = np.asarray(warm['accel'], dtype=float).reshape(3).copy()
        gyro = np.asarray(warm['gyro'], dtype=float).reshape(3).copy()
        return {'sat_ids': tuple(warm['sat_ids']), 'residual': torch.as_tensor(residual, dtype=torch.float64, device=DEVICE), 'x_pred': torch.as_tensor(x_pred.copy(), dtype=torch.float64, device=DEVICE), 'x_post': torch.as_tensor(x_post.copy(), dtype=torch.float64, device=DEVICE), 'state_innovation': torch.as_tensor(state_innovation.copy(), dtype=torch.float64, device=DEVICE), 'state_residual': torch.as_tensor(state_residual.copy(), dtype=torch.float64, device=DEVICE), 'accel': torch.as_tensor(accel, dtype=torch.float64, device=DEVICE), 'gyro': torch.as_tensor(gyro, dtype=torch.float64, device=DEVICE)}
    training_imr_gyro_t = torch.as_tensor(imr_gyro, dtype=torch.float64, device=DEVICE)
    training_imr_accel_t = torch.as_tensor(imr_accel, dtype=torch.float64, device=DEVICE)
    lever_arm_train_t = torch.as_tensor(lever_arm_b_m, dtype=torch.float64, device=DEVICE).reshape(3)
    fusion_truth_position_t = torch.as_tensor(fusion_imu_truth_position, dtype=torch.float64, device=DEVICE)
    fusion_truth_velocity_t = torch.as_tensor(fusion_imu_truth_velocity, dtype=torch.float64, device=DEVICE)
    fusion_truth_dcm_t = torch.as_tensor(fusion_truth_body_to_ecef, dtype=torch.float64, device=DEVICE)
    zero_error_state_t = torch.zeros(INS_STATE_DIM, dtype=torch.float64, device=DEVICE)


    def _detach_torch_navigation_state(nav_state: TorchNavigationState) -> TorchNavigationState:
        return TorchNavigationState(nav_state.position_ecef_m.detach(), nav_state.velocity_ecef_mps.detach(), nav_state.body_to_ecef_dcm.detach())

    def _detach_training_context(context):
        detached = {}
        for key, value in context.items():
            detached[key] = value.detach() if isinstance(value, torch.Tensor) else value
        return detached

    def _detach_recurrent_state(recurrent_state):
        if recurrent_state is None:
            return None
        return tuple((state.detach() for state in recurrent_state))

    def _prepare_torch_measurements(measurements):
        measurements = tuple(retain_clock_observable_measurements(measurements))
        n = len(measurements)
        if n == 0:
            return ((), None, None, None, None, None, None)
        variances = np.asarray([float(m.sigma_code_m) ** 2 for m in measurements], dtype=float)
        projector = _clock_projector(measurements, variances)
        return (tuple((m.sat_id for m in measurements)), torch.as_tensor(np.stack([m.satellite_position_reception_ecef_m for m in measurements], axis=0), dtype=torch.float64, device=DEVICE), torch.as_tensor([m.pseudorange_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.satellite_clock_bias_s for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.ionosphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor([m.troposphere_delay_m for m in measurements], dtype=torch.float64, device=DEVICE), torch.as_tensor(projector, dtype=torch.float64, device=DEVICE))

    def _torch_measurement_innovation(nav_state: TorchNavigationState, prepared_measurements) -> tuple[torch.Tensor, tuple[str, ...]]:
        sat_ids, satellite_position, observed, satellite_clock, ionosphere, troposphere, projector_t = prepared_measurements
        if not sat_ids:
            return (nav_state.position_ecef_m.new_empty(0), ())
        lever_e = nav_state.body_to_ecef_dcm @ lever_arm_train_t
        antenna_position = nav_state.position_ecef_m + lever_e
        geometric = torch.linalg.vector_norm(antenna_position.unsqueeze(0) - satellite_position, dim=1)
        predicted_zero_clock = geometric - SPEED_OF_LIGHT_MPS * satellite_clock + ionosphere + troposphere
        return (projector_t @ (observed - predicted_zero_clock), sat_ids)

    def _torch_pad_training_epoch(fixed_nn: torch.Tensor, obs_nn: torch.Tensor, current_mask: torch.Tensor, channel_mask: torch.Tensor, innovation_now: torch.Tensor, slot_capacity: int):
        n = int(current_mask.numel())
        pad = int(slot_capacity - n)
        if pad == 0:
            return (fixed_nn, obs_nn, current_mask, channel_mask, innovation_now)
        obs_pad = torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=obs_nn.dtype, device=obs_nn.device)
        mask_pad = torch.zeros(pad, dtype=torch.bool, device=DEVICE)
        channel_pad = torch.zeros((pad, OBSERVATION_FEATURE_DIM), dtype=torch.bool, device=DEVICE)
        innovation_pad = torch.zeros(pad, dtype=innovation_now.dtype, device=innovation_now.device)
        return (fixed_nn, torch.cat((obs_nn, obs_pad), dim=0), torch.cat((current_mask, mask_pad), dim=0), torch.cat((channel_mask, channel_pad), dim=0), torch.cat((innovation_now, innovation_pad), dim=0))

    def _rollout_actual_closed_loop_trajectory(start_sample: int, stop_sample: int, *, phase: str | None, backward_scale: float | None=None, trajectory_id: int | None=None, optimizer_step_index: int | None=None, initialization_mode: str='stored_causal', recurrent_detach_step: int=0, initial_rollout_state=None, allow_empty_training_window: bool=False):
        training = phase is not None
        if initial_rollout_state is not None:
            nav_train = initial_rollout_state['nav_train']
            previous_context = initial_rollout_state['previous_context']
            last_feature_accel = initial_rollout_state['last_feature_accel']
            last_feature_gyro = initial_rollout_state['last_feature_gyro']
            recurrent_state = initial_rollout_state['recurrent_state']
        elif initialization_mode == 'kalmannet_v2_truth':
            boundary_row = history_rows[start_sample]
            boundary_fusion_index = int(boundary_row['fusion_index'])
            nav_train = TorchNavigationState(fusion_truth_position_t[boundary_fusion_index].clone(), fusion_truth_velocity_t[boundary_fusion_index].clone(), fusion_truth_dcm_t[boundary_fusion_index].clone())
            stored_boundary_context = _torch_training_context_from_stored(start_sample)
            boundary_measurements = boundary_row['fixed_measurements']
            boundary_prepared = _prepare_torch_measurements(boundary_measurements)
            boundary_residual, boundary_sat_ids = _torch_measurement_innovation(nav_train, boundary_prepared)
            previous_context = {'sat_ids': tuple(boundary_sat_ids), 'residual': boundary_residual, 'x_pred': zero_error_state_t.clone(), 'x_post': zero_error_state_t.clone(), 'state_innovation': zero_error_state_t.clone(), 'state_residual': zero_error_state_t.clone(), 'accel': stored_boundary_context['accel'], 'gyro': stored_boundary_context['gyro']}
            last_feature_accel = previous_context['accel']
            last_feature_gyro = previous_context['gyro']
            recurrent_state = None
        else:
            stored_nav = history_rows[start_sample]['posterior_nav']
            nav_train = TorchNavigationState(
                torch.as_tensor(stored_nav.position_ecef_m, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(stored_nav.velocity_ecef_mps, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(stored_nav.body_to_ecef_dcm, dtype=torch.float64, device=DEVICE).clone(),
            )
            previous_context = _torch_training_context_from_stored(start_sample)
            last_feature_accel = previous_context['accel']
            last_feature_gyro = previous_context['gyro']
            recurrent_state = None
        recurrent_capacity = int(network_nmax)
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
                    segment_tuple = tuple((int(i), float(dt)) for i, dt in row['preceding_interval_segments'])

                    def interval_core(position, velocity, dcm):
                        state = TorchNavigationState(position, velocity, dcm)
                        feature_gyro = training_imr_gyro_t[segment_tuple[0][0]]
                        feature_accel = training_imr_accel_t[segment_tuple[0][0]]
                        for imu_index, dt in segment_tuple:
                            propagation_gyro = training_imr_gyro_t[imu_index].reshape(3)
                            propagation_accel = training_imr_accel_t[imu_index].reshape(3)
                            feature_gyro, feature_accel = propagation_gyro, propagation_accel
                            dt = float(dt)
                            C0 = state.body_to_ecef_dcm
                            earth_rotvec = C0.new_tensor(-OMEGA_IE_E * dt)
                            C1 = _torch_so3_exponential(earth_rotvec) @ C0 @ _torch_so3_exponential(propagation_gyro * dt)
                            Cmid = 0.5 * (C0 + C1)
                            omega = C0.new_tensor(OMEGA_IE_E)
                            r_e = state.position_ecef_m.reshape(3)
                            x, y, z = r_e.unbind()
                            radius = torch.linalg.vector_norm(r_e).clamp_min(1.0)
                            z2_r2 = z * z / (radius * radius)
                            j2 = 1.5 * J2_UNITLESS * (EARTH_SEMI_MAJOR_AXIS_M / radius) ** 2
                            xy_factor = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
                            z_factor = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
                            scale = -EARTH_GRAVITATIONAL_PARAMETER_M3PS2 / radius ** 3
                            gravity = scale * torch.stack((x * xy_factor, y * xy_factor, z * z_factor))
                            effective_gravity = gravity - torch.linalg.cross(omega, torch.linalg.cross(omega, r_e))
                            acceleration = Cmid @ propagation_accel + effective_gravity - 2.0 * torch.linalg.cross(omega, state.velocity_ecef_mps)
                            velocity = state.velocity_ecef_mps + acceleration * dt
                            position = state.position_ecef_m + 0.5 * (state.velocity_ecef_mps + velocity) * dt
                            state = TorchNavigationState(position, velocity, C1)
                        return state.position_ecef_m, state.velocity_ecef_mps, state.body_to_ecef_dcm, feature_gyro, feature_accel

                    inputs = (nav_train.position_ecef_m, nav_train.velocity_ecef_mps, nav_train.body_to_ecef_dcm)
                    if training and any(value.requires_grad for value in inputs):
                        position, velocity, dcm, last_feature_gyro, last_feature_accel = checkpoint(interval_core, *inputs, use_reentrant=False)
                    else:
                        position, velocity, dcm, last_feature_gyro, last_feature_accel = interval_core(*inputs)
                    nav_train = TorchNavigationState(position, velocity, dcm)
                fusion_index = int(row['fusion_index'])
                prior_position_error = fusion_truth_position_t[fusion_index] - nav_train.position_ecef_m
                max_prior_position_error_t = torch.maximum(max_prior_position_error_t, torch.linalg.vector_norm(prior_position_error).detach())
                measurements = row['fixed_measurements']
                if not measurements:
                    skipped_no_measurements += 1
                    continue
                fixed_y_check = np.asarray([m.pseudorange_m for m in measurements], dtype=float)
                prepared_measurements = _prepare_torch_measurements(measurements)
                innovation_now, current_sat_ids = _torch_measurement_innovation(nav_train, prepared_measurements)
                if innovation_now.numel():
                    max_innovation_abs_t = torch.maximum(max_innovation_abs_t, torch.max(torch.abs(innovation_now)).detach())
                n_current = len(current_sat_ids)
                innovation_k = innovation_now.reshape(n_current)
                delta_accel = last_feature_accel.reshape(3) - previous_context['accel']
                delta_gyro = last_feature_gyro.reshape(3) - previous_context['gyro']
                fixed_k = torch.cat((delta_accel, delta_gyro, previous_context['state_residual'], previous_context['state_innovation']))
                previous_index = {sat_id: index for index, sat_id in enumerate(previous_context['sat_ids'])}
                residual_values = []
                residual_valid = []
                zero = innovation_k.new_zeros(())
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
                current_mask = torch.ones(n_current, dtype=torch.bool, device=DEVICE)
                obs_raw = torch.stack((residual_k, innovation_k), dim=1)
                channel_k = torch.stack((residual_mask, current_mask), dim=1)
                obs_k = torch.where(channel_k, obs_raw, torch.zeros_like(obs_raw))
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _torch_pad_training_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, recurrent_capacity)
                output = model(fixed_nn.to(dtype=torch.float32).unsqueeze(0), obs_nn.to(dtype=torch.float32).unsqueeze(0), mask_nn.unsqueeze(0), channel_nn.unsqueeze(0), recurrent_state=recurrent_state)
                recurrent_state = output.recurrent_state
                correction = fig8_state_update(output, innovation_nn.to(dtype=torch.float32).unsqueeze(0))[0]
                correction64 = correction.to(dtype=torch.float64)
                valid_gain = output.kalman_gain[0, :, :len(current_sat_ids)]
                gain_fro = torch.linalg.matrix_norm(valid_gain).detach()
                max_gain_fro_t = gain_fro if max_gain_fro_t is None else torch.maximum(max_gain_fro_t, gain_fro)
                relative_attitude = fusion_truth_dcm_t[fusion_index].reshape(3, 3) @ nav_train.body_to_ecef_dcm.reshape(3, 3).T
                Rm = relative_attitude.reshape(3, 3)
                vee = torch.stack((Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]))
                sin_theta = 0.5 * torch.linalg.vector_norm(vee)
                cos_theta = 0.5 * (torch.trace(Rm) - 1.0)
                theta = torch.atan2(sin_theta, cos_theta)
                denom = (2.0 * sin_theta).clamp_min(1e-15)
                exact_factor = theta / denom
                small_factor = 0.5 + theta * theta / 12.0
                factor = torch.where(sin_theta < 1e-07, small_factor, exact_factor)
                attitude_target = factor * vee / ATTITUDE_FEEDBACK_SIGN
                target_state_9_now = torch.cat((fusion_truth_position_t[fusion_index] - nav_train.position_ecef_m, fusion_truth_velocity_t[fusion_index] - nav_train.velocity_ecef_mps, attitude_target))
                state_error_9 = target_state_9_now - correction64[:SUPERVISED_STATE_DIM]
                state_loss = torch.sum(state_error_9 ** 2)
                position_loss = torch.sum(state_error_9[:3] ** 2)
                if not bool(torch.isfinite(state_loss).detach().cpu()):
                    max_innovation_abs_m = float(max_innovation_abs_t.cpu())
                    max_gain_fro_norm = 0.0 if max_gain_fro_t is None else float(max_gain_fro_t.cpu())
                    raise FloatingPointError(f'non-finite Yan Eq.(30) loss inside KalmanNet-style BPTT trajectory={trajectory_id}, bounds=[{start_sample},{stop_sample}), history_row={row_index}, fusion_index={fusion_index}, optimizer_step={optimizer_step_index}, max_innovation={max_innovation_abs_m:.6g}, max_gain={max_gain_fro_norm:.6g}, max_position_correction={float(max_position_correction_t.cpu()):.6g}, max_velocity_correction={float(max_velocity_correction_t.cpu()):.6g}, max_attitude_correction={float(max_attitude_correction_t.cpu()):.6g}')
                state_losses.append(state_loss)
                total_state_sum_t += state_loss.detach()
                total_position_sum_t += position_loss.detach()
                total_count += 1
                max_position_correction_t = torch.maximum(max_position_correction_t, torch.linalg.vector_norm(correction64[0:3]).detach())
                max_velocity_correction_t = torch.maximum(max_velocity_correction_t, torch.linalg.vector_norm(correction64[3:6]).detach())
                max_attitude_correction_t = torch.maximum(max_attitude_correction_t, torch.linalg.vector_norm(correction64[6:9]).detach())
                correction64 = correction64.reshape(INS_STATE_DIM)
                nav_train = TorchNavigationState(
                    nav_train.position_ecef_m + correction64[0:3],
                    nav_train.velocity_ecef_mps + correction64[3:6],
                    _torch_so3_exponential(ATTITUDE_FEEDBACK_SIGN * correction64[6:9]) @ nav_train.body_to_ecef_dcm,
                )
                posterior_residual, posterior_sat_ids = _torch_measurement_innovation(nav_train, prepared_measurements)
                x_pred_context = zero_error_state_t.reshape(INS_STATE_DIM)
                x_post_context = correction64.reshape(INS_STATE_DIM)
                previous_context = {
                    'sat_ids': tuple(current_sat_ids),
                    'residual': posterior_residual.reshape(-1),
                    'x_pred': x_pred_context,
                    'x_post': x_post_context,
                    'state_innovation': x_post_context - x_pred_context,
                    'state_residual': torch.zeros_like(x_post_context) if previous_context is None else x_post_context - previous_context['x_post'],
                    'accel': last_feature_accel.reshape(3),
                    'gyro': last_feature_gyro.reshape(3),
                }
                if training:
                    nav_train = _detach_torch_navigation_state(nav_train)
                    previous_context = _detach_training_context(previous_context)
                    last_feature_accel = last_feature_accel.detach()
                    last_feature_gyro = last_feature_gyro.detach()
                learned_updates_since_detach += 1
                if training and recurrent_detach_step > 0 and (learned_updates_since_detach >= recurrent_detach_step) and (sample_index < stop_sample - 1):
                    recurrent_state = _detach_recurrent_state(recurrent_state)
                    learned_updates_since_detach = 0
                    tbptt_detach_count += 1
        if total_count <= 0:
            trajectory_eq30 = float('nan')
        else:
            trajectory_mean_loss = torch.stack(state_losses).mean()
            trajectory_eq30 = float(trajectory_mean_loss.detach().cpu())
            if training:
                scaled_loss = trajectory_mean_loss * float(backward_scale)
                scaled_loss.backward()
        rollout_state = None
        if training:
            rollout_state = {'nav_train': _detach_torch_navigation_state(nav_train), 'previous_context': _detach_training_context(previous_context), 'last_feature_accel': last_feature_accel.detach(), 'last_feature_gyro': last_feature_gyro.detach(), 'recurrent_state': _detach_recurrent_state(recurrent_state)}
        return {'trajectory_id': None if trajectory_id is None else int(trajectory_id), 'start_sample': int(start_sample), 'stop_sample': int(stop_sample), 'trajectory_eq30': trajectory_eq30, 'state_sum': float(total_state_sum_t.cpu()), 'position_sum': float(total_position_sum_t.cpu()), 'learned_updates': int(total_count), 'skipped_no_measurements': int(skipped_no_measurements), 'max_prior_position_error_m': float(max_prior_position_error_t.cpu()), 'max_innovation_abs_m': float(max_innovation_abs_t.cpu()), 'max_gain_fro_norm': 0.0 if max_gain_fro_t is None else float(max_gain_fro_t.cpu()), 'max_position_correction_norm_m': float(max_position_correction_t.cpu()), 'max_velocity_correction_norm_mps': float(max_velocity_correction_t.cpu()), 'max_attitude_correction_norm_rad': float(max_attitude_correction_t.cpu()), 'navigation_detach_step': 1 if training else 0, 'recurrent_detach_step': int(recurrent_detach_step), 'tbptt_detach_count': int(tbptt_detach_count), 'rollout_state': rollout_state}


    def _run_training_phase(phase: str, optimizer: torch.optim.Optimizer, epoch: int, epoch_batches, *, sequence_bounds=None):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if phase == 'filter':
            model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
            for module in filter_modules:
                module.train()
            for parameter in filter_parameters:
                parameter.requires_grad_(True)
        else:
            for module in representation_modules:
                module.train()
            model.lstm.train()
            model.lstm.lstm.dropout = 0.0
            for parameter in representation_parameters:
                parameter.requires_grad_(True)
        bounds = training_trajectories if sequence_bounds is None else list(sequence_bounds)
        sequence_count = len(bounds)
        total_state_sum = 0.0
        total_position_sum = 0.0
        total_count = 0
        total_skipped = 0
        batch_objectives = []
        batch_eq30_means = []
        gradient_norms = []
        gradient_norms_after_clip = []
        clipped_batch_count = 0
        batch_partitions = []
        optimizer_step_count = 0
        recurrent_detach_count = 0
        sequence_state_sums = {sequence_id: 0.0 for sequence_id in range(sequence_count)}
        sequence_update_counts = {sequence_id: 0 for sequence_id in range(sequence_count)}
        maxima = {'max_prior_position_error_m': 0.0, 'max_innovation_abs_m': 0.0, 'max_gain_fro_norm': 0.0, 'max_position_correction_norm_m': 0.0, 'max_velocity_correction_norm_mps': 0.0, 'max_attitude_correction_norm_rad': 0.0}
        with torch.no_grad():
            gain_head_l2_before_epoch = math.sqrt(sum(float(torch.sum(parameter.detach().double() ** 2).cpu()) for parameter in model.gain_head.parameters()))
        for batch_index, batch_ids in enumerate(epoch_batches):
            batch_ids = np.asarray(batch_ids, dtype=int).reshape(-1)
            actual_batch_size = int(batch_ids.size)
            batch_partitions.append([int(v) for v in batch_ids.tolist()])
            carried_states = {int(sequence_id): None for sequence_id in batch_ids.tolist()}
            max_sequence_length = max((bounds[int(sequence_id)][1] - bounds[int(sequence_id)][0] for sequence_id in batch_ids.tolist()))
            window_count = int(math.ceil(max_sequence_length / OPTIMIZER_WINDOW_SIZE))
            for window_index in range(window_count):
                active_ids = []
                for sequence_id_raw in batch_ids:
                    sequence_id = int(sequence_id_raw)
                    sequence_start, sequence_stop = bounds[sequence_id]
                    window_start = sequence_start + window_index * OPTIMIZER_WINDOW_SIZE
                    if window_start < sequence_stop:
                        active_ids.append(sequence_id)
                if not active_ids:
                    continue
                optimizer.zero_grad(set_to_none=True)
                window_metrics = []
                gradient_ids = []
                for sequence_id in active_ids:
                    sequence_start, sequence_stop = bounds[sequence_id]
                    window_start = sequence_start + window_index * OPTIMIZER_WINDOW_SIZE
                    window_stop = min(window_start + OPTIMIZER_WINDOW_SIZE, sequence_stop)
                    if any((history_rows[sample_index + 1]['fixed_measurements'] for sample_index in range(window_start, window_stop))):
                        gradient_ids.append(sequence_id)
                for sequence_id in active_ids:
                    sequence_start, sequence_stop = bounds[sequence_id]
                    window_start = sequence_start + window_index * OPTIMIZER_WINDOW_SIZE
                    window_stop = min(window_start + OPTIMIZER_WINDOW_SIZE, sequence_stop)
                    metrics = _rollout_actual_closed_loop_trajectory(window_start, window_stop, phase=phase, backward_scale=1.0 / max(len(gradient_ids), 1), trajectory_id=sequence_id, optimizer_step_index=optimizer_step_count, initialization_mode='kalmannet_v2_truth', recurrent_detach_step=RECURRENT_TBPTT_DETACH_STEP, initial_rollout_state=carried_states[sequence_id], allow_empty_training_window=True)
                    carried_states[sequence_id] = metrics['rollout_state']
                    recurrent_detach_count += int(metrics['tbptt_detach_count'])
                    window_metrics.append(metrics)
                    total_state_sum += metrics['state_sum']
                    total_position_sum += metrics['position_sum']
                    total_count += metrics['learned_updates']
                    total_skipped += metrics['skipped_no_measurements']
                    sequence_state_sums[sequence_id] += metrics['state_sum']
                    sequence_update_counts[sequence_id] += metrics['learned_updates']
                    for key in maxima:
                        maxima[key] = max(maxima[key], metrics[key])
                learned_window_metrics = [metrics for metrics in window_metrics if metrics['learned_updates'] > 0]
                if not learned_window_metrics:
                    continue
                regularization = _eq32_regularization()
                if GAMMA_L2:
                    regularization.backward()
                window_data_loss = float(np.mean([metrics['trajectory_eq30'] for metrics in learned_window_metrics]))
                window_objective = window_data_loss + float(regularization.detach().cpu())
                active_parameters = [parameter for parameter in (filter_parameters if phase == 'filter' else representation_parameters) if parameter.grad is not None]
                gradient_norm_before_clip = math.sqrt(sum(float(torch.sum(parameter.grad.detach().double() ** 2).cpu()) for parameter in active_parameters))
                if gradient_norm_before_clip > GRADIENT_CLIP_NORM:
                    scale = GRADIENT_CLIP_NORM / max(gradient_norm_before_clip, 1e-300)
                    with torch.no_grad():
                        for parameter in active_parameters:
                            parameter.grad.mul_(scale)
                gradient_norm_after_clip = math.sqrt(sum(float(torch.sum(parameter.grad.detach().double() ** 2).cpu()) for parameter in active_parameters))
                gradient_l2_norm = gradient_norm_before_clip
                gradient_norms.append(float(gradient_norm_before_clip))
                gradient_norms_after_clip.append(float(gradient_norm_after_clip))
                if gradient_norm_before_clip > GRADIENT_CLIP_NORM:
                    clipped_batch_count += 1
                optimizer.step()
                optimizer_step_count += 1
                batch_objectives.append(float(window_objective))
                batch_eq30_means.append(float(window_data_loss))
        all_trajectory_eq30 = [sequence_state_sums[sequence_id] / sequence_update_counts[sequence_id] for sequence_id in range(sequence_count) if sequence_update_counts[sequence_id] > 0]
        flattened = [v for batch in batch_partitions for v in batch]
        with torch.no_grad():
            gain_head_l2_after_epoch = math.sqrt(sum(float(torch.sum(parameter.detach().double() ** 2).cpu()) for parameter in model.gain_head.parameters()))
        out = {'eq30': float(total_state_sum / total_count), 'position_rmse_m': math.sqrt(total_position_sum / total_count), 'trajectory_mean_eq30': float(np.mean(all_trajectory_eq30)), 'objective': float(np.mean(batch_objectives)), 'mean_batch_eq30': float(np.mean(batch_eq30_means)), 'learned_updates': int(total_count), 'skipped_no_measurements': int(total_skipped), 'optimizer_steps': int(optimizer_step_count), 'trajectory_count': int(len(flattened)), 'available_trajectory_count': int(sequence_count), 'sequence_batch_size': int(max((len(batch) for batch in batch_partitions))), 'sampled_trajectory_ids': [int(v) for v in flattened], 'batch_partitions': batch_partitions, 'epoch_dataset_coverage_fraction': float(len(set(flattened)) / sequence_count), 'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE), 'navigation_detach_step': 1, 'recurrent_detach_step': int(RECURRENT_TBPTT_DETACH_STEP), 'recurrent_detach_count': int(recurrent_detach_count), 'gradient_l2_norm': float(np.mean(gradient_norms)), 'gradient_l2_norm_max': float(np.max(gradient_norms)), 'gradient_l2_norm_after_clip': float(np.mean(gradient_norms_after_clip)), 'gradient_l2_norm_after_clip_max': float(np.max(gradient_norms_after_clip)), 'gradient_clip_max_norm': float(GRADIENT_CLIP_NORM), 'gradient_clipped_optimizer_steps': int(clipped_batch_count), 'gain_head_parameter_l2_before': float(gain_head_l2_before_epoch), 'gain_head_parameter_l2_after': float(gain_head_l2_after_epoch), 'epoch': int(epoch)}
        out.update(maxima)
        return out

    def _evaluate_actual_closed_loop_training_objective():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        metrics = _rollout_actual_closed_loop_trajectory(0, training_sample_count, phase=None, backward_scale=None, trajectory_id=None, optimizer_step_index=None)
        metrics['eq30'] = metrics['state_sum'] / metrics['learned_updates']
        metrics['position_rmse_m'] = math.sqrt(metrics['position_sum'] / metrics['learned_updates'])
        metrics['objective'] = metrics['eq30'] + float(_eq32_regularization().detach().cpu())
        return metrics

    def _evaluate_actual_closed_loop_validation_objective():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        metrics = _rollout_actual_closed_loop_trajectory(validation_start_sample, data01_total_sample_count, phase=None, backward_scale=None, trajectory_id=None, optimizer_step_index=None, initialization_mode='kalmannet_v2_truth')
        metrics['eq30'] = metrics['state_sum'] / metrics['learned_updates']
        metrics['position_rmse_m'] = math.sqrt(metrics['position_sum'] / metrics['learned_updates'])
        metrics['objective'] = metrics['eq30']
        return metrics

    def _safe_recursive_monitor(scope: str, evaluator):
        try:
            metrics = evaluator()
        except FloatingPointError as error:
            message = str(error)
            print(f'WARNING: {scope} diverged and is ineligible for checkpoint selection: {message}')
            return {'eq30': float('inf'), 'position_rmse_m': float('inf'), 'objective': float('inf'), 'learned_updates': 0, 'diverged': True, 'divergence_reason': message}
        metrics['diverged'] = False
        metrics['divergence_reason'] = None
        return metrics

    pretraining_initialized_metrics = _safe_recursive_monitor('pre-training initialized-model Data01 rollout', _evaluate_actual_closed_loop_training_objective)
    print(f"pre-training initialized-model Data01 rollout: Eq30={pretraining_initialized_metrics['eq30']:.6g}, posRMSE={pretraining_initialized_metrics['position_rmse_m']:.3f} m, maxK={pretraining_initialized_metrics.get('max_gain_fro_norm', float('nan')):.6g}, maxPosCorr={pretraining_initialized_metrics.get('max_position_correction_norm_m', float('nan')):.6g} m, diverged={pretraining_initialized_metrics['diverged']}")
    saved_zero_gain_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    try:
        with torch.no_grad():
            model.gain_head.weight.zero_()
            if model.gain_head.bias is not None:
                model.gain_head.bias.zero_()
        pretraining_zero_gain_metrics = _evaluate_actual_closed_loop_training_objective()
    finally:
        model.load_state_dict({key: value.to(DEVICE) for key, value in saved_zero_gain_state.items()})
    print(f"pre-training TRUE zero-K Data01 baseline: Eq30={pretraining_zero_gain_metrics['eq30']:.6g}, posRMSE={pretraining_zero_gain_metrics['position_rmse_m']:.3f} m, maxInnovation={pretraining_zero_gain_metrics['max_innovation_abs_m']:.3f} m, maxK={pretraining_zero_gain_metrics['max_gain_fro_norm']:.3g}, maxPosCorr={pretraining_zero_gain_metrics['max_position_correction_norm_m']:.3g} m")
    best_model_state = None
    best_selection_eq30 = float('inf')
    best_selection_rmse_m = float('inf')
    best_selection_stage = None
    best_selection_epoch = None
    print('\n=== DATA01 TRAINING ===')
    training_history.append({'stage': 'pretraining_scaled_kaiming_data01_diagnostic_only', 'eligible_for_checkpoint_selection': False, 'gain_head_initial_scale': float(GAIN_HEAD_INITIAL_SCALE), 'eq30': float(pretraining_initialized_metrics['eq30']), 'position_rmse_m': float(pretraining_initialized_metrics['position_rmse_m']), 'diverged': bool(pretraining_initialized_metrics['diverged']), 'divergence_reason': pretraining_initialized_metrics['divergence_reason']})
    training_history.append({'stage': 'pretraining_true_zero_gain_data01_baseline_diagnostic_only', 'eligible_for_checkpoint_selection': False, 'eq30': float(pretraining_zero_gain_metrics['eq30']), 'position_rmse_m': float(pretraining_zero_gain_metrics['position_rmse_m']), 'max_innovation_abs_m': float(pretraining_zero_gain_metrics['max_innovation_abs_m']), 'max_gain_fro_norm': float(pretraining_zero_gain_metrics['max_gain_fro_norm']), 'max_position_correction_norm_m': float(pretraining_zero_gain_metrics['max_position_correction_norm_m']), 'learned_updates': int(pretraining_zero_gain_metrics['learned_updates'])})
    final_training_metrics = None
    epochs_ran = 0
    print('\n--- Short-trajectory alternating training (theta -> psi; not Latent encoder warm start) ---')
    for epoch in range(1, KNET_V2_ALTERNATING_EPOCHS + 1):
        epochs_ran += 1
        trajectory_ids = list(range(trajectory_count))
        random.shuffle(trajectory_ids)
        epoch_batches = [np.asarray(trajectory_ids[start:start + TRAJECTORY_BATCH_SIZE], dtype=int) for start in range(0, trajectory_count, TRAJECTORY_BATCH_SIZE)]
        filter_phase = _run_training_phase('filter', filter_optimizer, epoch, epoch_batches)
        representation_phase = _run_training_phase('representation', representation_optimizer, epoch, epoch_batches)
        final_training_metrics = representation_phase
        current_filter_lr = float(filter_optimizer.param_groups[0]['lr'])
        current_representation_lr = float(representation_optimizer.param_groups[0]['lr'])
        train_recursive_monitor = _safe_recursive_monitor(f'ALT epoch {epoch} Data01 training monitor', _evaluate_actual_closed_loop_training_objective)
        validation_recursive_monitor = _safe_recursive_monitor(f'ALT epoch {epoch} Data01 validation monitor', _evaluate_actual_closed_loop_validation_objective)
        validation_eq30 = float(validation_recursive_monitor['eq30'])
        validation_rmse_m = float(validation_recursive_monitor['position_rmse_m'])
        if math.isfinite(validation_eq30) and math.isfinite(validation_rmse_m) and (validation_eq30 < best_selection_eq30):
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_selection_eq30 = validation_eq30
            best_selection_rmse_m = validation_rmse_m
            best_selection_stage = 'short_trajectory_alternating_Data01_validation_recursive'
            best_selection_epoch = int(epoch)
        training_history.append({'stage': 'kalmannet_v2_short_trajectory_alternating', 'epoch': epoch, 'global_training_epoch': epochs_ran, 'sequence_length': int(training_sample_count), 'short_stage_trajectory_length': int(KNET_V2_TRAJECTORY_LENGTH), 'short_stage_initialization': 'ground_truth_navigation_state', 'short_stage_trained_sample_count': int(v2_trained_sample_count), 'short_stage_initialization_sample_count': int(v2_initialization_sample_count), 'short_stage_samples_excluded_from_loss': int(v2_unused_sample_count), 'short_stage_context_only_boundary_samples': int(_interchunk_context_only_samples), 'short_stage_unused_tail_samples': int(_tail_unused_samples), 'trajectory_count_available': int(trajectory_count), 'trajectory_batch_size': int(TRAJECTORY_BATCH_SIZE), 'algorithm2_q_batches': int(len(epoch_batches)), 'algorithm2_epoch_batch_partitions': filter_phase['batch_partitions'], 'algorithm2_same_partition_theta_psi': True, 'algorithm2_full_dataset_coverage': float(filter_phase['epoch_dataset_coverage_fraction']), 'filter_phase_sampled_trajectory_ids': filter_phase['sampled_trajectory_ids'], 'filter_phase_total_objective': filter_phase['objective'], 'filter_phase_eq30': filter_phase['eq30'], 'filter_phase_trajectory_mean_eq30': filter_phase['trajectory_mean_eq30'], 'filter_phase_position_rmse_m': filter_phase['position_rmse_m'], 'filter_phase_learned_updates': filter_phase['learned_updates'], 'filter_phase_optimizer_steps': filter_phase['optimizer_steps'], 'filter_phase_gradient_l2_norm': filter_phase['gradient_l2_norm'], 'filter_phase_gradient_l2_norm_after_clip': filter_phase['gradient_l2_norm_after_clip'], 'filter_phase_gradient_clip_max_norm': filter_phase['gradient_clip_max_norm'], 'filter_phase_gradient_clipped_optimizer_steps': filter_phase['gradient_clipped_optimizer_steps'], 'filter_phase_gain_head_l2_before': filter_phase['gain_head_parameter_l2_before'], 'filter_phase_gain_head_l2_after': filter_phase['gain_head_parameter_l2_after'], 'filter_phase_max_innovation_abs_m': filter_phase['max_innovation_abs_m'], 'filter_phase_max_gain_fro_norm': filter_phase['max_gain_fro_norm'], 'filter_phase_max_position_correction_norm_m': filter_phase['max_position_correction_norm_m'], 'representation_phase_sampled_trajectory_ids': representation_phase['sampled_trajectory_ids'], 'representation_phase_total_objective': representation_phase['objective'], 'representation_phase_eq30': representation_phase['eq30'], 'representation_phase_trajectory_mean_eq30': representation_phase['trajectory_mean_eq30'], 'representation_phase_position_rmse_m': representation_phase['position_rmse_m'], 'representation_phase_learned_updates': representation_phase['learned_updates'], 'representation_phase_optimizer_steps': representation_phase['optimizer_steps'], 'representation_phase_gradient_l2_norm': representation_phase['gradient_l2_norm'], 'representation_phase_gradient_l2_norm_after_clip': representation_phase['gradient_l2_norm_after_clip'], 'representation_phase_gradient_clip_max_norm': representation_phase['gradient_clip_max_norm'], 'representation_phase_gradient_clipped_optimizer_steps': representation_phase['gradient_clipped_optimizer_steps'], 'representation_phase_gain_head_l2_before': representation_phase['gain_head_parameter_l2_before'], 'representation_phase_gain_head_l2_after': representation_phase['gain_head_parameter_l2_after'], 'representation_phase_max_innovation_abs_m': representation_phase['max_innovation_abs_m'], 'representation_phase_max_gain_fro_norm': representation_phase['max_gain_fro_norm'], 'representation_phase_max_position_correction_norm_m': representation_phase['max_position_correction_norm_m'], 'Data01_train_recursive_eq30': float(train_recursive_monitor['eq30']), 'Data01_train_recursive_position_rmse_m': float(train_recursive_monitor['position_rmse_m']), 'Data01_validation_recursive_eq30': float(validation_recursive_monitor['eq30']), 'Data01_validation_recursive_position_rmse_m': float(validation_recursive_monitor['position_rmse_m']), 'Data01_train_recursive_diverged': bool(train_recursive_monitor['diverged']), 'Data01_validation_recursive_diverged': bool(validation_recursive_monitor['diverged']), 'Data01_validation_divergence_reason': validation_recursive_monitor['divergence_reason'], 'validation_to_train_rmse_ratio': (float(validation_recursive_monitor['position_rmse_m'] / max(train_recursive_monitor['position_rmse_m'], 1e-12)) if math.isfinite(validation_recursive_monitor['position_rmse_m']) and math.isfinite(train_recursive_monitor['position_rmse_m']) else float('inf')), 'best_Data01_validation_recursive_eq30_so_far': float(best_selection_eq30), 'best_Data01_validation_recursive_position_rmse_m_so_far': float(best_selection_rmse_m), 'filter_block_learning_rate': current_filter_lr, 'representation_block_learning_rate': current_representation_lr})
        print(f"ALT epoch {epoch:03d}/{KNET_V2_ALTERNATING_EPOCHS}: theta_Eq30={filter_phase['trajectory_mean_eq30']:.6g}, theta_RMSE={filter_phase['position_rmse_m']:.3f}m, theta_grad={filter_phase['gradient_l2_norm']:.3g}->{filter_phase['gradient_l2_norm_after_clip']:.3g}, theta_Kparam={filter_phase['gain_head_parameter_l2_before']:.3g}->{filter_phase['gain_head_parameter_l2_after']:.3g}, theta_maxK={filter_phase['max_gain_fro_norm']:.3g}; psi_Eq30={representation_phase['trajectory_mean_eq30']:.6g}, psi_RMSE={representation_phase['position_rmse_m']:.3f}m, psi_grad={representation_phase['gradient_l2_norm']:.3g}->{representation_phase['gradient_l2_norm_after_clip']:.3g}; train_recursive_RMSE={train_recursive_monitor['position_rmse_m']:.3f}m; val_recursive_RMSE={validation_recursive_monitor['position_rmse_m']:.3f}m; best_val_RMSE={best_selection_rmse_m:.3f}m")
    print('\n--- KalmanNet V1 full-sequence fine-tuning ---')
    if KNET_V1_FINE_TUNE_EPOCHS == 0:
        print('V1 fine-tuning disabled by MKNET_V1_FINE_TUNE_EPOCHS=0 (short-trajectory-stage-only diagnostic).')
    for epoch in range(1, KNET_V1_FINE_TUNE_EPOCHS + 1):
        epochs_ran += 1
        filter_phase = _run_training_phase('filter', filter_optimizer, epoch, [np.asarray([0], dtype=int)], sequence_bounds=[(0, training_sample_count)])
        representation_phase = _run_training_phase('representation', representation_optimizer, epoch, [np.asarray([0], dtype=int)], sequence_bounds=[(0, training_sample_count)])
        final_training_metrics = representation_phase
        current_filter_lr = float(filter_optimizer.param_groups[0]['lr'])
        current_representation_lr = float(representation_optimizer.param_groups[0]['lr'])
        train_recursive_monitor = _safe_recursive_monitor(f'V1 epoch {epoch} Data01 training monitor', _evaluate_actual_closed_loop_training_objective)
        validation_recursive_monitor = _safe_recursive_monitor(f'V1 epoch {epoch} Data01 validation monitor', _evaluate_actual_closed_loop_validation_objective)
        validation_eq30 = float(validation_recursive_monitor['eq30'])
        validation_rmse_m = float(validation_recursive_monitor['position_rmse_m'])
        if math.isfinite(validation_eq30) and math.isfinite(validation_rmse_m) and (validation_eq30 < best_selection_eq30):
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_selection_eq30 = validation_eq30
            best_selection_rmse_m = validation_rmse_m
            best_selection_stage = 'kalmannet_v1_Data01_validation_recursive'
            best_selection_epoch = int(epoch)
        training_history.append({'stage': 'kalmannet_v1_full_sequence_finetune_yan_masked_cla_latent_alternating', 'epoch': epoch, 'global_training_epoch': epochs_ran, 'sequence_length': int(training_sample_count), 'initialization': 'ground_truth_navigation_state', 'trajectory_count': 1, 'trajectory_batch_size': 1, 'filter_phase_total_objective': filter_phase['objective'], 'filter_phase_eq30': filter_phase['eq30'], 'filter_phase_trajectory_mean_eq30': filter_phase['trajectory_mean_eq30'], 'filter_phase_position_rmse_m': filter_phase['position_rmse_m'], 'filter_phase_learned_updates': filter_phase['learned_updates'], 'filter_phase_optimizer_steps': filter_phase['optimizer_steps'], 'filter_phase_gradient_l2_norm': filter_phase['gradient_l2_norm'], 'filter_phase_gradient_l2_norm_after_clip': filter_phase['gradient_l2_norm_after_clip'], 'filter_phase_gradient_clip_max_norm': filter_phase['gradient_clip_max_norm'], 'filter_phase_gradient_clipped_optimizer_steps': filter_phase['gradient_clipped_optimizer_steps'], 'filter_phase_gain_head_l2_before': filter_phase['gain_head_parameter_l2_before'], 'filter_phase_gain_head_l2_after': filter_phase['gain_head_parameter_l2_after'], 'filter_phase_max_innovation_abs_m': filter_phase['max_innovation_abs_m'], 'filter_phase_max_gain_fro_norm': filter_phase['max_gain_fro_norm'], 'filter_phase_max_position_correction_norm_m': filter_phase['max_position_correction_norm_m'], 'representation_phase_total_objective': representation_phase['objective'], 'representation_phase_eq30': representation_phase['eq30'], 'representation_phase_trajectory_mean_eq30': representation_phase['trajectory_mean_eq30'], 'representation_phase_position_rmse_m': representation_phase['position_rmse_m'], 'representation_phase_learned_updates': representation_phase['learned_updates'], 'representation_phase_optimizer_steps': representation_phase['optimizer_steps'], 'representation_phase_gradient_l2_norm': representation_phase['gradient_l2_norm'], 'representation_phase_gradient_l2_norm_after_clip': representation_phase['gradient_l2_norm_after_clip'], 'representation_phase_gradient_clip_max_norm': representation_phase['gradient_clip_max_norm'], 'representation_phase_gradient_clipped_optimizer_steps': representation_phase['gradient_clipped_optimizer_steps'], 'representation_phase_gain_head_l2_before': representation_phase['gain_head_parameter_l2_before'], 'representation_phase_gain_head_l2_after': representation_phase['gain_head_parameter_l2_after'], 'representation_phase_max_innovation_abs_m': representation_phase['max_innovation_abs_m'], 'representation_phase_max_gain_fro_norm': representation_phase['max_gain_fro_norm'], 'representation_phase_max_position_correction_norm_m': representation_phase['max_position_correction_norm_m'], 'Data01_train_recursive_eq30': float(train_recursive_monitor['eq30']), 'Data01_train_recursive_position_rmse_m': float(train_recursive_monitor['position_rmse_m']), 'Data01_validation_recursive_eq30': float(validation_recursive_monitor['eq30']), 'Data01_validation_recursive_position_rmse_m': float(validation_recursive_monitor['position_rmse_m']), 'Data01_train_recursive_diverged': bool(train_recursive_monitor['diverged']), 'Data01_validation_recursive_diverged': bool(validation_recursive_monitor['diverged']), 'Data01_validation_divergence_reason': validation_recursive_monitor['divergence_reason'], 'validation_to_train_rmse_ratio': (float(validation_recursive_monitor['position_rmse_m'] / max(train_recursive_monitor['position_rmse_m'], 1e-12)) if math.isfinite(validation_recursive_monitor['position_rmse_m']) and math.isfinite(train_recursive_monitor['position_rmse_m']) else float('inf')), 'best_Data01_validation_recursive_eq30_so_far': float(best_selection_eq30), 'best_Data01_validation_recursive_position_rmse_m_so_far': float(best_selection_rmse_m), 'filter_block_learning_rate': current_filter_lr, 'representation_block_learning_rate': current_representation_lr})
        print(f"V1 epoch {epoch:03d}/{KNET_V1_FINE_TUNE_EPOCHS}: theta_Eq30={filter_phase['trajectory_mean_eq30']:.6g}, theta_RMSE={filter_phase['position_rmse_m']:.3f}m, theta_grad={filter_phase['gradient_l2_norm']:.3g}->{filter_phase['gradient_l2_norm_after_clip']:.3g}, theta_Kparam={filter_phase['gain_head_parameter_l2_before']:.3g}->{filter_phase['gain_head_parameter_l2_after']:.3g}; psi_Eq30={representation_phase['trajectory_mean_eq30']:.6g}, psi_RMSE={representation_phase['position_rmse_m']:.3f}m, psi_grad={representation_phase['gradient_l2_norm']:.3g}->{representation_phase['gradient_l2_norm_after_clip']:.3g}; train_recursive_RMSE={train_recursive_monitor['position_rmse_m']:.3f}m; val_recursive_RMSE={validation_recursive_monitor['position_rmse_m']:.3f}m; best_val_RMSE={best_selection_rmse_m:.3f}m")
    model.load_state_dict({key: value.to(DEVICE) for key, value in best_model_state.items()})
    print(f'restored best TRAINED Data01 validation checkpoint: stage={best_selection_stage}, epoch={best_selection_epoch}, val_Eq30={best_selection_eq30:.6g}, val_posRMSE={best_selection_rmse_m:.3f} m')
    final_training_metrics = _evaluate_actual_closed_loop_training_objective()
    final_validation_metrics = _evaluate_actual_closed_loop_validation_objective()
    zero_gain_rmse = float(pretraining_zero_gain_metrics['position_rmse_m'])
    trained_rmse = float(final_training_metrics['position_rmse_m'])
    validation_rmse = float(final_validation_metrics['position_rmse_m'])
    model.eval()
    teacher_state_sum_t = zero_error_state_t.new_zeros(())
    teacher_position_sum_t = zero_error_state_t.new_zeros(())
    teacher_recurrent_state = None
    with torch.inference_mode():
        for sample_index in range(training_sample_count):
            fixed_eval = torch.as_tensor(fixed_network[sample_index], dtype=torch.float32, device=DEVICE)
            obs_eval = torch.as_tensor(observations_network[sample_index], dtype=torch.float32, device=DEVICE)
            mask_eval = torch.as_tensor(satellite_masks[sample_index], dtype=torch.bool, device=DEVICE)
            channel_eval = torch.as_tensor(channel_masks[sample_index], dtype=torch.bool, device=DEVICE)
            innovation_eval = torch.as_tensor(innovation_padded[sample_index], dtype=torch.float32, device=DEVICE)
            fixed_eval, obs_eval, mask_eval, channel_eval, innovation_eval = _torch_pad_training_epoch(fixed_eval, obs_eval, mask_eval, channel_eval, innovation_eval, network_nmax)
            output = model(fixed_eval.unsqueeze(0), obs_eval.unsqueeze(0), mask_eval.unsqueeze(0), channel_eval.unsqueeze(0), recurrent_state=teacher_recurrent_state)
            teacher_recurrent_state = output.recurrent_state
            correction = fig8_state_update(output, innovation_eval.unsqueeze(0))[0].double()
            residual_9 = target_base_t[sample_index] - correction[:SUPERVISED_STATE_DIM]
            teacher_state_sum_t += torch.sum(residual_9 ** 2)
            teacher_position_sum_t += torch.sum(residual_9[:3] ** 2)
    teacher_state_sum = float(teacher_state_sum_t.cpu())
    teacher_position_sum = float(teacher_position_sum_t.cpu())
    final_teacher_forced_metrics = {
        'eq30': teacher_state_sum / training_sample_count,
        'position_rmse_m': math.sqrt(teacher_position_sum / training_sample_count),
    }
    training_performance_summary = {'split': {'mode': 'chronological_holdout', 'Data01_total_samples': int(data01_total_sample_count), 'training_samples': int(training_sample_count), 'validation_samples': int(validation_sample_count), 'validation_fraction': float(VALIDATION_FRACTION), 'Data02_used_for_validation': False}, 'zero_gain_closed_loop_train': {'eq30': float(pretraining_zero_gain_metrics['eq30']), 'position_rmse_m': zero_gain_rmse, 'max_innovation_abs_m': float(pretraining_zero_gain_metrics['max_innovation_abs_m'])}, 'trained_closed_loop_train': {'eq30': float(final_training_metrics['eq30']), 'position_rmse_m': trained_rmse, 'rmse_ratio_vs_zero_gain': trained_rmse / max(zero_gain_rmse, 1e-12), 'max_innovation_abs_m': float(final_training_metrics['max_innovation_abs_m']), 'max_gain_fro_norm': float(final_training_metrics['max_gain_fro_norm']), 'max_position_correction_norm_m': float(final_training_metrics['max_position_correction_norm_m'])}, 'validation_closed_loop': {'eq30': float(final_validation_metrics['eq30']), 'position_rmse_m': validation_rmse, 'validation_to_train_rmse_ratio': validation_rmse / max(trained_rmse, 1e-12), 'validation_minus_train_rmse_m': validation_rmse - trained_rmse, 'max_innovation_abs_m': float(final_validation_metrics['max_innovation_abs_m']), 'max_gain_fro_norm': float(final_validation_metrics['max_gain_fro_norm']), 'max_position_correction_norm_m': float(final_validation_metrics['max_position_correction_norm_m'])}, 'one_step_train': {'eq30': float(final_teacher_forced_metrics['eq30']), 'position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m'])}, 'selected_checkpoint': {'stage': best_selection_stage, 'epoch': int(best_selection_epoch), 'selection_metric': 'Data01_validation_recursive_Eq30', 'Data01_validation_recursive_eq30': float(best_selection_eq30), 'Data01_validation_recursive_position_rmse_m': float(best_selection_rmse_m)}}
    print('Data01 train/validation performance:', json.dumps(training_performance_summary, indent=2))
    model.eval()
    best_epoch = int(best_selection_epoch)
    best_val_state_loss = float(best_selection_eq30)
    best_val_position_rmse_m = float(best_selection_rmse_m)
    selected_training_stage = str(best_selection_stage)

    def _make_online_context(sat_ids, posterior_residual, x_pred, x_post, accel, gyro, previous_context=None):
        x_pred = np.asarray(x_pred, dtype=float).reshape(INS_STATE_DIM)
        x_post = np.asarray(x_post, dtype=float).reshape(INS_STATE_DIM)
        state_innovation = x_post - x_pred
        state_residual = np.zeros(INS_STATE_DIM) if previous_context is None else x_post - np.asarray(previous_context['x_post'], dtype=float).reshape(INS_STATE_DIM)
        context = {'sat_ids': tuple(sat_ids), 'residual': np.asarray(posterior_residual, dtype=float).copy(), 'x_pred': x_pred.copy(), 'x_post': x_post.copy(), 'state_innovation': state_innovation.copy(), 'state_residual': state_residual.copy(), 'accel': np.asarray(accel, dtype=float).reshape(3).copy(), 'gyro': np.asarray(gyro, dtype=float).reshape(3).copy()}
        return context

    def _online_feature_arrays(previous_online, current_sat_ids, innovation_now, current_accel, current_gyro):
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
        obs_k_raw = np.where(channel_k, obs_k_raw, 0.0)
        return (fixed_k_raw, obs_k_raw, current_mask, channel_k, innovation_k, fixed_k_raw, obs_k_raw)

    def _pad_neural_epoch(fixed_k, obs_k, current_mask, channel_k, innovation_k, slot_capacity: int):
        fixed_k = np.asarray(fixed_k, dtype=float).reshape(FIXED_FEATURE_DIM)
        obs_k = np.asarray(obs_k, dtype=float).reshape(-1, OBSERVATION_FEATURE_DIM)
        current_mask = np.asarray(current_mask, dtype=bool).reshape(-1)
        channel_k = np.asarray(channel_k, dtype=bool).reshape(-1, OBSERVATION_FEATURE_DIM)
        innovation_k = np.asarray(innovation_k, dtype=float).reshape(-1)
        count = len(current_mask)
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
    model.eval()
    torch.save({'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 'checkpoint_schema': 'v102_9_state_masked_cla_latent_audit_1600', 'nmax': int(network_nmax), 'training_observed_nmax': int(nmax), 'data01_sequence_nmax': int(data01_sequence_nmax), 'data02_sequence_nmax': None, 'nmax_source': 'Data01_actual_neural_observation_max_only', 'state_order': '[delta_p,delta_v,delta_theta]', 'kalman_gain_state_dimension': INS_STATE_DIM, 'direct_state_label_dimension': SUPERVISED_STATE_DIM, 'feature_normalization': 'none_raw_inputs', 'alternating_partition': 'psi=masked_conv;theta=masked_lstm+attention+gain_head', 'latent_kalmannet_warm_start': 'not_applied_no_published_supervised_target_for_intermediate_Yan_CNN_features', 'training_stage_selected': selected_training_stage, 'training_epochs': best_epoch, 'requires_recursive_Data01_gate_before_Data02': True, 'sequence_training': 'Yan_Fig7_Fig8_order_adapted_to_user_requested_9_state_branch;short_trajectory_Algorithm2_style_random_partition_all_Q_batches_same_partition_theta_then_psi_then_optional_V1_full_sequence;learned_posterior_to_ECEF_INS_to_next_features;no_internal_classical_KF_reset;no_Phi_H_surrogate;fixed_observation_snapshot_recursive_innovation', 'training_sequence_length': int(training_sample_count), 'validation_sequence_length': int(validation_sample_count), 'data01_total_sequence_length': int(data01_total_sample_count), 'validation_fraction': float(VALIDATION_FRACTION), 'validation_protocol': 'chronological_tail_holdout_recursive_eval_best_Eq30', 'training_sequence_count': int(trajectory_count), 'sequence_batch_size': int(TRAJECTORY_BATCH_SIZE), 'short_trajectory_truth_restart_diagnostic': short_trajectory_restart_diagnostic, 'optimizer_step_inside_sequence': True, 'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE), 'navigation_state_detach_step': 1, 'recurrent_state_detach_step': int(RECURRENT_TBPTT_DETACH_STEP), 'cross_fusion_navigation_gradient': 'stopped_after_each_learned_update_while_numerical_state_continues', 'cross_fusion_neural_gradient': 'LSTM_hidden_cell_BPTT_truncated_every_configured_recurrent_steps', 'gain_head_initialization': 'scaled_kaiming_uniform_weight_zero_bias', 'gain_head_initial_scale': float(GAIN_HEAD_INITIAL_SCALE), 'synthetic_prior_augmentation': False, 'lstm_sequence_state': 'Yan_Eq25_per_position_hc_carried_across_fusion_epochs_k;feature_position_t_not_used_as_LSTM_time_axis', 'external_filter_state': 'short_stage_recursive_within_truth_initialized_chunks;optional_V1_full_sequence_recursive;no_internal_classical_KF_reset', 'final_epoch': best_epoch, 'final_train_recursive_eq30_state_loss': float(final_training_metrics['eq30']), 'final_train_recursive_position_rmse_m': float(final_training_metrics['position_rmse_m']), 'final_validation_recursive_eq30_state_loss': float(final_validation_metrics['eq30']), 'final_validation_recursive_position_rmse_m': float(final_validation_metrics['position_rmse_m']), 'selected_validation_recursive_eq30': best_val_state_loss, 'selected_validation_recursive_position_rmse_m': best_val_position_rmse_m, 'final_clean_teacher_forced_eq30': float(final_teacher_forced_metrics['eq30']), 'final_clean_teacher_forced_position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m']), 'state_target_scaling': 'none_raw_physical_units_per_Yan_Eq30', 'measurement_mode': 'pseudorange_only', 'observation_residual_feature': 'Yan_Eq11_Delta_y_k_minus_1;same_epoch_Delta_y_k_not_used_because_it_requires_the_current_posterior_and_would_make_K_k_input_circular'}, OUTPUT_DIR / 'best_model.pt')
    (OUTPUT_DIR / 'history.json').write_text(json.dumps(training_history, indent=2), encoding='utf-8')
    if RUN_RECURSIVE_DATA01_DIAGNOSTIC:
        print('\n=== DATA01 RECURSIVE PERFORMANCE ===')
        data01_diag_leo = LEODownlinkSimulator(tle_provider, leo_klobuchar, seed=LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
        diag_nav = initial_nav.copy()
        diag_P = P0.copy()
        diag_rows = []
        diag_previous = None
        diag_warm_start = None
        diag_recurrent_state = None
        classical_position_by_fusion_index = {int(row['fusion_index']): gnss_antenna_position(row['posterior_nav'], lever_arm_b_m) for row in history_rows}
        diag_last_gyro, diag_last_accel = compensate_imu(imr_gyro[0], imr_accel[0])
        diag_feature_gyro, diag_feature_accel = (diag_last_gyro, diag_last_accel)
        van_loan_stats_before_data01_diag = dict(_VAN_LOAN_STATS)
        try:
            for event in build_exact_fusion_timeline(imr_time, fusion_time, through_last_fusion=True):
                if isinstance(event, PropagationSegment):
                    imu_index = event.imu_index
                    dt = event.end_time_gpst_s - event.start_time_gpst_s
                    diag_last_gyro, diag_last_accel = compensate_imu(imr_gyro[imu_index], imr_accel[imu_index])
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
                diag_model = build_measurement_model(diag_nav, diag_measurements, lever_arm_b_m)
                diag_input_model = diag_model
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
                diag_P = learned_gain_covariance_update(diag_P, active_gain, diag_model.H, diag_model.R, diag_correction)
                diag_nav = inject_error_state(diag_nav, diag_correction)
                posterior_position = gnss_antenna_position(diag_nav, lever_arm_b_m)
                posterior_error_3d_m = float(np.linalg.norm(posterior_position - truth_position_now))
                posterior_residual = build_innovation_only(diag_nav, diag_measurements, lever_arm_b_m)
                diag_previous = _make_online_context(current_sat_ids, posterior_residual, np.zeros(INS_STATE_DIM), diag_correction, diag_feature_accel, diag_feature_gyro, previous_context=diag_previous)
                diag_rows.append({'time': t, 'position': posterior_position.copy(), 'truth': truth_position_now.copy(), 'prior_error_3d_m': prior_error_3d_m, 'posterior_error_3d_m': posterior_error_3d_m, 'classical_posterior_error_3d_m': classical_error_3d_m, 'gain_fro_norm': float(np.linalg.norm(active_gain)), 'correction_position_norm_m': float(np.linalg.norm(diag_correction[:3])), 'input_innovation_max_abs_m': float(np.max(np.abs(diag_input_model.innovation))) if len(diag_input_model.innovation) else 0.0, 'obs_max_abs_train_ratio': float(feature_shift['obs_max_abs_train_ratio'])})
        finally:
            _VAN_LOAN_STATS.clear()
            _VAN_LOAN_STATS.update(van_loan_stats_before_data01_diag)
        diag_fields = ('time', 'prior_error_3d_m', 'posterior_error_3d_m', 'classical_posterior_error_3d_m', 'gain_fro_norm', 'correction_position_norm_m', 'input_innovation_max_abs_m', 'obs_max_abs_train_ratio')
        for row_index, row in enumerate(diag_rows):
            missing = [name for name in diag_fields if name not in row]
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
        gate_learned = np.asarray(diag_error_3d, dtype=float).reshape(-1)
        gate_classical = np.asarray(diag_classical_error_3d, dtype=float).reshape(-1)
        if gate_learned.shape != gate_classical.shape or gate_learned.size == 0:
            data01_acceptance_gate = {'passed': False, 'reason': 'missing_or_misaligned_same_epoch_errors', 'epochs_compared': 0, 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
        else:
            gate_finite = np.isfinite(gate_learned) & np.isfinite(gate_classical)
            if not np.all(gate_finite):
                data01_acceptance_gate = {'passed': False, 'reason': 'nonfinite_or_missing_same_epoch_error', 'epochs_compared': int(np.count_nonzero(gate_finite)), 'learned_rmse_3d_m': None, 'classical_rmse_3d_m': None}
            else:
                gate_learned_rmse = float(np.sqrt(np.mean(gate_learned ** 2)))
                gate_classical_rmse = float(np.sqrt(np.mean(gate_classical ** 2)))
                gate_passed = bool(gate_learned_rmse <= gate_classical_rmse)
                data01_acceptance_gate = {'passed': gate_passed, 'reason': 'learned_rmse_not_worse_than_classical' if gate_passed else 'learned_rmse_worse_than_classical', 'epochs_compared': int(gate_learned.size), 'learned_rmse_3d_m': gate_learned_rmse, 'classical_rmse_3d_m': gate_classical_rmse, 'criterion': 'learned_same_epoch_rmse_3d_m <= classical_same_epoch_rmse_3d_m', 'paper_basis': 'Yan_Fig9a_qualitative_learned_error_below_traditional_KF', 'absolute_threshold_used': False}
        diag_time = np.asarray([row['time'] for row in diag_rows], dtype=float)
        diag_obs_ratio = np.asarray([row['obs_max_abs_train_ratio'] for row in diag_rows], dtype=float)

        def _diag_first(mask):
            indices = np.flatnonzero(np.asarray(mask, dtype=bool))
            if not len(indices):
                return None
            i = int(indices[0])
            return {'row_index': i, 'time_gpst_s': float(diag_time[i]), 'posterior_error_3d_m': float(diag_error_3d[i])}
        data01_recursive_summary = {'epochs': int(len(diag_rows)), 'causal_warm_start': diag_warm_start, 'rmse_n_e_d_3d_m': [float(x) for x in diag_rmse], 'recursive_same_epoch_acceptance_gate': data01_acceptance_gate, 'first_error_gt_100m': _diag_first(diag_error_3d > 100.0), 'first_error_gt_1km': _diag_first(diag_error_3d > 1000.0), 'first_error_gt_100km': _diag_first(diag_error_3d > 100000.0), 'first_obs_feature_gt_10x_training_absmax': _diag_first(diag_obs_ratio > 10.0), 'max_input_innovation_abs_m': float(np.nanmax([row['input_innovation_max_abs_m'] for row in diag_rows])), 'max_gain_fro_norm': float(np.nanmax([row['gain_fro_norm'] for row in diag_rows])), 'max_position_correction_norm_m': float(np.nanmax([row['correction_position_norm_m'] for row in diag_rows])), 'scope': 'frozen_weights_same_Data01_recursive_diagnostic_before_Data02'}
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
    test_antenna_truth = GroundTruth(test_antenna_truth_week.astype(float) * GPS_WEEK_S + test_antenna_truth_array[:, 1], test_antenna_truth_array[:, 2:5], test_antenna_truth_array[:, 5:8], test_antenna_truth_array[:, 8], test_antenna_truth_array[:, 9], test_antenna_truth_array[:, 10])
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
    test_imu_truth = GroundTruth(test_imu_truth_week.astype(float) * GPS_WEEK_S + test_imu_truth_array[:, 1], test_imu_truth_array[:, 2:5], test_imu_truth_array[:, 5:8], test_imu_truth_array[:, 8], test_imu_truth_array[:, 9], test_imu_truth_array[:, 10])
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
    test_anchor_time = float(first_test_rinex_epoch.time_gpst_s)
    test_imr_time_all = anchor_imr_tow_to_gpst_seconds(test_imr_tow_all, test_anchor_time)
    test_antenna_truth_time = np.asarray(test_antenna_truth.time_gpst_s, dtype=float).reshape(-1)
    test_imu_truth_time = np.asarray(test_imu_truth.time_gpst_s, dtype=float).reshape(-1)
    test_common_start = max(float(test_imr_time_all[0]), float(test_antenna_truth_time[0]), float(test_imu_truth_time[0]))
    test_common_end = min(float(test_imr_time_all[-1]), float(test_antenna_truth_time[-1]), float(test_imu_truth_time[-1]))
    test_start = int(np.searchsorted(test_imr_time_all, test_common_start, side='left'))
    test_usable_imu_start = float(test_imr_time_all[test_start])
    test_gnss_epochs = tuple(test_rinex.iter_epochs(allowed_constellations={'G', 'C'}, start_time_gpst_s=test_usable_imu_start, end_time_gpst_s=test_common_end, max_epochs=MAX_TEST_FUSION_EPOCHS, require_measurements=True))
    test_fusion_time = np.asarray([epoch.time_gpst_s for epoch in test_gnss_epochs], dtype=float)
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
    test_initial_antenna_from_imu = test_imu_position[0] + test_C_b_e @ test_lever_arm_b_m
    test_truth_reference_error_m = np.linalg.norm(test_antenna_position[0] - test_initial_antenna_from_imu)
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
    zero_error_state_np = np.zeros(INS_STATE_DIM)
    previous_online = None
    test_recurrent_capacity = int(network_nmax)
    test_warm_start = None
    test_recurrent_state = None
    data02_observed_nmax = 0
    data02_nmax_before_capacity = 0
    data02_capacity_selection_epochs = 0
    data02_capacity_excluded_measurements = 0
    last_gyro, last_accel = compensate_imu(test_imr_gyro[0], test_imr_accel[0])
    last_feature_gyro, last_feature_accel = (last_gyro, last_accel)
    test_timeline = build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True)
    for event in test_timeline:
        if isinstance(event, PropagationSegment):
            imu_index = event.imu_index
            dt = event.end_time_gpst_s - event.start_time_gpst_s
            last_gyro, last_accel = compensate_imu(test_imr_gyro[imu_index], test_imr_accel[imu_index])
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
        input_measurement_model = build_measurement_model(nav, measurements, test_lever_arm_b_m)
        data02_nmax_before_capacity = max(data02_nmax_before_capacity, len(measurements))
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
        measurement_model = build_measurement_model(nav, measurements, test_lever_arm_b_m)
        n = len(measurements)
        current_sat_ids = measurement_model.sat_ids
        innovation_now = measurement_model.innovation
        fixed_k, obs_k, current_mask, channel_k, innovation_k, fixed_k_raw, obs_k_raw = _online_feature_arrays(previous_online, current_sat_ids, innovation_now, last_feature_accel, last_feature_gyro)
        feature_shift = _feature_shift_diagnostics(fixed_k_raw, obs_k_raw, channel_k)
        input_innovation = np.asarray(input_measurement_model.innovation, dtype=float)
        input_innovation_rms_m = float(np.sqrt(np.mean(input_innovation ** 2))) if input_innovation.size else 0.0
        input_innovation_max_abs_m = float(np.max(np.abs(input_innovation))) if input_innovation.size else 0.0
        knet_innovation_rms_m = float(np.sqrt(np.mean(innovation_k ** 2))) if innovation_k.size else 0.0
        knet_innovation_max_abs_m = float(np.max(np.abs(innovation_k))) if innovation_k.size else 0.0
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
        P = learned_gain_covariance_update(P, active_gain, measurement_model.H, measurement_model.R, correction)
        nav = inject_error_state(nav, correction)
        posterior_position = gnss_antenna_position(nav, test_lever_arm_b_m)
        posterior_error_3d_m = float(np.linalg.norm(posterior_position - truth_position_now))
        posterior_residual = build_innovation_only(nav, measurements, test_lever_arm_b_m)
        previous_online = _make_online_context(current_sat_ids, posterior_residual, learned_error_state_pred, learned_error_state_post, last_feature_accel, last_feature_gyro, previous_context=previous_online)
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
    for row_index, row in enumerate(online_rows):
        missing = [name for name in diagnostic_fields if name not in row]
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
    divergence_diagnostics = {'first_error_gt_100m': _first_true_event(error_3d > 100.0), 'first_error_gt_1km': _first_true_event(error_3d > 1000.0), 'first_error_gt_100km': _first_true_event(error_3d > 100000.0), 'first_error_growth_gt_10x': _first_true_event(growth_ratio > 10.0), 'first_obs_feature_gt_10x_training_absmax': _first_true_event(diagnostic_arrays['obs_max_abs_train_ratio'] > 10.0), 'max_input_innovation_abs_m': float(np.nanmax(diagnostic_arrays['input_innovation_max_abs_m'])), 'max_knet_innovation_abs_m': float(np.nanmax(diagnostic_arrays['knet_innovation_max_abs_m'])), 'max_gain_fro_norm': float(np.nanmax(diagnostic_arrays['gain_fro_norm'])), 'max_position_correction_norm_m': float(np.nanmax(diagnostic_arrays['correction_position_norm_m'])), 'max_fixed_feature_abs_training_ratio': float(np.nanmax(diagnostic_arrays['fixed_max_abs_train_ratio'])), 'max_observation_feature_abs_training_ratio': float(np.nanmax(diagnostic_arrays['obs_max_abs_train_ratio']))}
    np.savez(OUTPUT_DIR / 'test_evaluation.npz', dataset_dir=np.asarray(str(test_dataset_dir)), time_gpst_s=online_time, estimate_ecef_m=estimate, truth_ecef_m=truth_aligned, ned_error_m=ned_error, error_3d_m=error_3d, horizontal_error_m=horizontal_error, vertical_error_m=vertical_error, **diagnostic_arrays, error_growth_ratio=growth_ratio, rmse_north_east_down_3d_m=rmse, cdf_probability=cdf_probability, north_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 0])), east_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 1])), down_cdf_absolute_error_m=np.sort(np.abs(ned_error[:, 2])), three_d_cdf_error_m=np.sort(error_3d))
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
    proposed_van_loan_stats = dict(_VAN_LOAN_STATS)
    classical_baseline_rmse = None
    if RUN_CLASSICAL_TEST_BASELINE:
        baseline_leo_simulator = LEODownlinkSimulator(tle_provider, test_leo_klobuchar, seed=TEST_LEO_SEED, tx_epsilon_position_m=LEO_TX_EPSILON_POSITION_M, tx_max_iterations=LEO_TX_MAX_ITERATIONS, minimum_elevation_deg=LEO_MIN_ELEVATION_DEG, prefilter_guard_deg=LEO_PREFILTER_GUARD_DEG, use_ionosphere=USE_IONOSPHERE, use_troposphere=USE_TROPOSPHERE)
        baseline_nav = test_initial_nav.copy()
        baseline_P = test_P0.copy()
        baseline_rows = []
        baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr_gyro[0], test_imr_accel[0])
        for baseline_event in build_exact_fusion_timeline(test_imr_time, test_fusion_time, through_last_fusion=True):
            if isinstance(baseline_event, PropagationSegment):
                imu_index = baseline_event.imu_index
                dt = baseline_event.end_time_gpst_s - baseline_event.start_time_gpst_s
                baseline_last_gyro, baseline_last_accel = compensate_imu(test_imr_gyro[imu_index], test_imr_accel[imu_index])
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
                baseline_model = build_measurement_model(baseline_nav, baseline_measurements, test_lever_arm_b_m)
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
    summary = {'paper_exact': False, 'intentional_project_differences': {'measurement_mode': 'pseudorange_only', 'leo_orbit': 'TLE/SGP4_instead_of_STK_HPOP', 'leo_clock': 'ideal_zero_unpublished_simulated_clock_generator', 'navigation_error_state': '9_state_[delta_p,delta_v,delta_theta]_instead_of_Yan_15_state', 'imu_bias_states': 'removed_by_user_request', 'Fig8_eta': 'removed_by_user_request'}, 'paper_components': {'leo_Eq1_to_Eq5': {'deterministic_iono_tropo_and_light_time': True, 'stochastic_variance_policy': 'unchanged_from_v6_by_user_request', 'variance_terms_used': ['ionosphere', 'troposphere', 'MP_NLOS'], 'MP_NLOS': 'Ref35_Eq18_elevation_only_same_as_v6', 'sampling': 'independent_zero_mean_Gaussian_same_as_v6', 'not_added_to_LEO_variance': ['URA', 'receiver_noise'], 'paper_difference': 'Yan_Eq4_CN0_dependent_MP_NLOS_not_used_in_LEO_branch'}, 'INS_Eq6_to_Eq9': 'user_requested_9_state_ECEF_error_model_[delta_p,delta_v,delta_theta];IMU_bias_states_and_eta_removed;lever_arm_retained', 'Eq7_Fig8_state_update': {'state_order': '[delta_p,delta_v,delta_theta]', 'classical_TC_gain_rows': INS_STATE_DIM, 'masked_CLA_gain_rows': INS_STATE_DIM, 'direct_truth_label_rows': SUPERVISED_STATE_DIM, 'paper_status': 'explicit_user_requested_reduction_from_Yan_15_state_to_9_state'}, 'MaskedCLA_Eq10_to_Eq29': 'implemented_with_v29_mask_semantics_CNN_position_axis_preserved_Eq23_mask_gates_each_LSTM_position_and_current_epoch_attention_cross_epoch_hc_only', 'training_Eq30_to_Eq32': 'Yan_state_MSE_plus_actual_Fig7_Fig8_closed_loop_forward_plus_Latent_KalmanNet_alternating', 'Fig8_eta': {'enabled': False, 'paper_status': 'Yan_Fig8_includes_eta_but_user_requested_removal_in_this_branch'}, 'stability': {'gradient_clipping': {'enabled': True, 'norm_type': 2.0, 'max_norm': float(GRADIENT_CLIP_NORM), 'implementation': 'overflow_safe_float64_global_norm_scaling', 'source': 'SongJgit_KalmanNet4SensorFusion', 'paper_status': 'Yan_does_not_report_gradient_clipping'}, 'tbptt': {'navigation_detach_step': 1, 'recurrent_detach_step': int(RECURRENT_TBPTT_DETACH_STEP), 'optimizer_window_size': int(OPTIMIZER_WINDOW_SIZE), 'forward_state_continuity': True, 'gradient_history_truncated': True, 'source': 'SongJgit_KalmanNet4SensorFusion_detach_step_pattern', 'paper_status': 'Yan_does_not_publish_TBPTT_schedule'}, 'learning_rate': {'active': float(LEARNING_RATE), 'Yan_reported': float(YAN_REPORTED_LEARNING_RATE), 'default_source': 'LatentKalmanNet_and_KalmanNet4SensorFusion_public_configs'}, 'checkpoint_selection': {'untrained_model_eligible': False, 'zero_gain_baseline_eligible': False, 'criterion': 'best_finite_chronological_Data01_validation_recursive_Eq30_after_optimizer_update', 'source': 'KalmanNet_KalmanNet4SensorFusion_LatentKalmanNet_validation_checkpoint_pattern'}, 'Fig10_stability_diagnostic': 'omitted_by_user_request', 'recursive_Data01_diagnostic_before_Data02': 'same_epoch_learned_vs_classical_3D_RMSE_nonblocking', 'exact_75_25_stress_generator': 'not_reconstructed_distribution_parameters_unpublished'}}, 'unavoidable_unpublished_completions': ['offline_Eq10_to_Eq14_history_generator_conventional_TC_reference_pass', 'causal_boundary_timing_for_state_innovation_and_state_residual', 'exact_CNN_tensorization_and_FC_dimensions', 'full_INS_navigation_BPTT_is_KalmanNet_guided_completion_not_published_by_Yan', 'project_short_trajectory_training_with_ground_truth_chunk_initial_state', 'project_short_trajectory_alternating_then_optional_V1_full_sequence_tuning_not_published_by_Yan_or_LatentKalmanNet', 'short_stage_and_V1_epoch_counts_are_explicit_overridable_completion_not_published_by_KalmanNet_or_Yan', 'V1_continues_same_network_and_Adam_states_after_V2_no_reinitialization', 'V1_full_sequence_initial_state_is_ground_truth_navigation_state_not_classical_KF_posterior', 'navigation_state_stop_gradient_each_update_and_LSTM_detach_step_2_from_KalmanNet4SensorFusion_not_published_by_Yan', 'stability_default_lr_1e-3_from_public_LatentKalmanNet_and_KalmanNet4SensorFusion_configs_Yan_reports_1e-2', 'short_stage_chunk_boundary_Yan_lagged_feature_context_is_project_adaptation_no_classical_KF_state_reset', 'scaled_Kaiming_KG_output_weight_and_zero_bias_adapted_from_KalmanNet4SensorFusion_for_direct_raw_innovation_Yan_does_not_publish_FC_initialization', 'global_L2_gradient_clipping_max_norm_1_from_KalmanNet4SensorFusion_Yan_unpublished', 'checkpoint_selection_excludes_untrained_and_zero_gain_baseline_following_KalmanNet_family_pattern', 'short_stage_random_partition_all_Q_batches_reused_theta_then_psi_per_Latent_KalmanNet_Algorithm2', 'exact_Fig8_FC_tensorization_unpublished', 'Data02_Nk_gt_Data01_Nmax_capacity_selection_by_sigma_code_without_truth_Yan_unpublished', 'LEO_variance_intentionally_kept_as_v6_instead_of_Yan_Eq4', 'random_LEO_masking_angle_distribution_from_Ref43', 'learned_gain_covariance_bookkeeping'], 'dataset_protocol': {'training_dataset': str(ROVE_GROUND_TRUTH_PATH.parent), 'test_dataset': str(TEST_ROVE_GROUND_TRUTH_PATH.parent), 'test_used_for_training': False, 'test_used_for_architecture_dimensioning': False, 'training_fusion_epoch_cap': None if MAX_FUSION_EPOCHS is None else int(MAX_FUSION_EPOCHS), 'test_fusion_epoch_cap': None if MAX_TEST_FUSION_EPOCHS is None else int(MAX_TEST_FUSION_EPOCHS), 'shortened_diagnostic_run': bool(MAX_FUSION_EPOCHS is not None or MAX_TEST_FUSION_EPOCHS is not None)}, 'training': {'Data01_total_samples': int(data01_total_sample_count), 'samples': int(training_sample_count), 'validation_samples': int(validation_sample_count), 'validation_fraction': float(VALIDATION_FRACTION), 'validation_mode': 'chronological_tail_holdout_recursive_no_grad', 'Nmax': int(network_nmax), 'training_observed_Nmax': int(nmax), 'Data01_sequence_Nmax': int(data01_sequence_nmax), 'Data02_sequence_Nmax': int(data02_observed_nmax), 'Data02_Nmax_before_capacity': int(data02_nmax_before_capacity), 'Nmax_source': 'Data01_actual_neural_observation_max_only', 'Data02_capacity_policy': {'enabled': True, 'capacity': int(network_nmax), 'selection': 'smallest_sigma_code_m_with_sat_id_tie_break_preserve_original_order_and_clock_observability', 'epochs_capped': int(data02_capacity_selection_epochs), 'measurements_excluded': int(data02_capacity_excluded_measurements), 'uses_test_truth': False, 'paper_status': 'project_completion_Yan_does_not_publish_Nk_gt_training_Nmax_policy'}, 'v2_alternating_epochs': int(KNET_V2_ALTERNATING_EPOCHS), 'v1_finetune_epochs': int(KNET_V1_FINE_TUNE_EPOCHS), 'total_alternating_epochs': int(epochs_ran), 'recursive_Data01_acceptance_gate': data01_acceptance_gate, 'gamma_l2_completion': float(GAMMA_L2), 'gain_head_initialization': {'method': 'scaled_kaiming_uniform_weight_zero_bias', 'scale': float(GAIN_HEAD_INITIAL_SCALE), 'reason': 'direct_linear_gain_times_raw_pseudorange_innovation'}, 'feature_normalization': 'none_raw_inputs', 'lstm_axis_semantics': {'paper_index_k': 'fusion_epoch_temporal_axis', 'paper_index_t': 'padded_feature_position_axis', 'recurrent_state_shape': '[num_layers,B,D,hidden_size]', 'mask_rule': 'Yan_Eq25_retain_previous_position_state_when_M_k_t_is_zero'}, 'alternating_partition': {'psi_representation': 'masked_conv_feature_extractor', 'theta_filter': 'masked_lstm_masked_attention_gain_head', 'source': 'Latent_KalmanNet_Algorithm2_encoder_vs_recurrent_filter_decomposition', 'batch_schedule': 'random_partition_D_into_Q_batches_once_per_short_stage_epoch;theta_all_Q_then_psi_same_all_Q', 'full_dataset_coverage_per_phase': True, 'paper_status': 'batch_schedule_matches_Latent_KalmanNet_Algorithm2;Yan_requires_alternating_but_does_not_publish_exact_module_partition', 'frozen_counterpart_mode': {'filter_phase': 'psi_frozen_eval;theta_train_with_configured_LSTM_dropout', 'representation_phase': 'psi_trainable;theta_frozen;LSTM_kept_train_mode_for_cuDNN_backward_with_dropout_zero;attention_and_gain_head_eval'}, 'warm_start': 'not_applied_because_Yan_has_no_supervised_target_for_intermediate_CNN_features'}, 'sequence_training': {'scope': 'Data01_only', 'method': 'Yan_Fig7_Fig8_actual_closed_loop_forward', 'raw_measurements_fixed': True, 'estimator_dependent_features_recomputed': True, 'navigation_propagation': '9_state_no_bias_no_eta_ECEF_INS_mechanization', 'linearized_Phi_H_training_surrogate': False, 'cross_fusion_navigation_gradient': 'stopped_after_each_learned_update_while_numerical_state_continues', 'cross_fusion_neural_gradient': 'LSTM_h_c_BPTT_detached_every_configured_recurrent_steps', 'training_sequence_length': int(training_sample_count), 'validation_sequence_length': int(validation_sample_count), 'data01_total_sequence_length': int(data01_total_sample_count), 'validation_fraction': float(VALIDATION_FRACTION), 'validation_protocol': 'chronological_tail_holdout_recursive_eval_best_Eq30', 'training_sequence_count': int(trajectory_count), 'sequence_batch_size': int(TRAJECTORY_BATCH_SIZE), 'short_trajectory_truth_restart_diagnostic': short_trajectory_restart_diagnostic, 'sequence_order': 'chronological_within_each_short_trajectory;each_short_trajectory_truth_reinitialized;each_short_stage_epoch_randomly_partitions_all_trajectories_once;same_ordered_batches_reused_for_theta_then_psi', 'sequence_boundary_context': 'each_short_training_chunk_reinitialized_to_ground_truth_navigation_state;stored_causal_IMU_context_used;periodic_truth_restart_is_project_training_adaptation', 'internal_classical_KF_reset': False, 'optimizer_step_inside_sequence': True, 'optimizer_step_schedule': 'one_step_per_four-update_window_with_carried_numerical_state', 'external_filter_recurrence': 'short_stage_recursive_within_each_truth_initialized_chunk;optional_V1_recursive_across_full_training_sequence;validation_and_test_recursive_without_periodic_truth_restarts', 'lagged_state_feature_timing': 'completed_previous_fusion_context_as_in_online_Data02', 'observation_residual_feature_timing': 'Yan_Eq11_Delta_y_k_minus_1;posterior_Delta_y_k_is_stored_after_epoch_k_and_used_at_k_plus_1;same_epoch_use_rejected_as_circular', 'masked_CLA_recurrent_state': 'Yan_Eq25_per_position_h_c_carried_across_fusion_epochs_k;attention_is_current_epoch_position_fusion', 'input_tensorization': 'Yan_Eq15_structure_adapted_to_9_state;fixed24_once;only_residual_and_innovation_blocks_zero_padded_to_fixed_Nmax;no_satellite_token_broadcast', 'synthetic_prior_augmentation': False, 'Data02_used': False}, 'final_actual_closed_loop_Eq30': float(final_training_metrics['eq30']), 'final_actual_closed_loop_total_objective': float(final_training_metrics['objective']), 'final_actual_closed_loop_position_rmse_m': float(final_training_metrics['position_rmse_m']), 'final_clean_teacher_forced_Eq30': float(final_teacher_forced_metrics['eq30']), 'final_clean_teacher_forced_position_rmse_m': float(final_teacher_forced_metrics['position_rmse_m']), 'final_selected_training_stage': selected_training_stage, 'final_selected_validation_eq30': float(best_val_state_loss), 'final_selected_validation_position_rmse_m': float(best_val_position_rmse_m), 'final_validation_recursive_eq30': float(final_validation_metrics['eq30']), 'final_validation_recursive_position_rmse_m': float(final_validation_metrics['position_rmse_m'])}, 'test': {'epochs': int(len(online_rows)), 'causal_warm_start': test_warm_start, 'neural_recurrent_state': 'carried_across_online_fusion_epochs_after_causal_warm_start', 'rmse_ned3d_m': [float(v) for v in rmse], 'imu_bias_states_enabled': False, 'eta_enabled': False, 'divergence_diagnostics': divergence_diagnostics, 'classical_baseline_rmse_ned3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None, 'Data02_observed_Nmax': int(data02_observed_nmax), 'Data02_Nmax_before_capacity': int(data02_nmax_before_capacity), 'network_Nmax_Data01_only': int(network_nmax), 'capacity_selection_epochs': int(data02_capacity_selection_epochs), 'capacity_excluded_measurements': int(data02_capacity_excluded_measurements)}, 'runtime_checks': {'van_loan_fast_calls': int(proposed_van_loan_stats['taylor_calls']), 'van_loan_exact_calls': int(proposed_van_loan_stats['exact_fallback_calls']), 'van_loan_max_validation_phi_abs': float(proposed_van_loan_stats['max_validation_phi_abs']), 'van_loan_max_validation_qd_abs': float(proposed_van_loan_stats['max_validation_qd_abs'])}}
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    test_measurement_counts = np.asarray([row['n_total'] for row in online_rows], dtype=float)
    test_gnss_counts = np.asarray([row['n_gnss'] for row in online_rows], dtype=float)
    test_leo_counts = np.asarray([row['n_leo'] for row in online_rows], dtype=float)
    test_performance_summary = {'epochs': int(len(online_rows)), 'rmse_n_e_d_3d_m': [float(v) for v in rmse], 'error_3d_m': {'median': float(np.median(error_3d)), 'p95': float(np.percentile(error_3d, 95.0)), 'p99': float(np.percentile(error_3d, 99.0)), 'max': float(np.max(error_3d))}, 'measurements_used': {'min': int(np.min(test_measurement_counts)), 'median': float(np.median(test_measurement_counts)), 'max': int(np.max(test_measurement_counts)), 'gnss_median': float(np.median(test_gnss_counts)), 'leo_median': float(np.median(test_leo_counts))}, 'divergence': divergence_diagnostics, 'classical_baseline_rmse_n_e_d_3d_m': [float(v) for v in classical_baseline_rmse] if classical_baseline_rmse is not None else None, 'network_capacity': {'Nmax_Data01_only': int(network_nmax), 'Data02_observed_Nmax': int(data02_observed_nmax), 'Data02_Nmax_before_capacity': int(data02_nmax_before_capacity), 'epochs_capped': int(data02_capacity_selection_epochs), 'measurements_excluded': int(data02_capacity_excluded_measurements), 'selection_policy': 'smallest_sigma_code_m_then_sat_id_tie_break_preserve_original_order'}}
    print('\n=== DATA02 INDEPENDENT PERFORMANCE ===')
    print(json.dumps(test_performance_summary, indent=2))
