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
import os
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


TORCH_FLOAT_DTYPE = torch.float64
TORCH_FLOAT_DTYPE_NAME = 'float64'
torch.set_default_dtype(TORCH_FLOAT_DTYPE)

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
FEATURE_SEMANTICS_VERSION = 2
MEASUREMENT_PREPROCESSING_POLICY = 'current_navigation_state_shared_train_validation_test'
FDE_RECURRENT_POLICY = 'reset_after_non_neural_or_missing_measurement_update'
TRAINING_STRATEGY = (
    'paper_aligned_alternating_filter_then_representation_'
    'v2_independent_sequence_minibatches'
)
ALTERNATING_PHASE_ORDER = ('filter', 'representation')
ALTERNATING_PARAMETER_PARTITION = {
    'representation': ('conv',),
    'filter': ('lstm', 'attention', 'gain_head'),
}
SUPERVISED_LOSS_STATE_ORDER = '[delta_p,delta_v,delta_theta]'
REFERENCE_TRAIN_BATCH_SIZE = 20
REFERENCE_BATCHED_LEARNING_RATE = 1e-3
ALTERNATING_LOSS_POLICY = (
    'shared_end_to_end_state_mse_mean_over_valid_sequence_timesteps_plus_'
    'single_l2_penalty_per_minibatch_window'
)


class TrainingSequencePlan(NamedTuple):
    start: int
    stop: int
    windows: tuple[tuple[int, int], ...]


def plan_training_sequences(
    sample_count: int,
    sequence_length: int,
    optimizer_window_size: int,
) -> tuple[TrainingSequencePlan, ...]:
    """Split training samples into independent sequences and bounded TBPTT windows.

    Reference implementation:
    Song et al., KalmanNet4SensorFusion, ``nclt_split`` and
    ``LitFusionKalmanNet.training_step``.  The dataset first creates independent
    training subsequences, while the trainer then creates shorter optimization
    windows inside each subsequence.  No optimizer window crosses a sequence
    boundary.

    https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/dataset/nclt_dataset.py
    https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/trainer/fusion_trainer.py
    """
    sample_count = int(sample_count)
    sequence_length = int(sequence_length)
    optimizer_window_size = int(optimizer_window_size)
    if sample_count < 0:
        raise ValueError('sample_count must be non-negative')
    if sequence_length <= 0:
        raise ValueError('sequence_length must be positive')
    if optimizer_window_size <= 0:
        raise ValueError('optimizer_window_size must be positive')

    plan = []
    for sequence_start in range(0, sample_count, sequence_length):
        sequence_stop = min(sequence_start + sequence_length, sample_count)
        windows = tuple(
            (
                window_start,
                min(window_start + optimizer_window_size, sequence_stop),
            )
            for window_start in range(
                sequence_start,
                sequence_stop,
                optimizer_window_size,
            )
        )
        plan.append(TrainingSequencePlan(sequence_start, sequence_stop, windows))
    return tuple(plan)


def plan_training_batches(
    sequence_plan: tuple[TrainingSequencePlan, ...],
    requested_batch_size: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[tuple[TrainingSequencePlan, ...], ...]:
    """Shuffle independent sequences and partition them into mini-batches.

    KalmanNet samples multiple independent trajectories for one batch, while
    KalmanNet4SensorFusion lets its DataLoader shuffle non-overlapping training
    subsequences.  This function applies the same dataset-level operation to
    ``TrainingSequencePlan`` objects; it never splits or overlaps a sequence.

    References:
    https://github.com/KalmanNet/KalmanNet_TSP/blob/main/Pipelines/Pipeline_EKF.py
    https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/dataset/nclt_dataset.py
    """
    requested_batch_size = int(requested_batch_size)
    if requested_batch_size <= 0:
        raise ValueError('training batch size must be positive')
    sequence_plan = tuple(sequence_plan)
    if not sequence_plan:
        return ()

    rng = np.random.default_rng(np.random.SeedSequence((int(seed), int(epoch))))
    order = rng.permutation(len(sequence_plan)).tolist()
    shuffled = tuple(sequence_plan[index] for index in order)
    return tuple(
        shuffled[start:start + requested_batch_size]
        for start in range(0, len(shuffled), requested_batch_size)
    )


def iter_training_batch_windows(
    sequence_batch: tuple[TrainingSequencePlan, ...],
):
    """Yield same-offset TBPTT windows with separate state per sequence."""
    sequence_batch = tuple(sequence_batch)
    maximum_window_count = max(
        (len(sequence.windows) for sequence in sequence_batch),
        default=0,
    )
    for window_index in range(maximum_window_count):
        members = tuple(
            (sequence, sequence.windows[window_index])
            for sequence in sequence_batch
            if window_index < len(sequence.windows)
        )
        if members:
            yield members


def batch_window_loss_weights(batch_window_members) -> tuple[float, ...]:
    """Return Eq. (32) weights based on each window's valid time samples."""
    members = tuple(batch_window_members)
    valid_counts = tuple(
        int(window_stop) - int(window_start)
        for _, (window_start, window_stop) in members
    )
    if not valid_counts or any(count <= 0 for count in valid_counts):
        raise ValueError('batch windows must contain positive valid sample counts')
    total_count = sum(valid_counts)
    return tuple(count / total_count for count in valid_counts)


def resolve_run_profile(profile: str) -> dict[str, int | None]:
    """Return explicit smoke or paper-scale limits without silently mixing them."""
    normalized = str(profile).strip().lower()
    if normalized == 'smoke':
        return {
            'max_train_fusion_epochs': 1001,
            'max_test_fusion_epochs': 50,
            'training_epochs': 50,
        }
    if normalized == 'paper':
        return {
            'max_train_fusion_epochs': None,
            'max_test_fusion_epochs': None,
            'training_epochs': 500,
        }
    raise ValueError("MKNET_RUN_PROFILE must be either 'smoke' or 'paper'")


def resolve_alternating_learning_rates(
    base_learning_rate: float,
    environment=None,
) -> dict[str, float]:
    """Resolve the two Algorithm-2 rates from one documented base value."""
    environment = os.environ if environment is None else environment
    rates = {
        'filter': float(environment.get(
            'MKNET_FILTER_LR',
            str(base_learning_rate),
        )),
        'representation': float(environment.get(
            'MKNET_REPRESENTATION_LR',
            str(base_learning_rate),
        )),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in rates.values()):
        raise ValueError('alternating learning rates must be positive finite values')
    return rates


def resolve_batched_training_config(
    sequence_count: int,
    environment=None,
) -> dict:
    """Resolve documented KalmanNet mini-batch and learning-rate defaults.

    The original KalmanNet configuration uses a batch of 20 independent
    trajectories and Adam with learning rate 1e-3.  KalmanNet4SensorFusion and
    the published Latent-KalmanNet experiments likewise use mini-batch mean MSE
    with Adam learning rates in the 1e-3 regime.  No linear learning-rate
    scaling rule is applied because none is specified by these references.
    """
    environment = os.environ if environment is None else environment
    sequence_count = int(sequence_count)
    if sequence_count <= 0:
        raise ValueError('at least one independent training sequence is required')
    requested_batch_size = int(environment.get(
        'MKNET_TRAIN_BATCH_SIZE',
        str(REFERENCE_TRAIN_BATCH_SIZE),
    ))
    if requested_batch_size <= 0:
        raise ValueError('MKNET_TRAIN_BATCH_SIZE must be positive')
    learning_rates = resolve_alternating_learning_rates(
        REFERENCE_BATCHED_LEARNING_RATE,
        environment,
    )
    return {
        'requested_batch_size': requested_batch_size,
        'effective_batch_size': min(requested_batch_size, sequence_count),
        'learning_rates': learning_rates,
    }


def batched_training_checkpoint_contract(
    batched_config: dict,
    *,
    sequence_count: int,
    gamma: float,
) -> dict:
    """Persist the evidence-backed batch, trajectory, LR, and loss semantics."""
    sequence_count = int(sequence_count)
    gamma = float(gamma)
    if sequence_count <= 0:
        raise ValueError('sequence_count must be positive')
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError('gamma must be finite and non-negative')
    return {
        # Data01 is one physical recording.  Its non-overlapping subsequences
        # are independent recurrent training units, not falsely reported as
        # independent physical trajectories.
        'physical_training_trajectory_count': 1,
        'recurrent_training_sequence_count': sequence_count,
        'training_batch_size_requested': int(
            batched_config['requested_batch_size']
        ),
        'training_batch_size_effective': int(
            batched_config['effective_batch_size']
        ),
        'training_sequence_shuffle': 'epoch_seeded_without_replacement',
        'learning_rate_policy': (
            'KalmanNet_batched_reference_default_1e-3_no_linear_scaling'
        ),
        'alternating_loss_policy': ALTERNATING_LOSS_POLICY,
        'alternating_loss_data_term': (
            'mean_end_to_end_state_squared_error_over_valid_batch_timesteps'
        ),
        'alternating_loss_regularization': (
            'gamma_times_complete_network_l2_once_per_minibatch_window'
        ),
        'alternating_loss_regularization_coefficient': gamma,
        'alternating_loss_regularization_coefficient_source': (
            'existing_project_setting_not_numerically_published_by_Yan_Eq32'
        ),
    }


def alternating_checkpoint_contract(learning_rates: dict[str, float]) -> dict:
    """Return metadata that prevents joint and alternating checkpoints mixing."""
    expected_keys = set(ALTERNATING_PHASE_ORDER)
    if set(learning_rates) != expected_keys:
        raise ValueError(
            'alternating learning_rates must contain exactly '
            f'{sorted(expected_keys)}'
        )
    normalized_rates = {
        phase: float(learning_rates[phase])
        for phase in ALTERNATING_PHASE_ORDER
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in normalized_rates.values()
    ):
        raise ValueError('alternating learning rates must be positive finite values')
    return {
        'alternating_optimization': True,
        'alternating_phase_order': list(ALTERNATING_PHASE_ORDER),
        'alternating_parameter_partition': {
            key: list(value)
            for key, value in ALTERNATING_PARAMETER_PARTITION.items()
        },
        'learning_rates': normalized_rates,
    }

# Yan Sec. II-D does not publish a numerical false-alarm probability.
# Ref. [33], which Yan cites for the DIA/statistical-testing details, uses
# alpha = 1e-3 in its main positioning-safety experiments (Figs. 4-6).
# This is therefore a reference-backed reproduction choice, not a value
# claimed to be explicitly reported by Yan et al.
FDE_SIGNIFICANCE_ALPHA = 1e-3

# Yan Fig. 20 places both horizontal and vertical alert-limit boundaries at
# 30 m.  The body text discusses AL but does not print these numeric values;
# they are read directly from the published Stanford diagrams.
PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M = 30.0
PAPER_FIG20_VERTICAL_ALERT_LIMIT_M = 30.0
PAPER_FIG20_REFERENCE_HORIZONTAL_NO_PERCENT = 86.950
PAPER_FIG20_REFERENCE_VERTICAL_NO_PERCENT = 98.189

try:
    from fde_dia_v2_phase6 import DIAAdapter, FDEDetector, FDEExcluder, FDEIdentifier, IntegrityMonitor
except ModuleNotFoundError:
    DIAAdapter = None
    FDEDetector = None
    FDEExcluder = None
    FDEIdentifier = None
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
# Yan Eq. (7): [delta_p, delta_v, delta_theta, b_a, b_g].
INS_STATE_DIM = 15
# Ground truth provides position, velocity and attitude only.
DIRECT_STATE_LABEL_DIM = 9
_BIAS_ABLATION_MASKS = {
    'none':       [1.0] * 9 + [0.0] * 3 + [0.0] * 3,
    'accel_only': [1.0] * 9 + [1.0] * 3 + [0.0] * 3,
    'gyro_only':  [1.0] * 9 + [0.0] * 3 + [1.0] * 3,
    'both':       [1.0] * 9 + [1.0] * 3 + [1.0] * 3,
}
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
    # Current estimated IMU biases. These are the b_a and b_g states in
    # Yan Eq. (7), and are subtracted from the IMU measurements in Eq. (8).
    accel_bias_body_mps2: Array
    gyro_bias_body_radps: Array


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
    # Yan Eq. (8), restricted to the bias states present in Eq. (7).
    corrected_gyro = np.asarray(angular_rate_body_radps, dtype=float).reshape(3) - nav.gyro_bias_body_radps
    corrected_accel = np.asarray(specific_force_body_mps2, dtype=float).reshape(3) - nav.accel_bias_body_mps2
    C1 = _project_rotation(_earth_rotation(dt) @ C0 @ _so3_exp(corrected_gyro * dt))
    Cmid = 0.5 * (C0 + C1)
    r_e = np.asarray(nav.position_ecef_m, dtype=float).reshape(3)
    acceleration = Cmid @ corrected_accel + _gravity_j2(r_e) - np.array([-EARTH_ROTATION_RATE_RADPS * (EARTH_ROTATION_RATE_RADPS * r_e[0]), EARTH_ROTATION_RATE_RADPS * (-EARTH_ROTATION_RATE_RADPS * r_e[1]), 0.0]) - 2.0 * np.array([-EARTH_ROTATION_RATE_RADPS * nav.velocity_ecef_mps[1], EARTH_ROTATION_RATE_RADPS * nav.velocity_ecef_mps[0], 0.0])
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return NavState(position, velocity, C1, nav.accel_bias_body_mps2.copy(), nav.gyro_bias_body_radps.copy())


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

# 15-state error dynamics/KF.


# ECEF error dynamics for Yan Eq. (7): [delta_p, delta_v, delta_theta, b_a, b_g].
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
    corrected_specific_force = np.asarray(specific_force_body_mps2, dtype=float).reshape(3) - nav.accel_bias_body_mps2
    force_e = C @ corrected_specific_force
    x, y, z = force_e
    F[3:6, 6:9] = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    # Bias couplings follow the Eq. (8) compensation and the existing error-state sign convention.
    F[3:6, 9:12] = -C
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    F[6:9, 12:15] = C
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
    # Predicted innovation and its linearized model after removal of the
    # receiver-clock nuisance modes.  FDE phase 2 uses these quantities
    # before any learned Kalman gain/state update.
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]

    # Pre-projection quantities are retained for auditability and for the
    # fault-hypothesis construction required in later FDE/DIA phases.  They
    # are not used directly by the phase-2 global test because receiver clock
    # states are not part of Yan Eq. (7)'s 15-state vector in this project.
    raw_innovation: Array
    H_raw: Array
    R_raw: Array
    clock_projector: Array
    constellations: tuple[str, ...]


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


def prepare_fusion_measurements(
    gnss_processor,
    gnss_epoch,
    leo_measurements,
    nav: NavState,
    lever_arm_b_m: Array,
    *,
    capacity: int | None,
) -> tuple[PseudoObs, ...]:
    """Apply one state-dependent measurement policy in every data split.

    Satellite transmit time, elevation filtering, atmospheric corrections and
    measurement variances all depend on the supplied navigation state.  Keeping
    this boundary shared prevents training from using a frozen INS-only
    preprocessing trajectory while validation/test use the online estimate.
    """
    measurements = clock_observable(
        tuple(gnss_processor.prepare_epoch(gnss_epoch, nav, lever_arm_b_m))
        + tuple(leo_measurements)
    )
    measurements = tuple(sorted(
        measurements,
        key=lambda measurement: (
            str(measurement.constellation),
            str(measurement.sat_id),
        ),
    ))
    if capacity is not None and len(measurements) > int(capacity):
        measurements = select_measurements(measurements, int(capacity))
    return tuple(measurements)


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
        empty_h = np.zeros((0, INS_STATE_DIM))
        empty_r = np.zeros((0, 0))
        return TCModel(
            np.empty(0), empty_h, empty_r, (),
            np.empty(0), empty_h.copy(), empty_r.copy(), np.zeros((0, 0)), (),
        )
    raw_innovation = np.empty(n)
    H_raw = np.zeros((n, INS_STATE_DIM))
    variances = np.empty(n)
    lever_e = nav.body_to_ecef_dcm @ np.asarray(lever_arm_b_m).reshape(3)
    antenna_position = nav.position_ecef_m + lever_e
    x, y, z = lever_e
    lever_skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    sat_ids: list[str] = []
    constellations: list[str] = []
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
        constellations.append(m.constellation)
    projector = _clock_projection(measurements, variances)
    innovation = projector @ raw_innovation
    H = projector @ H_raw
    R_raw = np.diag(variances)
    R = projector @ R_raw @ projector.T
    R = 0.5 * (R + R.T)
    return TCModel(
        innovation, H, R, tuple(sat_ids),
        raw_innovation.copy(), H_raw.copy(), R_raw.copy(), projector.copy(),
        tuple(constellations),
    )


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
# Integrity monitoring / FDE.
INTEGRITY = IntegrityMonitor(ecef_llh, ecef_to_ned) if IntegrityMonitor is not None else None
FDE_DETECTOR = FDEDetector(FDE_SIGNIFICANCE_ALPHA) if FDEDetector is not None else None
FDE_IDENTIFIER = FDEIdentifier() if FDEIdentifier is not None else None
FDE_EXCLUDER = FDEExcluder() if FDEExcluder is not None else None
DIA_ADAPTER = DIAAdapter() if DIAAdapter is not None else None


# Classical TC/KF update.
def kf_update(P: Array, innovation: Array, H: Array, R: Array):
    PHt = P @ H.T
    S = H @ PHt + R
    S = 0.5 * (S + S.T)
    K = PHt @ np.linalg.pinv(S, rcond=1e-12)
    if not np.all(np.isfinite(K)):
        raise FloatingPointError('non-finite 15-state classical Kalman gain')
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
    out = NavState(
        nav.position_ecef_m.copy(),
        nav.velocity_ecef_mps.copy(),
        nav.body_to_ecef_dcm.copy(),
        nav.accel_bias_body_mps2.copy(),
        nav.gyro_bias_body_radps.copy(),
    )
    out.position_ecef_m += dx[0:3]
    out.velocity_ecef_mps += dx[3:6]
    out.body_to_ecef_dcm = _project_rotation(_so3_exp(ATTITUDE_FEEDBACK_SIGN * dx[6:9]) @ out.body_to_ecef_dcm)
    out.accel_bias_body_mps2 += dx[9:12]
    out.gyro_bias_body_radps += dx[12:15]
    return out


# Differentiable Data01 state.
class TorchNavState(NamedTuple):
    position_ecef_m: torch.Tensor
    velocity_ecef_mps: torch.Tensor
    body_to_ecef_dcm: torch.Tensor
    accel_bias_body_mps2: torch.Tensor
    gyro_bias_body_radps: torch.Tensor


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
    corrected_gyro = angular_rate_body_radps.reshape(3) - nav.gyro_bias_body_radps
    corrected_accel = specific_force_body_mps2.reshape(3) - nav.accel_bias_body_mps2
    earth_rotvec = C0.new_tensor(-OMEGA_IE_E * dt)
    C1 = _torch_so3_exp(earth_rotvec) @ C0 @ _torch_so3_exp(corrected_gyro * dt)
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
    acceleration = Cmid @ corrected_accel + gravity - torch.linalg.cross(omega, torch.linalg.cross(omega, r_e)) - 2.0 * torch.linalg.cross(omega, nav.velocity_ecef_mps)
    velocity = nav.velocity_ecef_mps + acceleration * dt
    position = nav.position_ecef_m + 0.5 * (nav.velocity_ecef_mps + velocity) * dt
    return TorchNavState(position, velocity, C1, nav.accel_bias_body_mps2, nav.gyro_bias_body_radps)


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
FIXED_FEATURE_DIM = 6 + 2 * INS_STATE_DIM   # 36 for Yan Eq. (7) 15-state navigation
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
        # Stability initialization adapted from the reference KalmanNet GRU:
        # input weights use Xavier, recurrent weights are orthogonal, and all
        # recurrent biases start at zero.  This changes initialization only;
        # the Yan CNN-LSTM-attention-gain architecture is unchanged.
        for parameter_name, parameter in self.lstm.named_parameters():
            if 'weight_ih' in parameter_name:
                nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in parameter_name:
                nn.init.orthogonal_(parameter)
            elif 'bias' in parameter_name:
                nn.init.zeros_(parameter)

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

        # Stable initialization only; Eqs. (26)-(29) are unchanged.
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.v.weight)

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

        # Near-zero symmetric Kalman-gain initialization.  Unlike exact-zero
        # initialization, this preserves a gradient path into the upstream CLA
        # blocks from the first optimizer step while keeping the initial learned
        # correction small.  The Yan CNN-LSTM-attention-gain architecture is
        # unchanged.
        nn.init.xavier_uniform_(self.gain_head.weight, gain=1e-2)
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


def build_alternating_parameter_groups(model: MaskedCLA) -> dict[str, list[nn.Parameter]]:
    """Build the two Algorithm-2 parameter blocks and validate the partition."""
    groups = {
        'representation': list(model.conv.parameters()),
        'filter': [
            *model.lstm.parameters(),
            *model.attention.parameters(),
            *model.gain_head.parameters(),
        ],
    }
    representation_ids = {id(parameter) for parameter in groups['representation']}
    filter_ids = {id(parameter) for parameter in groups['filter']}
    all_ids = {id(parameter) for parameter in model.parameters()}

    if not representation_ids or not filter_ids:
        raise RuntimeError('alternating parameter blocks must both be non-empty')
    overlap = representation_ids.intersection(filter_ids)
    if overlap:
        raise RuntimeError('alternating parameter blocks overlap')
    covered = representation_ids.union(filter_ids)
    if covered != all_ids:
        missing_names = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) not in covered
        ]
        raise RuntimeError(
            'alternating parameter blocks do not cover the complete model: '
            + ', '.join(missing_names)
        )
    return groups


def configure_alternating_phase(
    model: MaskedCLA,
    phase: str,
    parameter_groups: dict[str, list[nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze one block and train the other without severing input gradients."""
    if phase not in ALTERNATING_PHASE_ORDER:
        raise ValueError(f'unsupported alternating phase: {phase!r}')

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    if phase == 'filter':
        active_parameters = parameter_groups['filter']
        inactive_parameters = parameter_groups['representation']
        model.lstm.train()
        model.attention.train()
        model.gain_head.train()
    else:
        active_parameters = parameter_groups['representation']
        inactive_parameters = parameter_groups['filter']
        # The frozen filter remains in evaluation mode.  Autograd still
        # propagates through it to the active convolutional representation,
        # while recurrent dropout cannot inject phase-specific noise.
        model.conv.train()

    for parameter in active_parameters:
        parameter.requires_grad_(True)
    return active_parameters, inactive_parameters


@torch.no_grad()
def snapshot_parameter_values(parameters) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def assert_parameter_gradients_absent(parameters, *, phase: str) -> None:
    unexpected = [
        index
        for index, parameter in enumerate(parameters)
        if parameter.grad is not None
    ]
    if unexpected:
        raise RuntimeError(
            f'inactive parameters received gradients during {phase} phase: '
            f'indices={unexpected}'
        )


@torch.no_grad()
def assert_parameter_values_unchanged(
    before: list[torch.Tensor],
    parameters,
    *,
    phase: str,
) -> None:
    parameter_list = list(parameters)
    if len(before) != len(parameter_list):
        raise RuntimeError(f'inactive parameter count changed during {phase} phase')
    changed = [
        index
        for index, (old, parameter) in enumerate(zip(before, parameter_list))
        if not torch.equal(old, parameter.detach())
    ]
    if changed:
        raise RuntimeError(
            f'inactive parameters changed during {phase} phase: indices={changed}'
        )


def documented_alternating_objective(
    batch_data_loss: torch.Tensor,
    all_parameters,
    *,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the shared alternating objective from the two cited papers.

    Yan et al. Eq. (32) uses mean end-to-end state squared error plus one L2
    penalty on the complete parameter set.  Latent-KalmanNet Eqs. (13)-(15)
    and Algorithm 2 use that same end-to-end objective for both alternating
    phases; the inactive block is frozen rather than assigned a different
    scientific loss.  Consequently the scalar regularizer includes the whole
    network, while autograd updates only parameters whose ``requires_grad`` is
    true in the current phase.
    """
    if not isinstance(batch_data_loss, torch.Tensor) or batch_data_loss.ndim != 0:
        raise ValueError('batch_data_loss must be a scalar tensor')
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError('gamma must be finite and non-negative')
    parameters = tuple(all_parameters)
    if not parameters:
        raise ValueError('the alternating objective requires network parameters')
    regularization = batch_data_loss.new_zeros(())
    for parameter in parameters:
        values = parameter.to(dtype=batch_data_loss.dtype)
        regularization = regularization + torch.sum(values * values)
    regularization = gamma * regularization
    return batch_data_loss + regularization, regularization


def run_alternating_cycle(
    train_block,
    training_batches=None,
) -> dict[str, dict]:
    """Run one Algorithm-2 cycle with one shared random batch partition."""
    if training_batches is None:
        return {
            phase: train_block(phase)
            for phase in ALTERNATING_PHASE_ORDER
        }
    return {
        phase: train_block(phase, training_batches)
        for phase in ALTERNATING_PHASE_ORDER
    }


def _load_test_only_model(
    checkpoint_path: Path | str,
    device: str | torch.device,
    *,
    expected_nmax: int | None = None,
    expected_bias_ablation_mode: str | None = None,
) -> tuple[MaskedCLA, dict]:
    """Load and validate the saved Kaggle model without creating an optimizer step."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'test-only checkpoint not found: {path}. '
            'Set MKNET_CHECKPOINT_PATH to the Kaggle best_model.pt path.'
        )

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with older Kaggle PyTorch releases which predate the
        # weights_only argument.  The path must still be a trusted checkpoint.
        payload = torch.load(path, map_location=device)

    if not isinstance(payload, dict):
        raise TypeError('test-only checkpoint must contain a dictionary payload')
    required = {
        'model_state_dict',
        'nmax',
        'state_order',
        'architecture',
        'feature_normalization',
        'feature_semantics_version',
        'measurement_preprocessing_policy',
        'fde_recurrent_policy',
        'bias_ablation_mode',
        'supervised_loss_state_order',
        'training_strategy',
        'alternating_optimization',
        'alternating_phase_order',
        'alternating_parameter_partition',
        'learning_rates',
        'torch_float_dtype',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f'test-only checkpoint is missing fields: {missing}')

    checkpoint_nmax = int(payload['nmax'])
    if checkpoint_nmax <= 0:
        raise ValueError('test-only checkpoint nmax must be positive')
    if expected_nmax is not None and checkpoint_nmax != int(expected_nmax):
        raise ValueError(
            f'checkpoint nmax={checkpoint_nmax} does not match '
            f'Data01 feature packing nmax={int(expected_nmax)}'
        )
    if payload['architecture'] != 'Yan_masked_CNN24_k3_LSTM5x64_attention_FC_KG':
        raise ValueError(f"unsupported checkpoint architecture: {payload['architecture']!r}")
    if payload['feature_normalization'] != FEATURE_NORMALIZATION_MODE:
        raise ValueError(
            'checkpoint feature normalization is incompatible with this test-only script'
        )
    if int(payload['feature_semantics_version']) != FEATURE_SEMANTICS_VERSION:
        raise ValueError('checkpoint feature semantics are incompatible with this script')
    if payload['measurement_preprocessing_policy'] != MEASUREMENT_PREPROCESSING_POLICY:
        raise ValueError('checkpoint measurement preprocessing policy is incompatible')
    if payload['fde_recurrent_policy'] != FDE_RECURRENT_POLICY:
        raise ValueError('checkpoint FDE recurrent-state policy is incompatible')
    if payload['supervised_loss_state_order'] != SUPERVISED_LOSS_STATE_ORDER:
        raise ValueError('checkpoint supervised state order is incompatible')
    if payload['training_strategy'] != TRAINING_STRATEGY:
        raise ValueError('checkpoint training strategy is incompatible')
    if payload['alternating_optimization'] is not True:
        raise ValueError('checkpoint does not use alternating optimization')
    if tuple(payload['alternating_phase_order']) != ALTERNATING_PHASE_ORDER:
        raise ValueError('checkpoint alternating phase order is incompatible')
    expected_partition = {
        key: list(value)
        for key, value in ALTERNATING_PARAMETER_PARTITION.items()
    }
    if payload['alternating_parameter_partition'] != expected_partition:
        raise ValueError('checkpoint alternating parameter partition is incompatible')
    checkpoint_rates = payload['learning_rates']
    if not isinstance(checkpoint_rates, dict) or set(checkpoint_rates) != set(ALTERNATING_PHASE_ORDER):
        raise ValueError('checkpoint alternating learning rates are incompatible')
    if any(
        not math.isfinite(float(checkpoint_rates[phase]))
        or float(checkpoint_rates[phase]) <= 0.0
        for phase in ALTERNATING_PHASE_ORDER
    ):
        raise ValueError('checkpoint alternating learning rates are incompatible')
    if payload['torch_float_dtype'] != TORCH_FLOAT_DTYPE_NAME:
        raise ValueError(
            f"checkpoint torch_float_dtype={payload['torch_float_dtype']!r} is incompatible; "
            f"this script requires {TORCH_FLOAT_DTYPE_NAME!r}"
        )
    if (
        expected_bias_ablation_mode is not None
        and payload['bias_ablation_mode'] != str(expected_bias_ablation_mode)
    ):
        raise ValueError(
            f"checkpoint bias_ablation_mode={payload['bias_ablation_mode']!r} does not "
            f"match requested mode={str(expected_bias_ablation_mode)!r}"
        )
    if payload['state_order'] != '[delta_p,delta_v,delta_theta,b_a,b_g]':
        raise ValueError('checkpoint state order is incompatible with this test-only script')

    model = MaskedCLA(nmax=checkpoint_nmax, dropout=0.2).to(
        device=device,
        dtype=TORCH_FLOAT_DTYPE,
    )
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    return model, payload


# Kalman correction K_k * innovation.
# For b_a and b_g this is the increment applied to the predicted bias state;
# the stored NavState bias remains the absolute/current bias estimate.
def state_update(network_output: CLAOutput, innovation: torch.Tensor) -> torch.Tensor:
    # The CLA, innovation, and navigation path all use float64 so the learned
    # correction does not cross a lower-precision boundary.
    gain = network_output.kalman_gain.to(dtype=innovation.dtype)
    return torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def clip_grad_norm_float64_(
    parameters,
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients using a float64 total norm and return the pre-clip norm."""
    parameter_list = list(parameters)
    gradients = [
        parameter.grad
        for parameter in parameter_list
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=TORCH_FLOAT_DTYPE)

    gradient_norms = []
    for gradient in gradients:
        detached = gradient.detach()
        if not bool(torch.all(torch.isfinite(detached)).cpu()):
            raise RuntimeError('gradient contains non-finite values before clipping')
        gradient_norms.append(
            torch.linalg.vector_norm(detached.to(dtype=TORCH_FLOAT_DTYPE))
        )

    total_norm = torch.linalg.vector_norm(torch.stack(gradient_norms))
    if not bool(torch.isfinite(total_norm).cpu()):
        raise RuntimeError('float64 total gradient norm is non-finite')

    max_norm_tensor = total_norm.new_tensor(float(max_norm))
    clip_coefficient = torch.clamp(
        max_norm_tensor / (total_norm + 1e-6),
        max=1.0,
    )
    for gradient in gradients:
        gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


def recurrent_state_after_update(update_kind: str, recurrent_state):
    """Keep recurrence only across consecutive neural measurement updates."""
    return recurrent_state if str(update_kind) == 'neural_no_fault' else None


def summarize_branch_rmse(ned_error: Array, update_modes: Array) -> dict[str, dict]:
    """Report Data02 errors without mixing neural, DIA, and INS-only branches."""
    errors = np.asarray(ned_error, dtype=float).reshape(-1, 3)
    modes = np.asarray(update_modes, dtype=str).reshape(-1)
    if errors.shape[0] != modes.size:
        raise ValueError('ned_error and update_modes must contain the same number of epochs')

    summary = {}
    for mode in ('neural_no_fault', 'dia_fault', 'unresolved_ins_only'):
        selected = errors[modes == mode]
        if selected.size == 0:
            summary[mode] = {'count': 0, 'rmse_ned3d_m': None}
            continue
        component_rmse = np.sqrt(np.mean(selected ** 2, axis=0))
        rmse_3d = math.sqrt(float(np.mean(np.sum(selected ** 2, axis=1))))
        summary[mode] = {
            'count': int(selected.shape[0]),
            'rmse_ned3d_m': [float(v) for v in np.append(component_rmse, rmse_3d)],
        }
    return summary


def _numpy_nav_from_torch(nav_state: TorchNavState) -> NavState:
    """Detach only the nondifferentiable measurement-preparation boundary."""
    return NavState(
        nav_state.position_ecef_m.detach().cpu().numpy().astype(float).copy(),
        nav_state.velocity_ecef_mps.detach().cpu().numpy().astype(float).copy(),
        nav_state.body_to_ecef_dcm.detach().cpu().numpy().astype(float).copy(),
        nav_state.accel_bias_body_mps2.detach().cpu().numpy().astype(float).copy(),
        nav_state.gyro_bias_body_radps.detach().cpu().numpy().astype(float).copy(),
    )


def _apply_bias_ablation_numpy(correction: Array, mode: str) -> Array:
    """Apply the training-time bias ablation contract during online inference."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _BIAS_ABLATION_MASKS:
        raise ValueError(
            f'bias ablation mode must be one of {tuple(_BIAS_ABLATION_MASKS)}, '
            f'got {normalized_mode!r}'
        )
    values = np.asarray(correction, dtype=float).reshape(INS_STATE_DIM)
    mask = np.asarray(_BIAS_ABLATION_MASKS[normalized_mode], dtype=float)
    return values * mask


# Detach every tensor carried across a TBPTT boundary without changing its
# forward value.  This mirrors the reference model's _detach() lifecycle for
# the recurrent state and recursive filter history.
def _detach_rollout_state(
    nav_state: TorchNavState,
    previous_context: dict,
    feature_accel: torch.Tensor,
    feature_gyro: torch.Tensor,
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None,
):
    detached_nav = TorchNavState(*(value.detach() for value in nav_state))
    detached_context = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in previous_context.items()
    }
    detached_recurrent = (
        None
        if recurrent_state is None
        else tuple(value.detach() for value in recurrent_state)
    )
    return (
        detached_nav,
        detached_context,
        feature_accel.detach(),
        feature_gyro.detach(),
        detached_recurrent,
    )


# Compact failure-only tensor summary for forward divergence diagnostics.
def _tensor_finite_stats(value: torch.Tensor) -> dict:
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    return {
        'shape': list(tensor.shape),
        'element_count': int(tensor.numel()),
        'nonfinite_count': int((~finite).sum().cpu()),
        'max_abs_finite': (
            float(torch.max(torch.abs(finite_values)).cpu())
            if finite_values.numel()
            else 0.0
        ),
    }


# Pipeline
if __name__ == '__main__':
    if FDE_DETECTOR is None or FDE_IDENTIFIER is None or FDE_EXCLUDER is None or DIA_ADAPTER is None:
        raise ModuleNotFoundError(
            'Phase-6 FDE/DIA requires fde_dia_v2_phase6.py next to this script'
        )
    # Run mode.
    #
    # TEST_ONLY = False:
    #   Ignore any existing checkpoint, initialize MaskedCLA from scratch,
    #   train on Data01, save a new best_model.pt, reload that saved checkpoint,
    #   then run the Data02 test.
    #
    # TEST_ONLY = True:
    #   Skip training completely, require an existing checkpoint, load it,
    #   and run the Data02 test without overwriting the checkpoint.
    TEST_ONLY = False

    OUTPUT_DIR = Path('/kaggle/working/direct_test')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GENERATED_CHECKPOINT_PATH = OUTPUT_DIR / 'best_model.pt'
    CHECKPOINT_PATH = Path(os.environ.get(
        'MKNET_CHECKPOINT_PATH',
        str(GENERATED_CHECKPOINT_PATH),
    ))
    RUN_PROFILE = os.environ.get('MKNET_RUN_PROFILE', 'smoke').strip().lower()
    RUN_LIMITS = resolve_run_profile(RUN_PROFILE)
    MAX_FUSION_EPOCHS = RUN_LIMITS['max_train_fusion_epochs']
    MAX_TEST_FUSION_EPOCHS = RUN_LIMITS['max_test_fusion_epochs']
    TRAINING_EPOCHS = int(os.environ.get(
        'MKNET_TRAINING_EPOCHS',
        str(RUN_LIMITS['training_epochs']),
    ))
    if TRAINING_EPOCHS <= 0 and not TEST_ONLY:
        raise ValueError('MKNET_TRAINING_EPOCHS must be positive during fresh training')
    # Yan et al. Table III / Fig. 15 report 0.01.  Keep that paper value as
    # provenance, while the actual batched optimizer rates are resolved after
    # the independent training sequences are known.  The documented batched
    # KalmanNet default is 1e-3 (resolve_batched_training_config).
    PAPER_INITIAL_LEARNING_RATE = 1e-2
    # KalmanNet4SensorFusion separates the 50-step training subsequence from
    # its 4-step optimizer window and 2-step detach interval.  Its validation
    # and test datasets deliberately keep seq_len=None (full trajectories).
    # Source: configs/nclt/fusion/wheel_gpsfusion_origin.py, lines 14-17, 43-56.
    # https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/configs/nclt/fusion/wheel_gpsfusion_origin.py
    TRAINING_SEQUENCE_LENGTH = int(os.environ.get(
        'MKNET_TRAINING_SEQUENCE_LENGTH',
        '50',
    ))
    if TRAINING_SEQUENCE_LENGTH <= 0:
        raise ValueError('MKNET_TRAINING_SEQUENCE_LENGTH must be positive')
    OPTIMIZER_WINDOW_SIZE = 4
    TBPTT_DETACH_STEP = 2
    GRADIENT_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 6
    VALIDATION_FRACTION = 0.20
    # Yan Eq. (32) publishes the L2 coefficient symbol gamma but does not give
    # its numerical value.  Retain the project's existing 1e-5 explicitly as
    # an unpublished reproduction setting; do not label it paper-exact.
    GAMMA_L2 = 1e-5
    SEED = 0
    TEST_LEO_SEED = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Diagnostic A/B/C/D bias-gradient ablation.  The project default remains
    # 'none' because direct truth is available only for the first nine states.
    # This is an explicit project deviation from Yan Eq. (7), not paper-exact
    # 15-state supervision.  The selected mode is saved in every checkpoint.
    # none       -> learned delta_ba=0, delta_bg=0
    # accel_only -> learned delta_ba active, delta_bg=0
    # gyro_only  -> learned delta_ba=0, delta_bg active
    # both       -> learned delta_ba and delta_bg active
    BIAS_ABLATION_MODE = os.environ.get('BIAS_ABLATION_MODE', 'none').strip().lower()
    if BIAS_ABLATION_MODE not in _BIAS_ABLATION_MASKS:
        raise ValueError(
            f'BIAS_ABLATION_MODE must be one of {tuple(_BIAS_ABLATION_MASKS)}, '
            f'got {BIAS_ABLATION_MODE!r}'
        )
    BIAS_ABLATION_MASK_T = torch.tensor(
        _BIAS_ABLATION_MASKS[BIAS_ABLATION_MODE],
        dtype=torch.float64,
        device=DEVICE,
    )
    print(f'Bias-gradient ablation mode: {BIAS_ABLATION_MODE}')
    print(f'Run profile: {RUN_PROFILE}')
    if TEST_ONLY:
        print('Run mode: TEST ONLY')
        print(f'Checkpoint to load: {CHECKPOINT_PATH}')
    else:
        print('Run mode: TRAIN FROM SCRATCH + TEST')
        print('Existing checkpoints will not be inspected or loaded.')
        print(f'New checkpoint will be saved to: {GENERATED_CHECKPOINT_PATH}')
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
    initial_nav = NavState(initial_imu_truth_position.copy(), imu_velocity[0].copy(), C_b_e, np.zeros(3), np.zeros(3))
    P0 = np.diag(np.concatenate((
        imu_model_values['ISDV_Pos'],
        imu_model_values['ISDV_Vel'],
        imu_model_values['ISDV_Att'] * (np.pi / 180.0),
        imu_model_values['ISDV_AccelBias'],
        imu_model_values['ISDV_GyrosBias'] * (np.pi / 180.0),
    )) ** 2)
    Qc = np.diag(np.concatenate((
        imu_model_values['PNSD_Pos'],
        imu_model_values['PNSD_Vel'],
        imu_model_values['PNSD_Att'] * (np.pi / 180.0),
        imu_model_values['PNSD_AccelBias'],
        imu_model_values['PNSD_GyrosBias'] * (np.pi / 180.0),
    )) ** 2)
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
    # Static Data01 preprocessing.
    #
    # A full classical-KF trajectory is not needed for neural training.  Build
    # only network-independent fusion records here, plus one classical update at
    # the first usable fusion epoch to initialize the recursive KalmanNet state.
    # Subsequent innovations/residuals/state-difference features are generated
    # online inside _rollout() from the network's own navigation trajectory.
    preprocessing_nav = NavState(
        initial_nav.position_ecef_m.copy(),
        initial_nav.velocity_ecef_mps.copy(),
        initial_nav.body_to_ecef_dcm.copy(),
        initial_nav.accel_bias_body_mps2.copy(),
        initial_nav.gyro_bias_body_radps.copy(),
    )
    warm_start_covariance = P0.copy()
    training_warm_start = None
    fusion_records = []
    interval_segments_since_previous_usable_fusion = []
    last_gyro = np.asarray(imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(imr_acceleration_body_mps2[0], dtype=float).reshape(3)

    for event_kind, event_start, event_end, event_index in fusion_timeline(
        imr_time,
        fusion_time,
        through_last_fusion=True,
    ):
        if event_kind == 'imu':
            imu_index = event_index
            dt = event_end - event_start
            last_gyro = np.asarray(
                imr_angular_rate_body_radps[imu_index],
                dtype=float,
            ).reshape(3)
            last_accel = np.asarray(
                imr_acceleration_body_mps2[imu_index],
                dtype=float,
            ).reshape(3)

            # INS-only bootstrap reference.  It determines which raw epochs are
            # initially usable and supplies the one-time classical warm start;
            # it is not used as the neural rollout's measurement-preprocessing
            # state.
            preprocessing_nav = mechanize(
                preprocessing_nav,
                last_gyro,
                last_accel,
                dt,
            )
            interval_segments_since_previous_usable_fusion.append((imu_index, dt))

            # Covariance propagation is required only until the one-time
            # classical warm-start update has been formed.
            if training_warm_start is None:
                F_error = error_dynamics(preprocessing_nav, last_accel)
                Phi, Qd = van_loan(F_error, Qc, dt)
                warm_start_covariance = (
                    Phi @ warm_start_covariance @ Phi.T + Qd
                )
                warm_start_covariance = 0.5 * (
                    warm_start_covariance + warm_start_covariance.T
                )
            continue

        fusion_index = event_index
        t = event_start
        epoch = gnss_epochs[fusion_index]
        leo_measurements = leo_simulator.simulate_epoch(
            t,
            fusion_antenna_truth_position[fusion_index],
        )
        bootstrap_measurements = prepare_fusion_measurements(
            gnss_preprocessor,
            epoch,
            leo_measurements,
            preprocessing_nav,
            lever_arm_b_m,
            capacity=None,
        )
        if not bootstrap_measurements:
            continue

        fusion_record = {
            'time': float(t),
            'fusion_index': int(fusion_index),
            'gnss_epoch': epoch,
            'leo_measurements': tuple(leo_measurements),
            'bootstrap_measurement_count': int(len(bootstrap_measurements)),
            'preceding_interval_segments': tuple(
                interval_segments_since_previous_usable_fusion
            ),
            'accel': last_accel.copy(),
            'gyro': last_gyro.copy(),
        }
        fusion_records.append(fusion_record)

        if training_warm_start is None:
            # One classical update only: this supplies the first posterior state
            # and the lagged context required by Yan Eqs. (10)-(14).  No later
            # training epoch uses a precomputed classical innovation/residual.
            warm_model = measurement_model(
                preprocessing_nav,
                bootstrap_measurements,
                lever_arm_b_m,
            )
            x_pred = np.zeros(INS_STATE_DIM, dtype=float)
            x_pred[9:12] = preprocessing_nav.accel_bias_body_mps2
            x_pred[12:15] = preprocessing_nav.gyro_bias_body_radps

            correction, warm_start_covariance, _ = kf_update(
                warm_start_covariance,
                warm_model.innovation,
                warm_model.H,
                warm_model.R,
            )
            warm_nav = inject_error(preprocessing_nav, correction)

            x_post = correction.copy()
            x_post[9:12] = warm_nav.accel_bias_body_mps2
            x_post[12:15] = warm_nav.gyro_bias_body_radps
            posterior_residual = innovation_only(
                warm_nav,
                bootstrap_measurements,
                lever_arm_b_m,
            )

            training_warm_start = {
                'nav': NavState(
                    warm_nav.position_ecef_m.copy(),
                    warm_nav.velocity_ecef_mps.copy(),
                    warm_nav.body_to_ecef_dcm.copy(),
                    warm_nav.accel_bias_body_mps2.copy(),
                    warm_nav.gyro_bias_body_radps.copy(),
                ),
                'context': {
                    'sat_ids': tuple(warm_model.sat_ids),
                    'residual': posterior_residual.copy(),
                    'x_post': x_post.copy(),
                    'state_innovation': (x_post - x_pred).copy(),
                    'state_residual': np.zeros(INS_STATE_DIM, dtype=float),
                    'accel': last_accel.copy(),
                    'gyro': last_gyro.copy(),
                },
                'feature_accel': last_accel.copy(),
                'feature_gyro': last_gyro.copy(),
            }

        interval_segments_since_previous_usable_fusion = []

    if training_warm_start is None or len(fusion_records) < 4:
        raise ValueError(
            'Static Data01 preprocessing produced fewer than four usable fusion '
            'records; check GNSS products, TLE coverage, masks, and time synchronization'
        )

    network_nmax = max(
        int(record['bootstrap_measurement_count'])
        for record in fusion_records[1:]
    )
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
        state0 = 6
        state1 = state0 + INS_STATE_DIM
        state2 = state1 + INS_STATE_DIM
        return np.concatenate([
            _l2_norm(fixed_raw[0:3]),
            _l2_norm(fixed_raw[3:6]),
            _l2_norm(fixed_raw[state0:state1]),
            _l2_norm(fixed_raw[state1:state2]),
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
    data01_total_sample_count = len(fusion_records) - 1
    validation_sample_count = max(1, int(round(data01_total_sample_count * VALIDATION_FRACTION)))
    training_sample_count = data01_total_sample_count - validation_sample_count
    validation_start_sample = training_sample_count

    training_sequence_plan = plan_training_sequences(
        training_sample_count,
        TRAINING_SEQUENCE_LENGTH,
        OPTIMIZER_WINDOW_SIZE,
    )
    batched_training_config = resolve_batched_training_config(
        len(training_sequence_plan),
    )
    REQUESTED_TRAIN_BATCH_SIZE = int(
        batched_training_config['requested_batch_size']
    )
    EFFECTIVE_TRAIN_BATCH_SIZE = int(
        batched_training_config['effective_batch_size']
    )
    LEARNING_RATES = dict(batched_training_config['learning_rates'])
    FILTER_LEARNING_RATE = LEARNING_RATES['filter']
    REPRESENTATION_LEARNING_RATE = LEARNING_RATES['representation']
    print(
        'Documented sequence mini-batching | '
        f'sequences={len(training_sequence_plan)} | '
        f'requested batch={REQUESTED_TRAIN_BATCH_SIZE} | '
        f'effective batch={EFFECTIVE_TRAIN_BATCH_SIZE} | '
        f'filter lr={FILTER_LEARNING_RATE:g} | '
        f'representation lr={REPRESENTATION_LEARNING_RATE:g}'
    )

    def _build_truth_boundary_warm_start(boundary_sample: int, *, split_name: str):
        """Create one excluded boundary state for an independent sequence.

        KalmanNet4SensorFusion ``nclt_split`` takes the first ground-truth state
        of every training subsequence as ``initial_state`` and excludes that
        boundary sample from the supervised sequence.  Its trainer calls
        ``init_beliefs`` for each such sequence.  Yan Eqs. (10)-(14) additionally
        require lagged innovation/residual/state context, so this script applies
        the same one-step classical warm start already used at its first fusion
        epoch; it does not construct or reuse a full classical trajectory.

        https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/dataset/nclt_dataset.py
        https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/trainer/fusion_trainer.py
        """
        boundary_record = fusion_records[int(boundary_sample)]
        boundary_index = int(boundary_record['fusion_index'])
        seed_nav = NavState(
            fusion_imu_truth_position[boundary_index].copy(),
            fusion_imu_truth_velocity[boundary_index].copy(),
            fusion_truth_body_to_ecef[boundary_index].copy(),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        )
        seed_measurements = prepare_fusion_measurements(
            gnss_preprocessor,
            boundary_record['gnss_epoch'],
            boundary_record['leo_measurements'],
            seed_nav,
            lever_arm_b_m,
            capacity=network_nmax,
        )
        if not seed_measurements:
            raise ValueError(
                f'{split_name} boundary sample {int(boundary_sample)} has no '
                'usable measurements for cold start'
            )
        seed_model = measurement_model(
            seed_nav,
            seed_measurements,
            lever_arm_b_m,
        )
        seed_correction, _, _ = kf_update(
            P0.copy(),
            seed_model.innovation,
            seed_model.H,
            seed_model.R,
        )
        boundary_nav = inject_error(seed_nav, seed_correction)
        x_pred = np.zeros(INS_STATE_DIM, dtype=float)
        x_post = seed_correction.copy()
        x_post[9:12] = boundary_nav.accel_bias_body_mps2
        x_post[12:15] = boundary_nav.gyro_bias_body_radps
        return {
            'nav': boundary_nav,
            'context': {
                'sat_ids': tuple(seed_model.sat_ids),
                'residual': innovation_only(
                    boundary_nav,
                    seed_measurements,
                    lever_arm_b_m,
                ),
                'x_post': x_post,
                'state_innovation': x_post - x_pred,
                'state_residual': np.zeros(INS_STATE_DIM, dtype=float),
                'accel': boundary_record['accel'].copy(),
                'gyro': boundary_record['gyro'].copy(),
            },
            'feature_accel': boundary_record['accel'].copy(),
            'feature_gyro': boundary_record['gyro'].copy(),
        }

    # Preserve the existing causal warm start for the first sequence.  Every
    # later reference-defined subsequence receives its own truth-boundary state;
    # _warm_start_to_rollout_state then creates an empty recurrent state.
    training_sequence_warm_starts = {}
    if not TEST_ONLY:
        training_sequence_warm_starts[0] = training_warm_start
        training_sequence_warm_starts.update({
            item.start: _build_truth_boundary_warm_start(
                item.start,
                split_name='training',
            )
            for item in training_sequence_plan
            if item.start != 0
        })

    # Validation remains one uninterrupted held-out trajectory with an
    # independent boundary initialization.  This matches the reference config,
    # where train_dataset uses seq_len=50 but val/test use seq_len=None.
    validation_warm_start = _build_truth_boundary_warm_start(
        validation_start_sample,
        split_name='validation',
    )

    # Model lifecycle.
    if TEST_ONLY:
        # Existing checkpoint is mandatory in test-only mode.
        model, loaded_checkpoint = _load_test_only_model(
            CHECKPOINT_PATH,
            DEVICE,
            expected_nmax=network_nmax,
            expected_bias_ablation_mode=BIAS_ABLATION_MODE,
        )
        print(f'Loaded checkpoint: {CHECKPOINT_PATH}')
    else:
        # Fresh training mode: do not inspect or load any previous checkpoint.
        model = MaskedCLA(nmax=network_nmax, dropout=0.2).to(
            device=DEVICE,
            dtype=TORCH_FLOAT_DTYPE,
        )
        loaded_checkpoint = {}
        print('Initialized a new MaskedCLA model from scratch.')

    LSTM_CONFIGURED_DROPOUT = float(model.lstm.lstm.dropout)
    alternating_parameter_groups = build_alternating_parameter_groups(model)
    representation_parameters = alternating_parameter_groups['representation']
    filter_parameters = alternating_parameter_groups['filter']
    all_network_parameters = list(model.parameters())

    if TEST_ONLY:
        optimizers = {}
    else:
        # Latent-KalmanNet Algorithm 2 mapped onto Yan Fig. 8:
        # theta = masked LSTM + attention + FC Kalman-gain head
        # psi   = masked convolutional representation
        # The optimizers own disjoint parameter lists; no ADMM auxiliary
        # variables or layer-wise surrogate losses are introduced.
        optimizers = {
            'filter': torch.optim.Adam(
                filter_parameters,
                lr=FILTER_LEARNING_RATE,
            ),
            'representation': torch.optim.Adam(
                representation_parameters,
                lr=REPRESENTATION_LEARNING_RATE,
            ),
        }

    # Diagnostics only; these checks do not clamp or alter the optimizer,
    # loss, Kalman gain, innovation, correction, or navigation state.
    bias_gradient_diagnostics_path = (
        OUTPUT_DIR / f'bias_gradient_diagnostics_{BIAS_ABLATION_MODE}.jsonl'
    )
    bias_gradient_diagnostics_path.write_text('', encoding='utf-8')
    forward_finite_diagnostics_path = (
        OUTPUT_DIR / f'forward_finite_diagnostics_{BIAS_ABLATION_MODE}.jsonl'
    )
    forward_finite_diagnostics_path.write_text('', encoding='utf-8')

    def _check_forward_finite(sample_index, phase, stage, **named_tensors):
        failed_names = [
            name
            for name, value in named_tensors.items()
            if not bool(torch.all(torch.isfinite(value)).detach().cpu())
        ]
        if not failed_names:
            return
        record = {
            'sample_index': int(sample_index),
            'phase': 'validation' if phase is None else str(phase),
            'stage': str(stage),
            'failed_tensors': failed_names,
            'tensors': {
                name: _tensor_finite_stats(value)
                for name, value in named_tensors.items()
            },
        }
        with forward_finite_diagnostics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        raise FloatingPointError(
            f'non-finite forward tensor at sample {sample_index} | '
            f'phase={record["phase"]} | stage={stage} | '
            f'tensors={", ".join(failed_names)}'
        )

    def _gradient_diagnostics():
        nonfinite_parameters = []
        finite_total_sq = 0.0
        finite_max_abs = 0.0

        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            finite = torch.isfinite(grad)
            if not bool(torch.all(finite).cpu()):
                nonfinite_parameters.append({
                    'name': name,
                    'nonfinite_count': int((~finite).sum().cpu()),
                    'element_count': int(grad.numel()),
                })
            else:
                grad64 = grad.double()
                finite_total_sq += float(torch.sum(grad64 * grad64).cpu())
                if grad64.numel():
                    finite_max_abs = max(
                        finite_max_abs,
                        float(torch.max(torch.abs(grad64)).cpu()),
                    )

        group_stats = {}
        weight_grad = model.gain_head.weight.grad
        bias_grad = model.gain_head.bias.grad if model.gain_head.bias is not None else None
        groups = {
            'navigation_rows_0_8': (0, 9),
            'accel_bias_rows_9_11': (9, 12),
            'gyro_bias_rows_12_14': (12, 15),
        }

        if weight_grad is not None:
            weight_rows = weight_grad.reshape(INS_STATE_DIM, network_nmax, -1)
            bias_rows = (
                None if bias_grad is None
                else bias_grad.reshape(INS_STATE_DIM, network_nmax)
            )
            for label, (start_row, stop_row) in groups.items():
                values = [weight_rows[start_row:stop_row].reshape(-1)]
                if bias_rows is not None:
                    values.append(bias_rows[start_row:stop_row].reshape(-1))
                values = torch.cat(values).detach().double()
                finite = torch.isfinite(values)
                finite_values = values[finite]
                group_stats[label] = {
                    'l2_finite_part': (
                        float(torch.linalg.vector_norm(finite_values).cpu())
                        if finite_values.numel() else 0.0
                    ),
                    'max_abs_finite': (
                        float(torch.max(torch.abs(finite_values)).cpu())
                        if finite_values.numel() else 0.0
                    ),
                    'nonfinite_count': int((~finite).sum().cpu()),
                    'element_count': int(values.numel()),
                }
        else:
            for label in groups:
                group_stats[label] = {
                    'l2_finite_part': 0.0,
                    'max_abs_finite': 0.0,
                    'nonfinite_count': 0,
                    'element_count': 0,
                }

        return {
            'all_finite': len(nonfinite_parameters) == 0,
            'finite_total_l2': (
                math.sqrt(finite_total_sq)
                if not nonfinite_parameters else None
            ),
            'finite_max_abs': finite_max_abs,
            'nonfinite_parameters': nonfinite_parameters,
            'gain_head_groups': group_stats,
        }

    def _write_gradient_diagnostic(record):
        with bias_gradient_diagnostics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

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

        state0 = 6
        state1 = state0 + INS_STATE_DIM
        state2 = state1 + INS_STATE_DIM
        fixed_nn = torch.cat((
            fixed_raw[0:3] / torch.clamp(torch.linalg.vector_norm(fixed_raw[0:3]), min=FEATURE_L2_EPS),
            fixed_raw[3:6] / torch.clamp(torch.linalg.vector_norm(fixed_raw[3:6]), min=FEATURE_L2_EPS),
            fixed_raw[state0:state1] / torch.clamp(torch.linalg.vector_norm(fixed_raw[state0:state1]), min=FEATURE_L2_EPS),
            fixed_raw[state1:state2] / torch.clamp(torch.linalg.vector_norm(fixed_raw[state1:state2]), min=FEATURE_L2_EPS),
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
        def core(position, velocity, dcm, accel_bias, gyro_bias):
            state = TorchNavState(position, velocity, dcm, accel_bias, gyro_bias)
            feature_gyro = training_imr_gyro_t[segment_tuple[0][0]]
            feature_accel = training_imr_accel_t[segment_tuple[0][0]]
            for imu_index, dt in segment_tuple:
                feature_gyro = training_imr_gyro_t[imu_index].reshape(3)
                feature_accel = training_imr_accel_t[imu_index].reshape(3)
                state = _torch_mechanize(state, feature_gyro, feature_accel, dt)
            return state.position_ecef_m, state.velocity_ecef_mps, state.body_to_ecef_dcm, state.accel_bias_body_mps2, state.gyro_bias_body_radps, feature_gyro, feature_accel
        inputs = (nav_state.position_ecef_m, nav_state.velocity_ecef_mps, nav_state.body_to_ecef_dcm, nav_state.accel_bias_body_mps2, nav_state.gyro_bias_body_radps)
        if training and any(value.requires_grad for value in inputs):
            position, velocity, dcm, accel_bias, gyro_bias, feature_gyro, feature_accel = checkpoint(core, *inputs, use_reentrant=False)
        else:
            position, velocity, dcm, accel_bias, gyro_bias, feature_gyro, feature_accel = core(*inputs)
        return TorchNavState(position, velocity, dcm, accel_bias, gyro_bias), feature_gyro, feature_accel

    def _warm_start_to_rollout_state(warm_start):
        start_nav = warm_start['nav']
        warm_context = warm_start['context']
        return {
            'nav': TorchNavState(
                torch.as_tensor(start_nav.position_ecef_m, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.velocity_ecef_mps, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.body_to_ecef_dcm, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.accel_bias_body_mps2, dtype=torch.float64, device=DEVICE).clone(),
                torch.as_tensor(start_nav.gyro_bias_body_radps, dtype=torch.float64, device=DEVICE).clone(),
            ),
            'context': {
                'sat_ids': tuple(warm_context['sat_ids']),
                'residual': torch.as_tensor(warm_context['residual'], dtype=torch.float64, device=DEVICE),
                'x_post': torch.as_tensor(warm_context['x_post'], dtype=torch.float64, device=DEVICE),
                'state_innovation': torch.as_tensor(warm_context['state_innovation'], dtype=torch.float64, device=DEVICE),
                'state_residual': torch.as_tensor(warm_context['state_residual'], dtype=torch.float64, device=DEVICE),
                'accel': torch.as_tensor(warm_context['accel'], dtype=torch.float64, device=DEVICE),
                'gyro': torch.as_tensor(warm_context['gyro'], dtype=torch.float64, device=DEVICE),
            },
            'feature_accel': torch.as_tensor(warm_start['feature_accel'], dtype=torch.float64, device=DEVICE),
            'feature_gyro': torch.as_tensor(warm_start['feature_gyro'], dtype=torch.float64, device=DEVICE),
            'recurrent_state': None,
        }

    # Recursive TBPTT
    def _rollout(
        start_sample: int,
        stop_sample: int,
        *,
        phase: str | None=None,
        initial_state=None,
        loss_start_sample: int | None=None,
        backward_weight: float=1.0,
    ):
        training = phase is not None
        metric_start = start_sample if loss_start_sample is None else int(loss_start_sample)
        backward_weight = float(backward_weight)
        if training and (
            not math.isfinite(backward_weight) or backward_weight <= 0.0
        ):
            raise ValueError('training backward_weight must be positive and finite')

        if initial_state is None:
            if start_sample != 0:
                raise ValueError(
                    'A rollout without a carried state must start at sample 0 so '
                    'the one-time warm start and lagged context remain causal.'
                )

            initial_state = _warm_start_to_rollout_state(training_warm_start)
            nav_state = initial_state['nav']
            previous_context = initial_state['context']
            feature_accel = initial_state['feature_accel']
            feature_gyro = initial_state['feature_gyro']
            recurrent_state = initial_state['recurrent_state']
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
                row = fusion_records[sample_index + 1]
                if row['preceding_interval_segments']:
                    nav_state, feature_gyro, feature_accel = _propagate(nav_state, row['preceding_interval_segments'], training=training)

                _check_forward_finite(
                    sample_index,
                    phase,
                    'predicted_state',
                    position=nav_state.position_ecef_m,
                    velocity=nav_state.velocity_ecef_mps,
                    body_to_ecef_dcm=nav_state.body_to_ecef_dcm,
                    accel_bias=nav_state.accel_bias_body_mps2,
                    gyro_bias=nav_state.gyro_bias_body_radps,
                )

                fusion_index = int(row['fusion_index'])
                measurements = prepare_fusion_measurements(
                    gnss_preprocessor,
                    row['gnss_epoch'],
                    row['leo_measurements'],
                    _numpy_nav_from_torch(nav_state),
                    lever_arm_b_m,
                    capacity=network_nmax,
                )
                if not measurements:
                    raise FloatingPointError(
                        f'no usable state-dependent measurements at Data01 sample {sample_index}'
                    )
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
                _check_forward_finite(
                    sample_index,
                    phase,
                    'innovation',
                    innovation=innovation_now,
                )
                fixed_nn, obs_nn, mask_nn, channel_nn, innovation_nn = _network_input(previous_context, current_sat_ids, innovation_now, feature_accel, feature_gyro)
                output = model(
                    fixed_nn.to(dtype=TORCH_FLOAT_DTYPE).unsqueeze(0),
                    obs_nn.to(dtype=TORCH_FLOAT_DTYPE).unsqueeze(0),
                    mask_nn.unsqueeze(0),
                    channel_nn.unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
                recurrent_state = output.recurrent_state
                _check_forward_finite(
                    sample_index,
                    phase,
                    'network_output',
                    kalman_gain=output.kalman_gain,
                    recurrent_hidden=recurrent_state[0],
                    recurrent_cell=recurrent_state[1],
                )
                correction = state_update(output, innovation_nn.unsqueeze(0))[0]
                correction = correction.reshape(INS_STATE_DIM)
                correction = correction * BIAS_ABLATION_MASK_T
                _check_forward_finite(
                    sample_index,
                    phase,
                    'correction',
                    correction=correction,
                )

                # Predicted state in Yan Eq. (7) notation.  The closed-loop
                # navigation-error mean is zero after feedback, but the IMU
                # bias states are persistent estimates rather than reset errors.
                x_pred_current = torch.cat((
                    correction.new_zeros(9),
                    nav_state.accel_bias_body_mps2.reshape(3),
                    nav_state.gyro_bias_body_radps.reshape(3),
                ))

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

                # KF measurement update: b^+ = b^- + (K*innovation)_b.
                # The resulting posterior bias is the estimate used by Eq. (8).
                nav_state = TorchNavState(
                    nav_state.position_ecef_m + correction[0:3],
                    nav_state.velocity_ecef_mps + correction[3:6],
                    _torch_so3_exp(ATTITUDE_FEEDBACK_SIGN * correction[6:9]) @ nav_state.body_to_ecef_dcm,
                    nav_state.accel_bias_body_mps2 + correction[9:12],
                    nav_state.gyro_bias_body_radps + correction[12:15],
                )

                x_post_current = torch.cat((
                    correction[:9],
                    nav_state.accel_bias_body_mps2.reshape(3),
                    nav_state.gyro_bias_body_radps.reshape(3),
                ))

                posterior_residual, _ = _torch_innovation(nav_state, prepared)
                _check_forward_finite(
                    sample_index,
                    phase,
                    'posterior_state',
                    position=nav_state.position_ecef_m,
                    velocity=nav_state.velocity_ecef_mps,
                    body_to_ecef_dcm=nav_state.body_to_ecef_dcm,
                    accel_bias=nav_state.accel_bias_body_mps2,
                    gyro_bias=nav_state.gyro_bias_body_radps,
                    posterior_innovation=posterior_residual,
                )
                previous_context = {
                    'sat_ids': tuple(current_sat_ids),
                    'residual': posterior_residual,
                    'x_post': x_post_current,
                    'state_innovation': x_post_current - x_pred_current,
                    'state_residual': x_post_current - previous_context['x_post'],
                    'accel': feature_accel.reshape(3),
                    'gyro': feature_gyro.reshape(3),
                }

                # Reference-style TBPTT(k=2,w=4): keep the numerical state
                # continuous while cutting every carried autograd history after
                # two fusion steps.  Validation remains an uninterrupted causal
                # forward rollout because it has no backward graph.
                if training and (sample_index - start_sample + 1) % TBPTT_DETACH_STEP == 0:
                    (
                        nav_state,
                        previous_context,
                        feature_accel,
                        feature_gyro,
                        recurrent_state,
                    ) = _detach_rollout_state(
                        nav_state,
                        previous_context,
                        feature_accel,
                        feature_gyro,
                        recurrent_state,
                    )

        mean_loss = torch.stack(state_losses).mean()
        if training:
            # Across a mini-batch, each independent window contributes in
            # proportion to its valid timestep count.  Summing these weighted
            # backward calls is exactly the data term of Yan Eq. (32) and
            # Latent-KalmanNet Eq. (13), without retaining every sequence graph.
            (mean_loss * backward_weight).backward()
            (
                detached_nav,
                detached_context,
                detached_feature_accel,
                detached_feature_gyro,
                detached_recurrent_state,
            ) = _detach_rollout_state(
                nav_state,
                previous_context,
                feature_accel,
                feature_gyro,
                recurrent_state,
            )
            carried = {
                'nav': detached_nav,
                'context': detached_context,
                'feature_accel': detached_feature_accel,
                'feature_gyro': detached_feature_gyro,
                'recurrent_state': detached_recurrent_state,
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

    def _train_block(phase: str, training_batches):
        optimizer = optimizers[phase]
        active_parameters, inactive_parameters = configure_alternating_phase(
            model,
            phase,
            alternating_parameter_groups,
        )
        inactive_before = snapshot_parameter_values(inactive_parameters)

        state_sum = 0.0
        position_sum = 0.0
        count = 0
        regularization_losses = []
        gradients = []
        optimizer_step_count = 0

        # KalmanNet batches independent trajectories before one optimizer step.
        # KalmanNet4SensorFusion likewise batches independent, non-overlapping
        # subsequences and applies TBPTT windows inside each sequence.  Each
        # sequence below therefore owns a distinct carried navigation/context/
        # recurrent state, while gradients are averaged across the valid time
        # samples of all active sequence windows before a single optimizer step.
        # https://github.com/KalmanNet/KalmanNet_TSP/blob/main/Pipelines/Pipeline_EKF.py
        # https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/trainer/fusion_trainer.py
        for batch_index, sequence_batch in enumerate(training_batches):
            carried_by_sequence = {
                sequence.start: _warm_start_to_rollout_state(
                    training_sequence_warm_starts[sequence.start]
                )
                for sequence in sequence_batch
            }
            for batch_window_index, batch_window_members in enumerate(
                iter_training_batch_windows(sequence_batch)
            ):
                model.zero_grad(set_to_none=True)
                loss_weights = batch_window_loss_weights(batch_window_members)
                batch_data_loss = 0.0
                member_ranges = []

                for (
                    training_sequence,
                    (window_start, window_stop),
                ), loss_weight in zip(batch_window_members, loss_weights):
                    metrics = _rollout(
                        window_start,
                        window_stop,
                        phase=phase,
                        initial_state=carried_by_sequence[training_sequence.start],
                        backward_weight=loss_weight,
                    )
                    carried_by_sequence[training_sequence.start] = metrics[
                        'rollout_state'
                    ]
                    batch_data_loss += loss_weight * float(metrics['eq30'])
                    state_sum += metrics['state_sum']
                    position_sum += metrics['position_sum']
                    count += metrics['count']
                    member_ranges.append({
                        'sequence_start': int(training_sequence.start),
                        'sequence_stop': int(training_sequence.stop),
                        'window_start': int(window_start),
                        'window_stop': int(window_stop),
                        'loss_weight': float(loss_weight),
                    })

                # The weighted backward calls above jointly produce the batch
                # data gradient.  This is the shared end-to-end state MSE used
                # in both Algorithm-2 phases, not a phase-specific surrogate.
                eq30_diag = _gradient_diagnostics()
                _write_gradient_diagnostic({
                    'mode': BIAS_ABLATION_MODE,
                    'phase': phase,
                    'stage': 'after_batched_eq30_backward',
                    'batch_index': int(batch_index),
                    'batch_window_index': int(batch_window_index),
                    'batch_size': len(batch_window_members),
                    'members': member_ranges,
                    'eq30': float(batch_data_loss),
                    **eq30_diag,
                })
                if not eq30_diag['all_finite']:
                    bad = ', '.join(
                        item['name'] for item in eq30_diag['nonfinite_parameters']
                    )
                    raise FloatingPointError(
                        f'non-finite Eq.(30) gradient | '
                        f'mode={BIAS_ABLATION_MODE} | phase={phase} | '
                        f'batch={batch_index} | window={batch_window_index} | '
                        f'parameters={bad}'
                    )
                assert_parameter_gradients_absent(
                    inactive_parameters,
                    phase=phase,
                )

                # Yan Eq. (32) and Latent-KalmanNet Eqs. (13)-(15): add exactly
                # one complete-network L2 penalty to the mean data loss of this
                # mini-batch window.  Freezing determines which block receives
                # gradients; it does not change the scientific objective.
                detached_batch_data_loss = torch.as_tensor(
                    batch_data_loss,
                    dtype=TORCH_FLOAT_DTYPE,
                    device=DEVICE,
                )
                objective, regularization = documented_alternating_objective(
                    detached_batch_data_loss,
                    all_network_parameters,
                    gamma=GAMMA_L2,
                )
                regularization.backward()
                regularization_value = float(regularization.detach().cpu())
                objective_value = float(objective.detach().cpu())

                reg_diag = _gradient_diagnostics()
                _write_gradient_diagnostic({
                    'mode': BIAS_ABLATION_MODE,
                    'phase': phase,
                    'stage': 'after_regularization_backward',
                    'batch_index': int(batch_index),
                    'batch_window_index': int(batch_window_index),
                    'batch_size': len(batch_window_members),
                    'members': member_ranges,
                    'eq30': float(batch_data_loss),
                    'regularization': regularization_value,
                    'objective': objective_value,
                    **reg_diag,
                })
                if not reg_diag['all_finite']:
                    bad = ', '.join(
                        item['name'] for item in reg_diag['nonfinite_parameters']
                    )
                    raise FloatingPointError(
                        f'non-finite gradient after regularization | '
                        f'mode={BIAS_ABLATION_MODE} | phase={phase} | '
                        f'batch={batch_index} | window={batch_window_index} | '
                        f'parameters={bad}'
                    )
                assert_parameter_gradients_absent(
                    inactive_parameters,
                    phase=phase,
                )

                # Preserve global L2 clipping while computing the total norm in
                # float64, so large finite gradients are not misclassified as Inf.
                grad_norm_tensor = clip_grad_norm_float64_(
                    active_parameters,
                    max_norm=GRADIENT_CLIP_NORM,
                )
                grad_norm = float(grad_norm_tensor.detach().cpu())
                clip_coefficient = min(
                    1.0,
                    GRADIENT_CLIP_NORM / (grad_norm + 1e-6),
                )
                postclip_diag = _gradient_diagnostics()
                _write_gradient_diagnostic({
                    'mode': BIAS_ABLATION_MODE,
                    'phase': phase,
                    'stage': 'after_float64_gradient_clipping',
                    'batch_index': int(batch_index),
                    'batch_window_index': int(batch_window_index),
                    'batch_size': len(batch_window_members),
                    'members': member_ranges,
                    'preclip_total_l2': grad_norm,
                    'clip_coefficient': clip_coefficient,
                    **postclip_diag,
                })
                optimizer.step()
                optimizer_step_count += 1

                if not math.isfinite(objective_value):
                    raise FloatingPointError('non-finite training loss')
                regularization_losses.append(regularization_value)
                gradients.append(grad_norm)

        assert_parameter_values_unchanged(
            inactive_before,
            inactive_parameters,
            phase=phase,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

        data_loss = state_sum / count
        regularization_loss = float(np.mean(regularization_losses))

        return {
            'phase': phase,
            'loss': data_loss + regularization_loss,
            'data_loss': data_loss,
            'regularization_loss': regularization_loss,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'position_rmse_m': math.sqrt(position_sum / count),
            'gradient_l2_norm': float(np.mean(gradients)),
            'optimizer_step_count': int(optimizer_step_count),
            'requested_batch_size': REQUESTED_TRAIN_BATCH_SIZE,
            'effective_batch_size': EFFECTIVE_TRAIN_BATCH_SIZE,
        }

    # Causal validation.  Validation starts from its own one-time classical
    # context at the split boundary; it never inherits train navigation/LSTM state.
    def _evaluate(metric_start: int, metric_stop: int, *, warm_start=None):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
        if warm_start is None:
            rollout_start = 0
            initial_state = None
        else:
            rollout_start = int(metric_start)
            initial_state = _warm_start_to_rollout_state(warm_start)
        return _rollout(
            rollout_start,
            metric_stop,
            phase=None,
            initial_state=initial_state,
            loss_start_sample=metric_start,
        )

    training_history = []
    epochs_without_improvement = 0

    if TEST_ONLY:
        # Validate the loaded checkpoint once on the Data01 validation interval.
        # No backward pass or optimizer step is executed.
        initial_validation = _evaluate(
            validation_start_sample,
            data01_total_sample_count,
            warm_start=validation_warm_start,
        )
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_selection_eq30 = float(loaded_checkpoint.get(
            'validation_loss',
            initial_validation['eq30'],
        ))
        best_selection_rmse_m = float(loaded_checkpoint.get(
            'validation_position_rmse_m',
            initial_validation['position_rmse_m'],
        ))
        best_selection_epoch = int(loaded_checkpoint.get('selected_epoch', 0))
    else:
        # Fresh training: select the best *trained* epoch only.
        # An untrained epoch-0 model is never written as a successful checkpoint.
        best_model_state = None
        best_selection_eq30 = float('inf')
        best_selection_rmse_m = float('inf')
        best_selection_epoch = 0

        for epoch in range(1, TRAINING_EPOCHS + 1):
            try:
                # One epoch is one complete Algorithm-2 cycle: first optimize
                # theta and then psi over the exact same randomly shuffled
                # mini-batch partition (Latent-KalmanNet Algorithm 2).  The
                # permutation changes reproducibly between epochs.
                epoch_training_batches = plan_training_batches(
                    training_sequence_plan,
                    REQUESTED_TRAIN_BATCH_SIZE,
                    seed=SEED,
                    epoch=epoch,
                )
                cycle_metrics = run_alternating_cycle(
                    _train_block,
                    epoch_training_batches,
                )
                filter_phase = cycle_metrics['filter']
                representation_phase = cycle_metrics['representation']
                overall_train = _evaluate(0, training_sample_count)
                validation = _evaluate(
                    validation_start_sample,
                    data01_total_sample_count,
                    warm_start=validation_warm_start,
                )
            except FloatingPointError as error:
                print(f'Epoch {epoch:02d}: stopped - {error}')
                break

            overall_train_loss = float(overall_train['eq30'])
            validation_loss = float(validation['eq30'])
            validation_rmse = float(validation['position_rmse_m'])
            train_loss = float(np.mean((
                filter_phase['loss'],
                representation_phase['loss'],
            )))
            train_data_loss = float(np.mean((
                filter_phase['data_loss'],
                representation_phase['data_loss'],
            )))
            train_regularization_loss = float(np.mean((
                filter_phase['regularization_loss'],
                representation_phase['regularization_loss'],
            )))
            gradient_l2_norm = float(np.mean((
                filter_phase['gradient_l2_norm'],
                representation_phase['gradient_l2_norm'],
            )))

            if math.isfinite(validation_loss) and validation_loss < best_selection_eq30:
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_selection_eq30 = validation_loss
                best_selection_rmse_m = validation_rmse
                best_selection_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            training_history.append({
                'epoch': epoch,
                'alternating_phase_order': list(ALTERNATING_PHASE_ORDER),
                'training_sequence_batches': [
                    [int(sequence.start) for sequence in sequence_batch]
                    for sequence_batch in epoch_training_batches
                ],
                'train_loss': train_loss,
                'train_data_loss': train_data_loss,
                'train_regularization_loss': train_regularization_loss,
                'gradient_l2_norm': gradient_l2_norm,
                'filter_phase': filter_phase,
                'representation_phase': representation_phase,
                'overall_train_loss': overall_train_loss,
                'validation_loss': validation_loss,
                'validation_position_rmse_m': validation_rmse,
            })
            print(
                f"Epoch {epoch:02d}/{TRAINING_EPOCHS} | "
                f"theta/filter loss={filter_phase['loss']:.6g} "
                f"(data={filter_phase['data_loss']:.6g}, "
                f"reg={filter_phase['regularization_loss']:.3g}, "
                f"lr={filter_phase['learning_rate']:.3g}, "
                f"grad={filter_phase['gradient_l2_norm']:.3g}, "
                f"steps={filter_phase['optimizer_step_count']}) | "
                f"psi/representation loss={representation_phase['loss']:.6g} "
                f"(data={representation_phase['data_loss']:.6g}, "
                f"reg={representation_phase['regularization_loss']:.3g}, "
                f"lr={representation_phase['learning_rate']:.3g}, "
                f"grad={representation_phase['gradient_l2_norm']:.3g}, "
                f"steps={representation_phase['optimizer_step_count']}) | "
                f"overall train loss={overall_train_loss:.6g} | "
                f"validation loss={validation_loss:.6g} | "
                f"validation RMSE={validation_rmse:.3f} m"
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping after epoch {epoch}.')
                break

        if best_model_state is None:
            raise RuntimeError(
                'Fresh training did not produce any finite validation checkpoint; '
                'no best_model.pt will be created and Data02 testing is aborted.'
            )

    model.load_state_dict({
        key: value.to(DEVICE)
        for key, value in best_model_state.items()
    })
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

    # Checkpoint lifecycle.
    if TEST_ONLY:
        # Never overwrite the externally supplied checkpoint in test-only mode.
        print(f'TEST_ONLY=True: testing loaded checkpoint {CHECKPOINT_PATH}')
    else:
        checkpoint_payload = {
            'model_state_dict': {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            'nmax': int(network_nmax),
            'state_order': '[delta_p,delta_v,delta_theta,b_a,b_g]',
            'supervised_loss_state_order': SUPERVISED_LOSS_STATE_ORDER,
            'architecture': 'Yan_masked_CNN24_k3_LSTM5x64_attention_FC_KG',
            'feature_normalization': FEATURE_NORMALIZATION_MODE,
            'feature_semantics_version': FEATURE_SEMANTICS_VERSION,
            'measurement_preprocessing_policy': MEASUREMENT_PREPROCESSING_POLICY,
            'fde_recurrent_policy': FDE_RECURRENT_POLICY,
            'bias_ablation_mode': BIAS_ABLATION_MODE,
            'run_profile': RUN_PROFILE,
            'selected_epoch': best_epoch,
            'validation_loss': best_val_state_loss,
            'validation_position_rmse_m': best_val_position_rmse_m,
            'paper_initial_learning_rate': PAPER_INITIAL_LEARNING_RATE,
            'reference_batched_learning_rate': REFERENCE_BATCHED_LEARNING_RATE,
            'optimizer': 'two independent Adam optimizers',
            'training_strategy': TRAINING_STRATEGY,
            **alternating_checkpoint_contract(LEARNING_RATES),
            **batched_training_checkpoint_contract(
                batched_training_config,
                sequence_count=len(training_sequence_plan),
                gamma=GAMMA_L2,
            ),
            'alternating_encoder_pretraining': False,
            'alternating_encoder_pretraining_note': (
                'not applied because Yan does not publish a separate supervised '
                'target for the masked-CNN intermediate representation'
            ),
            'external_bcd_admm_auxiliary_variables': False,
            'external_bcd_admm_note': (
                'mDLAM/BCD/dlADMM auxiliary activations, penalties, and dual '
                'updates are excluded because they would replace Yan Eqs. (30) and (32)'
            ),
            'optimizer_window_size': OPTIMIZER_WINDOW_SIZE,
            'tbptt_detach_step': TBPTT_DETACH_STEP,
            'training_sequence_length': TRAINING_SEQUENCE_LENGTH,
            'training_sequence_count': len(training_sequence_plan),
            'training_sequence_policy': (
                'nonoverlapping_causal_first_then_truth_boundary_warm_start_tbptt'
            ),
            'training_sequence_reference': (
                'SongJgit/KalmanNet4SensorFusion:'
                'configs/nclt/fusion/wheel_gpsfusion_origin.py,'
                'Net/dataset/nclt_dataset.py,'
                'Net/trainer/fusion_trainer.py'
            ),
            'training_batch_reference': (
                'KalmanNet/KalmanNet_TSP:Pipelines/Pipeline_EKF.py;'
                'KalmanNet/Latent_KalmanNet_TSP:Algorithm_2_and_Eqs_13_15'
            ),
            'gradient_clip_norm': GRADIENT_CLIP_NORM,
            'torch_float_dtype': TORCH_FLOAT_DTYPE_NAME,
        }
        torch.save(checkpoint_payload, GENERATED_CHECKPOINT_PATH)
        (OUTPUT_DIR / 'history.json').write_text(
            json.dumps(training_history, indent=2),
            encoding='utf-8',
        )
        print(
            f'Fresh training complete: saved best epoch {best_epoch} '
            f'to {GENERATED_CHECKPOINT_PATH}'
        )

        # Test exactly the checkpoint written to disk, rather than only the
        # in-memory copy of the selected model.
        model, loaded_checkpoint = _load_test_only_model(
            GENERATED_CHECKPOINT_PATH,
            DEVICE,
            expected_nmax=network_nmax,
            expected_bias_ablation_mode=BIAS_ABLATION_MODE,
        )
        model.lstm.lstm.dropout = LSTM_CONFIGURED_DROPOUT
        print(f'Reloaded generated checkpoint for Data02 test: {GENERATED_CHECKPOINT_PATH}')

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
    test_Qc = np.diag(np.concatenate((
        test_imu_model_values['PNSD_Pos'],
        test_imu_model_values['PNSD_Vel'],
        test_imu_model_values['PNSD_Att'] * (np.pi / 180.0),
        test_imu_model_values['PNSD_AccelBias'],
        test_imu_model_values['PNSD_GyrosBias'] * (np.pi / 180.0),
    )) ** 2)

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

    nav = NavState(test_imu_position[0].copy(), test_imu_velocity[0].copy(), test_C_b_e, np.zeros(3), np.zeros(3))
    P = np.diag(np.concatenate((
        test_imu_model_values['ISDV_Pos'],
        test_imu_model_values['ISDV_Vel'],
        test_imu_model_values['ISDV_Att'] * (np.pi / 180.0),
        test_imu_model_values['ISDV_AccelBias'],
        test_imu_model_values['ISDV_GyrosBias'] * (np.pi / 180.0),
    )) ** 2)
    online_rows = []
    previous_online = None
    recurrent_state = None
    last_gyro = np.asarray(test_imr_angular_rate_body_radps[0], dtype=float).reshape(3)
    last_accel = np.asarray(test_imr_acceleration_body_mps2[0], dtype=float).reshape(3)
    last_feature_gyro, last_feature_accel = last_gyro, last_accel

    # Phase 5 activates the online branch:
    # no detected fault -> Masked KalmanNet update;
    # uniquely identified fault -> hard exclusion + Yan Eq. (34) DIA update;
    # unresolved detection -> no measurement update and integrity unavailable.
    fde_detection_rows = []
    fde_identification_rows = []
    fde_adaptation_rows = []
    fde_application_rows = []

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
        test_leo_measurements = test_leo_simulator.simulate_epoch(
            t,
            test_fusion_antenna_truth_position[fusion_index],
        )
        measurements = prepare_fusion_measurements(
            test_gnss_processor,
            test_gnss_epochs[fusion_index],
            test_leo_measurements,
            nav,
            test_lever_arm_b_m,
            capacity=None,
        )
        if not measurements:
            recurrent_state = recurrent_state_after_update(
                'no_measurements',
                recurrent_state,
            )
            continue

        # Yan Sec. II-D: FDE uses the INS-predicted innovation before the
        # learned KG is applied.  All currently usable observations participate
        # in Detection/Identification before the network-nmax selection.
        fde_model = measurement_model(nav, measurements, test_lever_arm_b_m)
        fde_detection = FDE_DETECTOR.detect(P, fde_model)
        fde_detection_rows.append((
            float(t),
            bool(fde_detection.detected),
            float(fde_detection.statistic),
            float(fde_detection.threshold),
            int(fde_detection.dof),
            int(len(fde_model.sat_ids)),
        ))

        if fde_detection.detected:
            fde_identification = FDE_IDENTIFIER.identify(
                fde_model,
                fde_detection.Q_nu_nu,
            )
        else:
            fde_identification = None

        fde_identification_rows.append((
            float(t),
            bool(fde_detection.detected),
            bool(fde_identification is not None and fde_identification.identified),
            "" if fde_identification is None else fde_identification.identified_sat_id,
            "" if fde_identification is None else fde_identification.identified_constellation,
            0.0 if fde_identification is None else float(fde_identification.local_statistic),
            0.0 if fde_identification is None else float(fde_identification.local_score),
            0.0 if fde_identification is None else float(fde_identification.estimated_fault_m),
            () if fde_identification is None else tuple(fde_identification.ambiguous_sat_ids),
            0 if fde_identification is None else int(fde_identification.candidate_count),
        ))

        if fde_identification is not None and fde_identification.identified:
            fde_adaptation = DIA_ADAPTER.adapt(
                P,
                fde_model,
                fde_detection.Q_nu_nu,
                fde_identification,
            )
            dia_reset_covariance = reset_covariance(
                fde_adaptation.covariance,
                fde_adaptation.state_correction,
            )
            fde_exclusion = FDE_EXCLUDER.exclude(
                measurements,
                fde_identification,
                clock_observable,
            )
        else:
            fde_adaptation = None
            dia_reset_covariance = None
            fde_exclusion = None

        fde_adaptation_rows.append((
            float(t),
            bool(fde_adaptation is not None and fde_adaptation.available),
            "" if fde_adaptation is None else fde_adaptation.identified_sat_id,
            np.zeros(INS_STATE_DIM, dtype=float)
                if fde_adaptation is None
                else fde_adaptation.nominal_state_correction.copy(),
            np.zeros(INS_STATE_DIM, dtype=float)
                if fde_adaptation is None
                else fde_adaptation.state_correction.copy(),
            np.zeros(INS_STATE_DIM, dtype=float)
                if fde_adaptation is None
                else fde_adaptation.state_adjustment.copy(),
            np.zeros((INS_STATE_DIM, INS_STATE_DIM), dtype=float)
                if fde_adaptation is None
                else fde_adaptation.nominal_covariance.copy(),
            np.zeros((INS_STATE_DIM, INS_STATE_DIM), dtype=float)
                if fde_adaptation is None
                else fde_adaptation.covariance.copy(),
            np.zeros((INS_STATE_DIM, INS_STATE_DIM), dtype=float)
                if dia_reset_covariance is None
                else dia_reset_covariance.copy(),
            0.0 if fde_adaptation is None else float(fde_adaptation.estimated_fault_m),
            0.0 if fde_adaptation is None else float(np.linalg.norm(fde_adaptation.L_i)),
        ))

        warm_start_epoch = previous_online is None
        previous_context_before_update = previous_online

        x_pred = np.zeros(INS_STATE_DIM, dtype=float)
        x_pred[9:12] = nav.accel_bias_body_mps2
        x_pred[12:15] = nav.gyro_bias_body_radps

        # Large explicit fault: Yan Eq. (34) is the state/covariance update.
        # The Masked-KalmanNet learned gain is deliberately NOT applied in the
        # same epoch: Yan says FDE acts before the learned KG, and neither Yan
        # nor Ref. [33] defines a learned-gain replacement inside L_i.
        if fde_detection.detected and fde_adaptation is not None and fde_adaptation.available:
            correction = np.asarray(fde_adaptation.state_correction, dtype=float).reshape(INS_STATE_DIM)
            P = np.asarray(dia_reset_covariance, dtype=float).copy()
            nav = inject_error(nav, correction)

            x_post = correction.copy()
            x_post[9:12] = nav.accel_bias_body_mps2
            x_post[12:15] = nav.gyro_bias_body_radps

            retained_measurements = tuple(fde_exclusion.retained_measurements)
            retained_sat_ids = tuple(m.sat_id for m in retained_measurements)
            posterior_residual = innovation_only(
                nav,
                retained_measurements,
                test_lever_arm_b_m,
            )
            previous_online = _online_context(
                retained_sat_ids,
                posterior_residual,
                x_pred,
                x_post,
                last_feature_accel,
                last_feature_gyro,
                previous_context=previous_context_before_update,
            )
            recurrent_state = recurrent_state_after_update(
                'dia_fault',
                recurrent_state,
            )

            posterior_position = (
                nav.position_ecef_m
                + nav.body_to_ecef_dcm @ np.asarray(test_lever_arm_b_m).reshape(3)
            )
            hpl_m, vpl_m = (
                INTEGRITY.protection_levels(P, posterior_position)
                if INTEGRITY is not None
                else (float('nan'), float('nan'))
            )
            fde_application_rows.append((
                float(t),
                'dia_fault',
                str(fde_exclusion.excluded_sat_id),
                tuple(fde_exclusion.observability_dropped_sat_ids),
                False,   # neural update applied
                True,    # DIA update applied
                True,    # integrity solution available
            ))

            # The first online fusion epoch remains a context/warm-start epoch,
            # consistent with the pre-FDE test path.
            if warm_start_epoch:
                continue

            online_rows.append((
                t,
                posterior_position.copy(),
                test_fusion_antenna_truth_position[fusion_index].copy(),
                hpl_m,
                vpl_m,
                'dia_fault',
                True,
            ))
            continue

        # Detection occurred but no unique hypothesis could be established.
        # Teunissen's DIA framework permits an undecided/unavailable region
        # rather than outputting a potentially unreliable adapted estimate.
        # In this integrated-navigation implementation that means: retain the
        # INS prediction, apply no GNSS/LEO state update, and mark integrity
        # unavailable for this epoch.
        if fde_detection.detected:
            x_post = x_pred.copy()
            previous_online = _online_context(
                (),
                np.empty(0, dtype=float),
                x_pred,
                x_post,
                last_feature_accel,
                last_feature_gyro,
                previous_context=previous_context_before_update,
            )
            recurrent_state = recurrent_state_after_update(
                'unresolved_ins_only',
                recurrent_state,
            )
            posterior_position = (
                nav.position_ecef_m
                + nav.body_to_ecef_dcm @ np.asarray(test_lever_arm_b_m).reshape(3)
            )
            fde_application_rows.append((
                float(t),
                'unresolved_ins_only',
                '',
                (),
                False,
                False,
                False,
            ))

            if warm_start_epoch:
                continue

            online_rows.append((
                t,
                posterior_position.copy(),
                test_fusion_antenna_truth_position[fusion_index].copy(),
                float('nan'),
                float('nan'),
                'unresolved_ins_only',
                False,
            ))
            continue

        # No detected large fault.  The first online epoch keeps the existing
        # classical-KF warm start; later epochs use the learned Masked-KalmanNet
        # gain, which provides Yan's "soft exclusion" for errors below FDE Td.
        if warm_start_epoch:
            correction, P, _ = kf_update(
                P,
                fde_model.innovation,
                fde_model.H,
                fde_model.R,
            )
            nav = inject_error(nav, correction)

            x_post = correction.copy()
            x_post[9:12] = nav.accel_bias_body_mps2
            x_post[12:15] = nav.gyro_bias_body_radps

            previous_online = _online_context(
                fde_model.sat_ids,
                innovation_only(nav, measurements, test_lever_arm_b_m),
                x_pred,
                x_post,
                last_feature_accel,
                last_feature_gyro,
            )
            recurrent_state = recurrent_state_after_update(
                'warm_classical_no_fault',
                recurrent_state,
            )
            fde_application_rows.append((
                float(t),
                'warm_classical_no_fault',
                '',
                (),
                False,
                False,
                True,
            ))
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
                torch.tensor(fixed_nn[None], dtype=TORCH_FLOAT_DTYPE, device=DEVICE),
                torch.tensor(obs_padded[None], dtype=TORCH_FLOAT_DTYPE, device=DEVICE),
                torch.tensor(mask_padded[None], dtype=torch.bool, device=DEVICE),
                torch.tensor(channel_padded[None], dtype=torch.bool, device=DEVICE),
                recurrent_state=recurrent_state,
            )
            recurrent_state = output.recurrent_state
            correction = state_update(
                output,
                torch.tensor(innovation_padded[None], dtype=torch.float64, device=DEVICE),
            )[0].cpu().numpy().astype(float)
            correction = _apply_bias_ablation_numpy(
                correction,
                BIAS_ABLATION_MODE,
            )
            recurrent_state = recurrent_state_after_update(
                'neural_no_fault',
                recurrent_state,
            )

        active_gain = output.kalman_gain[0].cpu().numpy().astype(float)[:, :len(measurements)]
        if not np.all(np.isfinite(correction)) or not np.all(np.isfinite(active_gain)):
            raise FloatingPointError('non-finite Data02 neural update')

        P = learned_covariance(P, active_gain, meas.H, meas.R, correction)
        nav = inject_error(nav, correction)

        x_post = correction.copy()
        x_post[9:12] = nav.accel_bias_body_mps2
        x_post[12:15] = nav.gyro_bias_body_radps

        posterior_position = (
            nav.position_ecef_m
            + nav.body_to_ecef_dcm @ np.asarray(test_lever_arm_b_m).reshape(3)
        )
        previous_online = _online_context(
            meas.sat_ids,
            innovation_only(nav, measurements, test_lever_arm_b_m),
            x_pred,
            x_post,
            last_feature_accel,
            last_feature_gyro,
            previous_context=previous_context_before_update,
        )
        hpl_m, vpl_m = (
            INTEGRITY.protection_levels(P, posterior_position)
            if INTEGRITY is not None
            else (float('nan'), float('nan'))
        )
        fde_application_rows.append((
            float(t),
            'neural_no_fault',
            '',
            (),
            True,
            False,
            True,
        ))
        online_rows.append((
            t,
            posterior_position.copy(),
            test_fusion_antenna_truth_position[fusion_index].copy(),
            hpl_m,
            vpl_m,
            'neural_no_fault',
            True,
        ))

    if not online_rows:
        raise RuntimeError('Data02 test produced no usable online navigation epochs')

    online_time = np.asarray([row[0] for row in online_rows], dtype=float)
    estimate = np.stack([row[1] for row in online_rows])
    truth_aligned = np.stack([row[2] for row in online_rows])
    ned_error = np.empty_like(estimate)
    for i, (est, truth_i) in enumerate(zip(estimate, truth_aligned)):
        lat, lon, _ = ecef_llh(truth_i)
        ned_error[i] = ecef_to_ned(lat, lon) @ (est - truth_i)
    error_3d = np.linalg.norm(ned_error, axis=1)
    rmse = np.append(np.sqrt(np.mean(ned_error ** 2, axis=0)), math.sqrt(float(np.mean(error_3d ** 2))))
    online_update_modes = np.asarray([row[5] for row in online_rows], dtype=str)
    branch_rmse = summarize_branch_rmse(ned_error, online_update_modes)

    # Yan Fig. 20 uses horizontal/vertical position error against HPL/VPL.
    horizontal_error_m = np.hypot(ned_error[:, 0], ned_error[:, 1])
    vertical_error_m = np.abs(ned_error[:, 2])
    hpl_array_m = np.asarray([row[3] for row in online_rows], dtype=float)
    vpl_array_m = np.asarray([row[4] for row in online_rows], dtype=float)
    integrity_available_array = np.asarray([row[6] for row in online_rows], dtype=bool)

    horizontal_stanford_category, horizontal_integrity = INTEGRITY.stanford_summary(
        horizontal_error_m,
        hpl_array_m,
        PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M,
        integrity_available=integrity_available_array,
    )
    vertical_stanford_category, vertical_integrity = INTEGRITY.stanford_summary(
        vertical_error_m,
        vpl_array_m,
        PAPER_FIG20_VERTICAL_ALERT_LIMIT_M,
        integrity_available=integrity_available_array,
    )

    horizontal_integrity['paper_fig20_reference_NO_percent'] = float(
        PAPER_FIG20_REFERENCE_HORIZONTAL_NO_PERCENT
    )
    horizontal_integrity['difference_from_paper_NO_percentage_points'] = float(
        horizontal_integrity['nominal_operation_percent_of_all']
        - PAPER_FIG20_REFERENCE_HORIZONTAL_NO_PERCENT
    )
    vertical_integrity['paper_fig20_reference_NO_percent'] = float(
        PAPER_FIG20_REFERENCE_VERTICAL_NO_PERCENT
    )
    vertical_integrity['difference_from_paper_NO_percentage_points'] = float(
        vertical_integrity['nominal_operation_percent_of_all']
        - PAPER_FIG20_REFERENCE_VERTICAL_NO_PERCENT
    )

    integrity_report = {
        'PL_model': {
            'type': 'SBAS_APV_style_covariance_bound',
            'K_horizontal': 6.0,
            'K_vertical': 5.33,
            'source_note': (
                'External SBAS/APV standard formula. Yan et al. show PL/Stanford '
                'results but do not publish the PL equation or K-factors; Yan Ref. [35] '
                'is a satellite-selection/error-model reference and is not the PL source.'
            ),
        },
        'alert_limits': {
            'horizontal_m': float(PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M),
            'vertical_m': float(PAPER_FIG20_VERTICAL_ALERT_LIMIT_M),
            'source_note': 'Read directly from the 30 m boundaries in Yan Fig. 20.',
        },
        'horizontal': horizontal_integrity,
        'vertical': vertical_integrity,
        'fault_label_metrics': {
            'available': False,
            'reason': (
                'The supplied real dataset has no ground-truth labels identifying which '
                'pseudorange epochs/satellites are faulty; therefore empirical FDE true-'
                'positive, false-positive, missed-detection and identification rates are '
                'not reported from the dataset.'
            ),
        },
    }
    (OUTPUT_DIR / 'integrity_metrics.json').write_text(
        json.dumps(integrity_report, indent=2),
        encoding='utf-8',
    )

    # Paper-style Stanford density plots.  No epoch with unavailable/unresolved
    # integrity is assigned an invented PL; those epochs are omitted from the
    # density plot and are reported in integrity_metrics.json.
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def _save_stanford_diagram(
        position_error_m: Array,
        protection_level_m: Array,
        categories: Array,
        alert_limit_m: float,
        title: str,
        output_path: Path,
    ) -> None:
        error = np.asarray(position_error_m, dtype=float).reshape(-1)
        protection = np.asarray(protection_level_m, dtype=float).reshape(-1)
        categories = np.asarray(categories, dtype=str).reshape(-1)
        finite = (
            np.isfinite(error)
            & np.isfinite(protection)
            & (categories != 'UNRESOLVED')
        )

        finite_error = error[finite]
        finite_protection = protection[finite]
        maximum = float(alert_limit_m) + 10.0
        if finite_error.size:
            maximum = max(
                maximum,
                float(np.max(finite_error)) * 1.05,
                float(np.max(finite_protection)) * 1.05,
            )
        maximum = max(10.0, math.ceil(maximum / 10.0) * 10.0)

        fig, ax = plt.subplots(figsize=(7, 6))
        if finite_error.size >= 2:
            histogram = ax.hist2d(
                finite_error,
                finite_protection,
                bins=80,
                range=[[0.0, maximum], [0.0, maximum]],
                norm=LogNorm(),
            )
            fig.colorbar(histogram[3], ax=ax, label='Epoch density')
        elif finite_error.size == 1:
            ax.scatter(finite_error, finite_protection)

        ax.plot([0.0, maximum], [0.0, maximum])
        ax.axvline(float(alert_limit_m))
        ax.axhline(float(alert_limit_m))
        ax.set_xlim(0.0, maximum)
        ax.set_ylim(0.0, maximum)
        ax.set_xlabel('Position error [m]')
        ax.set_ylabel('Protection level [m]')
        ax.set_title(title)

        counts = {
            label: int(np.count_nonzero(categories == label))
            for label in ('NO', 'MI', 'HO', 'SU', 'SU&MI', 'UNRESOLVED')
        }
        total = max(int(len(categories)), 1)
        ax.text(0.25 * alert_limit_m, 0.55 * alert_limit_m,
                f"NO\\n{100.0 * counts['NO'] / total:.3f}%")
        ax.text(0.60 * alert_limit_m, 0.18 * alert_limit_m,
                f"MI\\n{100.0 * counts['MI'] / total:.3f}%")
        ax.text(0.25 * alert_limit_m, min(maximum * 0.92, alert_limit_m + 0.5 * (maximum - alert_limit_m)),
                f"SU\\n{100.0 * counts['SU'] / total:.3f}%")
        ax.text(min(maximum * 0.82, alert_limit_m + 0.45 * (maximum - alert_limit_m)),
                0.45 * alert_limit_m,
                f"HO\\n{100.0 * counts['HO'] / total:.3f}%")
        ax.text(min(maximum * 0.78, alert_limit_m + 0.38 * (maximum - alert_limit_m)),
                min(maximum * 0.88, alert_limit_m + 0.38 * (maximum - alert_limit_m)),
                f"SU&MI\\n{100.0 * counts['SU&MI'] / total:.3f}%")
        if counts['UNRESOLVED']:
            ax.text(
                0.02 * maximum,
                0.98 * maximum,
                f"Unresolved FDE: {counts['UNRESOLVED']} epochs",
                va='top',
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    _save_stanford_diagram(
        vertical_error_m,
        vpl_array_m,
        vertical_stanford_category,
        PAPER_FIG20_VERTICAL_ALERT_LIMIT_M,
        'Vertical Stanford Integrity Diagram',
        OUTPUT_DIR / 'stanford_vertical.png',
    )
    _save_stanford_diagram(
        horizontal_error_m,
        hpl_array_m,
        horizontal_stanford_category,
        PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M,
        'Horizontal Stanford Integrity Diagram',
        OUTPUT_DIR / 'stanford_horizontal.png',
    )

    np.savez(
        OUTPUT_DIR / 'test_evaluation.npz',
        time_gpst_s=online_time,
        estimate_ecef_m=estimate,
        truth_ecef_m=truth_aligned,
        ned_error_m=ned_error,
        error_3d_m=error_3d,
        hpl_m=hpl_array_m,
        vpl_m=vpl_array_m,
        horizontal_error_m=horizontal_error_m,
        vertical_error_m=vertical_error_m,
        horizontal_alert_limit_m=np.asarray(PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M, dtype=float),
        vertical_alert_limit_m=np.asarray(PAPER_FIG20_VERTICAL_ALERT_LIMIT_M, dtype=float),
        horizontal_stanford_category=horizontal_stanford_category,
        vertical_stanford_category=vertical_stanford_category,
        online_update_mode=np.asarray([row[5] for row in online_rows], dtype=str),
        online_integrity_available=integrity_available_array,
        rmse_north_east_down_3d_m=rmse,
        fde_detection_time_gpst_s=np.asarray([row[0] for row in fde_detection_rows], dtype=float),
        fde_detected=np.asarray([row[1] for row in fde_detection_rows], dtype=bool),
        fde_statistic=np.asarray([row[2] for row in fde_detection_rows], dtype=float),
        fde_threshold=np.asarray([row[3] for row in fde_detection_rows], dtype=float),
        fde_dof=np.asarray([row[4] for row in fde_detection_rows], dtype=int),
        fde_measurement_count=np.asarray([row[5] for row in fde_detection_rows], dtype=int),
        fde_alpha=np.asarray(FDE_SIGNIFICANCE_ALPHA, dtype=float),
        fde_identification_time_gpst_s=np.asarray([row[0] for row in fde_identification_rows], dtype=float),
        fde_identification_attempted=np.asarray([row[1] for row in fde_identification_rows], dtype=bool),
        fde_identified=np.asarray([row[2] for row in fde_identification_rows], dtype=bool),
        fde_identified_sat_id=np.asarray([row[3] for row in fde_identification_rows], dtype=str),
        fde_identified_constellation=np.asarray([row[4] for row in fde_identification_rows], dtype=str),
        fde_local_statistic=np.asarray([row[5] for row in fde_identification_rows], dtype=float),
        fde_local_score=np.asarray([row[6] for row in fde_identification_rows], dtype=float),
        fde_estimated_fault_m=np.asarray([row[7] for row in fde_identification_rows], dtype=float),
        fde_ambiguous_sat_ids=np.asarray(
            [','.join(row[8]) for row in fde_identification_rows],
            dtype=str,
        ),
        fde_identification_candidate_count=np.asarray([row[9] for row in fde_identification_rows], dtype=int),
        dia_time_gpst_s=np.asarray([row[0] for row in fde_adaptation_rows], dtype=float),
        dia_available=np.asarray([row[1] for row in fde_adaptation_rows], dtype=bool),
        dia_identified_sat_id=np.asarray([row[2] for row in fde_adaptation_rows], dtype=str),
        dia_nominal_state_correction=np.stack([row[3] for row in fde_adaptation_rows]),
        dia_state_correction=np.stack([row[4] for row in fde_adaptation_rows]),
        dia_state_adjustment=np.stack([row[5] for row in fde_adaptation_rows]),
        dia_nominal_covariance=np.stack([row[6] for row in fde_adaptation_rows]),
        dia_covariance_pre_reset=np.stack([row[7] for row in fde_adaptation_rows]),
        dia_covariance_reset_candidate=np.stack([row[8] for row in fde_adaptation_rows]),
        dia_estimated_fault_m=np.asarray([row[9] for row in fde_adaptation_rows], dtype=float),
        dia_L_frobenius_norm=np.asarray([row[10] for row in fde_adaptation_rows], dtype=float),
        fde_application_time_gpst_s=np.asarray([row[0] for row in fde_application_rows], dtype=float),
        fde_application_mode=np.asarray([row[1] for row in fde_application_rows], dtype=str),
        fde_excluded_fault_sat_id=np.asarray([row[2] for row in fde_application_rows], dtype=str),
        fde_observability_dropped_sat_ids=np.asarray(
            [','.join(row[3]) for row in fde_application_rows],
            dtype=str,
        ),
        fde_neural_update_applied=np.asarray([row[4] for row in fde_application_rows], dtype=bool),
        fde_dia_update_applied=np.asarray([row[5] for row in fde_application_rows], dtype=bool),
        fde_integrity_available=np.asarray([row[6] for row in fde_application_rows], dtype=bool),
    )
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps({
        'selected_epoch': best_epoch,
        'validation_loss': best_val_state_loss,
        'validation_position_rmse_m': best_val_position_rmse_m,
        'test_epochs': len(online_rows),
        'test_position_rmse_ned3d_m': [float(v) for v in rmse],
        'test_position_rmse_by_update_branch': branch_rmse,
        'run_profile': RUN_PROFILE,
        'feature_semantics_version': FEATURE_SEMANTICS_VERSION,
        'measurement_preprocessing_policy': MEASUREMENT_PREPROCESSING_POLICY,
        'fde_recurrent_policy': FDE_RECURRENT_POLICY,
        'bias_ablation_mode': BIAS_ABLATION_MODE,
        'measurement_mode': 'pseudorange_only',
        'leo_orbit': 'TLE_SGP4',
        'navigation_state': '15_state_[delta_p,delta_v,delta_theta,b_a,b_g]',
        'supervised_loss_state': '9_state_[delta_p,delta_v,delta_theta]',
        'FDE_DIA': 'phase6_complete_online_fde_dia_plus_integrity_stanford_evaluation',
        'FDE_alpha': float(FDE_SIGNIFICANCE_ALPHA),
        'FDE_alpha_source': 'Ref33 main safety-analysis experiments; Yan Sec. II-D gives no numeric false-alarm probability',
        'FDE_detection_covariance': 'Q_nu_nu = H P_minus H^T + R in the clock-nuisance-projected residual subspace',
        'FDE_detection_epochs': int(len(fde_detection_rows)),
        'FDE_detected_epochs': int(sum(int(row[1]) for row in fde_detection_rows)),
        'FDE_identification_model': 'one 1-D raw pseudorange outlier hypothesis per observation; C_j = clock_projector @ e_j',
        'FDE_identified_epochs': int(sum(int(row[2]) for row in fde_identification_rows)),
        'FDE_ambiguous_epochs': int(sum(int(bool(row[8])) for row in fde_identification_rows)),
        'FDE_identified_by_constellation': {
            constellation: int(sum(1 for row in fde_identification_rows if row[2] and row[4] == constellation))
            for constellation in ('G', 'C', 'L')
        },
        'FDE_max_local_identification_score': float(max((row[6] for row in fde_identification_rows), default=0.0)),
        'FDE_max_abs_estimated_fault_m': float(max((abs(row[7]) for row in fde_identification_rows), default=0.0)),
        'DIA_equations': 'Yan Eq.(34) with measurement-fault L_i from Ref33 Appendix Eq.(39)',
        'DIA_nominal_update': 'Ref33 classical-KF x0_plus/P0_plus; no learned-KG substitution is made',
        'FDE_online_policy': 'H0 accepted -> Masked KalmanNet; unique Hi -> hard exclusion plus Yan Eq.(34) DIA; unresolved -> INS prediction only and integrity unavailable',
        'FDE_fault_epoch_network_policy': 'learned KG is not applied on DIA/unresolved epochs because Yan places FDE before learned-KG state updating',
        'FDE_unresolved_policy': 'conservative undecided handling: no GNSS/LEO update; carry INS prediction; integrity unavailable',
        'DIA_available_epochs': int(sum(int(row[1]) for row in fde_adaptation_rows)),
        'DIA_max_state_adjustment_norm': float(max((np.linalg.norm(row[5]) for row in fde_adaptation_rows), default=0.0)),
        'DIA_max_L_frobenius_norm': float(max((row[10] for row in fde_adaptation_rows), default=0.0)),
        'FDE_hard_exclusion_epochs': int(sum(row[1] == 'dia_fault' for row in fde_application_rows)),
        'FDE_neural_update_epochs': int(sum(row[1] == 'neural_no_fault' for row in fde_application_rows)),
        'FDE_unresolved_epochs': int(sum(row[1] == 'unresolved_ins_only' for row in fde_application_rows)),
        'FDE_integrity_unavailable_epochs': int(sum(not row[6] for row in fde_application_rows)),
        'FDE_observability_extra_drop_epochs': int(sum(bool(row[3]) for row in fde_application_rows)),
        'FDE_max_statistic_to_threshold_ratio': float(max((row[2] / row[3] for row in fde_detection_rows if math.isfinite(row[3]) and row[3] > 0.0), default=0.0)),
        'integrity_PL_model': 'SBAS_APV_style_covariance_bound_KH_6.0_KV_5.33_external_standard_not_Yan_Ref35',
        'integrity_horizontal_alert_limit_m': float(PAPER_FIG20_HORIZONTAL_ALERT_LIMIT_M),
        'integrity_vertical_alert_limit_m': float(PAPER_FIG20_VERTICAL_ALERT_LIMIT_M),
        'integrity_horizontal_NO_percent': float(horizontal_integrity['nominal_operation_percent_of_all']),
        'integrity_vertical_NO_percent': float(vertical_integrity['nominal_operation_percent_of_all']),
        'integrity_horizontal_PL_below_AL_percent': float(horizontal_integrity['pl_below_alert_limit_percent_of_all']),
        'integrity_vertical_PL_below_AL_percent': float(vertical_integrity['pl_below_alert_limit_percent_of_all']),
        'integrity_horizontal_unresolved_percent': float(horizontal_integrity['unresolved_percent_of_all']),
        'integrity_vertical_unresolved_percent': float(vertical_integrity['unresolved_percent_of_all']),
        'integrity_horizontal_containment_percent_classified': float(horizontal_integrity['protection_containment_percent_of_classified']),
        'integrity_vertical_containment_percent_classified': float(vertical_integrity['protection_containment_percent_of_classified']),
        'paper_fig20_reference_horizontal_NO_percent': float(PAPER_FIG20_REFERENCE_HORIZONTAL_NO_PERCENT),
        'paper_fig20_reference_vertical_NO_percent': float(PAPER_FIG20_REFERENCE_VERTICAL_NO_PERCENT),
        'FDE_dataset_detection_performance_metrics': 'not_available_without_ground_truth_fault_labels',
    }, indent=2), encoding='utf-8')

    print(f'Data02 test | position RMSE: N={rmse[0]:.3f} m, E={rmse[1]:.3f} m, D={rmse[2]:.3f} m, 3D={rmse[3]:.3f} m')
    for update_mode, branch in branch_rmse.items():
        values = branch['rmse_ned3d_m']
        if values is None:
            print(f'Data02 branch {update_mode} | epochs=0 | RMSE=n/a')
        else:
            print(
                f'Data02 branch {update_mode} | epochs={branch["count"]} | '
                f'RMSE N={values[0]:.3f} m, E={values[1]:.3f} m, '
                f'D={values[2]:.3f} m, 3D={values[3]:.3f} m'
            )
    print(
        'Integrity | '
        f"vertical NO={vertical_integrity['nominal_operation_percent_of_all']:.3f}% | "
        f"horizontal NO={horizontal_integrity['nominal_operation_percent_of_all']:.3f}% | "
        f"unresolved={horizontal_integrity['unresolved_percent_of_all']:.3f}%"
    )
