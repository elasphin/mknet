from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
from scipy.stats import chi2

Array = np.ndarray


# Ref. [35] Eqs. (57)-(58), used for the PL branch shown in Yan Fig. 2/Fig. 20.
PL_VERTICAL_MULTIPLIER = 5.33
PL_HORIZONTAL_MULTIPLIER = 6.0


class IntegrityMonitor:
    """Optional HPL/VPL computation kept with the FDE/DIA integrity module."""

    def __init__(self, ecef_to_llh: Callable, c_ecef_to_ned: Callable):
        self._ecef_to_llh = ecef_to_llh
        self._c_ecef_to_ned = c_ecef_to_ned

    def protection_levels(
        self,
        covariance_ecef: Array,
        position_ecef_m: Array,
    ) -> tuple[float, float]:
        """Compute HPL/VPL from the project covariance using Ref. [35] multipliers."""
        P = np.asarray(covariance_ecef, dtype=float)
        position = np.asarray(position_ecef_m, dtype=float).reshape(3)
        if P.shape[0] < 3 or P.shape[1] < 3:
            raise ValueError("covariance must contain at least a 3x3 position block")
        if not np.all(np.isfinite(P[:3, :3])) or not np.all(np.isfinite(position)):
            return float("inf"), float("inf")
        lat, lon, _ = self._ecef_to_llh(position)
        C = self._c_ecef_to_ned(lat, lon)
        P_ned = C @ P[:3, :3] @ C.T
        P_ned = 0.5 * (P_ned + P_ned.T)
        if not np.all(np.isfinite(P_ned)):
            return float("inf"), float("inf")

        pnn = float(P_ned[0, 0])
        pee = float(P_ned[1, 1])
        pdd = float(P_ned[2, 2])
        pne = float(P_ned[0, 1])

        horizontal_mean = 0.5 * pnn + 0.5 * pee
        half_difference = 0.5 * pnn - 0.5 * pee
        horizontal_radius = math.hypot(half_difference, pne)
        horizontal_major_variance = horizontal_mean + horizontal_radius

        if not math.isfinite(horizontal_major_variance):
            hpl = float("inf")
        else:
            hpl = PL_HORIZONTAL_MULTIPLIER * math.sqrt(
                max(horizontal_major_variance, 0.0)
            )

        if not math.isfinite(pdd):
            vpl = float("inf")
        else:
            vpl = PL_VERTICAL_MULTIPLIER * math.sqrt(max(pdd, 0.0))

        return float(hpl), float(vpl)


@dataclass(frozen=True)
class Ref33FDEResult:
    """Raw-innovation DIA decision used before the Masked-CLA update."""

    tested_measurements: tuple[Any, ...]
    tested_measurement_model: Any
    measurements: tuple[Any, ...]
    measurement_model: Any
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
    dia_state_correction: Array
    dia_covariance: Array
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
    pinv = basis / eigenvalues[positive] @ basis.T
    return 0.5 * (pinv + pinv.T), rank


def _ref33_detection_terms(
    prior_covariance: Array,
    measurement_model: Any,
    alpha: float,
) -> tuple[float, float, int, Array, Array]:
    """Ref. [33] global detection statistic on Yan's raw INS innovation."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("FDE significance alpha must lie strictly between 0 and 1")
    P = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(measurement_model.H, dtype=float)
    R = np.asarray(measurement_model.R, dtype=float)
    nu = np.asarray(measurement_model.innovation, dtype=float)
    Q_base = H @ P @ H.T + R
    Q_nu_nu = 0.5 * (Q_base + Q_base.T)
    Q_pinv, dof = _ref33_psd_pinv_and_rank(Q_nu_nu)
    if dof == 0:
        return 0.0, float("inf"), 0, Q_nu_nu, Q_pinv
    statistic = float(nu @ Q_pinv @ nu)
    threshold = float(chi2.ppf(1.0 - float(alpha), df=dof))
    if not math.isfinite(statistic) or not math.isfinite(threshold):
        raise FloatingPointError("non-finite Ref. [33] FDE statistic/threshold")
    return statistic, threshold, dof, Q_nu_nu, Q_pinv


def _ref33_identify_single_pseudorange_fault(measurement_model: Any, Q_pinv: Array):
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
        candidates.append(
            {
                "index": int(j),
                "C_i": C_i.copy(),
                "denominator": denominator,
                "numerator": numerator,
                "local_statistic": local_statistic,
                "local_score": float(chi2.cdf(local_statistic, df=1)),
                "estimated_fault_m": float(numerator / denominator),
            }
        )
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
    model: Any,
    Q_nu_nu: Array,
    Q_pinv: Array,
    fault_direction: Array | None,
    identity_state: Array,
) -> tuple[Array, Array]:
    """Compute the Yan Eq. (34)/Ref. [33] DIA diagnostic state and covariance."""
    Pm = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(model.H, dtype=float)
    R = np.asarray(model.R, dtype=float)
    nu = np.asarray(model.innovation, dtype=float)
    K0 = Pm @ H.T @ Q_pinv
    x0_plus = K0 @ nu
    I_KH = identity_state - K0 @ H
    P0_plus = I_KH @ Pm @ I_KH.T + K0 @ R @ K0.T
    P0_plus = 0.5 * (P0_plus + P0_plus.T)
    if fault_direction is None:
        return x0_plus, P0_plus
    C = np.asarray(fault_direction, dtype=float).reshape(-1, 1)
    denom = (C.T @ Q_pinv @ C).item()
    if denom <= np.finfo(float).tiny:
        return x0_plus, P0_plus
    C_plus = C.T @ Q_pinv / denom
    L_i = K0 @ C @ C_plus
    x_i = x0_plus - L_i @ nu
    P_i = P0_plus + L_i @ Q_nu_nu @ L_i.T
    return np.asarray(x_i, dtype=float), 0.5 * (P_i + P_i.T)


class FDEDIAEngine:
    """Optional Yan/Ref.[33] FDE-DIA block bound to the host estimator."""

    def __init__(
        self,
        build_measurement_model: Callable,
        retain_clock_observable_measurements: Callable,
        identity_state: Array,
    ):
        self._build_measurement_model = build_measurement_model
        self._retain_clock_observable_measurements = retain_clock_observable_measurements
        self._identity_state = np.asarray(identity_state, dtype=float)

    def decision(
        self,
        nav: Any,
        prior_covariance: Array,
        measurements,
        lever_arm_b_m: Array,
        alpha: float,
    ) -> Ref33FDEResult:
        """Run FDE/DIA on raw innovation before the learned gain update."""
        retain = self._retain_clock_observable_measurements
        build = self._build_measurement_model
        current = tuple(retain(measurements))
        tested_model = build(nav, current, lever_arm_b_m)
        empty_model = build(nav, (), lever_arm_b_m)
        zero_dx = np.zeros(self._identity_state.shape[0])
        P0 = np.asarray(prior_covariance, dtype=float).copy()
        if not current:
            return Ref33FDEResult(
                (), tested_model, (), empty_model, False, (), (), (), (),
                0.0, float("inf"), 0, False, 0.0, 0.0,
                np.zeros((0, 0)), zero_dx, P0,
            )

        statistic, threshold, dof, Q_nu_nu, Q_pinv = _ref33_detection_terms(
            prior_covariance, tested_model, alpha
        )
        nominal_dx, nominal_P = _yan_eq34_dia_adaptation(
            prior_covariance,
            tested_model,
            Q_nu_nu,
            Q_pinv,
            None,
            self._identity_state,
        )
        if dof == 0 or statistic <= threshold:
            return Ref33FDEResult(
                current, tested_model, current, tested_model, False, (), (), (), (),
                statistic, threshold, dof, False, 0.0, 0.0,
                Q_nu_nu, nominal_dx, nominal_P,
            )

        identification = _ref33_identify_single_pseudorange_fault(tested_model, Q_pinv)
        if identification is None:
            return Ref33FDEResult(
                current, tested_model, (), empty_model, True, (), (),
                tuple(m.sat_id for m in current), (),
                statistic, threshold, dof, True, 0.0, 0.0,
                Q_nu_nu, nominal_dx, nominal_P,
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
        usable = tuple(retain(remaining))
        filtered_model = build(nav, usable, lever_arm_b_m)
        excluded_sat_ids = tuple(current[j].sat_id for j in excluded_indices)
        dia_dx, dia_P = _yan_eq34_dia_adaptation(
            prior_covariance,
            tested_model,
            Q_nu_nu,
            Q_pinv,
            np.asarray(identification["C_i"], dtype=float),
            self._identity_state,
        )
        return Ref33FDEResult(
            current,
            tested_model,
            usable,
            filtered_model,
            True,
            (selected.sat_id,),
            (selected.constellation,),
            ambiguous_sat_ids,
            excluded_sat_ids,
            statistic,
            threshold,
            dof,
            not bool(usable),
            float(identification["local_score"]),
            float(identification["estimated_fault_m"]),
            Q_nu_nu,
            dia_dx,
            dia_P,
            0.0,
            float("inf"),
            0,
            True,
        )

    @staticmethod
    def new_stats() -> dict:
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

    @staticmethod
    def export_fields(result: Ref33FDEResult | None) -> dict:
        """Diagnostic fields only; None means FDE/DIA was not executed."""
        if result is None:
            return {
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
        return {
            "fde_detected": bool(result.detected),
            "fde_identified_sat_ids": tuple(result.identified_sat_ids),
            "fde_excluded_sat_ids": tuple(result.excluded_sat_ids),
            "fde_ambiguous_sat_ids": tuple(result.ambiguous_sat_ids),
            "fde_statistic": float(result.statistic),
            "fde_threshold": float(result.threshold),
            "fde_dof": int(result.dof),
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

    @staticmethod
    def accumulate_stats(stats: dict, result: Ref33FDEResult) -> None:
        stats["epochs_checked"] += 1
        stats["epochs_detected"] += int(result.detected)
        stats["identified_fault_modes"] += len(result.identified_sat_ids)
        stats["hard_exclusion_epochs"] += int(bool(result.excluded_sat_ids))
        stats["excluded_measurements"] += len(result.excluded_sat_ids)
        stats["unresolved_epochs"] += int(result.unresolved)
        stats["post_exclusion_failed_epochs"] += int(
            bool(result.excluded_sat_ids) and not result.post_exclusion_consistent
        )
        if math.isfinite(result.threshold) and result.threshold > 0.0:
            stats["max_statistic_to_threshold_ratio"] = max(
                stats["max_statistic_to_threshold_ratio"],
                float(result.statistic / result.threshold),
            )
        stats["max_abs_estimated_fault_m"] = max(
            stats["max_abs_estimated_fault_m"], abs(float(result.estimated_fault_m))
        )
        stats["max_local_identification_score"] = max(
            stats["max_local_identification_score"],
            float(result.local_identification_score),
        )
        for constellation in result.identified_constellations:
            stats["identified_by_constellation"].setdefault(constellation, 0)
            stats["identified_by_constellation"][constellation] += 1

    @staticmethod
    def validate_statistical_core(
        alpha: float,
        *,
        sample_count: int = 100000,
        seed: int = 24681357,
    ) -> dict:
        """Run synthetic regression checks for the project FDE statistical core."""
        if sample_count <= 0:
            raise ValueError("FDE self-check sample_count must be positive")
        projector = np.eye(3) - np.ones((3, 3)) / 3.0
        Q = projector @ np.diag([4.0, 4.0, 4.0]) @ projector.T
        Q = 0.5 * (Q + Q.T)
        Q_pinv, rank = _ref33_psd_pinv_and_rank(Q)
        if rank != 2:
            raise RuntimeError(f"FDE self-check expected rank 2, got {rank}")

        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        positive = eigenvalues > (
            100.0
            * np.finfo(float).eps
            * max(Q.shape)
            * max(np.max(np.abs(eigenvalues)), 1.0)
        )
        basis = eigenvectors[:, positive]
        sqrt_values = np.sqrt(eigenvalues[positive])
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((sample_count, rank))
        nu = z * sqrt_values @ basis.T
        statistics = np.einsum("bi,ij,bj->b", nu, Q_pinv, nu)
        threshold = float(chi2.ppf(1.0 - float(alpha), df=rank))
        empirical_false_alarm = float(np.mean(statistics > threshold))
        standard_error = math.sqrt(
            float(alpha) * (1.0 - float(alpha)) / float(sample_count)
        )
        z_score = (empirical_false_alarm - float(alpha)) / max(
            standard_error, 1e-15
        )
        if abs(z_score) > 8.0:
            raise RuntimeError(
                "projected Ref. [33] chi-square self-check failed: "
                f"alpha={alpha}, empirical={empirical_false_alarm}, z={z_score}"
            )

        fault_direction = projector[:, 0]
        injected_fault_m = 20.0
        fault_nu = nu + injected_fault_m * fault_direction[None, :]
        fault_statistics = np.einsum("bi,ij,bj->b", fault_nu, Q_pinv, fault_nu)
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

        projector2 = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=float)
        Q2 = projector2 @ np.eye(2) @ projector2.T
        Q2_pinv, rank2 = _ref33_psd_pinv_and_rank(Q2)
        dummy_model = _SelfCheckMeasurementModel(
            innovation=np.array([5.0, -5.0]),
            H=np.zeros((2, 9)),
            R=Q2,
            sat_ids=("G01", "G02"),
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


@dataclass(frozen=True)
class _SelfCheckMeasurementModel:
    innovation: Array
    H: Array
    R: Array
    sat_ids: tuple[str, ...]
    clock_projector: Array
