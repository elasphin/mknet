from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
from scipy.stats import chi2

Array = np.ndarray


# Yan et al. show PL-based Stanford diagrams but do not publish the PL
# equations or K-factors.  The values below are the standard SBAS/APV
# protection-level multipliers (K_H = 6.0, K_V = 5.33).  They are therefore
# an explicit external-standard implementation choice for reproducing the
# integrity branch, not values attributed to Yan Ref. [35].
PL_VERTICAL_MULTIPLIER = 5.33
PL_HORIZONTAL_MULTIPLIER = 6.0

STANFORD_NO = "NO"
STANFORD_MI = "MI"
STANFORD_HO = "HO"
STANFORD_SU = "SU"
STANFORD_SU_MI = "SU&MI"
STANFORD_UNRESOLVED = "UNRESOLVED"


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
        """Compute SBAS/APV-style HPL/VPL from the project position covariance."""
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


    @staticmethod
    def stanford_category(
        position_error_m: float,
        protection_level_m: float,
        alert_limit_m: float,
        *,
        integrity_available: bool = True,
    ) -> str:
        """Classify one epoch in the Stanford integrity diagram.

        Definitions follow the standard Stanford diagram geometry:
          NO:     PE <= PL <= AL
          MI:     PL < PE <= AL
          HO:     PL <= AL < PE
          SU:     AL < PL and PE <= PL
          SU&MI:  AL < PL < PE

        Epochs for which Phase-5 FDE could not produce an integrity solution
        are kept separate as UNRESOLVED instead of being forced into a
        Stanford region with an invented PL.
        """
        pe = float(position_error_m)
        pl = float(protection_level_m)
        al = float(alert_limit_m)

        if al <= 0.0 or not math.isfinite(al):
            raise ValueError("alert limit must be finite and positive")
        if (
            not bool(integrity_available)
            or not math.isfinite(pe)
            or not math.isfinite(pl)
            or pe < 0.0
            or pl < 0.0
        ):
            return STANFORD_UNRESOLVED

        if pl > al:
            return STANFORD_SU_MI if pe > pl else STANFORD_SU
        if pe > al:
            return STANFORD_HO
        if pe > pl:
            return STANFORD_MI
        return STANFORD_NO

    @classmethod
    def stanford_summary(
        cls,
        position_error_m: Array,
        protection_level_m: Array,
        alert_limit_m: float,
        *,
        integrity_available: Array | None = None,
    ) -> tuple[Array, dict[str, Any]]:
        """Classify all epochs and return paper-style integrity metrics."""
        error = np.asarray(position_error_m, dtype=float).reshape(-1)
        pl = np.asarray(protection_level_m, dtype=float).reshape(-1)
        if error.shape != pl.shape:
            raise ValueError("position error and protection level must have equal length")

        if integrity_available is None:
            available_flag = np.ones(error.shape, dtype=bool)
        else:
            available_flag = np.asarray(integrity_available, dtype=bool).reshape(-1)
            if available_flag.shape != error.shape:
                raise ValueError("integrity_available must match the position-error length")

        categories = np.asarray([
            cls.stanford_category(e, p, alert_limit_m, integrity_available=a)
            for e, p, a in zip(error, pl, available_flag)
        ], dtype=str)

        labels = (
            STANFORD_NO,
            STANFORD_MI,
            STANFORD_HO,
            STANFORD_SU,
            STANFORD_SU_MI,
            STANFORD_UNRESOLVED,
        )
        counts = {label: int(np.count_nonzero(categories == label)) for label in labels}
        total = int(len(categories))
        classified = total - counts[STANFORD_UNRESOLVED]
        percentages_all = {
            label: (100.0 * counts[label] / total if total else 0.0)
            for label in labels
        }

        finite_classified = categories != STANFORD_UNRESOLVED
        operational_available = finite_classified & np.isin(
            categories,
            (STANFORD_NO, STANFORD_MI, STANFORD_HO),
        )
        bounded = finite_classified & np.isin(
            categories,
            (STANFORD_NO, STANFORD_SU),
        )

        summary = {
            "alert_limit_m": float(alert_limit_m),
            "total_epochs": total,
            "classified_epochs": classified,
            "counts": counts,
            "percent_of_all_epochs": percentages_all,
            # Yan's quoted Fig. 20 "safe operation" percentages coincide with
            # the NO region percentages.
            "nominal_operation_percent_of_all": percentages_all[STANFORD_NO],
            # The article text describes availability through PL < AL.
            "pl_below_alert_limit_percent_of_all": (
                100.0 * float(np.count_nonzero(operational_available)) / total
                if total else 0.0
            ),
            "unresolved_percent_of_all": percentages_all[STANFORD_UNRESOLVED],
            # Empirical containment is a diagnostic of whether the computed PL
            # actually bounds the observed position error on classified epochs.
            "protection_containment_percent_of_classified": (
                100.0 * float(np.count_nonzero(bounded)) / classified
                if classified else 0.0
            ),
        }
        return categories, summary


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
    """Ref. [33] overall model test on the pre-KalmanNet predicted innovation.

    The host estimator may remove nuisance receiver-clock modes before this
    call.  In that case H, R and innovation must all be expressed in the same
    projected residual space; the effective chi-square degrees of freedom are
    the numerical rank of Q_nu_nu.
    """
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


@dataclass(frozen=True)
class FDEDetectionResult:
    """Phase-2 detection result; no identification/exclusion/adaptation yet."""

    detected: bool
    statistic: float
    threshold: float
    dof: int
    alpha: float
    Q_nu_nu: Array

    @property
    def statistic_to_threshold_ratio(self) -> float:
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            return 0.0
        return float(self.statistic / self.threshold)


class FDEDetector:
    """Yan Sec. II-D / Ref. [33] global detection stage only.

    Detection is intentionally separated from Identification and Adaptation so
    phase 2 cannot silently exclude observations or modify the navigation state.
    """

    def __init__(self, alpha: float):
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("FDE significance alpha must lie strictly between 0 and 1")
        self.alpha = alpha

    def detect(
        self,
        prior_covariance: Array,
        measurement_model: Any,
    ) -> FDEDetectionResult:
        statistic, threshold, dof, Q_nu_nu, _ = _ref33_detection_terms(
            prior_covariance,
            measurement_model,
            self.alpha,
        )
        return FDEDetectionResult(
            detected=bool(dof > 0 and statistic > threshold),
            statistic=float(statistic),
            threshold=float(threshold),
            dof=int(dof),
            alpha=float(self.alpha),
            Q_nu_nu=np.asarray(Q_nu_nu, dtype=float).copy(),
        )



@dataclass(frozen=True)
class FDEIdentificationResult:
    """Phase-3 Ref. [33] identification result.

    Identification is diagnostic only in phase 3: it does not remove
    observations and does not modify the navigation state/covariance.
    """
    identified: bool
    identified_index: int
    identified_sat_id: str
    identified_constellation: str
    local_statistic: float
    local_score: float
    estimated_fault_m: float
    ambiguous_sat_ids: tuple[str, ...]
    candidate_count: int


class FDEIdentifier:
    """Ref. [33] local Identification stage for one pseudorange fault.

    Each raw pseudorange outlier is a one-dimensional measurement-model
    misspecification. Because the host estimator removes receiver-clock
    nuisance modes with `clock_projector`, a fault in raw pseudorange j
    enters the tested residual space as

        C_j = clock_projector @ e_j,

    i.e. column j of the projector.

    For q_j = 1, Ref. [33] Eq. (7) reduces to
        lambda_j = (C_j^T Q^+ nu)^2 / (C_j^T Q^+ C_j)
        T_j      = CDF_chi2(df=1)(lambda_j)

    The hypothesis with maximum T_j is selected only when unique. If
    receiver-clock projection makes two raw-pseudorange hypotheses span
    the same residual subspace, phase 3 reports ambiguity rather than
    selecting one arbitrarily.
    """

    @staticmethod
    def identify(
        measurement_model: Any,
        Q_nu_nu: Array,
    ) -> FDEIdentificationResult:
        nu = np.asarray(measurement_model.innovation, dtype=float).reshape(-1)
        projector = np.asarray(measurement_model.clock_projector, dtype=float)
        sat_ids = tuple(measurement_model.sat_ids)
        constellations = tuple(measurement_model.constellations)

        if projector.shape != (len(nu), len(nu)):
            raise ValueError("clock_projector shape is inconsistent with innovation")
        if len(sat_ids) != len(nu) or len(constellations) != len(nu):
            raise ValueError("measurement metadata is inconsistent with innovation")

        Q_pinv, _ = _ref33_psd_pinv_and_rank(Q_nu_nu)
        qpinv_norm_2 = float(np.linalg.norm(Q_pinv, ord=2))
        eps = np.finfo(float).eps

        candidates = []
        for j in range(len(nu)):
            # Ref. [33] measurement-fault direction e_j transformed into the
            # residual space used by the global test.
            C_j = np.asarray(projector[:, j], dtype=float).reshape(-1)
            denominator = float(C_j @ Q_pinv @ C_j)
            denominator_tol = 100.0 * eps * max(
                qpinv_norm_2 * float(C_j @ C_j),
                np.finfo(float).tiny,
            )
            if not math.isfinite(denominator) or denominator <= denominator_tol:
                continue

            numerator = float(C_j @ Q_pinv @ nu)
            local_statistic = float(numerator * numerator / denominator)
            local_score = float(chi2.cdf(local_statistic, df=1))
            estimated_fault_m = float(numerator / denominator)

            if not (
                math.isfinite(local_statistic)
                and math.isfinite(local_score)
                and math.isfinite(estimated_fault_m)
            ):
                raise FloatingPointError("non-finite Ref. [33] identification term")

            candidates.append({
                "index": int(j),
                "C": C_j,
                "denominator": denominator,
                "local_statistic": local_statistic,
                "local_score": local_score,
                "estimated_fault_m": estimated_fault_m,
            })

        if not candidates:
            return FDEIdentificationResult(
                identified=False,
                identified_index=-1,
                identified_sat_id="",
                identified_constellation="",
                local_statistic=0.0,
                local_score=0.0,
                estimated_fault_m=0.0,
                ambiguous_sat_ids=(),
                candidate_count=0,
            )

        # Ref. [33]: the most likely H_i is the one with maximum transformed
        # local statistic T_i.
        best = max(candidates, key=lambda item: item["local_score"])
        best_score = float(best["local_score"])
        best_statistic = float(best["local_statistic"])
        best_C = np.asarray(best["C"], dtype=float)
        best_denominator = float(best["denominator"])

        # No paper-defined tie breaker is introduced. Statistically
        # indistinguishable/tied hypotheses are reported as ambiguous.
        score_tol = 1000.0 * eps * max(1.0, abs(best_score))
        subspace_tol = 1000.0 * eps * max(1, len(nu))
        ambiguous_indices = []

        for candidate in candidates:
            C_j = np.asarray(candidate["C"], dtype=float)
            den_j = float(candidate["denominator"])
            metric_cosine = abs(float(best_C @ Q_pinv @ C_j)) / math.sqrt(
                max(best_denominator * den_j, np.finfo(float).tiny)
            )
            same_subspace = (
                1.0 - min(max(metric_cosine, 0.0), 1.0)
                <= subspace_tol
            )
            tied_score = (
                abs(float(candidate["local_score"]) - best_score)
                <= score_tol
            )
            if same_subspace or tied_score:
                ambiguous_indices.append(int(candidate["index"]))

        ambiguous_indices = tuple(sorted(set(ambiguous_indices)))
        if len(ambiguous_indices) != 1:
            return FDEIdentificationResult(
                identified=False,
                identified_index=-1,
                identified_sat_id="",
                identified_constellation="",
                local_statistic=best_statistic,
                local_score=best_score,
                estimated_fault_m=float(best["estimated_fault_m"]),
                ambiguous_sat_ids=tuple(sat_ids[j] for j in ambiguous_indices),
                candidate_count=len(candidates),
            )

        j = int(best["index"])
        return FDEIdentificationResult(
            identified=True,
            identified_index=j,
            identified_sat_id=str(sat_ids[j]),
            identified_constellation=str(constellations[j]),
            local_statistic=best_statistic,
            local_score=best_score,
            estimated_fault_m=float(best["estimated_fault_m"]),
            ambiguous_sat_ids=(),
            candidate_count=len(candidates),
        )

@dataclass(frozen=True)
class DIAAdaptationResult:
    """Yan Eq. (34) / Ref. [33] measurement-fault adaptation result.

    The vectors are local 15-state Kalman corrections for the host
    injected-error implementation.  `covariance` is the DIA posterior
    covariance before the host error-state reset.
    """
    available: bool
    identified_index: int
    identified_sat_id: str
    nominal_state_correction: Array
    nominal_covariance: Array
    state_correction: Array
    covariance: Array
    state_adjustment: Array
    covariance_increment: Array
    L_i: Array
    fault_direction: Array
    estimated_fault_m: float


def _validate_symmetric_psd(matrix: Array, name: str) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError(f"{name} contains non-finite values")
    if matrix.size == 0:
        return matrix
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1000.0 * np.finfo(float).eps * max(matrix.shape) * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise FloatingPointError(
            f"{name} is materially indefinite: min eigenvalue={float(np.min(eigenvalues)):.6g}"
        )
    return matrix


class DIAAdapter:
    """Measurement-model DIA adaptation from Ref. [33] Appendix Eq. (39).

    For the identified measurement-fault hypothesis C_i:

        C_i^+ = (C_i^T Q_nu_nu^+ C_i)^-1 C_i^T Q_nu_nu^+
        L_i   = P^- H^T Q_nu_nu^+ C_i C_i^+

    and Yan Eq. (34) / Ref. [33] Fig. 1 gives

        x_i = x_0^+ - L_i nu
        P_i = P_0^+ + L_i Q_nu_nu L_i^T.

    The nominal x_0^+, P_0^+ here are the classical-KF quantities defined
    in Ref. [33] Fig. 1.  Phase 4 computes them exactly as the cited DIA
    theory requires.  It does NOT silently substitute the Masked-KalmanNet
    learned gain for the classical gain; coupling to the learned online
    update is intentionally deferred to phase 5 because Yan et al. do not
    provide an equation defining such a substitution.
    """

    @staticmethod
    def adapt(
        prior_covariance: Array,
        measurement_model: Any,
        Q_nu_nu: Array,
        identification: FDEIdentificationResult,
    ) -> DIAAdaptationResult:
        P_minus = np.asarray(prior_covariance, dtype=float)
        H = np.asarray(measurement_model.H, dtype=float)
        nu = np.asarray(measurement_model.innovation, dtype=float).reshape(-1)
        projector = np.asarray(measurement_model.clock_projector, dtype=float)

        if P_minus.ndim != 2 or P_minus.shape[0] != P_minus.shape[1]:
            raise ValueError("prior covariance must be square")
        n_state = int(P_minus.shape[0])
        if H.shape != (len(nu), n_state):
            raise ValueError("measurement Jacobian shape is inconsistent with state/innovation")
        if projector.shape != (len(nu), len(nu)):
            raise ValueError("clock_projector shape is inconsistent with innovation")

        zero_state = np.zeros(n_state, dtype=float)
        zero_covariance = np.zeros((n_state, n_state), dtype=float)
        zero_L = np.zeros((n_state, len(nu)), dtype=float)
        zero_direction = np.zeros(len(nu), dtype=float)

        if not identification.identified:
            return DIAAdaptationResult(
                available=False,
                identified_index=-1,
                identified_sat_id="",
                nominal_state_correction=zero_state.copy(),
                nominal_covariance=zero_covariance.copy(),
                state_correction=zero_state.copy(),
                covariance=zero_covariance.copy(),
                state_adjustment=zero_state.copy(),
                covariance_increment=zero_covariance.copy(),
                L_i=zero_L,
                fault_direction=zero_direction,
                estimated_fault_m=0.0,
            )

        j = int(identification.identified_index)
        if not 0 <= j < len(nu):
            raise IndexError("identified pseudorange index lies outside the measurement vector")

        Q = np.asarray(Q_nu_nu, dtype=float)
        if Q.shape != (len(nu), len(nu)):
            raise ValueError("Q_nu_nu shape is inconsistent with innovation")
        Q_pinv, rank = _ref33_psd_pinv_and_rank(Q)
        if rank == 0:
            raise FloatingPointError("DIA adaptation cannot be formed from rank-zero Q_nu_nu")

        # Ref. [33] Fig. 1 nominal KF measurement update under H0.
        K0 = P_minus @ H.T @ Q_pinv
        x0_plus = K0 @ nu

        # Source-faithful covariance expression:
        # P0+ = (I - P^- H^T Q^-1 H) P^- .
        identity_state = np.eye(n_state, dtype=float)
        P0_plus = (identity_state - K0 @ H) @ P_minus
        P0_plus = _validate_symmetric_psd(P0_plus, "Ref. [33] nominal P0_plus")

        # Phase-3 hypothesis C_i = clock_projector @ e_i.
        C_i = np.asarray(projector[:, j], dtype=float).reshape(-1, 1)
        denominator = float((C_i.T @ Q_pinv @ C_i).item())
        qpinv_norm_2 = float(np.linalg.norm(Q_pinv, ord=2))
        denominator_tol = 100.0 * np.finfo(float).eps * max(
            qpinv_norm_2 * float((C_i.T @ C_i).item()),
            np.finfo(float).tiny,
        )
        if not math.isfinite(denominator) or denominator <= denominator_tol:
            raise FloatingPointError("identified fault direction is not estimable in Q_nu_nu")

        # Ref. [33] Appendix Eq. (39).
        C_i_plus = (C_i.T @ Q_pinv) / denominator
        L_i = K0 @ C_i @ C_i_plus

        # The weighted least-squares estimate of the scalar pseudorange fault.
        estimated_fault_m = float((C_i_plus @ nu).item())
        if not math.isfinite(estimated_fault_m):
            raise FloatingPointError("non-finite DIA pseudorange-fault estimate")

        # Consistency check with the Phase-3 identification result.
        reference_fault_m = float(identification.estimated_fault_m)
        fault_scale = max(1.0, abs(reference_fault_m), abs(estimated_fault_m))
        if abs(estimated_fault_m - reference_fault_m) > 1e-10 * fault_scale:
            raise RuntimeError(
                "Phase-3 identification and Phase-4 DIA fault estimates disagree"
            )

        # Yan Eq. (34).
        state_adjustment = -(L_i @ nu)
        x_i = x0_plus + state_adjustment

        covariance_increment = L_i @ Q @ L_i.T
        covariance_increment = _validate_symmetric_psd(
            covariance_increment,
            "Yan Eq. (34) DIA covariance increment",
        )
        P_i = _validate_symmetric_psd(
            P0_plus + covariance_increment,
            "Yan Eq. (34) DIA posterior covariance",
        )

        if not np.all(np.isfinite(x_i)) or not np.all(np.isfinite(L_i)):
            raise FloatingPointError("non-finite Yan Eq. (34) DIA state/L_i")

        return DIAAdaptationResult(
            available=True,
            identified_index=j,
            identified_sat_id=str(identification.identified_sat_id),
            nominal_state_correction=np.asarray(x0_plus, dtype=float).copy(),
            nominal_covariance=np.asarray(P0_plus, dtype=float).copy(),
            state_correction=np.asarray(x_i, dtype=float).copy(),
            covariance=np.asarray(P_i, dtype=float).copy(),
            state_adjustment=np.asarray(state_adjustment, dtype=float).copy(),
            covariance_increment=np.asarray(covariance_increment, dtype=float).copy(),
            L_i=np.asarray(L_i, dtype=float).copy(),
            fault_direction=C_i.reshape(-1).copy(),
            estimated_fault_m=estimated_fault_m,
        )

@dataclass(frozen=True)
class FDEExclusionResult:
    """Hard exclusion result for a uniquely identified pseudorange fault."""
    applied: bool
    excluded_index: int
    excluded_sat_id: str
    retained_measurements: tuple
    observability_dropped_sat_ids: tuple[str, ...]


class FDEExcluder:
    """Remove the uniquely identified raw pseudorange before later use.

    Yan et al. state that the FDE branch handles large explicit faults by
    exclusion.  The host estimator additionally requires GPS/BDS receiver-clock
    observability, so after the identified observation is removed the existing
    `clock_observable` rule is re-applied.  Any extra measurements dropped by
    that rule are logged separately and are not labelled as faults.
    """

    @staticmethod
    def exclude(
        measurements,
        identification: FDEIdentificationResult,
        retain_clock_observable: Callable,
    ) -> FDEExclusionResult:
        current = tuple(measurements)
        if not identification.identified:
            return FDEExclusionResult(
                applied=False,
                excluded_index=-1,
                excluded_sat_id="",
                retained_measurements=current,
                observability_dropped_sat_ids=(),
            )

        j = int(identification.identified_index)
        if not 0 <= j < len(current):
            raise IndexError("identified pseudorange index lies outside measurements")

        excluded = current[j]
        remaining = tuple(m for index, m in enumerate(current) if index != j)
        retained = tuple(retain_clock_observable(remaining))

        retained_ids = {str(m.sat_id) for m in retained}
        observability_dropped = tuple(
            str(m.sat_id)
            for m in remaining
            if str(m.sat_id) not in retained_ids
        )

        return FDEExclusionResult(
            applied=True,
            excluded_index=j,
            excluded_sat_id=str(excluded.sat_id),
            retained_measurements=retained,
            observability_dropped_sat_ids=observability_dropped,
        )

