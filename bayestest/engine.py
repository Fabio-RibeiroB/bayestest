from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .models import (
    AnalysisInput,
    AnalysisResult,
    ComparisonResult,
    DecisionThresholds,
    GuardrailInput,
    GuardrailResult,
    Recommendation,
    SrmResult,
    VariantInput,
)


def parse_payload(payload: dict) -> AnalysisInput:
    variants = [VariantInput(**row) for row in payload["variants"]]
    guardrails = [GuardrailInput(**row) for row in payload.get("guardrails", [])]
    threshold_payload = payload.get("decision_thresholds", {})

    return AnalysisInput(
        experiment_name=payload["experiment_name"],
        method=payload["method"],
        variants=variants,
        primary_metric=payload.get("primary_metric", "conversion_rate"),
        alpha=payload.get("alpha", 0.05),
        look_index=payload.get("look_index", 1),
        max_looks=payload.get("max_looks", 1),
        information_fraction=payload.get("information_fraction"),
        guardrails=guardrails,
        decision_thresholds=DecisionThresholds(**threshold_payload),
        random_seed=payload.get("random_seed", 7),
        samples=payload.get("samples", 50000),
    )


def validate_input(inp: AnalysisInput) -> None:
    if not inp.variants or len(inp.variants) < 2:
        raise ValueError("At least 2 variants are required.")

    controls = [v for v in inp.variants if v.is_control]
    if len(controls) != 1:
        raise ValueError("Exactly one variant must have is_control=true.")

    for variant in inp.variants:
        if variant.visitors <= 0:
            raise ValueError(f"visitors must be positive for {variant.name}.")
        if variant.conversions < 0:
            raise ValueError(f"conversions must be non-negative for {variant.name}.")
        if variant.conversions > variant.visitors:
            raise ValueError(
                f"conversions cannot exceed visitors for {variant.name}."
            )

    if inp.alpha <= 0 or inp.alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")

    if inp.method not in {"bayesian", "frequentist_sequential", "frequentist_ttest"}:
        raise ValueError(
            "method must be 'bayesian', 'frequentist_sequential', or 'frequentist_ttest'."
        )
    if inp.primary_metric not in {"conversion_rate", "arpu"}:
        raise ValueError("primary_metric must be 'conversion_rate' or 'arpu'.")

    if inp.primary_metric == "arpu":
        for variant in inp.variants:
            if variant.revenue_sum is None or variant.revenue_sum_squares is None:
                raise ValueError(
                    f"ARPU requires revenue_sum and revenue_sum_squares for {variant.name}."
                )


def analyze(inp: AnalysisInput) -> AnalysisResult:
    validate_input(inp)

    control = [v for v in inp.variants if v.is_control][0]
    treatments = [v for v in inp.variants if not v.is_control]

    guardrail_results = evaluate_guardrails(inp)
    guardrails_passed = all(g.passed for g in guardrail_results)
    srm_result = evaluate_srm(inp.variants)

    if inp.method == "bayesian":
        if inp.primary_metric == "conversion_rate":
            comparisons = analyze_bayesian_conversion(inp, control, treatments)
        else:
            comparisons = analyze_bayesian_arpu(inp, control, treatments)
    elif inp.method == "frequentist_sequential":
        if inp.primary_metric == "conversion_rate":
            comparisons = analyze_frequentist_sequential_conversion(inp, control, treatments)
        else:
            comparisons = analyze_frequentist_sequential_arpu(inp, control, treatments)
    else:
        if inp.primary_metric == "conversion_rate":
            comparisons = analyze_frequentist_fixed_conversion(inp, control, treatments)
        else:
            comparisons = analyze_frequentist_ttest_arpu(inp, control, treatments)

    recommendation = recommend(inp, comparisons, guardrails_passed, srm_result)

    return AnalysisResult(
        experiment_name=inp.experiment_name,
        method=inp.method,
        control_variant=control.name,
        comparisons=comparisons,
        guardrails_passed=guardrails_passed,
        guardrails=guardrail_results,
        srm=srm_result,
        recommendation=recommendation,
    )


def evaluate_srm(variants: list[VariantInput]) -> SrmResult:
    observed = [v.visitors for v in variants]
    total = float(sum(observed))
    k = len(observed)
    expected = [total / k for _ in range(k)]

    chi2 = 0.0
    for obs, exp in zip(observed, expected):
        if exp > 0:
            chi2 += ((obs - exp) ** 2) / exp

    # Wilson-Hilferty approximation for chi-square upper-tail p-value.
    df = max(k - 1, 1)
    x = max(chi2 / df, 1e-12)
    z = ((x ** (1 / 3)) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p_value = 1 - NormalDist().cdf(z)
    passed = p_value >= 0.001
    reason = "No strong SRM evidence." if passed else "Potential sample ratio mismatch detected."

    return SrmResult(
        passed=passed,
        p_value=float(max(min(p_value, 1.0), 0.0)),
        observed=observed,
        expected=expected,
        reason=reason,
    )


def evaluate_guardrails(inp: AnalysisInput) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    for item in inp.guardrails:
        if item.control == 0:
            relative_change = item.treatment - item.control
        else:
            relative_change = (item.treatment - item.control) / abs(item.control)

        direction = item.direction.lower().strip()
        if direction not in {"increase", "decrease"}:
            raise ValueError(
                f"guardrail direction must be 'increase' or 'decrease' for {item.name}."
            )

        if direction == "decrease":
            passed = relative_change <= item.max_relative_change
            reason = (
                "Within allowed increase"
                if passed
                else "Guardrail worsened above allowed increase"
            )
        else:
            passed = relative_change >= -item.max_relative_change
            reason = (
                "Within allowed decrease"
                if passed
                else "Guardrail dropped below allowed decrease"
            )

        results.append(
            GuardrailResult(
                name=item.name,
                direction=direction,
                control=item.control,
                treatment=item.treatment,
                relative_change=float(relative_change),
                max_relative_change=item.max_relative_change,
                passed=passed,
                reason=reason,
            )
        )

    return results


def analyze_bayesian_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    rng = np.random.default_rng(inp.random_seed)

    alpha_prior = 1.0
    beta_prior = 1.0

    c_a = alpha_prior + control.conversions
    c_b = beta_prior + (control.visitors - control.conversions)
    control_samples = rng.beta(c_a, c_b, size=inp.samples)

    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_a = alpha_prior + treatment.conversions
        t_b = beta_prior + (treatment.visitors - treatment.conversions)
        treatment_samples = rng.beta(t_a, t_b, size=inp.samples)

        diff_samples = treatment_samples - control_samples
        rel_samples = np.divide(
            diff_samples,
            np.clip(control_samples, 1e-12, None),
        )

        p_win = float(np.mean(treatment_samples > control_samples))
        expected_loss = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))

        control_rate = control.conversions / control.visitors
        treatment_rate = treatment.conversions / treatment.visitors

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=treatment_rate - control_rate,
                relative_lift=(treatment_rate - control_rate) / max(control_rate, 1e-12),
                probability_beats_control=p_win,
                expected_loss=expected_loss,
                ci_low=float(np.quantile(rel_samples, 0.025)),
                ci_high=float(np.quantile(rel_samples, 0.975)),
            )
        )

    return out


def analyze_bayesian_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    rng = np.random.default_rng(inp.random_seed)

    control_samples = sample_mean_posterior(
        control.visitors,
        float(control.revenue_sum or 0.0),
        float(control.revenue_sum_squares or 0.0),
        inp.samples,
        rng,
    )

    out: list[ComparisonResult] = []
    for treatment in treatments:
        treatment_samples = sample_mean_posterior(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
            inp.samples,
            rng,
        )

        diff_samples = treatment_samples - control_samples
        rel_samples = np.divide(
            diff_samples,
            np.clip(control_samples, 1e-12, None),
        )

        p_win = float(np.mean(treatment_samples > control_samples))
        expected_loss = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))

        control_arpu = float(control.revenue_sum or 0.0) / control.visitors
        treatment_arpu = float(treatment.revenue_sum or 0.0) / treatment.visitors

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="arpu",
                control_rate=control_arpu,
                treatment_rate=treatment_arpu,
                absolute_lift=treatment_arpu - control_arpu,
                relative_lift=(treatment_arpu - control_arpu) / max(control_arpu, 1e-12),
                probability_beats_control=p_win,
                expected_loss=expected_loss,
                ci_low=float(np.quantile(rel_samples, 0.025)),
                ci_high=float(np.quantile(rel_samples, 0.975)),
            )
        )

    return out


def analyze_frequentist_sequential_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    info_fraction = inp.information_fraction
    if info_fraction is None:
        info_fraction = min(max(inp.look_index / max(inp.max_looks, 1), 1e-9), 1.0)
    else:
        info_fraction = min(max(info_fraction, 1e-9), 1.0)

    alpha_spent = obrien_fleming_alpha_spent(inp.alpha, info_fraction)

    out: list[ComparisonResult] = []
    for treatment in treatments:
        p_value, z_value, se = two_proportion_test(
            control.conversions,
            control.visitors,
            treatment.conversions,
            treatment.visitors,
        )
        significant = p_value <= alpha_spent

        control_rate = control.conversions / control.visitors
        treatment_rate = treatment.conversions / treatment.visitors

        z_crit = NormalDist().inv_cdf(1 - alpha_spent / 2)
        diff = treatment_rate - control_rate
        ci_low = diff - z_crit * se
        ci_high = diff + z_crit * se

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=diff,
                relative_lift=diff / max(control_rate, 1e-12),
                p_value=float(p_value),
                alpha_spent=float(alpha_spent),
                significant=bool(significant),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
            )
        )

    return out


def analyze_frequentist_sequential_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    info_fraction = inp.information_fraction
    if info_fraction is None:
        info_fraction = min(max(inp.look_index / max(inp.max_looks, 1), 1e-9), 1.0)
    else:
        info_fraction = min(max(info_fraction, 1e-9), 1.0)

    alpha_spent = obrien_fleming_alpha_spent(inp.alpha, info_fraction)
    out: list[ComparisonResult] = []

    for treatment in treatments:
        c_mean, c_var = mean_and_var_from_aggregates(
            control.visitors,
            float(control.revenue_sum or 0.0),
            float(control.revenue_sum_squares or 0.0),
        )
        t_mean, t_var = mean_and_var_from_aggregates(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
        )
        se = math.sqrt(max((c_var / control.visitors) + (t_var / treatment.visitors), 1e-18))
        z = (t_mean - c_mean) / se
        p_value = 2 * (1 - NormalDist().cdf(abs(z)))
        significant = p_value <= alpha_spent
        z_crit = NormalDist().inv_cdf(1 - alpha_spent / 2)
        diff = t_mean - c_mean

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="arpu",
                control_rate=c_mean,
                treatment_rate=t_mean,
                absolute_lift=diff,
                relative_lift=diff / max(c_mean, 1e-12),
                p_value=float(p_value),
                alpha_spent=float(alpha_spent),
                significant=bool(significant),
                ci_low=float(diff - z_crit * se),
                ci_high=float(diff + z_crit * se),
            )
        )

    return out


def analyze_frequentist_fixed_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    out: list[ComparisonResult] = []

    for treatment in treatments:
        p_value, _, se = two_proportion_test(
            control.conversions,
            control.visitors,
            treatment.conversions,
            treatment.visitors,
        )
        significant = p_value <= inp.alpha

        control_rate = control.conversions / control.visitors
        treatment_rate = treatment.conversions / treatment.visitors

        z_crit = NormalDist().inv_cdf(1 - inp.alpha / 2)
        diff = treatment_rate - control_rate
        ci_low = diff - z_crit * se
        ci_high = diff + z_crit * se

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=diff,
                relative_lift=diff / max(control_rate, 1e-12),
                p_value=float(p_value),
                alpha_spent=float(inp.alpha),
                significant=bool(significant),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
            )
        )

    return out


def analyze_frequentist_ttest_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    out: list[ComparisonResult] = []

    for treatment in treatments:
        c_mean, c_var = mean_and_var_from_aggregates(
            control.visitors,
            float(control.revenue_sum or 0.0),
            float(control.revenue_sum_squares or 0.0),
        )
        t_mean, t_var = mean_and_var_from_aggregates(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
        )

        p_value, df, se = welch_t_test(
            c_mean,
            c_var,
            control.visitors,
            t_mean,
            t_var,
            treatment.visitors,
        )
        significant = p_value <= inp.alpha
        diff = t_mean - c_mean
        t_crit = inverse_student_t_cdf(1 - inp.alpha / 2, df)

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="arpu",
                control_rate=c_mean,
                treatment_rate=t_mean,
                absolute_lift=diff,
                relative_lift=diff / max(c_mean, 1e-12),
                p_value=float(p_value),
                alpha_spent=float(inp.alpha),
                significant=bool(significant),
                ci_low=float(diff - t_crit * se),
                ci_high=float(diff + t_crit * se),
            )
        )

    return out


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    p1 = x1 / n1
    p2 = x2 / n2

    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n1 + 1 / n2), 1e-18))

    z = (p2 - p1) / se
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))

    unpooled_se = math.sqrt(
        max((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2), 1e-18)
    )

    return p_value, z, unpooled_se


def obrien_fleming_alpha_spent(alpha: float, info_fraction: float) -> float:
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    spent = 2 - 2 * NormalDist().cdf(z_alpha / math.sqrt(info_fraction))
    return min(max(spent, 1e-12), alpha)


def mean_and_var_from_aggregates(n: int, value_sum: float, value_sum_squares: float) -> tuple[float, float]:
    mean = value_sum / max(n, 1)
    if n <= 1:
        return mean, 0.0
    centered = value_sum_squares - (value_sum * value_sum) / n
    variance = max(centered / (n - 1), 0.0)
    return mean, variance


def sample_mean_posterior(
    n: int,
    value_sum: float,
    value_sum_squares: float,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # Normal-Inverse-Gamma weak prior over unknown mean/variance using aggregate stats.
    mu0 = 0.0
    kappa0 = 1e-6
    alpha0 = 1.0
    beta0 = 1.0

    x_bar = value_sum / max(n, 1)
    ss = max(value_sum_squares - n * (x_bar**2), 0.0)

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * ss + (kappa0 * n * ((x_bar - mu0) ** 2)) / (2 * kappa_n)

    gamma_samples = rng.gamma(shape=alpha_n, scale=1.0, size=num_samples)
    sigma2_samples = beta_n / np.clip(gamma_samples, 1e-18, None)
    mu_samples = rng.normal(mu_n, np.sqrt(np.clip(sigma2_samples / kappa_n, 1e-18, None)))
    return mu_samples


def welch_t_test(
    mean1: float,
    var1: float,
    n1: int,
    mean2: float,
    var2: float,
    n2: int,
) -> tuple[float, float, float]:
    se2 = max((var1 / max(n1, 1)) + (var2 / max(n2, 1)), 1e-18)
    se = math.sqrt(se2)
    t_value = (mean2 - mean1) / se

    num = se2 * se2
    den_left = 0.0 if n1 <= 1 else ((var1 / n1) ** 2) / (n1 - 1)
    den_right = 0.0 if n2 <= 1 else ((var2 / n2) ** 2) / (n2 - 1)
    den = max(den_left + den_right, 1e-18)
    df = max(num / den, 1.0)

    cdf = student_t_cdf(abs(t_value), df)
    p_value = 2 * (1 - cdf)
    return float(min(max(p_value, 0.0), 1.0)), float(df), float(se)


def student_t_cdf(t_value: float, degrees_freedom: float) -> float:
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be > 0")
    if t_value == 0:
        return 0.5

    x = degrees_freedom / (degrees_freedom + (t_value * t_value))
    reg = regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)
    if t_value > 0:
        return 1 - 0.5 * reg
    return 0.5 * reg


def inverse_student_t_cdf(probability: float, degrees_freedom: float) -> float:
    if probability <= 0 or probability >= 1:
        raise ValueError("probability must be in (0, 1)")

    if probability == 0.5:
        return 0.0

    sign = 1.0
    target = probability
    if probability < 0.5:
        sign = -1.0
        target = 1.0 - probability

    low = 0.0
    high = 32.0
    while student_t_cdf(high, degrees_freedom) < target:
        high *= 2.0
        if high > 1e6:
            break

    for _ in range(80):
        mid = 0.5 * (low + high)
        if student_t_cdf(mid, degrees_freedom) < target:
            low = mid
        else:
            high = mid

    return sign * (0.5 * (low + high))


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + (a * math.log(x))
        + (b * math.log(1 - x))
    )

    if x < (a + 1) / (a + b + 2):
        return bt * beta_continued_fraction(a, b, x) / a
    return 1 - (bt * beta_continued_fraction(b, a, 1 - x) / b)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - (qab * x / qap)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, 401):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 1e-12:
            break

    return h


def recommend(
    inp: AnalysisInput,
    comparisons: list[ComparisonResult],
    guardrails_passed: bool,
    srm: SrmResult,
) -> Recommendation:
    risk_flags: list[str] = []
    if not srm.passed:
        risk_flags.append("srm_detected")
    if not guardrails_passed:
        risk_flags.append("guardrail_failure")

    if not srm.passed:
        return Recommendation(
            action="investigate_data_quality",
            rationale="SRM check failed; assignment or tracking may be biased.",
            decision_confidence=0.99,
            next_best_action="Audit randomization, traffic filters, and event logging before shipping.",
            risk_flags=risk_flags,
        )

    if not guardrails_passed:
        return Recommendation(
            action="do_not_ship",
            rationale="One or more guardrails failed. Keep test running or roll back.",
            decision_confidence=0.95,
            next_best_action="Fix guardrail regressions or reduce impact before re-testing.",
            risk_flags=risk_flags,
        )

    best = max(comparisons, key=lambda x: x.absolute_lift)

    if inp.method == "bayesian":
        prob_threshold = inp.decision_thresholds.bayes_prob_beats_control
        loss_threshold = inp.decision_thresholds.max_expected_loss

        if (
            best.probability_beats_control is not None
            and best.expected_loss is not None
            and best.absolute_lift > 0
            and best.probability_beats_control >= prob_threshold
            and best.expected_loss <= loss_threshold
        ):
            return Recommendation(
                action=f"ship_{best.treatment}",
                rationale=(
                    f"{best.treatment} passes Bayesian thresholds "
                    f"(P(win)={best.probability_beats_control:.3f}, "
                    f"expected_loss={best.expected_loss:.6f})."
                ),
                decision_confidence=float(best.probability_beats_control),
                next_best_action="Roll out gradually and monitor guardrails.",
                risk_flags=risk_flags,
            )

        return Recommendation(
            action="continue_collecting_data",
            rationale="Bayesian confidence or expected loss thresholds were not met.",
            decision_confidence=float(best.probability_beats_control or 0.5),
            next_best_action="Collect more samples until decision thresholds are reached.",
            risk_flags=risk_flags,
        )

    if best.significant and best.absolute_lift > 0:
        return Recommendation(
            action=f"ship_{best.treatment}",
            rationale=(
                f"Sequential significance reached for {best.treatment} "
                f"(p={best.p_value:.5f} <= alpha_spent={best.alpha_spent:.5f})."
            ),
            decision_confidence=0.95,
            next_best_action="Roll out gradually and continue guardrail monitoring.",
            risk_flags=risk_flags,
        )

    if best.significant and best.absolute_lift < 0:
        return Recommendation(
            action="stop_and_rollback",
            rationale="Significant negative lift detected under sequential testing.",
            decision_confidence=0.95,
            next_best_action="Rollback treatment and investigate adverse drivers.",
            risk_flags=risk_flags,
        )

    return Recommendation(
        action="continue_collecting_data",
        rationale="Sequential significance boundary not reached yet.",
        decision_confidence=0.5,
        next_best_action="Wait for more information fraction or sample size.",
        risk_flags=risk_flags,
    )
