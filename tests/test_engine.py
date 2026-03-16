import math
import unittest

from scipy.stats import chi2
from statsmodels.stats.proportion import proportions_ztest

from bayestest.engine import (
    analyze,
    evaluate_srm,
    mean_and_var_from_aggregates,
    parse_payload,
    two_proportion_test,
)
from bayestest.models import VariantInput


class EngineTests(unittest.TestCase):
    def test_bayesian_recommends_ship_for_clear_winner(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 10000, "conversions": 500, "is_control": True},
                {"name": "treatment", "visitors": 10000, "conversions": 650, "is_control": False},
            ],
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.9,
                "max_expected_loss": 0.01,
            },
            "samples": 30000,
            "random_seed": 11,
        }

        result = analyze(parse_payload(payload))
        self.assertTrue(result.recommendation.action.startswith("ship_"))
        self.assertGreaterEqual(result.recommendation.decision_confidence, 0.9)

    def test_guardrail_failure_blocks_ship(self):
        payload = {
            "experiment_name": "exp",
            "method": "frequentist_sequential",
            "alpha": 0.05,
            "look_index": 10,
            "max_looks": 10,
            "variants": [
                {"name": "control", "visitors": 50000, "conversions": 3000, "is_control": True},
                {"name": "treatment", "visitors": 50000, "conversions": 3500, "is_control": False},
            ],
            "guardrails": [
                {
                    "name": "p95_latency_ms",
                    "control": 200,
                    "treatment": 240,
                    "direction": "decrease",
                    "max_relative_change": 0.1,
                }
            ],
        }

        result = analyze(parse_payload(payload))
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertIn("guardrail_failure", result.recommendation.risk_flags)

    def test_srm_failure_blocks_decision(self):
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 9900, "conversions": 500, "is_control": True},
                {"name": "treatment", "visitors": 100, "conversions": 20, "is_control": False},
            ],
            "samples": 5000,
            "random_seed": 1,
        }

        result = analyze(parse_payload(payload))
        self.assertFalse(result.srm.passed)
        self.assertEqual(result.recommendation.action, "investigate_data_quality")

    def test_bayesian_arpu_probability_to_win(self):
        payload = {
            "experiment_name": "arpu_exp",
            "method": "bayesian",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 5000,
                    "conversions": 300,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 60000.0,
                    "is_control": True,
                },
                {
                    "name": "treatment",
                    "visitors": 5000,
                    "conversions": 320,
                    "revenue_sum": 11500.0,
                    "revenue_sum_squares": 77000.0,
                    "is_control": False,
                },
            ],
            "samples": 10000,
            "random_seed": 7,
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.8,
                "max_expected_loss": 0.2,
            },
        }
        result = analyze(parse_payload(payload))
        comp = result.comparisons[0]
        self.assertEqual(comp.metric, "arpu")
        self.assertIsNotNone(comp.probability_beats_control)
        self.assertGreater(comp.probability_beats_control, 0.5)

    def test_multi_variant_selects_best_treatment(self):
        payload = {
            "experiment_name": "multi_variant",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "variants": [
                {"name": "control", "visitors": 20000, "conversions": 900, "is_control": True},
                {"name": "treatment_a", "visitors": 20000, "conversions": 940, "is_control": False},
                {"name": "treatment_b", "visitors": 20000, "conversions": 1010, "is_control": False},
            ],
            "samples": 15000,
            "random_seed": 9,
            "decision_thresholds": {
                "bayes_prob_beats_control": 0.9,
                "max_expected_loss": 0.01,
            },
        }
        result = analyze(parse_payload(payload))
        self.assertEqual(len(result.comparisons), 2)
        self.assertEqual(result.recommendation.action, "ship_treatment_b")

    def test_srm_matches_scipy_chi_square_tail(self):
        variants = [
            VariantInput(name="control", visitors=5200, conversions=260, is_control=True),
            VariantInput(name="treatment", visitors=4800, conversions=250, is_control=False),
        ]

        result = evaluate_srm(variants)

        total = 5200 + 4800
        expected = total / 2
        chi2_stat = ((5200 - expected) ** 2) / expected + ((4800 - expected) ** 2) / expected
        expected_p_value = chi2.sf(chi2_stat, df=1)

        self.assertAlmostEqual(result.p_value, expected_p_value, places=4)
        self.assertEqual(result.passed, expected_p_value >= 0.001)

    def test_two_proportion_test_matches_statsmodels(self):
        x1, n1 = 500, 10_000
        x2, n2 = 650, 10_000

        p_value, z_value, unpooled_se = two_proportion_test(x1, n1, x2, n2)
        expected_z, expected_p_value = proportions_ztest([x2, x1], [n2, n1])

        p1 = x1 / n1
        p2 = x2 / n2
        expected_unpooled_se = math.sqrt(
            (p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)
        )

        self.assertAlmostEqual(z_value, expected_z, places=10)
        self.assertAlmostEqual(p_value, expected_p_value, places=10)
        self.assertAlmostEqual(unpooled_se, expected_unpooled_se, places=10)

    def test_mean_and_var_from_aggregates_matches_hand_worked_example(self):
        mean, variance = mean_and_var_from_aggregates(
            n=4,
            value_sum=14.0,
            value_sum_squares=54.0,
        )

        self.assertAlmostEqual(mean, 3.5)
        self.assertAlmostEqual(variance, 5 / 3)


if __name__ == "__main__":
    unittest.main()
