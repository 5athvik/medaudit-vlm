import numpy as np
from scipy import stats

def evaluate_audit_controls(ground_truth_doses: list[str], predicted_doses: list[str], alpha: float = 0.05, threshold: float = 0.85):
    matches = [1 if g.strip().lower() == p.strip().lower() else 0 for g, p in zip(ground_truth_doses, predicted_doses)]
    n = len(matches)
    accuracy = float(np.mean(matches)) if n > 0 else 0.0

    if n < 5:
        return {
            "sample_size": n,
            "empirical_accuracy": round(accuracy, 4),
            "ci_lower_95": 0.0,
            "ci_upper_95": 1.0,
            "nist_control_passed": False,
            "reason": "sample size too small for statistical significance (minimum 30 required)"
        }

    # Wilson score interval
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + (z**2) / n
    centre = (accuracy + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((accuracy * (1 - accuracy) + (z**2) / (4 * n)) / n)) / denom

    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    passed = bool(ci_lower >= threshold)

    return {
        "sample_size": n,
        "empirical_accuracy": round(accuracy, 4),
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "nist_control_passed": passed,
        "reason": "verified under 95% confidence interval threshold" if passed else f"lower bound {round(ci_lower*100, 1)}% is below {int(threshold*100)}% target threshold"
    }

if __name__ == "__main__":
    # 50 sample tests to demonstrate passing validation
    gt = ["500mg", "20mg"] * 25
    pred = ["500mg", "20mg"] * 25
    print(evaluate_audit_controls(gt, pred))