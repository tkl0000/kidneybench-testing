import numpy as np
import pandas as pd
from scipy import stats

def extract_metrics(filepath):
    df = pd.read_csv(filepath)

    avg_wait = df['patient_wait_time'].mean()
    stdev_wait = df['patient_wait_time'].std()
    stderr_wait = df['patient_wait_time'].sem()
    avg_matching_score = df['matching_score'].mean()
    stdev_matching_score = df['matching_score'].std()
    stderr_matching_score = df['matching_score'].sem()
    match_rate = (df["transplant_outcome"] == "Success").mean()
    # avg_score = df['matching_score'].mean()
    # avg_expected_life = df['expected_lifespan'].mean()
    # avg_organ_quality = df['organ_quality'].mean()
    # avg_donor_age = df['donor_age'].mean()
    # avg_patient_urgency = df['patient_urgency'].mean()
    
    return {
        "match_rate" : match_rate,
        "avg_wait" : avg_wait,
        "stdev_wait" : stdev_wait,
        "stderr_wait" : stderr_wait,
        "avg_matching_score" : avg_matching_score,
        "stdev_matching_score" : stdev_matching_score,
        "stderr_matching_score" : stderr_matching_score,
    }

def extract_wait_times(filepath):
    df = pd.read_csv(filepath)
    return df['patient_wait_time'].values

def extract_matching_scores(filepath):
    df = pd.read_csv(filepath)
    return df['matching_score'].values

def bootstrap_ci(diff_array, num_samples=10000, ci=0.95):
    bootstrapped_means = []
    n = len(diff_array)
    for _ in range(num_samples):
        sample = np.random.choice(diff_array, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return (lower, upper)

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt(((np.std(a, ddof=1) ** 2) + (np.std(b, ddof=1) ** 2)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std

def compare_metrics(metrics_a: dict, metrics_b: dict):
    keys = set(metrics_a.keys()) & set(metrics_b.keys())
    vals_a = np.array([metrics_a[k] for k in keys])
    vals_b = np.array([metrics_b[k] for k in keys])
    diffs = vals_a - vals_b

    print("Running statistical tests on", len(keys), "paired samples...\n")

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(vals_a, vals_b)
    print(f"Paired t-test: t = {t_stat:.4f}, p = {t_pval:.4f}")

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(diffs)
    print(f"Bootstrap 95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Mann-Whitney U-test
    u_stat, u_pval = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    print(f"Mann-Whitney U-test: U = {u_stat:.4f}, p = {u_pval:.4f}")

    # Kolmogorov–Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(vals_a, vals_b)
    print(f"KS test: D = {ks_stat:.4f}, p = {ks_pval:.4f}")

    # Cohen's d
    d = cohens_d(vals_a, vals_b)
    print(f"Cohen's d: {d:.4f}")