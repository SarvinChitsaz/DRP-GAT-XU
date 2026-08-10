import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon

def bootstrap_r2_ci(predictions, labels, n_bootstrap=1000, seed=42):
    predictions = np.array(predictions)
    labels = np.array(labels)
    n_samples = len(labels)

    np.random.seed(seed)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = r2_score(labels[indices], predictions[indices])
        bootstrap_scores.append(score)

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)
    return lower, upper


def compare_models_wilcoxon(predictions_a, labels_a, predictions_b, labels_b):
    errors_a = [(p - l) ** 2 for p, l in zip(predictions_a, labels_a)]
    errors_b = [(p - l) ** 2 for p, l in zip(predictions_b, labels_b)]

    min_len = min(len(errors_a), len(errors_b))
    stat, p_value = wilcoxon(errors_a[:min_len], errors_b[:min_len])
    return stat, p_value


def mean_absolute_error_by_group(df, group_column, predicted_column, actual_column):
    df = df.copy()
    df["abs_error"] = (df[predicted_column] - df[actual_column]).abs()
    return df.groupby(group_column)["abs_error"].agg(["mean", "std", "count"]).sort_values("mean")
