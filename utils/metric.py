from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,recall_score,precision_score,accuracy_score
import numpy as np
from scipy.stats import norm, pearsonr
from scipy import stats

def calculate_rmse(y_predict, y_true):
    return np.sqrt(np.mean((y_true - y_predict)**2))


def calc_corr2(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    sq = (sum([(x - a_avg) ** 2 for x in a]) * sum([(y - b_avg) ** 2 for y in b])) ** 0.5
    corr_factor = cov_ab / (sq+1e-8)
    return corr_factor


def weighted_correlation(Y_hat, Y, groups):
    
    unique_groups = np.unique(groups)
    total_n = len(Y)
    
    weighted_sum = 0
    sum_n = 0
    
    for group in unique_groups:
        indices = np.where(groups == group)
        Y_group = Y[indices]
        Y_hat_group = Y_hat[indices]
        nh = len(Y_group)
        
        if nh > 1:  # Pearson correlation requires at least two data points
            correlation, _ = pearsonr(Y_group, Y_hat_group)
            weighted_sum += correlation * nh
            sum_n += nh

    if sum_n == 0:
        raise ValueError("Total number of samples must not be zero, and each group must contain at least two samples.")

    return weighted_sum / sum_n


def CI95(mae_values):

    mean_mae = np.mean(mae_values)
    std_dev = np.std(mae_values)
    n = len(mae_values)
    std_err = std_dev / np.sqrt(n)
    margin_of_error = 1.96 * std_err
    lower_bound = mean_mae - margin_of_error
    upper_bound = mean_mae + margin_of_error
    return lower_bound,upper_bound


def calculate_pcc_ci(predictions, labels, confidence_level=0.95):
    """
    Calculates the 95% confidence interval (CI) of the Pearson correlation coefficient (PCC).

    Args:
        predictions (list): List of prediction values.
        labels (list): List of label values.
        confidence_level (float, optional): Confidence level for the interval (default: 0.95).

    Returns:
        tuple: Tuple containing the lower and upper bounds of the confidence interval.

    """
    # Calculate the PCC
    pcc = np.corrcoef(predictions, labels)[0, 1]

    n_samples = len(predictions)

    # Fisher transformation
    Z = 0.5 * np.log((1 + pcc) / (1 - pcc))

    # Standard error of the Fisher transformation
    S = 1 / np.sqrt(n_samples - 3)

    # Calculate the margin of error
    zl = Z - 1.96 * S
    zu = Z + 1.96 * S

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = (np.exp(2 * zl) - 1) / (np.exp(2 * zl) + 1)
    upper_bound = (np.exp(2 * zu) - 1) / (np.exp(2 * zu) + 1)

    return lower_bound, upper_bound


def calculate_r2_ci(predictions, labels, confidence_level=0.95):
    """
    Calculates the 95% confidence interval (CI) of the R-squared (R2) score using the Fisher transformation.

    Args:
        predictions (list): List of prediction values.
        labels (list): List of label values.
        confidence_level (float, optional): Confidence level for the interval (default: 0.95).

    Returns:
        tuple: Tuple containing the lower and upper bounds of the confidence interval.

    """
    r_squared = r2_score(labels, predictions)
    n_samples = len(predictions)

    # Fisher transformation
    Z = 0.5 * np.log((1 + r_squared) / (1 - r_squared))
    print('test:', r_squared)

    # Standard error of the Fisher transformation
    S = 1 / np.sqrt(n_samples - 3)

    # Calculate the margin of error
    zl = Z - 1.96 * S
    zu = Z + 1.96 * S

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = (np.exp(2 * zl) - 1) / (np.exp(2 * zl) + 1)
    upper_bound = (np.exp(2 * zu) - 1) / (np.exp(2 * zu) + 1)

    return lower_bound, upper_bound


def bootstrap_wR_ci(Y_hat, Y, groups, n_iterations=1000, ci=95):
    bootstrap_estimates = []
    np.random.seed(42)  # for reproducible results

    for _ in range(n_iterations):
        # Generate bootstrap sample: resample with replacement
        indices = np.random.choice(len(Y), size=len(Y), replace=True)
        bootstrap_Y = Y[indices]
        bootstrap_Y_hat = Y_hat[indices]
        bootstrap_groups = groups[indices]

        # Calculate weighted correlation for the bootstrap sample
        estimate = weighted_correlation(bootstrap_Y, bootstrap_Y_hat, bootstrap_groups)
        bootstrap_estimates.append(estimate)

    # Calculate the empirical CI
    lower_bound = np.percentile(bootstrap_estimates, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_estimates, 100 - (100 - ci) / 2)
    
    return lower_bound, upper_bound


def bootstrap_rmse_ci(y_pred, y_true, n_iterations=1000, ci=95):
    """Calculate the RMSE and its confidence interval using bootstrap."""
    rmse_estimates = []
    np.random.seed(42)  # for reproducible results

    for _ in range(n_iterations):
        # Generate bootstrap sample: resample with replacement
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        bootstrap_y_true = y_true[indices]
        bootstrap_y_pred = y_pred[indices]

        # Calculate RMSE for the bootstrap sample
        rmse = calculate_rmse(bootstrap_y_true, bootstrap_y_pred)
        rmse_estimates.append(rmse)

    # Calculate the empirical CI
    lower_bound = np.percentile(rmse_estimates, (100 - ci) / 2)
    upper_bound = np.percentile(rmse_estimates, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound


def return_all_metric(label, prediction, next_label):
    """
    Calculates and return all metric values including MAE, R2, PCC along with their 95% confidence interval (CI).

    Args:
        predictions (list): List of prediction values.
        labels (list): List of label values.

    """
    MAE = mean_absolute_error(label.squeeze(),prediction.squeeze())
    r2 = r2_score(label.squeeze(),prediction.squeeze())
    pccs = calc_corr2(prediction.squeeze().tolist(), label.squeeze().tolist())
    rmse_value = calculate_rmse(label.squeeze(),prediction.squeeze())
    wR = weighted_correlation(prediction.squeeze(), label.squeeze(), next_label.squeeze())

    # 95CI #######
    MAE_CI95 = CI95(np.abs(prediction.squeeze() - label.squeeze()))
    R2_CI95 = calculate_r2_ci(prediction.squeeze(), label.squeeze())
    pcc_CI95 = calculate_pcc_ci(prediction.squeeze(), label.squeeze())
    rmse_CI95 = bootstrap_rmse_ci(prediction.squeeze(), label.squeeze())
    wR_CI95 = bootstrap_wR_ci(prediction.squeeze(), label.squeeze(), next_label.squeeze())
    

    return MAE, r2, pccs, rmse_value, wR, MAE_CI95, R2_CI95, pcc_CI95, rmse_CI95, wR_CI95

    

