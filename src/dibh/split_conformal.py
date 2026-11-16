from math import ceil
from numpy import quantile
# design scratch
# residuals: calibration split residuals
# alpha: confidence level in [0, 1]
def calculate_q_hat (residuals, alpha):
    p_star = adjust_quantile(alpha, len(residuals))
    return(quantile(residuals, p_star, method = "inverted_cdf"))

def adjust_quantile (alpha, n):
    # missing handling for outputs > 1
    return ceil((1 - alpha) * (n + 1)) / n

# m: a model fit with a predict method
# y_c: calibration targets
# x_c: calibration predictors
# alpha: confidence level
# x_t: test predictors
def split_conformal(m, y_c, x_c, alpha, x_t):
    y_c_hat = m.predict(x_c)
    residuals = abs(y_c_hat - y_c)
    q_hat = calculate_q_hat(residuals, alpha)
    y_t_hat = m.predict(x_t)
    lower = y_t_hat - q_hat
    upper = y_y_hat + q_hat
    return([lower, upper])

class TwoNN:
    def __init__(self, y, x):
        self.
