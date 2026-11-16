from numpy import quantile
from dibh.split_conformal import adjust_quantile
from dibh.split_conformal import calculate_q_hat


residuals = [1, 2, 3, 4, 5]
alpha = 0.2

p_star = adjust_quantile(alpha, len(residuals))
print(quantile(residuals, 1 - alpha, method = "inverted_cdf"))
print(p_star)

q_hat = calculate_q_hat(residuals, alpha)

print(q_hat)
