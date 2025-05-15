import numpy as np


def divergence_score(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    prob_good = y_prob[y_true == 0]
    prob_bad = y_prob[y_true == 1]

    mu_good, mu_bad = np.mean(prob_good), np.mean(prob_bad)
    var_good, var_bad = np.var(prob_good), np.var(prob_bad)

    if var_good == 0 or var_bad == 0:
        return 0.0

    divergence = (mu_good - mu_bad) ** 2 / (0.5 * (var_good + var_bad))
    return divergence
