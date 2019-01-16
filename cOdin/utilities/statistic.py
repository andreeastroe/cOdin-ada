# provide functionality for running statistical tests (1 points);
import numpy as np
import scipy.stats as sts

def bartlett_test(n, l, x, e):
    m, q = np.shape(l)
    v = np.corrcoef(x, rowvar=False)
    psi = np.diag(e)
    v_ = l @ np.transpose(l) + psi
    I_ = np.linalg.inv(v_) @ v
    det_v_ = np.linalg.det(I_)
    trace_I = np.trace(I_)
    chi2_computed = (n - 1 - (2 * m + 4 * q - 5) / 2) * (trace_I - np.log(det_v_) - m)
    dof = ((m - q) * (m - q) - m - q) / 2
    chi2_estimated = sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated

def bartlett_factor(x):
    n, m = np.shape(x)
    r = np.corrcoef(x, rowvar=False)
    chi2_computed = -(n - 1 - (2 * m + 5) / 6) * np.log(np.linalg.det(r))
    dof = m * (m - 1) / 2
    chi2_estimated = 1 - sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated

def bartlett_wilks(r, n, p, q, m):
    r_inv = np.flipud(r)
    l = np.flipud(np.cumprod(1 - r_inv * r))
    dof = (p - np.arange(m)) * (q - np.arange(m))
    chi2_computed = (-n + 1 + (p + q + 1) / 2) * np.log(l)
    chi2_estimated = 1 - sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated
