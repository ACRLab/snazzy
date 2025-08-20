from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import erf

from golf_analysis import Experiment


# Skewed Gaussian function
def skewed_gaussian(x, A, mu, sigma, skew):
    norm_gauss = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    skew_factor = 1 + erf(skew * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gauss * skew_factor


# Ripple function (damped sine wave)
def ripple(x, A_r, mu_r, tau_r, f_r):
    ripple_wave = (
        A_r * np.exp(-(x - mu_r) / tau_r) * np.sin(2 * np.pi * f_r * (x - mu_r))
    )
    ripple_wave[x < mu_r] = 0  # Ensure ripples only happen after the peak
    return ripple_wave


# Full model with multiple peaks and optional ripples
def multi_skewed_gaussian_with_ripples(x, *params):
    num_peaks = len(params) // 4  # Each peak has 4 parameters: A, mu, sigma, skew
    total_signal = np.zeros_like(x)

    for i in range(num_peaks):
        A, mu, sigma, skew = params[i * 4 : (i + 1) * 4]
        total_signal += skewed_gaussian(x, A, mu, sigma, skew)

        # # Add ripples if necessary
        # if should_add_ripples(y, peaks, properties)[i]:
        #     total_signal += ripple(x, A_r=0.3 * A, mu_r=mu + 2, tau_r=5, f_r=2)

    return total_signal


# Automatically detect peaks
def detect_peaks(y, min_height=None, min_prominence=None):
    """Automatically adjust peak detection parameters based on the signal's properties."""
    if min_height is None:
        min_height = np.percentile(
            y, 95
        )  # Adaptive threshold based on top 10% intensities
    if min_prominence is None:
        min_prominence = (np.max(y) - np.min(y)) * 0.05  # Adaptive prominence

    peaks, properties = find_peaks(
        y, height=min_height, distance=60, prominence=min_prominence, width=5
    )
    return peaks, properties


# Automatically initialize fitting parameters based on peaks
def generate_initial_guesses(x, y):
    peaks, properties = detect_peaks(y, min_height=0.05)
    initial_guesses = []

    for i, peak in enumerate(peaks):
        A_guess = properties["peak_heights"][i]
        mu_guess = x[peak]
        sigma_guess = properties["widths"][i] / 2
        skew_guess = 0  # Assume symmetric initially

        initial_guesses.extend([A_guess, mu_guess, sigma_guess, skew_guess])

    return initial_guesses


# Determine which peaks should have ripples
def should_add_ripples(y, peaks, properties):
    ripple_flags = np.zeros(len(peaks), dtype=bool)
    prominence_threshold = np.percentile(properties["prominences"], 75)  # Top 25% peaks
    for i, prom in enumerate(properties["prominences"]):
        if prom > prominence_threshold:
            ripple_flags[i] = True
    return ripple_flags


# Fit a single trace
def fit_trace(x, y):
    """Fully automated process for fitting a single trace"""
    peaks, properties = detect_peaks(y, min_height=0.05)
    print(len(peaks))
    print(properties)
    if len(peaks) == 0:
        return None  # Skip if no peaks are found

    initial_guesses = generate_initial_guesses(x, y)

    try:
        popt, _ = curve_fit(
            multi_skewed_gaussian_with_ripples, x, y, p0=initial_guesses
        )
        return popt, peaks
    except RuntimeError:
        return None, None  # If fitting fails, return None


exp = Experiment(
    Path("/home/cdp58/Documents/repos/pasna_analysis/data/20240514_testing"),
    to_exclude=[1, 2],
    # Path("/home/cdp58/Documents/repos/pasna_analysis/data/202409011-vglutdf")
)
tr = exp.embryos["emb3"].trace.dff[900:]
x = exp.embryos["emb3"].trace.time[900:]
# Fit the synthetic trace
fitted_params, peaks = fit_trace(x, tr)
print(fitted_params)

# Generate fitted curve
if fitted_params is not None:
    fitted_signal = multi_skewed_gaussian_with_ripples(x, *fitted_params)
else:
    print("Fitting failed.")

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(x, tr, label="Real Signal", color="blue", alpha=0.7)
if fitted_params is not None:
    plt.plot(x, fitted_signal, label="Fitted Model", color="red", linestyle="dashed")
    plt.plot(x[peaks], tr[peaks], "r.")
plt.xlabel("Time (or x-axis)")
plt.ylabel("Signal Intensity")
plt.legend()
plt.title("Fitted Signal with Multiple Asymmetric Gaussians and Ripples")
plt.show()
