import numpy as np
import scipy.signal as spsig


class FrequencyAnalysis:
    """
    Methods related to performing frequency-domain analysis.
    """

    @staticmethod
    def get_filtered_signal(signal, freq_cutoff, fs=1 / 6, low_pass=True):
        """Applies a low or high pass filter and returns the inverse result.
        Filters by zeroing all components above or below threshold based on `low_pass` flag.

        Parameters:
            signal (ndarray):
                Signal that will be filtered.
            freq_cutoff (float):
                Frequency threshold.
            fs (float):
                Sampling frequency, ie. interval of acquisition between frames.
            low_pass (boolean):
                Flag that controls if the method will apply a low pass filter or high pass filter.
        """
        N = len(signal)
        freqs = np.fft.rfftfreq(N, 1 / fs)
        fft = np.fft.rfft(signal)
        if low_pass:
            mask = freqs < freq_cutoff
        else:
            mask = freqs > freq_cutoff
        filtered_fft = fft * mask
        filtered_signal = np.fft.irfft(filtered_fft, n=N)

        return filtered_signal

    @staticmethod
    def calculate_STFT(signal, fs=1 / 6, fft_size=600, noverlap=450):
        """Calculate the Short Time Fourier Transform for a signal.

        It replaces None values with 1e-11.

        Parameters:
            signal (ndarray):
                Preprocessed signal.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            noverlap (int):
                Num frames to overlap between segments.

        Returns:
            f (ndarray):
                STFT frequency bins.
            t (ndarray):
                STFT time columns.
            magnitude (ndarray):
                STFT magnitude (excludes phase).
        """
        signal = np.where(signal == None, 1e-11, signal).astype(float)
        return spsig.stft(
            signal, fs, detrend="constant", nperseg=fft_size, noverlap=noverlap
        )

    @staticmethod
    def apply_hipass_filter(signal, cutoff, fs=1 / 6, numtaps=501):
        """Apply a high pass finite impulse response filter to the signal.

        Parameters:
            signal (ndarray)
                signal values.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            numtaps (int):
                Num of coefficients in the filter.

        Returns:
            hipass_dff (ndarray):
                input signal with filter applied.
        """
        delay = int((numtaps - 1) / 2)
        padded_dff = np.pad(signal, (0, delay), mode="constant")

        fir_hipass = spsig.firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
        padded_hipass_dff = spsig.lfilter(fir_hipass, [1.0], padded_dff)
        adjusted_hipass_dff = padded_hipass_dff[delay:]

        return adjusted_hipass_dff

    @staticmethod
    def apply_lopass_filter(signal, cutoff, fs=1 / 6, numtaps=501):
        """Apply a low pass finite impulse response filter to the signal.

        Parameters:
            signal (ndarray)
                signal values.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            numtaps (int):
                Num of coefficients in the filter

        Returns:
            lopass_dff (ndarray):
                input signal with filter applied
        """
        delay = int((numtaps - 1) / 2)
        padded_dff = np.pad(signal, (0, delay), mode="constant")

        fir_lopass = spsig.firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=True)
        padded_lopass_dff = spsig.lfilter(fir_lopass, [1.0], padded_dff)
        adjusted_lopass_dff = padded_lopass_dff[delay : delay + len(signal)]

        return adjusted_lopass_dff
