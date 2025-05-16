import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Simulate ECG signal
duration = 10  # Duration in seconds
sampling_rate = 500  # Sampling rate in Hz

ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate)

# Plot the simulated ECG signal
plt.figure(figsize=(10, 5))
plt.plot(ecg_signal)
plt.title("Simulated ECG Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the filter
def apply_filter(data, lowcut=0.5, highcut=50.0, fs=500):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal

filtered_ecg = apply_filter(ecg_signal, lowcut=0.5, highcut=50.0, fs=500)

# Plot the filtered ECG signal
plt.figure(figsize=(10, 5))
plt.plot(filtered_ecg)
plt.title("Filtered ECG Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Detect R-peaks and calculate heart rate
def detect_r_peaks(ecg_signal, sampling_rate=500):
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    r_peaks = info["ECG_R_Peaks"]
    rr_intervals = np.diff(r_peaks) / sampling_rate
    heart_rate = 60 / rr_intervals
    return heart_rate, r_peaks

heart_rate, r_peaks = detect_r_peaks(filtered_ecg, sampling_rate=500)

# Plot the filtered ECG signal with R-peaks
plt.figure(figsize=(10, 5))
plt.plot(filtered_ecg, label="Filtered ECG")
plt.scatter(r_peaks, filtered_ecg[r_peaks], color='red', label="R-peaks")
plt.title("Filtered ECG Signal with R-Peaks")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and display the average heart rate
avg_heart_rate = np.mean(heart_rate)
print(f"Average Heart Rate: {avg_heart_rate:.2f} bpm")
print(f"Heart Rate Values: {heart_rate}")