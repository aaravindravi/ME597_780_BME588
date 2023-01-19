"""
Utilities for processing the EEG dataset.
"""
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CHANNEL_MAPPING = {'PO7': 0,
                   'PO3': 1,
                   'POZ': 2,
                   'PO4': 3,
                   'O1': 4,
                   'OZ': 5,
                   'O2': 6}

FFT_PARAMS = {
    'resolution': 0.1,
    'start_frequency': 5,
    'end_frequency': 35.0,
    'sampling_rate': 256
}

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    '''
    Returns bandpass filtered data between the frequency ranges specified in the input.

    Args:
        data (numpy.ndarray): array of samples. 
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        sample_rate (float): sampling rate (Hz).
        order (int): order of the bandpass filter.

    Returns:
        (numpy.ndarray): bandpass filtered data.
    '''
    
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_filtered_eeg(eeg, lowcut, highcut, order, sample_rate):
    '''
    Returns bandpass filtered eeg for all channels and trials.

    Args:
        eeg (numpy.ndarray): raw eeg data of shape (num_classes, num_channels, num_samples, num_trials).
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        order (int): order of the bandpass filter.
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): bandpass filtered eeg of shape (num_classes, num_channels, num_samples, num_trials).
    '''
    
    num_classes = eeg.shape[0]
    num_chan = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]
    
    trial_len = int(38+0.135*sample_rate+4*sample_rate-1) - int(38+0.135*sample_rate)
    filtered_data = np.zeros((eeg.shape[0], eeg.shape[1], trial_len, eeg.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                signal_to_filter = np.squeeze(eeg[target, channel, int(38+0.135*sample_rate):
                                               int(38+0.135*sample_rate+4*sample_rate-1), 
                                               trial])
                filtered_data[target, channel, :, trial] = butter_bandpass_filter(signal_to_filter, lowcut, 
                                                                                  highcut, sample_rate, order)
    return filtered_data

def buffer(data, duration, data_overlap):
    '''
    Returns segmented data based on the provided input window duration and overlap.

    Args:
        data (numpy.ndarray): array of samples. 
        duration (int): window length (number of samples).
        data_overlap (int): number of samples of overlap.

    Returns:
        (numpy.ndarray): segmented data of shape (number_of_segments, duration).
    '''
    
    number_segments = int(math.ceil((len(data) - data_overlap)/(duration - data_overlap)))
    temp_buf = [data[i:i+duration] for i in range(0, len(data), (duration - int(data_overlap)))]
    temp_buf[number_segments-1] = np.pad(temp_buf[number_segments-1],
                                         (0, duration-temp_buf[number_segments-1].shape[0]),
                                         'constant')
    segmented_data = np.vstack(temp_buf[0:number_segments])
    
    return segmented_data

def get_segmented_epochs(data, window_len, shift_len, sample_rate):
    '''
    Returns epoched eeg data based on the window duration and step size.

    Args:
        data (numpy.ndarray): array of samples. 
        window_len (int): window length (seconds).
        shift_len (int): step size (seconds).
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): epoched eeg data of shape. 
        (num_classes, num_channels, num_trials, number_of_segments, duration).
    '''
    
    num_classes = data.shape[0]
    num_chan = data.shape[1]
    num_trials = data.shape[3]
    
    duration = int(window_len*sample_rate)
    data_overlap = (window_len - shift_len)*sample_rate
    
    number_of_segments = int(math.ceil((data.shape[2] - data_overlap)/
                                       (duration - data_overlap)))
    
    segmented_data = np.zeros((data.shape[0], data.shape[1], 
                               data.shape[3], number_of_segments, duration))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                segmented_data[target, channel, trial, :, :] = buffer(data[target, channel, :, trial], 
                                                                      duration, data_overlap) 
    
    return segmented_data

def magnitude_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    '''
    
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros(((fft_index_end - fft_index_start), 
                              segmented_data.shape[1], segmented_data.shape[0], 
                              segmented_data.shape[2], segmented_data.shape[3]))
    
    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT)/fft_len
                    magnitude_spectrum = 2*np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[fft_index_start:fft_index_end,]
    
    return features_data


def plot_time_series(time_series_to_plot, stim_index, channel, sample_rate):
    '''
    Plots the time series of each trial in this dataset.

    Args:
        time_series_to_plot (numpy.ndarray): Time series to plot (n_classes, n_chan, n_samples, n_trials).
        sample_rate (int): sampling frequency in Hz.
    '''
    num_classes = time_series_to_plot.shape[0]
    n_ch = time_series_to_plot.shape[1]
    total_trial_len = time_series_to_plot.shape[2]
    num_trials = time_series_to_plot.shape[3]
    total_trial_s = total_trial_len/sample_rate
    
    fig, ax = plt.subplots(3, 5, figsize=(22, 15), sharex=True, gridspec_kw={'wspace': 0.2})
    ax = ax.reshape(-1)
    
    time_axis = np.linspace(0, total_trial_s, total_trial_len)
    for trial_index in range(num_trials):
        raw_eeg_trial = time_series_to_plot[stim_index, CHANNEL_MAPPING[channel], :, trial_index]
        ax[trial_index].plot(time_axis, raw_eeg_trial)
        ax[trial_index].set_xlabel('Time (s)')
        ax[trial_index].set_title(f'Trial {trial_index+1}')
    ax[0].set_ylabel('Amplitude (uV)')
    ax[5].set_ylabel('Amplitude (uV)')
    ax[10].set_ylabel('Amplitude (uV)');
    
    
def plot_spectrum(magnitude_spectrum, num_classes, subject, channel_idx, flicker_freq):
    '''
    Plots the average magnitude spectrum across trials.

    Args:
        magnitude_spectrum (numpy.ndarray): Magnitude spectrum computed across trials of shape (n_freq, n_chan, n_classes, n_trials, 1) 
        num_classes (int): number of classes in the dataset.
        subject (str): subject_id.
        channel_idx (int): channel index to plot. 
        flicker_freq (numpy.ndarray): stimulus frequencies of shape (n_classes,). 
    '''
    fig, ax = plt.subplots(4, 3, figsize=(16, 14), gridspec_kw=dict(hspace=0.5, wspace=0.2))
    ax = ax.reshape(-1)
    for class_idx in range(num_classes):
        stim_freq = flicker_freq[class_idx]
        fft_axis = np.linspace(FFT_PARAMS['start_frequency'], FFT_PARAMS['end_frequency'], magnitude_spectrum.shape[0])
        ax[class_idx].plot(fft_axis, np.mean(np.squeeze(magnitude_spectrum[:, channel_idx, class_idx, :, :]), axis=1))
        ax[class_idx].axvline(stim_freq, linestyle=':', linewidth=0.7, c='k', label=f'f = {stim_freq} Hz')
        ax[class_idx].axvline(2*stim_freq, linestyle=':', linewidth=0.7, c='b', label=f'2*f = {2*stim_freq} Hz')
        ax[class_idx].set_xlabel('Frequency (Hz)') 
        ax[class_idx].set_ylabel('Amplitude (uV)')
        ax[class_idx].set_title(f'{subject} f = {flicker_freq[class_idx]} Hz')
        ax[class_idx].set_xlim(fft_axis[0], fft_axis[-1])
    plt.show()
    print()
    
    
def plot_performance(ground_truth_labels_dict, predicted_labels_dict, subject_id):
    '''
    Plots the classification performance based on labels provided.

    Args:
        ground_truth_labels_dict (dict): dictionary of true labels for each subject_id.
        predicted_labels_dict (dict): dictionary of predicted labels for each subject_id.
        subject_id (str): subject_id to plot.
    Returns:
        (float): predicted accuracy.
    '''
    accuracy = accuracy_score(ground_truth_labels_dict[subject_id], predicted_labels_dict[subject_id])*100
    cmat = confusion_matrix(ground_truth_labels_dict[subject_id], predicted_labels_dict[subject_id])
    sns.heatmap(pd.DataFrame(cmat), annot=True, cmap='Greens');
    print(f'Subject ID: {subject_id} - Accuracy: {accuracy} %')
    
    return accuracy