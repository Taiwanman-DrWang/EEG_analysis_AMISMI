import streamlit as st
import numpy as np
import scipy.signal as signal
import pandas as pd
import mne
import tempfile
import os

class StateSpaceKalmanFilter:
    """一維高斯卡爾曼濾波器"""
    def __init__(self, sigma_v=0.0209, sigma_eps=1.0):
        self.sigma_v2 = sigma_v ** 2
        self.sigma_eps2 = sigma_eps ** 2
        self.x_est = 0.0
        self.P = 1.0
        self.dt2 = 1.0

    def update(self, y):
        # 若輸入為 NaN (代表該秒為極端雜訊)，則保持前一次的狀態不更新
        if np.isnan(y):
            return self.x_est
            
        x_pred = self.x_est
        P_pred = self.P + self.dt2 * self.sigma_v2
        K = P_pred / (P_pred + self.sigma_eps2)
        self.x_est = x_pred + K * (y - x_pred)
        self.P = (1 - K) * P_pred
        return self.x_est

def analyze_eeg_dynamics(edf_path, channel_name='EEG L1(Fp1)', fs_target=250, calib_start_sec=0, calib_duration_sec=300):
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    try:
        raw.pick_channels([channel_name])
    except ValueError:
        raise ValueError(f"找不到通道名稱 {channel_name}，請確認 EDF 檔案內的實際命名。")

    if raw.info['sfreq'] != fs_target:
        raw.resample(fs_target)

    data = raw.get_data()[0]
    fs = fs_target
    sec_num = len(data) // fs

    # 設置極端雜訊門檻 (排除電燒或斷線)
    artifact_mask = (np.abs(data) > 200e-6) | (data == 0)

    # ==========================
    # 1. AMI (Alpha Modulation Index) 計算
    # ==========================
    b_a, a_a = signal.butter(2, [8 / (fs/2), 14 / (fs/2)], btype='bandpass')
    alpha_signal = signal.filtfilt(b_a, a_a, data)
    abs_alpha = np.abs(alpha_signal)

    # 根據使用者設定的區間進行校準
    start_idx = int(calib_start_sec * fs)
    end_idx = int((calib_start_sec + calib_duration_sec) * fs)
    end_idx = min(end_idx, len(abs_alpha))
    
    calib_alpha = abs_alpha[start_idx:end_idx]
    calib_mask = artifact_mask[start_idx:end_idx]
    
    # 僅使用非雜訊的片段找閾值
    valid_calib_alpha = calib_alpha[~calib_mask]
    peaks, _ = signal.find_peaks(valid_calib_alpha)
    
    # 避免閾值因為過度平靜而趨近於 0，設定最低合理閾值保底 (例如 1 µV)
    if len(peaks) > 0:
        M_threshold = max(np.percentile(valid_calib_alpha[peaks], 50), 1e-6)
    else:
        M_threshold = 1e-6

    all_peaks, _ = signal.find_peaks(abs_alpha)
    valid_peaks = all_peaks[abs_alpha[all_peaks] > M_threshold]

    window_samples = int(0.05 * fs)
    is_up_state = np.zeros_like(abs_alpha, dtype=bool)
    for p in valid_peaks:
        s = max(0, p - window_samples)
        e = min(len(abs_alpha), p + window_samples)
        is_up_state[s:e] = True

    y_dup = np.zeros(sec_num)
    y_ddown = np.zeros(sec_num)
    for i in range(sec_num):
        # 檢查該秒是否有超過 10% 的時間是極端雜訊
        if np.sum(artifact_mask[i*fs : (i+1)*fs]) > (fs * 0.1):
            y_dup[i] = np.nan
            y_ddown[i] = np.nan
        else:
            up_ratio = np.sum(is_up_state[i*fs : (i+1)*fs]) / fs
            y_dup[i] = up_ratio
            y_ddown[i] = 1.0 - up_ratio

    kf_up = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)
    kf_down = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)
    ami_series = np.zeros(sec_num)
    
    for i in range(sec_num):
        sm_up = max(kf_up.update(y_dup[i]), 1e-6)
        sm_down = max(kf_down.update(y_ddown[i]), 1e-6)
        # 若遇到 NaN 雜訊段，讓輸出直接歸零，以利臨床辨識
        if np.isnan(y_dup[i]):
            ami_series[i] = 0
        else:
            ami_series[i] = (sm_up**0.5) / ((sm_up**0.5) + (sm_down**0.5))

    # ==========================
    # 2. SMI (Slow Modulation Index) 計算
    # ==========================
    b_s, a_s = signal.butter(2, [0.3 / (fs/2), 4.0 / (fs/2)], btype='bandpass')
    slow_signal = signal.filtfilt(b_s, a_s, data)
    calib_slow = slow_signal[start_idx:end_idx]
    valid_calib_slow = calib_slow[~calib_mask]
    
    M_C = 0
    if len(valid_calib_slow) > 1:
        crossings_0 = np.sum((valid_calib_slow[:-1] < M_C) & (valid_calib_slow[1:] > M_C))
        target_crossings = crossings_0 * 0.7
        M_prime = 0
        for thresh in np.linspace(0, np.max(valid_calib_slow), 100):
            if np.sum((valid_calib_slow[:-1] < thresh) & (valid_calib_slow[1:] > thresh)) <= target_crossings:
                M_prime = thresh
                break
        M_O = 0.5 * M_C + 0.5 * M_prime
    else:
        M_O = 1e-6

    y_fslow = np.zeros(sec_num)
    y_dsupp = np.zeros(sec_num)
    
    for i in range(sec_num):
        if np.sum(artifact_mask[i*fs : (i+1)*fs]) > (fs * 0.1):
            y_fslow[i] = np.nan
            y_dsupp[i] = np.nan
        else:
            s_idx = max(0, (i-5)*fs)
            e_idx = min(len(slow_signal), (i+5)*fs)
            segment = slow_signal[s_idx:e_idx]
            crosses = np.sum((segment[:-1] < M_C) & (segment[1:] > M_C))
            y_fslow[i] = crosses / ((e_idx - s_idx)/fs)
            
            curr_sec = slow_signal[i*fs : (i+1)*fs]
            y_dsupp[i] = np.sum(curr_sec < M_O) / fs

    # 使用校準區間的平均值作為基準
    valid_f = y_fslow[int(calib_start_sec):int(calib_start_sec+calib_duration_sec)]
    valid_d = y_dsupp[int(calib_start_sec):int(calib_start_sec+calib_duration_sec)]
    f_mean = np.nanmean(valid_f) if np.nanmean(valid_f) > 0 else 1.0
    d_mean = np.nanmean(valid_d) if np.nanmean(valid_d) > 0 else 0.1

    kf_f = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=10.0)
    kf_d = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=10.0)
    smi_series = np.zeros(sec_num)
    
    for i in range(sec_num):
        sm_f = max(kf_f.update(y_fslow[i]), 1e-6)
        sm_d = max(kf_d.update(y_dsupp[i]), 1e-6)
        
        if np.isnan(y_fslow[i]):
            smi_series[i] = 0
        else:
            term_f = (sm_f / f_mean) ** 2
            term_d = (sm_d / d_mean) ** 2
            smi_series[i] = term_f / (term_f + term_d)

    return ami_series, smi_series

# --- Streamlit 網頁介面 ---
st.set_page_config(page_title="EEG 狀態空間動態分析", layout="wide")
st.title("大腦麻醉狀態動態過渡分析 (加強抗雜訊版)")

uploaded_file = st.file_uploader("請上傳 EDF 腦波檔案", type=['edf'])

col1, col2, col3 = st.columns(3)
with col1:
    channel_input = st.text_input("EEG 通道名稱", value="EEG Fp1")
with col2:
    calib_start = st.number_input("校準起點 (避開開頭雜訊，單位：秒)", value=300, step=60)
with col3:
    calib_dur = st.number_input("校準時長 (建議300秒)", value=300, step=60)

if uploaded_file is not None:
    if st.button("開始運算"):
        with st.spinner("正在解析 EDF 並執行濾波，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                ami_res, smi_res = analyze_eeg_dynamics(
                    tmp_file_path, 
                    channel_name=channel_input,
                    calib_start_sec=calib_start,
                    calib_duration_sec=calib_dur
                )
                st.success("運算完成！")
                
                df_results = pd.DataFrame({
                    "AMI (Alpha Modulation Index)": ami_res,
                    "SMI (Slow Modulation Index)": smi_res
                })
                
                st.line_chart(df_results)
                
            except Exception as e:
                st.error(f"發生錯誤：{e}")
            finally:
                os.remove(tmp_file_path)
