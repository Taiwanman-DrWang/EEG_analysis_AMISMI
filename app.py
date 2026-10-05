import streamlit as st
import numpy as np
import scipy.signal as signal
import pandas as pd
import mne
import tempfile
import os

class StateSpaceKalmanFilter:
    """一維高斯卡爾曼濾波器 (Adam et al., 2023)"""
    def __init__(self, sigma_v=0.0209, sigma_eps=1.0):
        self.sigma_v2 = sigma_v ** 2
        self.sigma_eps2 = sigma_eps ** 2
        self.x_est = 0.0
        self.P = 1.0
        self.dt2 = 1.0

    def update(self, y):
        x_pred = self.x_est
        P_pred = self.P + self.dt2 * self.sigma_v2
        K = P_pred / (P_pred + self.sigma_eps2)
        self.x_est = x_pred + K * (y - x_pred)
        self.P = (1 - K) * P_pred
        return self.x_est

def analyze_eeg_dynamics(edf_path, channel_name='EEG L1(Fp1)', fs_target=250):
    """讀取 EDF 並同時計算 AMI 與 SMI"""
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

    # ==========================
    # 1. AMI (Alpha Modulation Index) 計算
    # ==========================
    b_a, a_a = signal.butter(2, [8 / (fs/2), 14 / (fs/2)], btype='bandpass')
    alpha_signal = signal.filtfilt(b_a, a_a, data)
    abs_alpha = np.abs(alpha_signal)

    calib_samples = min(300 * fs, len(abs_alpha))
    peaks, _ = signal.find_peaks(abs_alpha[:calib_samples])
    M_threshold = np.percentile(abs_alpha[peaks], 50) if len(peaks) > 0 else 0

    all_peaks, _ = signal.find_peaks(abs_alpha)
    valid_peaks = all_peaks[abs_alpha[all_peaks] > M_threshold]

    window_samples = int(0.05 * fs)
    is_up_state = np.zeros_like(abs_alpha, dtype=bool)
    for p in valid_peaks:
        start = max(0, p - window_samples)
        end = min(len(abs_alpha), p + window_samples)
        is_up_state[start:end] = True

    y_dup = np.zeros(sec_num)
    y_ddown = np.zeros(sec_num)
    for i in range(sec_num):
        up_ratio = np.sum(is_up_state[i*fs : (i+1)*fs]) / fs
        y_dup[i] = up_ratio
        y_ddown[i] = 1.0 - up_ratio

    # AMI 卡爾曼濾波 (sigma_eps = 1.0)
    kf_up = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)
    kf_down = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)
    ami_series = np.zeros(sec_num)
    for i in range(sec_num):
        sm_up = max(kf_up.update(y_dup[i]), 1e-6)
        sm_down = max(kf_down.update(y_ddown[i]), 1e-6)
        ami_series[i] = (sm_up**0.5) / ((sm_up**0.5) + (sm_down**0.5))

    # ==========================
    # 2. SMI (Slow Modulation Index) 計算
    # ==========================
    b_s, a_s = signal.butter(2, [0.3 / (fs/2), 4.0 / (fs/2)], btype='bandpass')
    slow_signal = signal.filtfilt(b_s, a_s, data)
    calib_slow = slow_signal[:calib_samples]
    
    # 尋找閾值 M_C 與 M_O
    M_C = 0
    crossings_0 = np.sum((calib_slow[:-1] < M_C) & (calib_slow[1:] > M_C))
    target_crossings = crossings_0 * 0.7
    M_prime = 0
    for thresh in np.linspace(0, np.max(calib_slow), 100):
        if np.sum((calib_slow[:-1] < thresh) & (calib_slow[1:] > thresh)) <= target_crossings:
            M_prime = thresh
            break
    M_O = 0.5 * M_C + 0.5 * M_prime

    y_fslow = np.zeros(sec_num)
    y_dsupp = np.zeros(sec_num)
    
    # 擷取慢波頻率與靜默期
    for i in range(sec_num):
        # 使用 10 秒滑動視窗來穩定計算慢波頻率
        start = max(0, (i-5)*fs)
        end = min(len(slow_signal), (i+5)*fs)
        segment = slow_signal[start:end]
        crosses = np.sum((segment[:-1] < M_C) & (segment[1:] > M_C))
        y_fslow[i] = crosses / ((end - start)/fs)
        
        # 計算當前秒數內的靜默期比例
        curr_sec = slow_signal[i*fs : (i+1)*fs]
        y_dsupp[i] = np.sum(curr_sec < M_O) / fs

    f_mean = np.mean(y_fslow[:300]) if np.mean(y_fslow[:300]) > 0 else 1.0
    d_mean = np.mean(y_dsupp[:300]) if np.mean(y_dsupp[:300]) > 0 else 0.1

    # SMI 卡爾曼濾波 (依據文獻，SMI 的 sigma_eps 設為 10.0)
    kf_f = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=10.0)
    kf_d = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=10.0)
    smi_series = np.zeros(sec_num)
    
    for i in range(sec_num):
        sm_f = max(kf_f.update(y_fslow[i]), 1e-6)
        sm_d = max(kf_d.update(y_dsupp[i]), 1e-6)
        
        term_f = (sm_f / f_mean) ** 2
        term_d = (sm_d / d_mean) ** 2
        smi_series[i] = term_f / (term_f + term_d)

    return ami_series, smi_series

# --- Streamlit 網頁介面 ---
st.set_page_config(page_title="EEG 狀態空間動態分析", layout="wide")
st.title("大腦麻醉狀態動態過渡分析 (AMI & SMI)")
st.markdown("基於 Adam et al. (2023) 的演算法，同步追蹤 Alpha 波幅調變與慢波頻率變化。")

uploaded_file = st.file_uploader("請上傳 EDF 腦波檔案", type=['edf'])
channel_input = st.text_input("輸入欲分析的 EEG 通道名稱 (例如：EEG Fp1 或 Fp1)", value="EEG Fp1")

if uploaded_file is not None:
    if st.button("開始運算"):
        with st.spinner("正在解析 EDF 並執行雙重卡爾曼濾波運算，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                ami_res, smi_res = analyze_eeg_dynamics(tmp_file_path, channel_name=channel_input)
                st.success("運算完成！")
                
                # 將結果整理成 DataFrame 方便同時繪圖
                df_results = pd.DataFrame({
                    "AMI (Alpha Modulation Index)": ami_res,
                    "SMI (Slow Modulation Index)": smi_res
                })
                
                st.subheader("大腦狀態調變指數趨勢圖 (X軸：秒，Y軸：0~1)")
                st.line_chart(df_results)
                
            except Exception as e:
                st.error(f"發生錯誤：{e}")
            finally:
                os.remove(tmp_file_path)
