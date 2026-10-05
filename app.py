import streamlit as st
import numpy as np
import scipy.signal as signal
import mne
import tempfile
import os

class StateSpaceKalmanFilter:
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

def calculate_ami(edf_path, channel_name='EEG Fp1', fs_target=250):
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

    b, a = signal.butter(2, [8 / (fs/2), 14 / (fs/2)], btype='bandpass')
    alpha_signal = signal.filtfilt(b, a, data)
    abs_alpha = np.abs(alpha_signal)

    calib_samples = min(300 * fs, len(abs_alpha))
    peaks, _ = signal.find_peaks(abs_alpha[:calib_samples])
    if len(peaks) == 0:
        return np.zeros(len(abs_alpha) // fs)
    # 這裡使用自適應閾值概念，取整體中位數作為動態校準基準
    M_threshold = np.percentile(abs_alpha[peaks], 50)

    all_peaks, _ = signal.find_peaks(abs_alpha)
    valid_peaks = all_peaks[abs_alpha[all_peaks] > M_threshold]

    window_samples = int(0.05 * fs)
    is_up_state = np.zeros_like(abs_alpha, dtype=bool)
    for p in valid_peaks:
        start = max(0, p - window_samples)
        end = min(len(abs_alpha), p + window_samples)
        is_up_state[start:end] = True

    sec_num = len(abs_alpha) // fs
    y_dup = np.zeros(sec_num)
    y_ddown = np.zeros(sec_num)

    for i in range(sec_num):
        segment = is_up_state[i*fs : (i+1)*fs]
        up_ratio = np.sum(segment) / fs
        y_dup[i] = up_ratio
        y_ddown[i] = 1.0 - up_ratio

    kf_up = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)
    kf_down = StateSpaceKalmanFilter(sigma_v=0.0209, sigma_eps=1.0)

    ami_series = np.zeros(sec_num)
    gamma_param = 0.5

    for i in range(sec_num):
        sm_up = max(kf_up.update(y_dup[i]), 1e-6)
        sm_down = max(kf_down.update(y_ddown[i]), 1e-6)
        ami_series[i] = (sm_up**gamma_param) / ((sm_up**gamma_param) + (sm_down**gamma_param))

    return ami_series

st.set_page_config(page_title="EEG AMI 術中動態分析", layout="wide")
st.title("大腦麻醉狀態動態過渡分析 (AMI)")
st.markdown("基於 Adam et al. (2023) 的演算法，利用卡爾曼濾波穩定追蹤 PAC")

uploaded_file = st.file_uploader("請上傳 EDF 腦波檔案", type=['edf'])
channel_input = st.text_input("輸入欲分析的 EEG 通道名稱 (例如：EEG Fp1 或 Fp1)", value="EEG Fp1")

if uploaded_file is not None:
    if st.button("開始運算"):
        with st.spinner("正在解析 EDF 並執行卡爾曼濾波運算，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                ami_result = calculate_ami(tmp_file_path, channel_name=channel_input)
                st.success("運算完成！")
                st.subheader("Alpha 調變指數 (AMI) 趨勢圖 (X軸：秒)")
                st.line_chart(ami_result)
            except Exception as e:
                st.error(f"發生錯誤：{e}")
            finally:
                os.remove(tmp_file_path)
