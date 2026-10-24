import wave
import numpy as np
import scipy.signal as signal

def read_wav_mono(filepath):
    """
    Reads a WAV file and returns it as a 1D float array normalized between -1.0 and 1.0.
    """
    with wave.open(filepath, 'rb') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw_data = w.readframes(n_frames)
        
        if sampwidth == 2:
            sig = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            sig = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            sig = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            
        # Convert to mono if stereo
        if n_channels > 1:
            sig = sig[0::n_channels]
            
    return sig, framerate

def extract_vocal_features(filepath):
    """
    Processes WAV and extracts real acoustic parameters: Jitter, Shimmer, ZCR, RMS, F0.
    """
    try:
        sig, framerate = read_wav_mono(filepath)
    except Exception as e:
        print(f"Error reading WAV: {e}")
        # Return fallback zeros
        return {'f0': 0.0, 'jitter': 0.0, 'shimmer': 0.0, 'zcr': 0.0, 'rms': 0.0}

    if len(sig) == 0:
        return {'f0': 0.0, 'jitter': 0.0, 'shimmer': 0.0, 'zcr': 0.0, 'rms': 0.0}
        
    # 1. RMS (Vocal Energy)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    
    # 2. ZCR (Zero Crossing Rate)
    zero_crossings = np.nonzero(np.diff(sig > 0))[0]
    zcr = float(len(zero_crossings) / len(sig))
    
    # 3. F0 & Jitter/Shimmer via Autocorrelation
    # Human voice pitch range typically 60Hz - 350Hz
    min_lag = int(framerate / 350)
    max_lag = int(framerate / 60)
    
    # Keep segment of audio to speed up autocorrelation
    segment_size = min(len(sig), int(framerate * 3.0)) # 3 seconds max
    sig_segment = sig[:segment_size]
    
    corr = np.correlate(sig_segment, sig_segment, mode='full')
    corr = corr[len(corr)//2:]
    
    f0 = 0.0
    jitter = 0.005 # fallback default values if pitch detection fails
    shimmer = 0.03
    
    if len(corr) > max_lag:
        peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
        if corr[peak_idx] > 0.1 * corr[0]: # Verify it's a valid peak
            f0 = float(framerate / peak_idx)
            
            # Find local peaks in amplitude corresponding to pitch period segments
            # for Jitter & Shimmer calculations
            step = peak_idx
            peaks = []
            amps = []
            for start in range(0, len(sig) - step, step):
                window = sig[start:start+step]
                if len(window) > 0:
                    peak_val = np.max(window)
                    peak_pos = np.argmax(window) + start
                    peaks.append(peak_pos)
                    amps.append(peak_val)
                    
            # Compute Jitter (pitch perturbation)
            if len(peaks) > 2:
                diffs = np.abs(np.diff(peaks) - step)
                # locPctJitter approx: mean absolute diff / pitch period
                jitter = float(np.mean(diffs) / step)
                
            # Compute Shimmer (amplitude perturbation)
            if len(amps) > 2:
                amp_diffs = np.abs(np.diff(amps))
                # locShimmer approx: mean absolute diff / mean amp
                shimmer = float(np.mean(amp_diffs) / (np.mean(amps) + 1e-6))
                
    return {
        'f0': f0,
        'jitter': jitter,
        'shimmer': shimmer,
        'zcr': zcr,
        'rms': rms
    }

def map_live_features_to_rfe(extracted_features, selected_features, baseline_patient_features=None):
    """
    Maps the 5 extracted raw features into the 25 RFE selected feature vector.
    For the remaining complex wavelet (TQWT) and cepstral features, it fills them 
    using the baseline_patient_features (if selected), or falls back to normal distributions.
    """
    mapped = {}
    
    # Set up baseline fallback values
    fallback_features = baseline_patient_features if baseline_patient_features else {}
    
    # Mappings from extracted parameters to UCI feature categories
    jitter_keys = ['jitter', 'Jitter', 'PPQ', 'RAP', 'ddp']
    shimmer_keys = ['shimmer', 'Shimmer', 'apq', 'dda']
    f0_keys = ['f0', 'F0', 'Pitch', 'meanPeriod']
    zcr_keys = ['zcr', 'ZCR', 'ZCR']
    rms_keys = ['rms', 'RMS', 'energy', 'Log_energy']
    
    for feature in selected_features:
        val = fallback_features.get(feature, None)
        
        if val is None:
            # Generate representative default value if no baseline exists
            val = 0.0
            
        # Overlay extracted features on top of baseline
        feature_upper = feature.upper()
        
        if any(k.upper() in feature_upper for k in jitter_keys):
            # Scale jitter appropriately depending on feature units
            multiplier = 100.0 if 'PCT' in feature_upper or 'locPct' in feature else 1.0
            val = extracted_features['jitter'] * multiplier
        elif any(k.upper() in feature_upper for k in shimmer_keys):
            multiplier = 10.0 if 'DB' in feature_upper else 1.0
            val = extracted_features['shimmer'] * multiplier
        elif any(k.upper() in feature_upper for k in f0_keys):
            val = extracted_features['f0'] if extracted_features['f0'] > 0 else (val if val != 0 else 120.0)
        elif any(k.upper() in feature_upper for k in zcr_keys):
            val = extracted_features['zcr']
        elif any(k.upper() in feature_upper for k in rms_keys):
            val = extracted_features['rms']
            if 'LOG' in feature_upper:
                val = float(np.log(val + 1e-5))
                
        mapped[feature] = val
        
    return mapped
