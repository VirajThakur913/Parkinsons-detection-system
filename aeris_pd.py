"""
AERIS-PD: Adaptive Explainable Reliability-Based Intelligent Screening Algorithm for Parkinson's Disease
=======================================================================================================
A comprehensive, end-to-end framework integrating:
  Stage 1: Input Quality Assessment (Voice & Motor)
  Stage 2: Advanced Preprocessing & Kinematic Extraction
  Stage 3: Adaptive Weighted Feature Selection (AWFS)
  Stage 4: Hybrid Heterogeneous Machine-Learning Prediction Ensemble
  Stage 5: Reliability-Based Multimodal Fusion (RBMF)
  Stage 6: Uncertainty-Aware Error Detection (UAED)
  Stage 7: Multimodal Explainable AI (SHAP)
"""

import os
import sys

# Suppress Loky subprocess wmic query and matplotlib font cache warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "temp"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import cv2

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.feature import hog

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# STAGE 1: INPUT QUALITY ASSESSMENT (IQA)
# ==============================================================================
class InputQualityAssessor:
    """
    Evaluates raw vocal recordings and handwriting/spiral drawing images
    to verify clinical sample adequacy before model inference.
    """
    def __init__(self, min_voice_dur=1.0, max_voice_dur=30.0, max_clip_ratio=0.03,
                 min_img_dim=64, min_stroke_density=0.003, max_stroke_density=0.75):
        self.min_voice_dur = min_voice_dur
        self.max_voice_dur = max_voice_dur
        self.max_clip_ratio = max_clip_ratio
        self.min_img_dim = min_img_dim
        self.min_stroke_density = min_stroke_density
        self.max_stroke_density = max_stroke_density

    def assess_voice_quality(self, signal_arr, sample_rate):
        """
        Calculates Voice Quality Index Q_v in [0, 1] using:
        - Signal duration check
        - Audio clipping ratio
        - Signal-to-Noise Ratio (SNR) estimate
        - RMS Vocal Energy
        """
        flags = []
        scores = []

        if signal_arr is None or len(signal_arr) == 0:
            return 0.0, False, {'status': 'REJECTED', 'flags': ['Audio missing or empty']}

        duration = len(signal_arr) / float(sample_rate)
        if duration < self.min_voice_dur:
            flags.append(f"Recording too short ({duration:.2f}s < {self.min_voice_dur}s)")
            scores.append(0.3)
        elif duration > self.max_voice_dur:
            flags.append(f"Recording exceeds recommended duration ({duration:.2f}s)")
            scores.append(0.7)
        else:
            scores.append(1.0)

        # Clipping ratio: proportion of samples exceeding 98.5% of dynamic range
        clip_count = np.sum(np.abs(signal_arr) >= 0.985)
        clip_ratio = clip_count / float(len(signal_arr))
        if clip_ratio > self.max_clip_ratio:
            flags.append(f"Audio clipping detected ({clip_ratio*100:.1f}% samples clipped)")
            scores.append(max(0.0, 1.0 - (clip_ratio / 0.10)))
        else:
            scores.append(1.0 - (clip_ratio / self.max_clip_ratio) * 0.2)

        # RMS Energy check (avoids dead silence / blank microphone)
        rms = float(np.sqrt(np.mean(signal_arr ** 2)))
        if rms < 0.003:
            flags.append("Insufficient vocal energy / silent recording")
            scores.append(0.1)
        else:
            rms_score = min(1.0, rms / 0.05)
            scores.append(rms_score)

        # Estimate Signal-to-Noise Ratio (SNR)
        frame_len = int(sample_rate * 0.03)  # 30ms frames
        if len(signal_arr) >= frame_len:
            frames = [signal_arr[i:i+frame_len] for i in range(0, len(signal_arr) - frame_len, frame_len)]
            frame_energies = [np.mean(f**2) for f in frames]
            frame_energies.sort()
            noise_energy = np.mean(frame_energies[:max(1, len(frame_energies)//10)]) + 1e-9
            signal_energy = np.mean(frame_energies[len(frame_energies)//2:]) + 1e-9
            snr_db = 10.0 * np.log10(signal_energy / noise_energy)
            if snr_db < 8.0:
                flags.append(f"High background noise detected (SNR: {snr_db:.1f} dB)")
                scores.append(max(0.2, snr_db / 15.0))
            else:
                scores.append(min(1.0, snr_db / 25.0))
        else:
            snr_db = 0.0
            scores.append(0.5)

        q_v = float(np.mean(scores))
        is_acceptable = q_v >= 0.45 and len(flags) <= 1

        details = {
            'quality_score': q_v,
            'is_acceptable': is_acceptable,
            'duration_sec': duration,
            'clipping_ratio': clip_ratio,
            'rms_energy': rms,
            'snr_db': snr_db,
            'flags': flags
        }
        return q_v, is_acceptable, details

    def assess_motor_quality(self, image_gray):
        """
        Calculates Motor Quality Index Q_m in [0, 1] using:
        - Image dimension & resolution adequacy
        - Pixel contrast & dynamic range
        - Stroke density / coverage
        - Contour continuity & completeness
        """
        flags = []
        scores = []

        if image_gray is None or image_gray.size == 0:
            return 0.0, False, {'status': 'REJECTED', 'flags': ['Image missing or unreadable']}

        h, w = image_gray.shape[:2]
        if h < self.min_img_dim or w < self.min_img_dim:
            flags.append(f"Image resolution too low ({w}x{h} < {self.min_img_dim}x{self.min_img_dim})")
            scores.append(0.3)
        else:
            scores.append(1.0)

        # Contrast check (standard deviation of pixel intensity)
        std_intensity = float(np.std(image_gray))
        if std_intensity < 12.0:
            flags.append("Image lacks sufficient contrast / almost uniform surface")
            scores.append(0.2)
        else:
            scores.append(min(1.0, std_intensity / 45.0))

        # Otsu thresholding to segment stroke
        try:
            thresh = threshold_otsu(image_gray)
            if np.mean(image_gray) > 127:
                binary = image_gray < thresh
            else:
                binary = image_gray > thresh
        except Exception:
            binary = image_gray > 127

        stroke_pixel_count = np.sum(binary)
        stroke_density = stroke_pixel_count / float(h * w)

        if stroke_density < self.min_stroke_density:
            flags.append(f"Drawing incomplete or blank (stroke coverage {stroke_density*100:.2f}%)")
            scores.append(0.2)
        elif stroke_density > self.max_stroke_density:
            flags.append("Image heavily obscured or excessively dark")
            scores.append(0.2)
        else:
            scores.append(1.0)

        # Contour continuity check
        binary_uint8 = binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            flags.append("No traceable contours detected")
            scores.append(0.1)
        else:
            # Check length of the largest primary contour
            max_contour_len = max([cv2.arcLength(c, False) for c in contours])
            if max_contour_len < 30.0:
                flags.append("Stroke too brief / negligible path length")
                scores.append(0.3)
            else:
                scores.append(min(1.0, max_contour_len / 150.0))

        q_m = float(np.mean(scores))
        is_acceptable = q_m >= 0.45 and len(flags) <= 1

        details = {
            'quality_score': q_m,
            'is_acceptable': is_acceptable,
            'dimensions': (w, h),
            'contrast_std': std_intensity,
            'stroke_density': stroke_density,
            'contour_count': len(contours),
            'flags': flags
        }
        return q_m, is_acceptable, details


# ==============================================================================
# STAGE 2: ADVANCED PREPROCESSING & KINEMATIC MOTOR FEATURE EXTRACTION
# ==============================================================================
class MotorKinematicExtractor:
    """
    Extracts comprehensive motor kinematic and morphology features:
    - Adaptive Otsu binarization (polarity invariant)
    - Zhang-Suen morphological skeletonization
    - Tremor oscillation variance
    - Stroke length & path curvature
    - Spiral deviation from centroid
    - HOG Structural Descriptors (252-dim)
    """
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def preprocess(self, image_gray):
        """Standardizes grayscale, denoises, and produces skeletonized stroke."""
        if image_gray.shape != self.target_size:
            image_gray = cv2.resize(image_gray, self.target_size)

        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        try:
            thresh = threshold_otsu(blurred)
            if np.mean(blurred) > 127:
                binary = blurred < thresh
            else:
                binary = blurred > thresh
        except Exception:
            binary = blurred > 127

        thinned = skeletonize(binary).astype(np.uint8) * 255
        otsu_binary = binary.astype(np.uint8) * 255
        return thinned, otsu_binary

    def extract_features(self, thinned_img):
        """
        Extracts 252 HOG features + 6 Kinematic contour/tremor descriptors.
        Total features: 258.
        """
        # 1. HOG Descriptors
        hog_features, _ = hog(
            thinned_img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=True
        )

        # 2. Kinematic & Tremor Descriptors
        coords = np.column_stack(np.where(thinned_img > 0))
        if len(coords) < 5:
            kinematic = np.zeros(6, dtype=np.float32)
            return np.concatenate([hog_features, kinematic])

        # Feature A: Stroke length (total skeleton pixels)
        stroke_length = float(len(coords))

        # Feature B: Centroid coordinates and spiral deviation
        centroid_y, centroid_x = np.mean(coords, axis=0)
        center_y, center_x = self.target_size[0] / 2.0, self.target_size[1] / 2.0
        centroid_deviation = float(np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2))

        # Feature C: Tremor oscillation / radial distance variance
        distances_from_center = np.sqrt((coords[:, 1] - centroid_x)**2 + (coords[:, 0] - centroid_y)**2)
        radial_variance = float(np.std(distances_from_center))

        # Feature D: Stroke bounding box aspect ratio
        min_y, min_x = np.min(coords, axis=0)
        max_y, max_x = np.max(coords, axis=0)
        box_w = max(1.0, float(max_x - min_x))
        box_h = max(1.0, float(max_y - min_y))
        aspect_ratio = float(box_w / box_h)

        # Feature E: Compactness / stroke dispersion
        stroke_dispersion = float(stroke_length / (box_w * box_h + 1e-5))

        # Feature F: Path irregularity / curvature gradient change
        sorted_coords = coords[np.argsort(coords[:, 1])]  # sort by X
        diffs = np.diff(sorted_coords, axis=0)
        angles = np.arctan2(diffs[:, 0], diffs[:, 1] + 1e-5)
        angular_jitter = float(np.std(angles))

        kinematic = np.array([
            stroke_length / 1000.0,
            centroid_deviation / 64.0,
            radial_variance / 32.0,
            aspect_ratio,
            stroke_dispersion * 100.0,
            angular_jitter
        ], dtype=np.float32)

        return np.concatenate([hog_features, kinematic])


# ==============================================================================
# STAGE 3: ADAPTIVE WEIGHTED FEATURE SELECTION (AWFS)
# ==============================================================================
class AdaptiveWeightedFeatureSelector:
    """
    AWFS enhances feature selection beyond single-method RFE by fusing:
      - Random Forest Gini Impurity Importance (I_RF)
      - Mutual Information with target class (I_MI)
      - Cross-Validation Fold Selection Stability (S_stab)
      - Collinearity & Redundancy Pruning (|r_ij| < theta_corr)
    """
    def __init__(self, n_features_to_select=25, w_rf=0.35, w_mi=0.35, w_stab=0.20,
                 w_corr_penalty=0.10, corr_threshold=0.88, n_folds=5, random_state=42):
        self.n_features = n_features_to_select
        self.w_rf = w_rf
        self.w_mi = w_mi
        self.w_stab = w_stab
        self.w_corr_penalty = w_corr_penalty
        self.corr_threshold = corr_threshold
        self.n_folds = n_folds
        self.random_state = random_state

        self.selected_indices_ = None
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(self, X, y, feature_names=None, groups=None):
        X_mat = np.array(X)
        y_arr = np.array(y)
        n_samples, n_total_features = X_mat.shape

        if feature_names is None or len(feature_names) != n_total_features:
            feature_names = [f"f_{i}" for i in range(n_total_features)]
        else:
            feature_names = list(feature_names)

        print(f"[AWFS] Evaluating {n_total_features} features for optimal subset of {self.n_features}...")

        # 1. Random Forest Gini Importance
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state, n_jobs=1)
        rf.fit(X_mat, y_arr)
        rf_imp = rf.feature_importances_
        rf_norm = (rf_imp - np.min(rf_imp)) / (np.max(rf_imp) - np.min(rf_imp) + 1e-9)

        # 2. Mutual Information
        mi_scores = mutual_info_classif(X_mat, y_arr, random_state=self.random_state)
        mi_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-9)

        # 3. Cross-Validation Selection Stability
        fold_counts = np.zeros(n_total_features)
        if groups is not None and len(np.unique(groups)) >= self.n_folds:
            cv = GroupKFold(n_splits=self.n_folds)
            splits = list(cv.split(X_mat, y_arr, groups=groups))
        else:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_mat, y_arr))

        for train_idx, _ in splits:
            rf_f = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state, n_jobs=1)
            rf_f.fit(X_mat[train_idx], y_arr[train_idx])
            top_k_fold = np.argsort(rf_f.feature_importances_)[-int(self.n_features * 1.5):]
            fold_counts[top_k_fold] += 1

        stability_score = fold_counts / float(self.n_folds)

        # 4. Composite AWFS Metric
        raw_composite = (self.w_rf * rf_norm) + (self.w_mi * mi_norm) + (self.w_stab * stability_score)

        # 5. Collinearity-Aware Greedy Selection
        # Compute correlation matrix
        corr_matrix = np.abs(np.corrcoef(X_mat, rowvar=False))
        np.fill_diagonal(corr_matrix, 0.0)

        # Sort features by initial composite score descending
        ranked_indices = np.argsort(raw_composite)[::-1]

        selected_indices = []
        for idx in ranked_indices:
            if len(selected_indices) >= self.n_features:
                break

            # Check maximum collinearity with already selected features
            if len(selected_indices) == 0:
                selected_indices.append(idx)
            else:
                max_corr = np.max(corr_matrix[idx, selected_indices])
                if max_corr < self.corr_threshold:
                    selected_indices.append(idx)

        # Fallback if strict threshold selects fewer than n_features
        if len(selected_indices) < self.n_features:
            remaining = [i for i in ranked_indices if i not in selected_indices]
            selected_indices.extend(remaining[:self.n_features - len(selected_indices)])

        self.selected_indices_ = np.array(selected_indices)
        self.selected_features_ = [feature_names[i] for i in selected_indices]
        self.feature_scores_ = raw_composite[selected_indices]

        print(f"[AWFS] Successfully selected {len(self.selected_features_)} non-redundant, stable features.")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        return np.array(X)[:, self.selected_indices_]


# ==============================================================================
# STAGE 4: HYBRID ENSEMBLE CLASSIFIER (LR + XGBOOST + RANDOM FOREST)
# ==============================================================================
class HybridEnsembleClassifier:
    """
    Heterogeneous soft-voting ensemble uniting:
      1. Calibrated Logistic Regression (Linear decision boundary)
      2. XGBoost Classifier (Gradient boosted trees with regularization)
      3. Random Forest Classifier (Bagged randomized decision trees)
    Validation-learned weights alpha = (alpha_LR, alpha_XGB, alpha_RF).
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.lr_model = LogisticRegression(max_iter=1000, C=0.3, random_state=random_state)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='logloss'
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_split=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )
        self.weights = np.array([0.25, 0.45, 0.30])  # [LR, XGB, RF]

    def fit(self, X, y, optimize_weights=True):
        X_mat = np.array(X)
        y_arr = np.array(y)

        # Train base models
        self.lr_model.fit(X_mat, y_arr)
        self.xgb_model.fit(X_mat, y_arr)
        self.rf_model.fit(X_mat, y_arr)

        if optimize_weights:
            # Optimize weights via 5-fold cross validation out-of-fold predictions
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            oof_lr = np.zeros(len(y_arr))
            oof_xgb = np.zeros(len(y_arr))
            oof_rf = np.zeros(len(y_arr))

            for train_idx, val_idx in skf.split(X_mat, y_arr):
                # Clone and fit
                lr = LogisticRegression(max_iter=1000, C=0.3, random_state=self.random_state).fit(X_mat[train_idx], y_arr[train_idx])
                xg = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=1.0,
                                        random_state=self.random_state, eval_metric='logloss').fit(X_mat[train_idx], y_arr[train_idx])
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state, n_jobs=1).fit(X_mat[train_idx], y_arr[train_idx])

                oof_lr[val_idx] = lr.predict_proba(X_mat[val_idx])[:, 1]
                oof_xgb[val_idx] = xg.predict_proba(X_mat[val_idx])[:, 1]
                oof_rf[val_idx] = rf.predict_proba(X_mat[val_idx])[:, 1]

            # Grid search for best weights maximizing ROC-AUC
            best_auc = -1.0
            best_w = self.weights
            for w1 in np.linspace(0.1, 0.6, 6):
                for w2 in np.linspace(0.2, 0.7, 6):
                    w3 = 1.0 - (w1 + w2)
                    if w3 < 0.1:
                        continue
                    prob_ens = w1 * oof_lr + w2 * oof_xgb + w3 * oof_rf
                    auc = roc_auc_score(y_arr, prob_ens)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = np.array([w1, w2, w3])

            self.weights = best_w / np.sum(best_w)
            print(f"[Hybrid Ensemble] Learned optimal model weights: LR={self.weights[0]:.2f}, XGB={self.weights[1]:.2f}, RF={self.weights[2]:.2f} (OOF AUC: {best_auc:.4f})")

        return self

    def predict_proba(self, X):
        X_mat = np.array(X)
        p_lr = self.lr_model.predict_proba(X_mat)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X_mat)[:, 1]
        p_rf = self.rf_model.predict_proba(X_mat)[:, 1]

        p_ens = (self.weights[0] * p_lr) + (self.weights[1] * p_xgb) + (self.weights[2] * p_rf)
        # Return standard (N, 2) probability format
        return np.column_stack([1.0 - p_ens, p_ens])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ==============================================================================
# STAGE 5: RELIABILITY-BASED MULTIMODAL FUSION (RBMF)
# ==============================================================================
class ReliabilityMultimodalFusion:
    """
    RBMF dynamically weights voice and motor predictions by evaluating:
      1. Physical input quality (Q_v, Q_m) from Stage 1
      2. Prediction confidence / certainty C_k = 2 * |P_k - 0.5|
      Formula:
        R_k = Q_k * [gamma + (1 - gamma) * C_k]
        P_final = (R_v * P_v + R_m * P_m) / (R_v + R_m + epsilon)
    """
    def __init__(self, gamma=0.35, epsilon=1e-7):
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_reliability(self, prob, quality_score):
        """Calculates single-modality reliability index R in [0, 1]."""
        if prob < 0.0 or quality_score <= 0.0:
            return 0.0
        # Prediction certainty: 0 at maximum ambiguity (P=0.5), 1 at full confidence (P=0 or 1)
        certainty = 2.0 * abs(prob - 0.5)
        # Bounded composite reliability
        reliability = quality_score * (self.gamma + (1.0 - self.gamma) * certainty)
        return float(np.clip(reliability, 0.0, 1.0))

    def fuse(self, p_voice, p_motor, q_voice=1.0, q_motor=1.0):
        """
        Executes reliability-weighted late fusion.
        Handles missing/single-modality cases gracefully.
        """
        has_voice = (p_voice >= 0.0 and q_voice > 0.0)
        has_motor = (p_motor >= 0.0 and q_motor > 0.0)

        if not has_voice and not has_motor:
            return 0.5, 0.0, 0.0

        if has_voice and not has_motor:
            r_v = self.compute_reliability(p_voice, q_voice)
            return p_voice, r_v, 0.0

        if has_motor and not has_voice:
            r_m = self.compute_reliability(p_motor, q_motor)
            return p_motor, 0.0, r_m

        # Both modalities available
        r_v = self.compute_reliability(p_voice, q_voice)
        r_m = self.compute_reliability(p_motor, q_motor)

        denom = r_v + r_m + self.epsilon
        p_final = float(((r_v * p_voice) + (r_m * p_motor)) / denom)
        return p_final, r_v, r_m


# ==============================================================================
# STAGE 6: UNCERTAINTY-AWARE ERROR DETECTION (UAED)
# ==============================================================================
class UncertaintyAwareErrorDetector:
    """
    Guards clinical safety by intercepting ambiguous, conflicting, or out-of-distribution cases:
      - OOD Detection: Isolation Forest feature space anomaly
      - Multimodal Conflict: |P_v - P_m| > theta_conflict
      - Borderline Ambiguity: |P_final - 0.50| < margin
      - Quality Degradation: Q_v < theta_q or Q_m < theta_q
    """
    def __init__(self, conflict_threshold=0.45, ambiguity_margin=0.06, min_quality_threshold=0.40):
        self.conflict_threshold = conflict_threshold
        self.ambiguity_margin = ambiguity_margin
        self.min_quality_threshold = min_quality_threshold
        self.voice_ood_detector = IsolationForest(contamination=0.03, random_state=42)
        self.motor_ood_detector = IsolationForest(contamination=0.03, random_state=42)
        self.is_fitted = False

    def fit(self, X_voice_train, X_motor_train):
        self.voice_ood_detector.fit(X_voice_train)
        self.motor_ood_detector.fit(X_motor_train)
        self.is_fitted = True
        return self

    def assess_uncertainty(self, p_voice, p_motor, p_final, q_voice, q_motor,
                           x_voice=None, x_motor=None):
        """
        Returns (is_uncertain, reason_list, clinical_recommendation).
        """
        reasons = []

        # 1. Input Quality check
        if q_voice > 0.0 and q_voice < self.min_quality_threshold:
            reasons.append(f"Vocal sample quality is substandard (Q_v={q_voice:.2f} < {self.min_quality_threshold})")
        if q_motor > 0.0 and q_motor < self.min_quality_threshold:
            reasons.append(f"Motor sample quality is substandard (Q_m={q_motor:.2f} < {self.min_quality_threshold})")

        # 2. Multimodal Conflict check
        if p_voice >= 0.0 and p_motor >= 0.0:
            divergence = abs(p_voice - p_motor)
            if divergence > self.conflict_threshold:
                reasons.append(f"Multimodal conflict detected: voice prediction ({p_voice*100:.1f}%) diverges sharply from motor prediction ({p_motor*100:.1f}%) by {divergence*100:.1f}%")

        # 3. Borderline Ambiguity check
        if abs(p_final - 0.50) < self.ambiguity_margin:
            reasons.append(f"Prediction falls in borderline indeterminate zone (P_final={p_final*100:.1f}%, margin +/-{self.ambiguity_margin*100:.1f}%)")

        # 4. Out-of-Distribution (OOD) check
        if self.is_fitted:
            if x_voice is not None and len(x_voice.shape) == 2:
                if self.voice_ood_detector.predict(x_voice)[0] == -1:
                    reasons.append("Acoustic features deviate from typical training distribution (OOD anomaly)")
            if x_motor is not None and len(x_motor.shape) == 2:
                if self.motor_ood_detector.predict(x_motor)[0] == -1:
                    reasons.append("Motor features deviate from typical training distribution (OOD anomaly)")

        is_uncertain = len(reasons) > 0
        if is_uncertain:
            recommendation = "Insufficient confidence. Please repeat the assessment or consult a healthcare professional."
        else:
            recommendation = "Prediction confidence verified. Suitable for clinical screening triage."

        return is_uncertain, reasons, recommendation


# ==============================================================================
# STAGE 7: EXPLAINABLE AI (SHAP ATTRIBUTION)
# ==============================================================================
class AerisShapExplainer:
    """
    Local and global feature attribution explainer:
    Measures probability shifts by perturbing features toward empirical background means.
    """
    def __init__(self, model, background_data, feature_names):
        self.model = model
        self.background_data = np.array(background_data)
        self.feature_names = list(feature_names)
        self.background_means = np.mean(self.background_data, axis=0)

    def explain_instance(self, instance_vector):
        inst = np.array(instance_vector).flatten()
        orig_prob = self.model.predict_proba([inst])[0, 1]

        contributions = []
        for i in range(len(inst)):
            perturbed = inst.copy()
            perturbed[i] = self.background_means[i]
            perturbed_prob = self.model.predict_proba([perturbed])[0, 1]
            contrib = orig_prob - perturbed_prob
            contributions.append(contrib)

        return np.array(contributions)

    def plot_local_explanation(self, instance_vector, output_path='static/plots/aeris_shap_local.png', top_n=10):
        contribs = self.explain_instance(instance_vector)
        indices = np.argsort(np.abs(contribs))[::-1][:top_n]

        top_contribs = contribs[indices]
        top_names = [self.feature_names[i].replace('_', ' ').title() for i in indices]

        plt.figure(figsize=(9, 5))
        colors = ['#e63946' if c >= 0 else '#457b9d' for c in top_contribs]
        y_pos = np.arange(len(top_contribs))
        plt.barh(y_pos, top_contribs, color=colors, height=0.6, alpha=0.85)

        for i, val in enumerate(top_contribs):
            offset = 0.005 if val >= 0 else -0.005
            ha = 'left' if val >= 0 else 'right'
            plt.text(val + offset, i, f"{val:+.3f}", va='center', ha=ha, fontsize=9, fontweight='bold')

        plt.yticks(y_pos, top_names, fontsize=10)
        plt.gca().invert_yaxis()
        plt.axvline(0, color='#888888', linestyle='--', alpha=0.6)
        plt.xlabel('Feature Contribution (SHAP Approximation)', fontsize=10)
        plt.title('AERIS-PD Local Explainability Attribution', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path


# ==============================================================================
# UNIFIED AERIS-PD MASTER PIPELINE
# ==============================================================================
class AERIS_PD_Pipeline:
    """
    Master pipeline orchestrating all 7 stages of AERIS-PD.
    """
    def __init__(self, n_voice_features=25, n_motor_features=25, random_state=42):
        self.random_state = random_state
        self.n_voice_features = n_voice_features
        self.n_motor_features = n_motor_features

        # Pipeline modules
        self.quality_assessor = InputQualityAssessor()
        self.kinematic_extractor = MotorKinematicExtractor()
        self.voice_awfs = AdaptiveWeightedFeatureSelector(n_features_to_select=n_voice_features, random_state=random_state)
        self.motor_awfs = AdaptiveWeightedFeatureSelector(n_features_to_select=n_motor_features, random_state=random_state)

        self.voice_scaler = StandardScaler()
        self.motor_scaler = StandardScaler()

        self.voice_model = HybridEnsembleClassifier(random_state=random_state)
        self.motor_model = HybridEnsembleClassifier(random_state=random_state)

        self.fusion_engine = ReliabilityMultimodalFusion()
        self.uncertainty_detector = UncertaintyAwareErrorDetector()

        self.voice_explainer = None
        self.motor_explainer = None

    def fit_voice(self, df_voice):
        print("\n=== Training AERIS-PD Voice Branch ===")
        groups = df_voice['id'].values
        y = df_voice['class'].values
        X = df_voice.drop(columns=['id', 'class'])

        # Filter quasi-constant
        quasi_mask = X.std() > 0.01
        X = X.loc[:, quasi_mask]

        # Patient-level split
        unique_pts = np.unique(groups)
        train_pts, test_pts = train_test_split(unique_pts, test_size=0.2, random_state=self.random_state)
        train_mask = np.isin(groups, train_pts)
        test_mask = np.isin(groups, test_pts)

        X_train_raw, X_test_raw = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]

        # Standard Scaling
        X_train_scaled = self.voice_scaler.fit_transform(X_train_raw)
        X_test_scaled = self.voice_scaler.transform(X_test_raw)
        feat_names = X.columns.tolist()

        # Apply SMOTE strictly on training data
        smote = SMOTE(random_state=self.random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        # Stage 3: AWFS
        self.voice_awfs.fit(X_train_bal, y_train_bal, feature_names=feat_names, groups=None)
        X_train_sel = self.voice_awfs.transform(X_train_bal)
        X_test_sel = self.voice_awfs.transform(X_test_scaled)

        # Stage 4: Hybrid Ensemble
        self.voice_model.fit(X_train_sel, y_train_bal, optimize_weights=True)

        # Explainability
        self.voice_explainer = AerisShapExplainer(self.voice_model, X_train_sel, self.voice_awfs.selected_features_)

        # Evaluate on independent patient test set
        preds = self.voice_model.predict(X_test_sel)
        probs = self.voice_model.predict_proba(X_test_sel)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"[Voice Branch Result] Patient-Independent Accuracy: {acc:.4f}, AUC: {auc:.4f}")

        return {
            'X_train_sel': X_train_sel, 'y_train': y_train_bal,
            'X_test_sel': X_test_sel, 'y_test': y_test,
            'preds': preds, 'probs': probs
        }

    def fit_motor(self, train_npz_path, test_npz_path):
        print("\n=== Training AERIS-PD Motor Branch ===")
        data_train = np.load(train_npz_path, allow_pickle=True)
        data_test = np.load(test_npz_path, allow_pickle=True)

        x_train_raw, y_train_str = data_train['arr_0'], data_train['arr_1']
        x_test_raw, y_test_str = data_test['arr_0'], data_test['arr_1']

        y_train = np.array([1 if str(label).lower() == 'parkinson' else 0 for label in y_train_str])
        y_test = np.array([1 if str(label).lower() == 'parkinson' else 0 for label in y_test_str])

        # Stage 2: Kinematic Preprocessing & Feature Extraction
        print("Extracting kinematic tremor & HOG features for handwriting...")
        X_train_feats = []
        for img in x_train_raw:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            thinned, _ = self.kinematic_extractor.preprocess(gray)
            feats = self.kinematic_extractor.extract_features(thinned)
            X_train_feats.append(feats)

        X_test_feats = []
        for img in x_test_raw:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            thinned, _ = self.kinematic_extractor.preprocess(gray)
            feats = self.kinematic_extractor.extract_features(thinned)
            X_test_feats.append(feats)

        X_train_feats = np.array(X_train_feats)
        X_test_feats = np.array(X_test_feats)

        # Scale features
        X_train_scaled = self.motor_scaler.fit_transform(X_train_feats)
        X_test_scaled = self.motor_scaler.transform(X_test_feats)

        # Stage 3: AWFS
        n_total_m = X_train_feats.shape[1]
        n_hog = n_total_m - 6
        motor_feat_names = [f"hog_{i}" for i in range(n_hog)] + [
            "stroke_length", "centroid_deviation", "radial_variance",
            "aspect_ratio", "stroke_dispersion", "angular_jitter"
        ]
        self.motor_awfs.fit(X_train_scaled, y_train, feature_names=motor_feat_names)
        X_train_sel = self.motor_awfs.transform(X_train_scaled)
        X_test_sel = self.motor_awfs.transform(X_test_scaled)

        # Stage 4: Hybrid Ensemble
        self.motor_model.fit(X_train_sel, y_train, optimize_weights=True)

        # Explainability
        self.motor_explainer = AerisShapExplainer(self.motor_model, X_train_sel, self.motor_awfs.selected_features_)

        # Evaluate
        preds = self.motor_model.predict(X_test_sel)
        probs = self.motor_model.predict_proba(X_test_sel)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"[Motor Branch Result] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

        # Fit Stage 6 OOD detector on clean training representations
        return {
            'X_train_sel': X_train_sel, 'y_train': y_train,
            'X_test_sel': X_test_sel, 'y_test': y_test,
            'preds': preds, 'probs': probs
        }

    def fit_pipeline(self, speech_csv_path, train_npz_path, test_npz_path):
        df_speech = pd.read_csv(speech_csv_path, header=1)
        voice_res = self.fit_voice(df_speech)
        motor_res = self.fit_motor(train_npz_path, test_npz_path)

        # Fit OOD detectors in Stage 6
        self.uncertainty_detector.fit(voice_res['X_train_sel'], motor_res['X_train_sel'])
        print("[AERIS-PD] Full pipeline successfully trained and calibrated.")
        return voice_res, motor_res

    def predict_screening(self, voice_features_dict=None, raw_voice_arr=None, voice_sr=16000,
                          motor_img_gray=None, raw_voice_quality=None, raw_motor_quality=None):
        """
        Complete 7-stage screening execution for an individual patient.
        """
        start_time = time.time()
        p_voice, q_voice = -1.0, 1.0
        p_motor, q_motor = -1.0, 1.0
        x_voice_sel, x_motor_sel = None, None

        # --- Stage 1 & Voice Inference ---
        if voice_features_dict is not None:
            if raw_voice_arr is not None:
                q_voice, is_acc_v, details_v = self.quality_assessor.assess_voice_quality(raw_voice_arr, voice_sr)
                if not is_acc_v:
                    return {
                        'status': 'REJECTED_SAMPLE',
                        'modality': 'voice',
                        'message': 'Voice sample rejected due to poor quality. Please repeat recording.',
                        'quality_details': details_v
                    }
            elif raw_voice_quality is not None:
                q_voice = raw_voice_quality

            # Vectorize features
            raw_cols = self.voice_scaler.feature_names_in_
            v_vec = np.array([float(voice_features_dict.get(col, 0.0)) for col in raw_cols]).reshape(1, -1)
            v_scaled = self.voice_scaler.transform(v_vec)
            x_voice_sel = self.voice_awfs.transform(v_scaled)
            p_voice = float(self.voice_model.predict_proba(x_voice_sel)[0, 1])

        # --- Stage 1 & Motor Inference ---
        if motor_img_gray is not None:
            q_motor, is_acc_m, details_m = self.quality_assessor.assess_motor_quality(motor_img_gray)
            if not is_acc_m:
                return {
                    'status': 'REJECTED_SAMPLE',
                    'modality': 'motor',
                    'message': 'Handwriting sample rejected due to poor quality. Please upload a clearer drawing.',
                    'quality_details': details_m
                }

            # Preprocessing & Kinematic Extraction
            thinned, otsu = self.kinematic_extractor.preprocess(motor_img_gray)
            m_feats = self.kinematic_extractor.extract_features(thinned).reshape(1, -1)
            m_scaled = self.motor_scaler.transform(m_feats)
            x_motor_sel = self.motor_awfs.transform(m_scaled)
            p_motor = float(self.motor_model.predict_proba(x_motor_sel)[0, 1])
        elif raw_motor_quality is not None:
            q_motor = raw_motor_quality

        # --- Stage 5: Reliability-Based Multimodal Fusion ---
        p_final, r_v, r_m = self.fusion_engine.fuse(p_voice, p_motor, q_voice, q_motor)

        # --- Stage 6: Uncertainty-Aware Error Detection ---
        is_uncertain, reasons, recommendation = self.uncertainty_detector.assess_uncertainty(
            p_voice, p_motor, p_final, q_voice, q_motor, x_voice_sel, x_motor_sel
        )

        inference_time_ms = (time.time() - start_time) * 1000.0

        # Risk Stratification
        if p_final >= 0.65:
            risk_tier = 'High Risk'
        elif p_final >= 0.35:
            risk_tier = 'Moderate Risk'
        else:
            risk_tier = 'Low Risk'

        result = {
            'status': 'UNCERTAIN' if is_uncertain else 'CONFIDENT',
            'p_final': p_final,
            'risk_level': risk_tier if not is_uncertain else 'Indeterminate / Consult Clinician',
            'p_voice': p_voice,
            'p_motor': p_motor,
            'reliability_voice': r_v,
            'reliability_motor': r_m,
            'quality_voice': q_voice,
            'quality_motor': q_motor,
            'is_uncertain': is_uncertain,
            'uncertainty_reasons': reasons,
            'clinical_recommendation': recommendation,
            'inference_time_ms': inference_time_ms
        }
        return result


# Save and load helpers
def save_aeris_pipeline(pipeline, filepath='models/aeris_pd_pipeline.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[AERIS-PD] Pipeline successfully saved to {filepath}")

def load_aeris_pipeline(filepath='models/aeris_pd_pipeline.pkl'):
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"[AERIS-PD] Pipeline successfully loaded from {filepath}")
    return pipeline
