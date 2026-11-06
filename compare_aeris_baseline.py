"""
Empirical Comparison: Baseline System vs. Proposed AERIS-PD Algorithm
=====================================================================
Evaluates both pipelines on the identical test partitions under rigorous
patient-independent evaluation conditions across 10 clinical metrics:
  1. Accuracy
  2. Precision
  3. Recall (Sensitivity)
  4. Specificity
  5. F1-Score
  6. ROC-AUC
  7. PR-AUC (Average Precision)
  8. False Positive Rate (FPR)
  9. False Negative Rate (FNR)
  10. Inference Latency (ms)
"""

import os
import sys

# Suppress loky CPU core query and set local cache for matplotlib
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "temp"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Import AERIS-PD algorithm
from aeris_pd import (
    AERIS_PD_Pipeline, save_aeris_pipeline,
    InputQualityAssessor, MotorKinematicExtractor
)


def compute_metrics(y_true, y_pred, y_prob):
    """Computes all 10 standard clinical diagnostic metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sens)': rec,
        'Specificity': spec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'FPR': fpr,
        'FNR': fnr,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
    }


def run_comparison():
    print("=" * 80)
    print("  PARKINSON'S DISEASE DETECTION: BASELINE vs. AERIS-PD BENCHMARK")
    print("=" * 80)

    # Paths
    speech_csv = 'data/pd_speech_features.csv'
    train_npz = 'dataset/train_set.npz'
    test_npz = 'dataset/test_set.npz'

    # --------------------------------------------------------------------------
    # 1. VOICE BRANCH BENCHMARK
    # --------------------------------------------------------------------------
    print("\n>>> PART 1: EVALUATING VOICE ACOUSTIC ANALYSIS BRANCH")
    df_speech = pd.read_csv(speech_csv, header=1)
    groups = df_speech['id'].values
    y_voice = df_speech['class'].values
    X_voice = df_speech.drop(columns=['id', 'class'])

    # Quasi-constant filter
    quasi_mask = X_voice.std() > 0.01
    X_voice = X_voice.loc[:, quasi_mask]

    # Patient-independent train/test split (80% train, 20% test)
    unique_pts = np.unique(groups)
    train_pts, test_pts = train_test_split(unique_pts, test_size=0.20, random_state=42)
    train_mask = np.isin(groups, train_pts)
    test_mask = np.isin(groups, test_pts)

    X_train_v, X_test_v = X_voice[train_mask], X_voice[test_mask]
    y_train_v, y_test_v = y_voice[train_mask], y_voice[test_mask]

    # Preprocessing (StandardScaler)
    scaler_base = StandardScaler()
    X_train_v_scaled = scaler_base.fit_transform(X_train_v)
    X_test_v_scaled = scaler_base.transform(X_test_v)

    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_v_bal, y_train_v_bal = smote.fit_resample(X_train_v_scaled, y_train_v)

    # --- BASELINE VOICE PIPELINE: RFE + Solo XGBoost ---
    print("\n--- Training Existing Baseline Voice Model (RFE + XGBoost) ---")
    t0 = time.time()
    rfe_estimator = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=25, step=5)
    rfe.fit(X_train_v_bal, y_train_v_bal)
    base_voice_features = [col for col, s in zip(X_voice.columns, rfe.support_) if s]

    X_train_v_rfe = X_train_v_bal[:, rfe.support_]
    X_test_v_rfe = X_test_v_scaled[:, rfe.support_]

    xgb_voice_base = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, eval_metric='logloss'
    )
    xgb_voice_base.fit(X_train_v_rfe, y_train_v_bal)
    base_v_train_time = time.time() - t0

    t0_inf = time.time()
    base_v_preds = xgb_voice_base.predict(X_test_v_rfe)
    base_v_probs = xgb_voice_base.predict_proba(X_test_v_rfe)[:, 1]
    base_v_lat_ms = (time.time() - t0_inf) / len(y_test_v) * 1000.0

    base_v_metrics = compute_metrics(y_test_v, base_v_preds, base_v_probs)
    base_v_metrics['Latency (ms)'] = base_v_lat_ms

    # --- AERIS-PD VOICE PIPELINE: AWFS + Hybrid Ensemble (LR + XGB + RF) ---
    print("\n--- Training AERIS-PD Voice Model (AWFS + Hybrid Ensemble) ---")
    aeris = AERIS_PD_Pipeline(n_voice_features=25, n_motor_features=60, random_state=42)
    t0 = time.time()
    aeris_v_res = aeris.fit_voice(df_speech)
    aeris_v_train_time = time.time() - t0

    t0_inf = time.time()
    aeris_v_preds = aeris.voice_model.predict(aeris_v_res['X_test_sel'])
    aeris_v_probs = aeris.voice_model.predict_proba(aeris_v_res['X_test_sel'])[:, 1]
    aeris_v_lat_ms = (time.time() - t0_inf) / len(y_test_v) * 1000.0

    aeris_v_metrics = compute_metrics(y_test_v, aeris_v_preds, aeris_v_probs)
    aeris_v_metrics['Latency (ms)'] = aeris_v_lat_ms

    # --------------------------------------------------------------------------
    # 2. MOTOR DRAWINGS BRANCH BENCHMARK
    # --------------------------------------------------------------------------
    print("\n>>> PART 2: EVALUATING MOTOR DRAWINGS BRANCH")
    data_train = np.load(train_npz, allow_pickle=True)
    data_test = np.load(test_npz, allow_pickle=True)
    x_train_raw, y_train_str = data_train['arr_0'], data_train['arr_1']
    x_test_raw, y_test_str = data_test['arr_0'], data_test['arr_1']

    y_train_m = np.array([1 if str(label).lower() == 'parkinson' else 0 for label in y_train_str])
    y_test_m = np.array([1 if str(label).lower() == 'parkinson' else 0 for label in y_test_str])

    # --- BASELINE MOTOR PIPELINE: Standard HOG (252) + Solo XGBoost ---
    print("\n--- Training Existing Baseline Motor Model (HOG + XGBoost) ---")
    from skimage.feature import hog
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize
    import cv2

    def base_preprocess(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        gray = cv2.resize(gray, (128, 128))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        try:
            th = threshold_otsu(blurred)
            binary = blurred < th if np.mean(blurred) > 127 else blurred > th
        except Exception:
            binary = blurred > 127
        thinned = skeletonize(binary).astype(np.uint8) * 255
        h_feat, _ = hog(thinned, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
        return h_feat

    X_train_m_base = np.array([base_preprocess(img) for img in x_train_raw])
    X_test_m_base = np.array([base_preprocess(img) for img in x_test_raw])

    t0 = time.time()
    xgb_motor_base = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    xgb_motor_base.fit(X_train_m_base, y_train_m)
    base_m_train_time = time.time() - t0

    t0_inf = time.time()
    base_m_preds = xgb_motor_base.predict(X_test_m_base)
    base_m_probs = xgb_motor_base.predict_proba(X_test_m_base)[:, 1]
    base_m_lat_ms = (time.time() - t0_inf) / len(y_test_m) * 1000.0

    base_m_metrics = compute_metrics(y_test_m, base_m_preds, base_m_probs)
    base_m_metrics['Latency (ms)'] = base_m_lat_ms

    # --- AERIS-PD MOTOR PIPELINE: Kinematic Tremor + HOG + AWFS + Hybrid Ensemble ---
    print("\n--- Training AERIS-PD Motor Model (Kinematic + HOG + AWFS + Hybrid Ensemble) ---")
    t0 = time.time()
    aeris_m_res = aeris.fit_motor(train_npz, test_npz)
    aeris_m_train_time = time.time() - t0

    t0_inf = time.time()
    aeris_m_preds = aeris.motor_model.predict(aeris_m_res['X_test_sel'])
    aeris_m_probs = aeris.motor_model.predict_proba(aeris_m_res['X_test_sel'])[:, 1]
    aeris_m_lat_ms = (time.time() - t0_inf) / len(y_test_m) * 1000.0

    aeris_m_metrics = compute_metrics(y_test_m, aeris_m_preds, aeris_m_probs)
    aeris_m_metrics['Latency (ms)'] = aeris_m_lat_ms

    # --------------------------------------------------------------------------
    # 3. MULTIMODAL FUSION & UNCERTAINTY BENCHMARK
    # --------------------------------------------------------------------------
    print("\n>>> PART 3: EVALUATING MULTIMODAL FUSION & UNCERTAINTY HANDLING")
    # For multimodal comparative benchmarking, pair test samples by true patient diagnostic class
    # (15 healthy motor drawings paired with 15 healthy voice recordings; 15 PD motor paired with 15 PD voice)
    idx_v_0 = np.where(y_test_v == 0)[0][:15]
    idx_v_1 = np.where(y_test_v == 1)[0][:15]
    idx_m_0 = np.where(y_test_m == 0)[0][:15]
    idx_m_1 = np.where(y_test_m == 1)[0][:15]

    matched_v_idx = np.concatenate([idx_v_0, idx_v_1])
    matched_m_idx = np.concatenate([idx_m_0, idx_m_1])
    y_multi = np.concatenate([y_test_m[idx_m_0], y_test_m[idx_m_1]])

    # Baseline Fusion: Fixed weights (0.4 Voice + 0.6 Motor)
    base_v_p = base_v_probs[matched_v_idx]
    base_m_p = base_m_probs[matched_m_idx]
    base_fused_probs = (0.4 * base_v_p) + (0.6 * base_m_p)
    base_fused_preds = (base_fused_probs >= 0.5).astype(int)
    base_multi_metrics = compute_metrics(y_multi, base_fused_preds, base_fused_probs)
    base_multi_metrics['Latency (ms)'] = base_v_lat_ms + base_m_lat_ms

    # AERIS-PD Reliability-Based Multimodal Fusion (RBMF) + UAED
    aeris.uncertainty_detector.fit(aeris_v_res['X_train_sel'], aeris_m_res['X_train_sel'])

    aeris_v_p = aeris_v_probs[matched_v_idx]
    aeris_m_p = aeris_m_probs[matched_m_idx]

    aeris_fused_probs = []
    aeris_fused_preds = []
    uncertain_mask = []

    for i in range(len(y_multi)):
        p_v = aeris_v_p[i]
        p_m = aeris_m_p[i]
        p_final, r_v, r_m = aeris.fusion_engine.fuse(p_v, p_m, q_voice=0.95, q_motor=0.95)
        is_unc, reasons, _ = aeris.uncertainty_detector.assess_uncertainty(
            p_v, p_m, p_final, q_voice=0.95, q_motor=0.95
        )
        uncertain_mask.append(is_unc)
        aeris_fused_probs.append(p_final)
        aeris_fused_preds.append(1 if p_final >= 0.5 else 0)

    aeris_fused_probs = np.array(aeris_fused_probs)
    aeris_fused_preds = np.array(aeris_fused_preds)
    uncertain_mask = np.array(uncertain_mask)
    uncertainty_count = int(np.sum(uncertain_mask))

    aeris_multi_metrics = compute_metrics(y_multi, aeris_fused_preds, aeris_fused_probs)
    aeris_multi_metrics['Latency (ms)'] = aeris_v_lat_ms + aeris_m_lat_ms

    # Confident Cohort Performance (UAED Selective Classification)
    conf_idx = np.where(~uncertain_mask)[0]
    if len(conf_idx) > 0:
        aeris_conf_metrics = compute_metrics(
            y_multi[conf_idx], aeris_fused_preds[conf_idx], aeris_fused_probs[conf_idx]
        )
    else:
        aeris_conf_metrics = aeris_multi_metrics

    # --------------------------------------------------------------------------
    # 4. TABULATE & DISPLAY COMPARATIVE RESULTS
    # --------------------------------------------------------------------------
    df_voice_comp = pd.DataFrame({
        'Metric': list(base_v_metrics.keys())[:9] + ['Latency (ms)'],
        'Baseline (RFE + XGBoost)': [f"{base_v_metrics[k]:.4f}" if isinstance(base_v_metrics[k], float) else base_v_metrics[k] for k in list(base_v_metrics.keys())[:9]] + [f"{base_v_lat_ms:.2f}"],
        'Proposed AERIS-PD (AWFS + Ensemble)': [f"{aeris_v_metrics[k]:.4f}" if isinstance(aeris_v_metrics[k], float) else aeris_v_metrics[k] for k in list(aeris_v_metrics.keys())[:9]] + [f"{aeris_v_lat_ms:.2f}"],
    })

    df_motor_comp = pd.DataFrame({
        'Metric': list(base_m_metrics.keys())[:9] + ['Latency (ms)'],
        'Baseline (HOG + XGBoost)': [f"{base_m_metrics[k]:.4f}" if isinstance(base_m_metrics[k], float) else base_m_metrics[k] for k in list(base_m_metrics.keys())[:9]] + [f"{base_m_lat_ms:.2f}"],
        'Proposed AERIS-PD (Kinematic + AWFS + Ens)': [f"{aeris_m_metrics[k]:.4f}" if isinstance(aeris_m_metrics[k], float) else aeris_m_metrics[k] for k in list(aeris_m_metrics.keys())[:9]] + [f"{aeris_m_lat_ms:.2f}"],
    })

    df_multi_comp = pd.DataFrame({
        'Metric': list(base_multi_metrics.keys())[:9] + ['Latency (ms)'],
        'Baseline (Fixed Late Fusion)': [f"{base_multi_metrics[k]:.4f}" if isinstance(base_multi_metrics[k], float) else base_multi_metrics[k] for k in list(base_multi_metrics.keys())[:9]] + [f"{base_multi_metrics['Latency (ms)']:.2f}"],
        'Proposed AERIS-PD (All Cohort)': [f"{aeris_multi_metrics[k]:.4f}" if isinstance(aeris_multi_metrics[k], float) else aeris_multi_metrics[k] for k in list(aeris_multi_metrics.keys())[:9]] + [f"{aeris_multi_metrics['Latency (ms)']:.2f}"],
        'Proposed AERIS-PD (UAED Confident)': [f"{aeris_conf_metrics[k]:.4f}" if isinstance(aeris_conf_metrics[k], float) else aeris_conf_metrics[k] for k in list(aeris_conf_metrics.keys())[:9]] + [f"{aeris_multi_metrics['Latency (ms)']:.2f}"],
    })

    print("\n" + "=" * 80)
    print("  TABLE 1: VOICE ACOUSTIC BRANCH COMPARISON (Patient-Independent Test Set)")
    print("=" * 80)
    print(df_voice_comp.to_string(index=False))

    print("\n" + "=" * 80)
    print("  TABLE 2: MOTOR DRAWING BRANCH COMPARISON (Test Set)")
    print("=" * 80)
    print(df_motor_comp.to_string(index=False))

    print("\n" + "=" * 80)
    print("  TABLE 3: MULTIMODAL FUSION COMPARISON (Voice + Motor Integrated)")
    print("=" * 80)
    print(df_multi_comp.to_string(index=False))
    print(f"\nAERIS-PD Uncertainty & Conflict Detection: Intercepted {uncertainty_count}/{len(y_multi)} ({uncertainty_count/len(y_multi)*100:.1f}%) ambiguous samples for secondary review.")
    print(f"Confident Cohort Size: {len(conf_idx)}/{len(y_multi)} ({len(conf_idx)/len(y_multi)*100:.1f}%) with {aeris_conf_metrics['Accuracy']*100:.1f}% accuracy.")

    # --------------------------------------------------------------------------
    # 5. GENERATE COMPARATIVE BENCHMARK PLOTS
    # --------------------------------------------------------------------------
    os.makedirs('static/plots', exist_ok=True)

    # Plot 1: ROC Curves Comparison
    plt.figure(figsize=(9, 7))
    fpr_bv, tpr_bv, _ = roc_curve(y_test_v, base_v_probs)
    fpr_av, tpr_av, _ = roc_curve(y_test_v, aeris_v_probs)
    fpr_bm, tpr_bm, _ = roc_curve(y_test_m, base_m_probs)
    fpr_am, tpr_am, _ = roc_curve(y_test_m, aeris_m_probs)

    plt.plot(fpr_bv, tpr_bv, 'r--', label=f"Baseline Voice (AUC = {base_v_metrics['ROC-AUC']:.3f})", alpha=0.7)
    plt.plot(fpr_av, tpr_av, 'r-', linewidth=2.5, label=f"AERIS-PD Voice [AWFS+Ens] (AUC = {aeris_v_metrics['ROC-AUC']:.3f})")
    plt.plot(fpr_bm, tpr_bm, 'b--', label=f"Baseline Motor (AUC = {base_m_metrics['ROC-AUC']:.3f})", alpha=0.7)
    plt.plot(fpr_am, tpr_am, 'b-', linewidth=2.5, label=f"AERIS-PD Motor [Kinematic+AWFS+Ens] (AUC = {aeris_m_metrics['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)

    plt.title("ROC Curves: Baseline vs. Proposed AERIS-PD Framework", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('static/plots/aeris_vs_baseline_roc.png', dpi=200)
    plt.close()
    print("\nSaved ROC curves to static/plots/aeris_vs_baseline_roc.png")

    # Plot 2: Bar Comparison of Key Metrics
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall (Sens)', 'Specificity', 'F1-Score', 'ROC-AUC']
    base_vals = [base_multi_metrics[m] for m in metrics_to_plot]
    aeris_vals = [aeris_multi_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, base_vals, width, label='Baseline System (Fixed Late Fusion)', color='#9e9e9e', alpha=0.85)
    plt.bar(x + width/2, aeris_vals, width, label='Proposed AERIS-PD (Adaptive Fusion)', color='#1e88e5', alpha=0.9)

    plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
    plt.title('Overall Multimodal Diagnostic Metric Comparison', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for i in range(len(metrics_to_plot)):
        plt.text(x[i] - width/2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha='center', fontsize=9)
        plt.text(x[i] + width/2, aeris_vals[i] + 0.02, f"{aeris_vals[i]:.2f}", ha='center', fontsize=9, fontweight='bold', color='#0d47a1')

    plt.tight_layout()
    plt.savefig('static/plots/aeris_vs_baseline_metrics.png', dpi=200)
    plt.close()
    print("Saved metric comparison to static/plots/aeris_vs_baseline_metrics.png")

    # Save AERIS-PD trained model pipeline
    save_aeris_pipeline(aeris, 'models/aeris_pd_pipeline.pkl')

    # Save summary dataframe to CSV
    os.makedirs('static/data', exist_ok=True)
    df_multi_comp.to_csv('static/data/aeris_baseline_comparison.csv', index=False)
    print("Benchmark successfully exported to static/data/aeris_baseline_comparison.csv!")

    return df_voice_comp, df_motor_comp, df_multi_comp


if __name__ == '__main__':
    run_comparison()
