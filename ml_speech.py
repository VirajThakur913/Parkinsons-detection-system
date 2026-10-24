import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import KMeansSMOTE, SMOTE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pickle

# Custom Model-Agnostic Feature Contribution Explainer (approximates SHAP)
class SimpleShapExplainer:
    def __init__(self, model, background_data, feature_names):
        self.model = model
        self.background_data = np.array(background_data)
        self.feature_names = feature_names
        self.background_means = np.mean(self.background_data, axis=0)
        
    def explain_instance(self, instance_vector):
        """
        Calculates local feature contributions (similar to LIME / Omission-based SHAP).
        Measures how much the prediction probability shifts when setting a feature
        to its background training mean value.
        """
        # Ensure it is 1D array
        inst = np.array(instance_vector).flatten()
        orig_prob = self.model.predict_proba([inst])[0, 1]
        
        contributions = []
        for i in range(len(inst)):
            perturbed = inst.copy()
            perturbed[i] = self.background_means[i]
            perturbed_prob = self.model.predict_proba([perturbed])[0, 1]
            # Contribution is: original prob - perturbed prob
            # If removing the feature decreases the prob of Parkinson's, the feature had positive impact
            contrib = orig_prob - perturbed_prob
            contributions.append(contrib)
            
        return np.array(contributions)
        
    def explain_dataset(self, X_df):
        all_contribs = []
        for _, row in X_df.iterrows():
            contribs = self.explain_instance(row.values)
            all_contribs.append(contribs)
        return np.array(all_contribs)

# Load dataset
def load_speech_data(filepath='data/pd_speech_features.csv'):
    # header=1 because first row contains high-level feature groups, second row contains feature names
    df = pd.read_csv(filepath, header=1)
    return df

# Optimize feature processing & split data to enhance model generalization
def preprocess_and_split(df):
    groups = df['id'].values
    y = df['class'].values
    X = df.drop(columns=['id', 'class'])
    
    # 1. Feature Optimization: Remove Constant / Quasi-constant features
    quasi_constant_filter = X.std() > 0.01
    X = X.loc[:, quasi_constant_filter]
    
    # 2. Feature Optimization: Remove highly correlated features to reduce dimensionality
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.92)]
    X = X.drop(columns=to_drop)
    
    feature_names = X.columns.tolist()
    
    # 3. Patient-Level Split (to prevent data leakage)
    unique_patients = np.unique(groups)
    train_pts, test_pts = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    train_idx = np.isin(groups, train_pts)
    test_idx = np.isin(groups, test_pts)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    
    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_df, X_test_df, y_train, y_test, groups_train, scaler

# KMeans-SMOTE-based balancing
def balance_classes(X_train, y_train):
    print("Class distribution before balancing:", np.bincount(y_train))
    try:
        balancer = KMeansSMOTE(cluster_balance_threshold=0.01, random_state=42, k_neighbors=3, density_power=1.0)
        X_res, y_res = balancer.fit_resample(X_train, y_train)
        print("Class distribution after KMeansSMOTE:", np.bincount(y_res))
    except Exception as e:
        print("KMeansSMOTE failed. Falling back to standard SMOTE. Error:", e)
        balancer = SMOTE(random_state=42)
        X_res, y_res = balancer.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE fallback:", np.bincount(y_res))
    return X_res, y_res

# RFE-based feature selection
def select_features_rfe(X_train, y_train, n_features=25):
    estimator = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=5)
    selector.fit(X_train, y_train)
    
    selected_mask = selector.support_
    selected_features = [feature for feature, support in zip(X_train.columns, selected_mask) if support]
    print(f"RFE selected {len(selected_features)} features.")
    return selected_features, selector

# Train and Evaluate models
def train_and_evaluate(X_train, y_train, X_test, y_test, groups_train, selected_features):
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    # 1. Model: Logistic Regression with RFE
    lr_model = LogisticRegression(max_iter=1000, C=0.2, penalty='l2', random_state=42)
    lr_model.fit(X_train_sel, y_train)
    lr_preds = lr_model.predict(X_test_sel)
    lr_probs = lr_model.predict_proba(X_test_sel)[:, 1]
    
    # 2. Model: XGBoost Classifier (with regularization to prevent overfitting)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_sel, y_train)
    xgb_preds = xgb_model.predict(X_test_sel)
    xgb_probs = xgb_model.predict_proba(X_test_sel)[:, 1]
    
    metrics = {}
    for name, preds, probs in [('Logistic_Regression', lr_preds, lr_probs), ('XGBoost', xgb_preds, xgb_probs)]:
        metrics[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'auc': roc_auc_score(y_test, probs)
        }
        print(f"\n{name} Performance:")
        print(f"Accuracy:  {metrics[name]['accuracy']:.4f}")
        print(f"Precision: {metrics[name]['precision']:.4f}")
        print(f"Recall:    {metrics[name]['recall']:.4f}")
        print(f"F1 Score:  {metrics[name]['f1']:.4f}")
        print(f"AUC ROC:   {metrics[name]['auc']:.4f}")
        
    return lr_model, xgb_model, metrics

# Custom SHAP plotting
def generate_shap_explains_custom(explainer, X_test_sel, model_name='XGBoost'):
    # Calculate local explanations
    print(f"Computing custom SHAP approximations for {model_name}...")
    shap_values = explainer.explain_dataset(X_test_sel)
    
    # Global feature importance: mean absolute SHAP value
    mean_abs_shaps = np.mean(np.abs(shap_values), axis=0)
    indices = np.argsort(mean_abs_shaps)
    
    top_n = min(15, len(explainer.feature_names))
    top_indices = indices[-top_n:]
    
    # 1. Bar Plot
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(top_n), mean_abs_shaps[top_indices], color='#008bfb', alpha=0.85, height=0.6)
    
    top_names = [explainer.feature_names[i].replace('_', ' ').title() for i in top_indices]
    plt.yticks(np.arange(top_n), top_names, fontsize=10)
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=10)
    plt.title(f"SHAP Global Feature Importance ({model_name})", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs('static/plots', exist_ok=True)
    plt.savefig(f'static/plots/shap_bar_{model_name.lower()}.png', dpi=150)
    plt.close()
    
    # 2. Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(top_indices):
        shaps_for_feat = shap_values[:, idx]
        vals_for_feat = X_test_sel.iloc[:, idx].values
        
        # Normalize for colormap (0=blue, 1=red)
        min_v, max_v = np.min(vals_for_feat), np.max(vals_for_feat)
        if max_v - min_v > 1e-5:
            norm_vals = (vals_for_feat - min_v) / (max_v - min_v)
        else:
            norm_vals = np.zeros_like(vals_for_feat)
            
        y_pos = np.ones_like(shaps_for_feat) * i + np.random.uniform(-0.15, 0.15, size=len(shaps_for_feat))
        
        # Scatter points colored coolwarm
        sc = plt.scatter(shaps_for_feat, y_pos, c=norm_vals, cmap='coolwarm', alpha=0.7, s=25, edgecolors='none')
        
    plt.yticks(np.arange(top_n), top_names, fontsize=10)
    plt.axvline(0, color='#888888', linestyle='--', alpha=0.5)
    plt.xlabel('SHAP value (impact on model output)', fontsize=10)
    plt.title(f"SHAP Summary Beeswarm ({model_name})", fontsize=12, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(sc, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.set_title('Feature value', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'static/plots/shap_summary_{model_name.lower()}.png', dpi=150)
    plt.close()
    
    return shap_values

def main():
    print("Starting Speech Pipeline...")
    df = load_speech_data()
    print("Data loaded. Shape:", df.shape)
    
    # Preprocess and split (Generalization: Patient-level splits)
    X_train, X_test, y_train, y_test, groups_train, scaler = preprocess_and_split(df)
    print("Train features shape:", X_train.shape, "Test features shape:", X_test.shape)
    
    # Balance Classes (KMeans-SMOTE)
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    # Feature Selection (RFE)
    selected_features, rfe_selector = select_features_rfe(X_train_balanced, y_train_balanced, n_features=25)
    
    # Train and Evaluate Models
    lr_model, xgb_model, metrics = train_and_evaluate(
        X_train_balanced, y_train_balanced, X_test, y_test, groups_train, selected_features
    )
    
    # Initialize custom explainers
    print("\nInitializing Custom Explainers...")
    X_train_sel = X_train_balanced[selected_features]
    X_test_sel = X_test[selected_features]
    
    xgb_explainer = SimpleShapExplainer(xgb_model, X_train_sel, selected_features)
    lr_explainer = SimpleShapExplainer(lr_model, X_train_sel, selected_features)
    
    # Generate global SHAP plots
    print("Generating SHAP plots...")
    generate_shap_explains_custom(xgb_explainer, X_test_sel, 'XGBoost')
    generate_shap_explains_custom(lr_explainer, X_test_sel, 'Logistic_Regression')
    
    # Save artifacts for deployment
    print("Saving models and metadata...")
    os.makedirs('models', exist_ok=True)
    with open('models/speech_pipeline.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'selected_features': selected_features,
            'lr_model': lr_model,
            'xgb_model': xgb_model,
            'xgb_explainer': xgb_explainer,
            'lr_explainer': lr_explainer,
            'metrics': metrics,
            'test_data': (X_test_sel, y_test)
        }, f)
        
    print("Speech Pipeline Complete!")

if __name__ == '__main__':
    main()
