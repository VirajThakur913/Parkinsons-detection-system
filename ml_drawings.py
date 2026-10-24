import os
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# Custom Zhang-Suen implementation as reference/fallback
def neighbours(x, y, image):
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [
        image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1],  # P2, P3, P4, P5
        image[x1][y], image[x1][y_1], image[x][y_1], image[x_1][y_1] # P6, P7, P8, P9
    ]

def transitions(n):
    # Order: P2, P3, P4, P5, P6, P7, P8, P9, P2
    n_seq = n + n[0:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n_seq, n_seq[1:]))

def zhang_suen_thinning(image):
    """
    Classic Zhang-Suen Thinning Algorithm.
    Note: Pure Python implementation is slow; skimage.morphology.skeletonize is preferred.
    """
    img = image.copy().astype(np.uint8)
    changing1 = changing2 = 1
    while changing1 or changing2:
        # Step 1
        changing1 = []
        rows, cols = img.shape
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if img[x][y] == 1:
                    n = neighbours(x, y, img)
                    if (2 <= sum(n) <= 6 and 
                        transitions(n) == 1 and 
                        n[0] * n[2] * n[4] == 0 and 
                        n[2] * n[4] * n[6] == 0):
                        changing1.append((x, y))
        for x, y in changing1:
            img[x][y] = 0
            
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if img[x][y] == 1:
                    n = neighbours(x, y, img)
                    if (2 <= sum(n) <= 6 and 
                        transitions(n) == 1 and 
                        n[0] * n[2] * n[6] == 0 and 
                        n[0] * n[4] * n[6] == 0):
                        changing2.append((x, y))
        for x, y in changing2:
            img[x][y] = 0
    return img

# Complete preprocessing function
def preprocess_drawing(image_gray):
    # 1. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # 2. Otsu thresholding to segment stroke (handles light paper photos and dark canvas)
    try:
        thresh = threshold_otsu(blurred)
        if np.mean(blurred) > 127:
            binary = blurred < thresh
        else:
            binary = blurred > thresh
    except ValueError:
        # Handle cases with uniform backgrounds
        if np.mean(blurred) > 127:
            binary = blurred < 127
        else:
            binary = blurred > 127
        
    # 3. Optimized thinning / skeletonization using skimage (C-optimized, 1000x faster than pure python zhang_suen)
    # The output is boolean, we convert it to uint8 (0 and 255)
    thinned = skeletonize(binary).astype(np.uint8) * 255
    
    # Return thinned image, and also standard Otsu binary for visualization
    otsu_binary = binary.astype(np.uint8) * 255
    return thinned, otsu_binary

# Feature extraction using Histogram of Oriented Gradients (HOG)
def extract_hog_features(image):
    # Ensure size is 128x128
    if image.shape != (128, 128):
        image = cv2.resize(image, (128, 128))
    
    features, hog_image = hog(
        image, 
        orientations=9, 
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        visualize=True
    )
    return features, hog_image

# Load and prepare the handwriting dataset
def load_and_preprocess_dataset():
    print("Loading drawings dataset...")
    # Load NPZ files
    data_train = np.load('dataset/train_set.npz', allow_pickle=True)
    x_train_raw = data_train['arr_0']
    y_train_raw = data_train['arr_1']
    
    data_test = np.load('dataset/test_set.npz', allow_pickle=True)
    x_test_raw = data_test['arr_0']
    y_test_raw = data_test['arr_1']
    
    print(f"Loaded train set shape: {x_train_raw.shape}, test set shape: {x_test_raw.shape}")
    
    X_train_hog = []
    X_test_hog = []
    
    # Preprocess all training images
    print("Preprocessing training drawings & extracting HOG features...")
    for i, img in enumerate(x_train_raw):
        # Ensure grayscale & 128x128
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        img_gray = cv2.resize(img_gray, (128, 128))
        
        thinned, _ = preprocess_drawing(img_gray)
        features, _ = extract_hog_features(thinned)
        X_train_hog.append(features)
        
    # Preprocess all testing images
    print("Preprocessing testing drawings & extracting HOG features...")
    for i, img in enumerate(x_test_raw):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        img_gray = cv2.resize(img_gray, (128, 128))
        
        thinned, _ = preprocess_drawing(img_gray)
        features, _ = extract_hog_features(thinned)
        X_test_hog.append(features)
        
    # Encode labels: healthy -> 0, parkinson -> 1
    y_train = np.array([1 if y == 'parkinson' else 0 for y in y_train_raw])
    y_test = np.array([1 if y == 'parkinson' else 0 for y in y_test_raw])
    
    return np.array(X_train_hog), np.array(X_test_hog), y_train, y_test

def main():
    print("Starting Handwriting Pipeline...")
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset()
    print(f"HOG Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    
    # Train high-generalization classifier (XGBoost)
    print("Training drawings model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, preds)
    print(f"\nDrawings Classifier Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Healthy', 'Parkinson']))
    
    # Save model and artifacts
    print("Saving drawings model...")
    os.makedirs('models', exist_ok=True)
    with open('models/drawings_pipeline.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'accuracy': accuracy
        }, f)
        
    print("Handwriting Pipeline Complete!")

if __name__ == '__main__':
    main()
