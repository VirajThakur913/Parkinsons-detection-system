import os
import subprocess
import sys
import shutil

def run_cmd(args):
    print(f"Running: {' '.join(args)}")
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(1)
    print(res.stdout)
    return res.stdout

def main():
    print("==================================================")
    print("          NEUROSENSE AI AUTOMATED ENGINE          ")
    print("==================================================")
    
    # 1. Identify python executable from venv
    python_exe = os.path.join('.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = 'python'
        print("Warning: Virtual environment python.exe not found at .venv/Scripts/python.exe. Using system python.")
    else:
        print(f"Using virtual environment python: {python_exe}")
        
    # 2. Setup directories
    print("\n[Step 1] Creating asset and output directories...")
    os.makedirs('static/sample_drawings', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('static/temp', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 3. Copy sample drawing presets for the dashboard registry
    print("\n[Step 2] Copying sample drawing assets for patient dashboard registry...")
    shutil.copy('dataset/test_image_healthy.png', 'static/sample_drawings/healthy_spiral.png')
    shutil.copy('dataset/test_image_parkinson.png', 'static/sample_drawings/parkinson_spiral.png')
    print("Drawing preset files copied.")
    
    # 4. Train Speech ML Model & pre-generate global SHAP explainers
    print("\n[Step 3] Executing Speech Pipeline (KMeans-SMOTE + RFE + LogReg/XGBoost + SHAP)...")
    run_cmd([python_exe, 'ml_speech.py'])
    
    # 5. Train Handwriting ML Model (Zhang-Suen / skimage skeletonization + HOG + XGBoost)
    print("\n[Step 4] Executing Handwriting Pipeline (Image Processing + HOG + XGBoost)...")
    run_cmd([python_exe, 'ml_drawings.py'])
    
    # 6. Launch Web App
    print("\n[Step 5] Launching Doctor Dashboard Server on http://127.0.0.1:5000 ...")
    try:
        subprocess.run([python_exe, 'app.py'])
    except KeyboardInterrupt:
        print("\nDashboard server stopped by user.")

if __name__ == '__main__':
    main()
