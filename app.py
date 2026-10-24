import os
import uuid
import base64
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from ml_drawings import preprocess_drawing, extract_hog_features
from ml_speech import SimpleShapExplainer
import __main__
__main__.SimpleShapExplainer = SimpleShapExplainer
import database
import report_generator
import signal_processing

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurosense_multimodal_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('static/temp', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)
os.makedirs('static/sample_drawings', exist_ok=True)

# Global variables for models
SPEECH_PIPELINE = None
DRAWINGS_PIPELINE = None

def load_models():
    global SPEECH_PIPELINE, DRAWINGS_PIPELINE
    if SPEECH_PIPELINE is not None and DRAWINGS_PIPELINE is not None:
        return
    
    # Load Speech Pipeline
    speech_path = 'models/speech_pipeline.pkl'
    if os.path.exists(speech_path):
        with open(speech_path, 'rb') as f:
            SPEECH_PIPELINE = pickle.load(f)
            print("Speech pipeline loaded successfully.")
            
    # Load Drawings Pipeline
    drawings_path = 'models/drawings_pipeline.pkl'
    if os.path.exists(drawings_path):
        with open(drawings_path, 'rb') as f:
            DRAWINGS_PIPELINE = pickle.load(f)
            print("Drawings pipeline loaded successfully.")

def get_feature_metadata(feature_name):
    """
    Returns (min, max, step) for a feature using the standard scaler training mean & std dev.
    """
    global SPEECH_PIPELINE
    if not SPEECH_PIPELINE or 'scaler' not in SPEECH_PIPELINE:
        return -1.0, 1.0, 0.01
        
    scaler = SPEECH_PIPELINE['scaler']
    feature_names = list(scaler.feature_names_in_)
    if feature_name in feature_names:
        idx = feature_names.index(feature_name)
        mean_val = float(scaler.mean_[idx])
        std_val = float(np.sqrt(scaler.var_[idx]))
        
        # Calculate 3-sigma range to cover 99.7% of realistic domain
        min_val = mean_val - 3.0 * std_val
        max_val = mean_val + 3.0 * std_val
        step_val = std_val / 50.0
        
        # Handle zero variance or tiny scales
        if step_val <= 0:
            step_val = 0.01
            
        return min_val, max_val, step_val
    return -1.0, 1.0, 0.01

# Create matplotlib local SHAP contribution plot for single sample
def create_local_shap_plot(shap_values, feature_values, feature_names):
    indices = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(12, len(feature_names))
    top_indices = indices[:top_n]
    
    top_shaps = shap_values[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    formatted_names = []
    for name in top_names:
        clean_name = name.replace('_', ' ').title()
        if len(clean_name) > 30:
            clean_name = clean_name[:27] + "..."
        formatted_names.append(clean_name)
        
    plt.figure(figsize=(9, 5))
    colors = ['#ff0051' if val >= 0 else '#008bfb' for val in top_shaps]
    
    y_pos = np.arange(top_n)
    plt.barh(y_pos, top_shaps, align='center', color=colors, alpha=0.85, edgecolor='none', height=0.6)
    
    for i, val in enumerate(top_shaps):
        align = 'left' if val < 0 else 'right'
        offset = -0.01 if val < 0 else 0.01
        plt.text(val + offset, i, f"{val:+.3f}", va='center', ha=align, color='white', fontweight='bold', fontsize=9)
        
    plt.yticks(y_pos, formatted_names, fontsize=10, color='#e0e0e0')
    plt.gca().invert_yaxis()
    plt.xlabel('SHAP Value (Impact on prediction score)', color='#e0e0e0', fontsize=10)
    plt.title('Individual Feature Contribution (SHAP Local Explanation)', color='white', fontsize=12, pad=15, fontweight='bold')
    
    plt.gca().set_facecolor('none')
    plt.gcf().patch.set_facecolor('none')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#555555')
    plt.gca().spines['bottom'].set_color('#555555')
    plt.tick_params(colors='#b0b0b0')
    plt.grid(axis='x', linestyle='--', alpha=0.15, color='#aaaaaa')
    plt.tight_layout()
    
    filename = f"shap_local_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join('static/temp', filename)
    plt.savefig(filepath, dpi=120)
    plt.close()
    
    return f"/static/temp/{filename}", filepath

@app.route('/')
def home():
    return render_template('index.html')

# --- AUTH API ---
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = database.verify_user(username, password)
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['patient_id'] = user['patient_id']
        
        database.log_audit_action(username, f"User logged in successfully (Role: {user['role']})")
        return jsonify({
            'success': True,
            'user': {
                'username': user['username'],
                'role': user['role'],
                'patient_id': user['patient_id']
            }
        })
        
    database.log_audit_action(username or 'Unknown', "Failed login attempt")
    return jsonify({'success': False, 'error': 'Invalid username or password'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    username = session.get('username', 'Unknown')
    database.log_audit_action(username, "User logged out")
    session.clear()
    return jsonify({'success': True})

@app.route('/api/current_user', methods=['GET'])
def get_current_user():
    if 'username' in session:
        return jsonify({
            'logged_in': True,
            'username': session['username'],
            'role': session['role'],
            'patient_id': session['patient_id']
        })
    return jsonify({'logged_in': False})

# --- CLINICAL REGISTRY APIs ---
@app.route('/api/patients', methods=['GET'])
def get_patients():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(database.get_all_patients())

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient_detail(patient_id):
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Patients can only fetch their own details
    if session['role'] == 'patient' and session['patient_id'] != patient_id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    patient = database.get_patient_by_id(patient_id)
    if not patient:
        if patient_id == 'WalkIn':
            database.add_patient('WalkIn', 'Walk-in Patient', 60, 'Male')
            patient = database.get_patient_by_id('WalkIn')
        else:
            return jsonify({'error': 'Patient not found'}), 404
        
    # Associate typical speech feature values from the CSV
    speech_features = {}
    try:
        selected = SPEECH_PIPELINE['selected_features']
        if patient_id == 'WalkIn':
            scaler = SPEECH_PIPELINE['scaler']
            feature_names = list(scaler.feature_names_in_)
            for f in selected:
                if f in feature_names:
                    idx = feature_names.index(f)
                    val = float(scaler.mean_[idx])
                    min_val, max_val, step_val = get_feature_metadata(f)
                    speech_features[f] = {
                        'value': val,
                        'min': min_val,
                        'max': max_val,
                        'step': step_val
                    }
        else:
            df = pd.read_csv('data/pd_speech_features.csv', header=1)
            # H001 maps to class=0 row index 0, P001 maps to class=1 row index 0
            if patient_id.startswith('H'):
                idx = int(patient_id[3:]) - 1
                sample = df[df['class'] == 0].iloc[idx]
            else:
                idx = int(patient_id[3:]) - 1
                sample = df[df['class'] == 1].iloc[idx]
                
            for f in selected:
                val = float(sample[f])
                min_val, max_val, step_val = get_feature_metadata(f)
                speech_features[f] = {
                    'value': val,
                    'min': min_val,
                    'max': max_val,
                    'step': step_val
                }
    except Exception as e:
        print("Error pulling speech preset features:", e)
        
    patient['speech_features'] = speech_features
    return jsonify(patient)

@app.route('/api/patient/register', methods=['POST'])
def register_patient():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.json
    p_id = data.get('id')
    name = data.get('name')
    age = int(data.get('age'))
    gender = data.get('gender')
    
    if not p_id or not name:
        return jsonify({'error': 'Missing ID or Name'}), 400
        
    # 1. Add to patients directory table
    success = database.add_patient(p_id, name, age, gender)
    if not success:
        return jsonify({'error': 'Patient ID already exists'}), 400
        
    # 2. Automatically register a user profile for this patient (username=ID, password=ID+123)
    p_username = p_id.lower()
    p_password = f"{p_id.lower()}123"
    database.register_user(p_username, p_password, 'patient', patient_id=p_id)
    
    database.log_audit_action(session['username'], f"Registered new patient {name} (ID: {p_id})")
    return jsonify({'success': True})

@app.route('/api/patient/history', methods=['GET'])
def get_patient_history():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Determine target patient
    if session['role'] == 'patient':
        patient_id = session['patient_id']
        history = database.get_patient_test_history(patient_id)
    else:
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({'error': 'Missing patient_id'}), 400
        if patient_id == 'all':
            history = database.get_all_recent_tests(50)
        else:
            history = database.get_patient_test_history(patient_id)
            
    return jsonify(history)

# --- SYSTEM ADMIN APIs ---
@app.route('/api/admin/audit_logs', methods=['GET'])
def api_audit_logs():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(database.get_audit_logs())

@app.route('/api/admin/stats', methods=['GET'])
def api_admin_stats():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    conn = database.get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM patients")
    total_patients = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM tests")
    total_tests = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM tests WHERE risk_level='High Risk'")
    high_risk = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tests WHERE risk_level='Moderate Risk'")
    mod_risk = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tests WHERE risk_level='Low Risk'")
    low_risk = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_patients': total_patients,
        'total_tests': total_tests,
        'risk_breakdown': {
            'low': low_risk,
            'moderate': mod_risk,
            'high': high_risk
        }
    })

# --- MULTIMODAL SCREENING API ---
@app.route('/api/predict/multimodal', methods=['POST'])
def predict_multimodal():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    if not SPEECH_PIPELINE or not DRAWINGS_PIPELINE:
        return jsonify({'error': 'ML models are not fully initialized.'}), 500
        
    data = request.json
    patient_id = data.get('patient_id')
    speech_inputs = data.get('speech_features', {})
    canvas_base64 = data.get('canvas_image') # Base64 drawing data URI
    mode = data.get('mode', 'both') # 'voice' | 'handwriting' | 'both'
    
    patient = database.get_patient_by_id(patient_id)
    if not patient:
        if patient_id == 'WalkIn':
            database.add_patient('WalkIn', 'Walk-in Patient', 60, 'Male')
            patient = database.get_patient_by_id('WalkIn')
        else:
            return jsonify({'error': 'Patient not found'}), 404
        
    test_id = f"TST_{uuid.uuid4().hex[:8]}"
    
    # ----------------------------------------------------
    # 1. Pipeline A: Voice Analysis
    # ----------------------------------------------------
    speech_prob = -1.0
    shap_plot_url = ""
    shap_plot_path = ""
    
    if mode in ('voice', 'both'):
        scaler = SPEECH_PIPELINE['scaler']
        selected_features = SPEECH_PIPELINE['selected_features']
        feature_names = scaler.feature_names_in_
        
        # Check if a custom audio file was uploaded (represented as raw signal stats)
        # If not, use standard slider feature mapping
        input_vector = []
        for name in feature_names:
            input_vector.append(float(speech_inputs.get(name, 0.0)))
            
        input_scaled = scaler.transform([input_vector])
        X_test_sel = pd.DataFrame(input_scaled, columns=feature_names)[selected_features]
        
        # Predict using XGBoost
        xgb_speech = SPEECH_PIPELINE['xgb_model']
        speech_prob = float(xgb_speech.predict_proba(X_test_sel)[0, 1])
        
        # Generate local SHAP plot
        explainer = SPEECH_PIPELINE['xgb_explainer']
        raw_shaps = explainer.explain_instance(X_test_sel.values[0])
        shap_plot_url, shap_plot_path = create_local_shap_plot(raw_shaps, X_test_sel.values[0], selected_features)
        
    # ----------------------------------------------------
    # 2. Pipeline B: Handwriting/Drawing Analysis
    # ----------------------------------------------------
    motor_prob = -1.0
    filepath_orig = ""
    filepath_thin = ""
    filename_orig = ""
    filename_thin = ""
    
    if mode in ('handwriting', 'both') and canvas_base64:
        # Decode canvas base64 image
        img_data = canvas_base64.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        filename_orig = f"canvas_{test_id}_orig.png"
        filepath_orig = os.path.join(app.config['UPLOAD_FOLDER'], filename_orig)
        with open(filepath_orig, 'wb') as f:
            f.write(img_bytes)
            
        # Read in grayscale & resize
        img_bgr = cv2.imread(filepath_orig)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (128, 128))
        
        # Preprocess (Otsu & Zhang-Suen)
        thinned, otsu = preprocess_drawing(img_gray)
        
        filename_otsu = f"canvas_{test_id}_otsu.png"
        filename_thin = f"canvas_{test_id}_thin.png"
        filepath_otsu = os.path.join(app.config['UPLOAD_FOLDER'], filename_otsu)
        filepath_thin = os.path.join(app.config['UPLOAD_FOLDER'], filename_thin)
        
        cv2.imwrite(filepath_otsu, otsu)
        cv2.imwrite(filepath_thin, thinned)
        
        # HOG feature extract & predict
        drawing_features, _ = extract_hog_features(thinned)
        xgb_drawings = DRAWINGS_PIPELINE['model']
        motor_prob = float(xgb_drawings.predict_proba([drawing_features])[0, 1])
        
    # ----------------------------------------------------
    # 3. Decision & Scores Fusion
    # ----------------------------------------------------
    if mode == 'voice':
        final_score = speech_prob
    elif mode == 'handwriting':
        final_score = motor_prob
    else:
        # Weight settings: 40% voice acoustic, 60% motor drawing
        w_voice = 0.4
        w_motor = 0.6
        final_score = (w_voice * speech_prob) + (w_motor * motor_prob)
    
    # Risk Classification
    if final_score >= 0.65:
        risk_level = 'High Risk'
    elif final_score >= 0.35:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'Low Risk'
        
    # ----------------------------------------------------
    # 4. Generate Medical PDF Report (`ReportLab`)
    # ----------------------------------------------------
    pdf_filename = f"report_{test_id}.pdf"
    pdf_path = os.path.join('static/reports', pdf_filename)
    pdf_url = f"/static/reports/{pdf_filename}"
    
    report_generator.generate_medical_report(
        test_id=test_id,
        patient_info=patient,
        voice_score=speech_prob,
        motor_score=motor_prob,
        final_score=final_score,
        risk_level=risk_level,
        original_img_path=filepath_orig,
        thinned_img_path=filepath_thin,
        shap_plot_path=shap_plot_path,
        output_pdf_path=pdf_path,
        mode=mode
    )
    
    # ----------------------------------------------------
    # 5. Database Commit & Logs
    # ----------------------------------------------------
    orig_url_val = f"/static/uploads/{filename_orig}" if filename_orig else ""
    thin_url_val = f"/static/uploads/{filename_thin}" if filename_thin else ""
    
    database.log_screening_test(
        test_id=test_id,
        patient_id=patient_id,
        voice_score=speech_prob,
        motor_score=motor_prob,
        final_score=final_score,
        risk_level=risk_level,
        original_url=orig_url_val,
        thinned_url=thin_url_val,
        pdf_url=pdf_url
    )
    
    database.log_audit_action(
        session['username'], 
        f"Executed {mode.upper()} screening layer for patient {patient['name']} (Verdict: {risk_level}, Score: {final_score*100:.1f}%)"
    )
    
    return jsonify({
        'test_id': test_id,
        'voice_score': speech_prob,
        'motor_score': motor_prob,
        'final_score': final_score,
        'risk_level': risk_level,
        'original_drawing_url': orig_url_val,
        'thinned_drawing_url': thin_url_val,
        'pdf_url': pdf_url,
        'shap_plot_url': shap_plot_url
    })

# Endpoint for live microphone recording feature mapping
@app.route('/api/predict/voice_file', methods=['POST'])
def predict_voice_file():
    if 'username' not in session or session['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    patient_id = request.form.get('patient_id')
    
    file_id = uuid.uuid4().hex[:8]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{file_id}.wav")
    file.save(filepath)
    
    # Extract live features
    extracted = signal_processing.extract_vocal_features(filepath)
    
    # Initialize baseline with scaler means
    scaler = SPEECH_PIPELINE['scaler']
    selected = SPEECH_PIPELINE['selected_features']
    feature_names = list(scaler.feature_names_in_)
    
    baseline = {}
    for feat in selected:
        if feat in feature_names:
            idx = feature_names.index(feat)
            baseline[feat] = float(scaler.mean_[idx])
            
    # Pull selected baseline patient features if patient selected to overlay patient values
    if patient_id and patient_id != 'WalkIn':
        try:
            # Re-read CSV patient preset to copy details
            df = pd.read_csv('data/pd_speech_features.csv', header=1)
            if patient_id.startswith('H'):
                idx = int(patient_id[3:]) - 1
                sample = df[df['class'] == 0].iloc[idx]
            else:
                idx = int(patient_id[3:]) - 1
                sample = df[df['class'] == 1].iloc[idx]
            
            for feat in selected:
                if feat in sample:
                    baseline[feat] = float(sample[feat])
        except Exception as e:
            print("Error reading patient presets:", e)
            
    # Map raw features to scale requirements
    mapped = signal_processing.map_live_features_to_rfe(extracted, selected, baseline)
    
    # Clean up audio file
    if os.path.exists(filepath):
        os.remove(filepath)
        
    # Return mapped features formatted with min, max, and step boundaries
    mapped_data = {}
    for feat in selected:
        val = mapped[feat]
        min_val, max_val, step_val = get_feature_metadata(feat)
        mapped_data[feat] = {
            'value': val,
            'min': min_val,
            'max': max_val,
            'step': step_val
        }
        
    return jsonify({
        'extracted': extracted,
        'mapped_features': mapped_data
    })

# Cleanup temp files
@app.route('/api/cleanup', methods=['POST'])
def cleanup_temp_files():
    import shutil
    try:
        shutil.rmtree('static/temp')
        os.makedirs('static/temp', exist_ok=True)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    database.init_db()
    load_models()
    print("NeuroSense AI server running at http://127.0.0.1:5000", flush=True)
    app.run(debug=False, host='127.0.0.1', port=5000)
