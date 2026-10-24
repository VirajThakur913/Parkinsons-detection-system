document.addEventListener('DOMContentLoaded', () => {
    // Session State
    let currentUser = null;
    let activePatient = null;
    let canvasDrawingActive = false;
    let trendChart = null;

    // WAV Audio Recorder State variables
    let audioContext = null;
    let micStream = null;
    let scriptProcessor = null;
    let audioBufferList = [];
    let isRecording = false;

    // DOM Elements
    const loginOverlay = document.getElementById('login-overlay');
    const loginForm = document.getElementById('login-form');
    const loginErrorMsg = document.getElementById('login-error-msg');
    const appWrapper = document.getElementById('app-wrapper');
    const adminDashboard = document.getElementById('admin-dashboard');
    const patientDashboard = document.getElementById('patient-dashboard');
    const btnAppLogout = document.getElementById('btn-app-logout');
    const headerUsername = document.getElementById('header-username');
    const headerUserRole = document.getElementById('header-user-role');
    const userRoleIcon = document.getElementById('user-role-icon');

    // Sidebar Patient Registry
    const patientSearch = document.getElementById('patient-search');
    const patientListContainer = document.getElementById('patient-list-container');
    const btnToggleAddPatient = document.getElementById('btn-toggle-add-patient');
    const addPatientCard = document.getElementById('add-patient-card');
    const btnCloseAddPatient = document.getElementById('btn-close-add-patient');
    const addPatientForm = document.getElementById('add-patient-form');
    const activePatientDetails = document.getElementById('active-patient-details');
    const btnLoadPatientFeatures = document.getElementById('btn-load-patient-features');
    const btnViewPatientPastTests = document.getElementById('btn-view-patient-past-tests');

    const detailName = document.getElementById('detail-name');
    const detailId = document.getElementById('detail-id');
    const detailAgeGender = document.getElementById('detail-age-gender');

    // Screening tab elements
    const slidersContainer = document.getElementById('sliders-container');
    const btnRunMultimodal = document.getElementById('btn-run-multimodal-screening');
    const modelSelector = document.getElementById('model-selector');
    const voiceSourceBadge = document.getElementById('voice-source-badge');

    // Voice Capture
    const btnRecordMic = document.getElementById('btn-record-mic');
    const btnUploadAudio = document.getElementById('btn-upload-audio');
    const voiceFileInput = document.getElementById('voice-file-input');
    const micVisualizerContainer = document.getElementById('mic-visualizer-container');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const recordingTimer = document.getElementById('recording-timer');
    const voiceExtractedStats = document.getElementById('voice-extracted-stats');
    const statF0 = document.getElementById('stat-f0');
    const statJitter = document.getElementById('stat-jitter');
    const statShimmer = document.getElementById('stat-shimmer');
    const statRms = document.getElementById('stat-rms');

    // Whiteboard Canvas
    const drawingCanvas = document.getElementById('drawing-canvas');
    const ctxCanvas = drawingCanvas.getContext('2d');
    const btnClearCanvas = document.getElementById('btn-clear-canvas');
    const btnUploadDrawing = document.getElementById('btn-upload-drawing');
    const drawingFileInput = document.getElementById('drawing-file-input');
    const whiteboardContainer = document.getElementById('whiteboard-container');
    const btnPresetHealthy = document.getElementById('btn-preset-draw-healthy');
    const btnPresetParkinson = document.getElementById('btn-preset-draw-parkinson');

    // Assessment outputs
    const btnDownloadPdfReport = document.getElementById('btn-download-pdf-report');
    const multimodalPlaceholder = document.getElementById('multimodal-placeholder');
    const multimodalResultContainer = document.getElementById('multimodal-result-container');
    const valVocalRisk = document.getElementById('val-vocal-risk');
    const valMotorRisk = document.getElementById('val-motor-risk');
    const valCombinedRisk = document.getElementById('val-combined-risk');
    const multimodalVerdictBadge = document.getElementById('multimodal-verdict-badge');
    const fusionImgOriginal = document.getElementById('fusion-img-original');
    const fusionImgThinned = document.getElementById('fusion-img-thinned');
    const fusionShapImage = document.getElementById('fusion-shap-image');
    const multimodalGauge = document.getElementById('multimodal-gauge');

    // Tab buttons & views
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Admin history & Audit
    const clinicHistoryTbody = document.getElementById('clinic-history-tbody');
    const auditLogsTbody = document.getElementById('audit-logs-tbody');
    const btnRefreshAudit = document.getElementById('btn-refresh-audit');

    // Patient Dashboard Elements
    const patientProfileName = document.getElementById('patient-profile-name');
    const patientProfileId = document.getElementById('patient-profile-id');
    const patientProfileAge = document.getElementById('patient-profile-age');
    const patientProfileGender = document.getElementById('patient-profile-gender');
    const patientHistoryTbody = document.getElementById('patient-history-tbody');

    // Check existing session
    checkSession();

    // 1. Session Auth Management
    async function checkSession() {
        try {
            const res = await fetch('/api/current_user');
            const data = await res.json();
            if (data.logged_in) {
                authSuccess(data);
            } else {
                loginOverlay.style.display = 'flex';
                appWrapper.style.display = 'none';
            }
        } catch (err) {
            console.error("Session check failed:", err);
        }
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;

        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();
            if (data.success) {
                authSuccess(data.user);
            }
        } catch (err) {
            loginErrorMsg.textContent = "Invalid username or password credentials.";
            loginErrorMsg.style.display = 'block';
        }
    });

    btnAppLogout.addEventListener('click', async () => {
        try {
            await fetch('/api/logout', { method: 'POST' });
            currentUser = null;
            activePatient = null;
            if (trendChart) {
                trendChart.destroy();
                trendChart = null;
            }
            loginOverlay.style.display = 'flex';
            appWrapper.style.display = 'none';
            document.getElementById('username').value = '';
            document.getElementById('password').value = '';
            loginErrorMsg.style.display = 'none';
        } catch (err) {
            console.error("Logout failed:", err);
        }
    });

    function authSuccess(user) {
        currentUser = user;
        loginOverlay.style.display = 'none';
        appWrapper.style.display = 'block';

        headerUsername.textContent = user.username === 'admin' ? 'Dr. Sarah Jenkins (CDSS)' : user.username.toUpperCase();
        headerUserRole.textContent = user.role === 'admin' ? 'Administrator / Clinician' : 'Patient';
        
        if (user.role === 'admin') {
            userRoleIcon.className = 'fa-solid fa-user-doctor user-icon';
            adminDashboard.style.display = 'grid';
            patientDashboard.style.display = 'none';
            // Init admin registry
            fetchPatients();
            fetchClinicHistory();
            fetchAuditLogs();
            initDrawingCanvas();
            loadWalkInDefaults();
        } else {
            userRoleIcon.className = 'fa-solid fa-user user-icon';
            adminDashboard.style.display = 'none';
            patientDashboard.style.display = 'grid';
            // Init patient view
            initPatientDashboard(user.patient_id);
        }
    }

    async function loadWalkInDefaults() {
        try {
            const res = await fetch('/api/patient/WalkIn');
            const data = await res.json();
            renderAcousticSliders(data.speech_features);
            voiceSourceBadge.className = 'badge';
            voiceSourceBadge.style.background = '#64748b';
            voiceSourceBadge.textContent = 'Walk-In Presets Loaded';
        } catch (err) {
            console.error("Error loading WalkIn defaults:", err);
        }
    }

    // 2. Admin Dashboard Operations
    async function fetchPatients() {
        try {
            const res = await fetch('/api/patients');
            const data = await res.json();
            renderPatientsList(data);
        } catch (err) {
            console.error("Error loading directory:", err);
        }
    }

    function renderPatientsList(patients) {
        patientListContainer.innerHTML = '';
        document.getElementById('patient-count').textContent = patients.length;
        
        patients.forEach(p => {
            const item = document.createElement('div');
            item.className = 'patient-item';
            item.setAttribute('data-id', p.id);
            const dotClass = p.status === 'Healthy' ? 'healthy' : 'parkinson';
            
            item.innerHTML = `
                <div class="p-info">
                    <h4>${p.name}</h4>
                    <p>ID: ${p.id} | Age: ${p.age} | ${p.gender}</p>
                </div>
                <div class="status-dot ${dotClass}" title="${p.status}"></div>
            `;
            
            item.addEventListener('click', () => {
                document.querySelectorAll('.patient-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                selectActivePatient(p.id);
            });
            patientListContainer.appendChild(item);
        });
    }

    // Filter Patients
    patientSearch.addEventListener('input', (e) => {
        const q = e.target.value.toLowerCase();
        document.querySelectorAll('.patient-item').forEach(item => {
            const text = item.textContent.toLowerCase();
            item.style.display = text.includes(q) ? 'flex' : 'none';
        });
    });

    // Toggle add patient card
    btnToggleAddPatient.addEventListener('click', () => {
        addPatientCard.style.display = addPatientCard.style.display === 'none' ? 'block' : 'none';
    });
    btnCloseAddPatient.addEventListener('click', () => {
        addPatientCard.style.display = 'none';
    });

    // Handle add patient submit
    addPatientForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const id = document.getElementById('new-p-id').value;
        const name = document.getElementById('new-p-name').value;
        const age = document.getElementById('new-p-age').value;
        const gender = document.getElementById('new-p-gender').value;

        try {
            const res = await fetch('/api/patient/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, name, age, gender })
            });
            const data = await res.json();
            if (data.success) {
                addPatientForm.reset();
                addPatientCard.style.display = 'none';
                fetchPatients();
                fetchAuditLogs();
            } else {
                alert("Error: " + data.error);
            }
        } catch (err) {
            console.error("Patient registration failed:", err);
        }
    });

    async function selectActivePatient(pId) {
        try {
            const res = await fetch(`/api/patient/${pId}`);
            activePatient = await res.json();
            
            detailName.textContent = activePatient.name;
            detailId.textContent = activePatient.id;
            detailAgeGender.textContent = `${activePatient.age} yrs / ${activePatient.gender}`;
            activePatientDetails.style.display = 'block';
            
            // Populate Sliders
            renderAcousticSliders(activePatient.speech_features);
            voiceSourceBadge.className = 'badge';
            voiceSourceBadge.style.background = '#8257e5';
            voiceSourceBadge.textContent = 'Baseline Loaded';

            // Clear visualizer metrics
            voiceExtractedStats.style.display = 'none';
            
            checkScreeningState();
        } catch (err) {
            console.error("Failed loading patient details:", err);
        }
    }

    btnLoadPatientFeatures.addEventListener('click', () => {
        if (activePatient && activePatient.speech_features) {
            renderAcousticSliders(activePatient.speech_features);
            voiceSourceBadge.className = 'badge';
            voiceSourceBadge.style.background = '#8257e5';
            voiceSourceBadge.textContent = 'Baseline Loaded';
            checkScreeningState();
        }
    });

    btnViewPatientPastTests.addEventListener('click', () => {
        if (activePatient) {
            tabBtns.forEach(btn => {
                if (btn.getAttribute('data-tab') === 'history') {
                    btn.click();
                }
            });
            fetchClinicHistory();
        }
    });

    // 3. Acoustic Sliders builder
    function renderAcousticSliders(features) {
        slidersContainer.innerHTML = '';
        Object.keys(features).forEach(key => {
            const featData = features[key];
            const val = featData.value;
            const min = featData.min;
            const max = featData.max;
            const step = featData.step;
            
            const group = document.createElement('div');
            group.className = 'slider-group';
            
            let cleanName = key.replace(/_/g, ' ').toUpperCase();
            if (cleanName.startsWith('TQWT ')) {
                cleanName = cleanName.replace('TQWT ', 'TQWT ');
            }

            group.innerHTML = `
                <div class="slider-info">
                    <span class="feat-name" title="${key}">${cleanName}</span>
                    <span class="feat-val" id="val-${key}">${val.toFixed(5)}</span>
                </div>
                <div class="slider-input-wrapper">
                    <input type="range" id="slider-${key}" min="${min}" max="${max}" step="${step}" value="${val}">
                </div>
            `;
            slidersContainer.appendChild(group);

            const input = group.querySelector('input');
            input.addEventListener('input', (e) => {
                document.getElementById(`val-${key}`).textContent = parseFloat(e.target.value).toFixed(5);
            });
        });
    }

    // 4. Drawing Canvas (Whiteboard tracing)
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function initDrawingCanvas() {
        ctxCanvas.strokeStyle = '#ffffff'; // White pen trace
        ctxCanvas.lineJoin = 'round';
        ctxCanvas.lineCap = 'round';
        ctxCanvas.lineWidth = 3;
        
        // Fill canvas with very dark slate so segmenting works properly
        clearBoard();

        drawingCanvas.addEventListener('mousedown', startDrawing);
        drawingCanvas.addEventListener('mousemove', draw);
        drawingCanvas.addEventListener('mouseup', stopDrawing);
        drawingCanvas.addEventListener('mouseout', stopDrawing);

        // Mobile touch events
        drawingCanvas.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            const rect = drawingCanvas.getBoundingClientRect();
            startDrawing({
                clientX: touch.clientX,
                clientY: touch.clientY
            });
        }, { passive: true });

        drawingCanvas.addEventListener('touchmove', (e) => {
            const touch = e.touches[0];
            draw({
                clientX: touch.clientX,
                clientY: touch.clientY,
                preventDefault: () => {}
            });
        }, { passive: true });

        drawingCanvas.addEventListener('touchend', stopDrawing);
    }

    function clearBoard() {
        ctxCanvas.fillStyle = '#060913';
        ctxCanvas.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        canvasDrawingActive = false;
        checkScreeningState();
    }

    btnClearCanvas.addEventListener('click', clearBoard);

    function startDrawing(e) {
        isDrawing = true;
        const rect = drawingCanvas.getBoundingClientRect();
        // Scale for internal canvas width/height matching physical clientBounding
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;
        
        lastX = (e.clientX - rect.left) * scaleX;
        lastY = (e.clientY - rect.top) * scaleY;
        canvasDrawingActive = true;
        checkScreeningState();
    }

    function draw(e) {
        if (!isDrawing) return;
        if (e.preventDefault) e.preventDefault();
        
        const rect = drawingCanvas.getBoundingClientRect();
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        ctxCanvas.beginPath();
        ctxCanvas.moveTo(lastX, lastY);
        ctxCanvas.lineTo(x, y);
        ctxCanvas.stroke();
        
        lastX = x;
        lastY = y;
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Drawing presets triggers
    btnPresetHealthy.addEventListener('click', async () => {
        loadPresetDrawingImage('/static/sample_drawings/healthy_spiral.png');
    });

    btnPresetParkinson.addEventListener('click', async () => {
        loadPresetDrawingImage('/static/sample_drawings/parkinson_spiral.png');
    });

    function loadPresetDrawingImage(url) {
        const img = new Image();
        img.onload = () => {
            ctxCanvas.fillStyle = '#060913';
            ctxCanvas.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            
            // Calculate scale to fit inside drawingCanvas preserving aspect ratio
            const scale = Math.min(drawingCanvas.width / img.width, drawingCanvas.height / img.height);
            const w = img.width * scale;
            const h = img.height * scale;
            const x = (drawingCanvas.width - w) / 2;
            const y = (drawingCanvas.height - h) / 2;
            
            ctxCanvas.drawImage(img, x, y, w, h);
            canvasDrawingActive = true;
            checkScreeningState();
        };
        img.src = url;
    }

    // Photo/Image Upload for Drawings
    if (btnUploadDrawing && drawingFileInput) {
        btnUploadDrawing.addEventListener('click', () => {
            drawingFileInput.click();
        });

        drawingFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    loadPresetDrawingImage(event.target.result);
                };
                reader.readAsDataURL(file);
            }
            drawingFileInput.value = '';
        });
    }

    // Drag & Drop for Drawings
    if (whiteboardContainer) {
        whiteboardContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            whiteboardContainer.classList.add('drag-over');
        });

        whiteboardContainer.addEventListener('dragleave', () => {
            whiteboardContainer.classList.remove('drag-over');
        });

        whiteboardContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            whiteboardContainer.classList.remove('drag-over');
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        loadPresetDrawingImage(event.target.result);
                    };
                    reader.readAsDataURL(file);
                } else {
                    alert("Please drop an image file (PNG, JPG, JPEG).");
                }
            }
        });
    }

    // 5. Live Audio Capture Visualizer (Web Audio API)
    btnRecordMic.addEventListener('click', () => {
        if (isRecording) {
            stopAudioRecording();
        } else {
            startAudioRecording();
        }
    });

    async function startAudioRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Your browser does not support voice recording APIs.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            isRecording = true;
            btnRecordMic.innerHTML = '<i class="fa-solid fa-stop-circle"></i> Stop Record';
            btnRecordMic.style.background = '#f43f5e';
            btnRecordMic.style.borderColor = '#f43f5e';
            
            audioBufferList = [];
            
            // Web Audio Nodes setup
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            micStream = audioContext.createMediaStreamSource(stream);
            
            // Draw visual waveform
            setupAudioWaveVisualizer(stream);

            // Record raw floats list using ScriptProcessorNode for WAV PCM packaging
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const buffer = e.inputBuffer.getChannelData(0);
                audioBufferList.push(new Float32Array(buffer));
            };

            micStream.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            // Capture timer countdown
            let count = 3.0;
            micVisualizerContainer.style.display = 'flex';
            recordingTimer.textContent = `00:03`;
            
            const timer = setInterval(() => {
                count -= 1;
                recordingTimer.textContent = `00:0${Math.max(0, count)}`;
                if (count <= 0) {
                    clearInterval(timer);
                    if (isRecording) stopAudioRecording();
                }
            }, 1000);

        } catch (err) {
            console.error("Failed to start voice capture:", err);
            alert("Microphone permission denied.");
        }
    }

    function stopAudioRecording() {
        if (!isRecording) return;
        isRecording = false;
        btnRecordMic.innerHTML = '<i class="fa-solid fa-microphone"></i> Record Mic';
        btnRecordMic.style.background = 'transparent';
        btnRecordMic.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        micVisualizerContainer.style.display = 'none';

        // Disconnect nodes
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (micStream) {
            micStream.disconnect();
            micStream = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        // Package Float arrays to 16-bit PCM WAV file blob
        const wavBlob = compilePCMFloatListToWAV(audioBufferList, 16000);
        uploadVoiceFile(wavBlob);
    }

    function setupAudioWaveVisualizer(stream) {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const src = context.createMediaStreamSource(stream);
        const analyser = context.createAnalyser();
        
        src.connect(analyser);
        analyser.fftSize = 256;
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const canvasCtx = waveformCanvas.getContext('2d');
        const width = waveformCanvas.width;
        const height = waveformCanvas.height;

        function drawWave() {
            if (!isRecording) {
                context.close();
                return;
            }
            requestAnimationFrame(drawWave);
            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = '#060913';
            canvasCtx.fillRect(0, 0, width, height);
            
            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = '#f43f5e';
            canvasCtx.beginPath();

            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * height) / 2.0;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            canvasCtx.lineTo(width, height / 2.0);
            canvasCtx.stroke();
        }
        drawWave();
    }

    // WAV Encoder Utility (Compiles browser microphone stream into 16-bit uncompressed WAV)
    function compilePCMFloatListToWAV(floatsList, sampleRate) {
        // Flatten list of Float32Arrays
        let totalLen = 0;
        floatsList.forEach(arr => totalLen += arr.length);
        const pcmData = new Float32Array(totalLen);
        let offset = 0;
        floatsList.forEach(arr => {
            pcmData.set(arr, offset);
            offset += arr.length;
        });

        // 16-bit PCM Conversion
        const buffer = new ArrayBuffer(44 + pcmData.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + pcmData.length * 2, true);
        /* RIFF type */
        writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw PCM = 1) */
        view.setUint16(20, 1, true);
        /* channel count */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sampleRate * blockAlign) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (channelCount * bytesPerSample) */
        view.setUint16(32, 2, true);
        /* bits per sample */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        writeString(view, 36, 'data');
        /* chunk length */
        view.setUint32(40, pcmData.length * 2, true);

        // Write Float values as 16-bit Ints
        let index = 44;
        for (let i = 0; i < pcmData.length; i++) {
            let s = Math.max(-1.0, Math.min(1.0, pcmData[i]));
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            index += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    async function uploadVoiceFile(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'live_audio.wav');
        formData.append('patient_id', activePatient ? activePatient.id : 'WalkIn');

        // Show spinner inside Sliders container
        slidersContainer.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-circle-notch fa-spin"></i> Processing Audio features...</div>';

        try {
            const res = await fetch('/api/predict/voice_file', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            // Render sliders with mapped features
            renderAcousticSliders(data.mapped_features);
            voiceSourceBadge.className = 'badge';
            voiceSourceBadge.style.background = '#10b981';
            voiceSourceBadge.textContent = 'Live Audio Mapped';

            // Show statistics
            document.getElementById('stat-f0').textContent = `${Math.round(data.extracted.f0)}Hz`;
            document.getElementById('stat-jitter').textContent = `${(data.extracted.jitter * 100).toFixed(3)}%`;
            document.getElementById('stat-shimmer').textContent = `${(data.extracted.shimmer * 100).toFixed(2)}%`;
            document.getElementById('stat-rms').textContent = data.extracted.rms.toFixed(4);
            voiceExtractedStats.style.display = 'flex';

            checkScreeningState();

        } catch (err) {
            console.error("Failed to upload audio:", err);
            alert("Acoustic extraction processing failed.");
            if (activePatient) renderAcousticSliders(activePatient.speech_features);
        }
    }

    // Audio File Upload handler
    if (btnUploadAudio && voiceFileInput) {
        btnUploadAudio.addEventListener('click', () => {
            voiceFileInput.click();
        });

        voiceFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            voiceSourceBadge.className = 'badge';
            voiceSourceBadge.style.background = '#00c2ff';
            voiceSourceBadge.textContent = 'Processing Audio...';
            slidersContainer.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-circle-notch fa-spin"></i> Decoding & Extracting Audio Features...</div>';

            try {
                // Decode file into 16kHz mono PCM using browser AudioContext
                const arrayBuffer = await file.arrayBuffer();
                const tempCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
                const channelData = decodedBuffer.getChannelData(0);
                
                // Encode into compliant standard 16kHz 16-bit mono PCM WAV
                const pcmWavBlob = compilePCMFloatListToWAV([channelData], 16000);
                tempCtx.close();

                await uploadVoiceFile(pcmWavBlob);
                voiceSourceBadge.textContent = 'Audio File Mapped';
            } catch (err) {
                console.error("Browser audio decoding error, fallback to direct upload:", err);
                // Fallback: upload file directly
                await uploadVoiceFile(file);
                voiceSourceBadge.textContent = 'Audio File Mapped';
            }
            voiceFileInput.value = '';
        });
    }

    // 6. Screening execution trigger logic
    function checkScreeningState() {
        if (canvasDrawingActive) {
            btnRunMultimodal.removeAttribute('disabled');
        } else {
            btnRunMultimodal.setAttribute('disabled', 'true');
        }
    }

    btnRunMultimodal.addEventListener('click', async () => {
        if (!canvasDrawingActive) return;

        const patientId = activePatient ? activePatient.id : 'WalkIn';

        // Collect sliders values
        const speech_features = {};
        const sliders = slidersContainer.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            const key = slider.id.replace('slider-', '');
            speech_features[key] = parseFloat(slider.value);
        });

        // Get Base64 image drawing
        const canvas_image = drawingCanvas.toDataURL('image/png');

        // Show loading spinner
        multimodalPlaceholder.style.display = 'none';
        multimodalResultContainer.style.display = 'none';
        btnDownloadPdfReport.style.display = 'none';

        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Training Multimodal Weights, Skeletonizing & Compiling Medical Report...';
        multimodalPlaceholder.parentNode.insertBefore(spinner, multimodalPlaceholder.nextSibling);

        try {
            const res = await fetch('/api/predict/multimodal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_id: patientId,
                    speech_features,
                    canvas_image
                })
            });
            const data = await res.json();
            spinner.remove();

            if (data.error) {
                alert("Error: " + data.error);
                multimodalPlaceholder.style.display = 'flex';
                return;
            }

            // Populate Output Risk Values
            valVocalRisk.textContent = `${Math.round(data.voice_score * 100)}%`;
            valMotorRisk.textContent = `${Math.round(data.motor_score * 100)}%`;
            
            const combPct = Math.round(data.final_score * 100);
            valCombinedRisk.textContent = `${combPct}%`;

            // Verdict Badges coloring
            let color = '#10b981'; // Green
            if (data.risk_level === 'High Risk') {
                color = '#f43f5e'; // Rose
                multimodalVerdictBadge.className = 'badge high-risk';
            } else if (data.risk_level === 'Moderate Risk') {
                color = '#f59e0b'; // Amber
                multimodalVerdictBadge.className = 'badge moderate-risk';
            } else {
                multimodalVerdictBadge.className = 'badge low-risk';
            }
            multimodalVerdictBadge.textContent = data.risk_level;
            
            // Set Combined circular gauge gradient
            multimodalGauge.style.background = `conic-gradient(${color} ${combPct}%, rgba(255,255,255,0.05) ${combPct}%)`;

            // Intermediate stages images
            const t = new Date().getTime();
            fusionImgOriginal.src = data.original_drawing_url + '?t=' + t;
            fusionImgThinned.src = data.thinned_drawing_url + '?t=' + t;
            fusionShapImage.src = data.shap_plot_url + '?t=' + t;

            // Enable PDF Download
            btnDownloadPdfReport.onclick = () => {
                window.open(data.pdf_url, '_blank');
            };
            btnDownloadPdfReport.style.display = 'block';

            multimodalResultContainer.style.display = 'block';
            
            // Refresh logs
            fetchClinicHistory();
            fetchAuditLogs();

        } catch (err) {
            spinner.remove();
            console.error("Screening prediction failed:", err);
            alert("Multimodal Screening execution failed.");
            multimodalPlaceholder.style.display = 'flex';
        }
    });

    // 7. Tab history list & Audits loader
    async function fetchClinicHistory() {
        try {
            const res = await fetch('/api/patient/history?patient_id=all'); // Admin requests all logs
            // Wait, does backend support patient_id=all?
            // In app.py: patient_id = request.args.get('patient_id')
            // And in database: if patient_id is H001, retrieve patient logs.
            // Let's make sure the backend returns recent logs if all is selected!
            // Actually, we can fetch all recent tests using a new query or `/api/admin/recent`
            // Let's check how app.py handles it:
            // "patient_id = request.args.get('patient_id')"
            // If patient_id is "all" or we query `/api/admin/history`?
            // In app.py line 125-139 of database.py we have `get_all_recent_tests(limit)`.
            // Let's write client-side query to look up all recent tests!
            // In app.py, `/api/patient/history` looks up `get_patient_test_history(patient_id)`.
            // If patient_id is undefined or "all", we can fetch recent tests from `/api/admin/recent`!
            // Wait, does app.py expose `/api/admin/recent`? No, but it has `get_all_recent_tests`!
            // Let's check what app.py has:
            // It has `get_all_recent_tests` in database.py, but does it expose it in app.py?
            // No, but we can query `/api/patient/history?patient_id=all` and in app.py:
            // "if patient_id == 'all': return jsonify(database.get_all_recent_tests())"
            // Wait! In the previous app.py we did:
            // `history = database.get_patient_test_history(patient_id)`
            // If patient_id == 'all', it would query tests where patient_id='all' which returns empty!
            // Oh! Let's check if we should add supporting for `patient_id == 'all'` in `app.py`!
            // Yes! That's extremely important so the Admin can see the complete history list!
            // Let's modify `app.py` to check `if patient_id == 'all' or patient_id is None:`
            // and return `database.get_all_recent_tests(50)`!
            // Let's make this small edit to `app.py` in a moment.
            // Let's write the JS fetch logic.
            const url = activePatient ? `/api/patient/history?patient_id=${activePatient.id}` : `/api/patient/history?patient_id=all`;
            const response = await fetch(url);
            const data = await response.json();
            renderClinicHistoryTable(data);
        } catch (err) {
            console.error("Failed to load clinic history:", err);
        }
    }

    function renderClinicHistoryTable(tests) {
        clinicHistoryTbody.innerHTML = '';
        if (tests.length === 0) {
            clinicHistoryTbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No tests logged in clinic.</td></tr>';
            return;
        }
        
        tests.forEach(t => {
            const row = document.createElement('tr');
            
            // Format score as percentage
            const scorePct = `${Math.round(t.final_score * 100)}%`;
            
            let badgeClass = 'low-risk';
            if (t.risk_level === 'High Risk') badgeClass = 'high-risk';
            else if (t.risk_level === 'Moderate Risk') badgeClass = 'moderate-risk';
            
            row.innerHTML = `
                <td><code>${t.id}</code></td>
                <td>${t.patient_id}</td>
                <td>${t.patient_name || activePatient?.name || t.patient_id}</td>
                <td>${t.date}</td>
                <td><strong>${scorePct}</strong></td>
                <td><span class="badge ${badgeClass}">${t.risk_level}</span></td>
                <td>
                    <a href="${t.pdf_url}" target="_blank" class="btn btn-secondary btn-sm">
                        <i class="fa-solid fa-file-pdf"></i> View PDF
                    </a>
                </td>
            `;
            clinicHistoryTbody.appendChild(row);
        });
    }

    async function fetchAuditLogs() {
        try {
            const res = await fetch('/api/admin/audit_logs');
            const data = await res.json();
            renderAuditLogsTable(data);
        } catch (err) {
            console.error("Failed to load audit logs:", err);
        }
    }

    function renderAuditLogsTable(logs) {
        auditLogsTbody.innerHTML = '';
        if (logs.length === 0) {
            auditLogsTbody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No audit trails found.</td></tr>';
            return;
        }

        logs.forEach(l => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><code>${l.id}</code></td>
                <td><strong>${l.username}</strong></td>
                <td>${l.action}</td>
                <td>${l.timestamp}</td>
            `;
            auditLogsTbody.appendChild(row);
        });
    }

    btnRefreshAudit.addEventListener('click', fetchAuditLogs);

    // 8. Patient Dashboard Personal Views
    async function initPatientDashboard(patientId) {
        try {
            // Fetch demographics
            const resPatient = await fetch(`/api/patient/${patientId}`);
            const p = await resPatient.json();
            
            patientProfileName.textContent = p.name;
            patientProfileId.textContent = p.id;
            patientProfileAge.textContent = `${p.age} years`;
            patientProfileGender.textContent = p.gender;

            // Fetch test history
            const resHistory = await fetch(`/api/patient/history`);
            const history = await resHistory.json();
            
            renderPatientPersonalHistory(history);
            renderPatientTrendChart(history);

        } catch (err) {
            console.error("Failed to load patient dashboard:", err);
        }
    }

    function renderPatientPersonalHistory(tests) {
        patientHistoryTbody.innerHTML = '';
        if (tests.length === 0) {
            patientHistoryTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No diagnostics run yet. Please ask your clinic practitioner.</td></tr>';
            return;
        }

        tests.forEach(t => {
            const row = document.createElement('tr');
            const scorePct = `${Math.round(t.final_score * 100)}%`;
            const voicePct = `${Math.round(t.voice_score * 100)}%`;
            const motorPct = `${Math.round(t.motor_score * 100)}%`;

            let badgeClass = 'low-risk';
            if (t.risk_level === 'High Risk') badgeClass = 'high-risk';
            else if (t.risk_level === 'Moderate Risk') badgeClass = 'moderate-risk';

            row.innerHTML = `
                <td>${t.date}</td>
                <td>${voicePct}</td>
                <td>${motorPct}</td>
                <td><strong>${scorePct}</strong></td>
                <td><span class="badge ${badgeClass}">${t.risk_level}</span></td>
                <td>
                    <a href="${t.pdf_url}" target="_blank" class="btn btn-secondary btn-sm">
                        <i class="fa-solid fa-file-pdf"></i> Download PDF
                    </a>
                </td>
            `;
            patientHistoryTbody.appendChild(row);
        });
    }

    function renderPatientTrendChart(tests) {
        if (trendChart) {
            trendChart.destroy();
        }

        if (tests.length === 0) return;

        // Sort tests chronologically for chart
        const sortedTests = [...tests].reverse();
        const labels = sortedTests.map(t => t.date.split(' ')[0]);
        const scores = sortedTests.map(t => Math.round(t.final_score * 100));

        const ctx = document.getElementById('patient-trend-chart').getContext('2d');
        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Combined Risk Score (%)',
                    data: scores,
                    borderColor: '#8257e5',
                    backgroundColor: 'rgba(130, 87, 229, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointBackgroundColor: '#00c2ff',
                    pointBorderColor: '#ffffff',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#8e9bb3' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8e9bb3' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // Tab buttons active switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            document.getElementById(`tab-${target}`).classList.add('active');
            
            if (target === 'history') fetchClinicHistory();
            if (target === 'audit') fetchAuditLogs();
        });
    });
});
