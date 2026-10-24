import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_medical_report(test_id, patient_info, voice_score, motor_score, final_score, risk_level, original_img_path, thinned_img_path, shap_plot_path, output_pdf_path, mode='both'):
    """
    Compiles a highly professional medical screening PDF report.
    """
    # Create parent folder if not exists
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Page setup - 0.5 inch margins for clinical grid alignment
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ClinicTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'ClinicSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=8,
        borderColor=colors.HexColor('#cbd5e1')
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    label_style = ParagraphStyle(
        'TableLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.HexColor('#475569')
    )
    
    value_style = ParagraphStyle(
        'TableValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#0f172a')
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("NEUROSENSE AI — CLINICAL ASSESSMENT REPORT", title_style))
    
    if mode == 'voice':
        subtitle_text = "Parkinson's Disease Vocal Acoustic Screening Layer"
    elif mode == 'handwriting':
        subtitle_text = "Parkinson's Disease Motor Control Screening Layer"
    else:
        subtitle_text = "Parkinson's Disease Multi-Modal Decision Support System"
        
    story.append(Paragraph(subtitle_text, subtitle_style))
    
    # 1. Demographics Grid
    demo_data = [
        [
            Paragraph("Patient Name:", label_style), Paragraph(patient_info['name'], value_style),
            Paragraph("Record ID:", label_style), Paragraph(patient_info['id'], value_style)
        ],
        [
            Paragraph("Age / Gender:", label_style), Paragraph(f"{patient_info['age']} / {patient_info['gender']}", value_style),
            Paragraph("Assessment Date:", label_style), Paragraph(datetime_now_str(), value_style)
        ],
        [
            Paragraph("Assessment ID:", label_style), Paragraph(test_id, value_style),
            Paragraph("Status:", label_style), Paragraph("Completed", value_style)
        ]
    ]
    
    demo_table = Table(demo_data, colWidths=[100, 170, 100, 170])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#f1f5f9')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(demo_table)
    story.append(Spacer(1, 15))
    
    # 2. Screening Result Summary Box
    if risk_level == 'High Risk':
        risk_color = colors.HexColor('#fee2e2')
        risk_text_color = colors.HexColor('#991b1b')
        risk_desc = "Clinical indicators reveal significant diagnostic anomalies consistent with fine-motor or vocal indicators of Parkinson's Disease. Further specialist neurologist evaluation is highly recommended."
    elif risk_level == 'Moderate Risk':
        risk_color = colors.HexColor('#fef3c7')
        risk_text_color = colors.HexColor('#92400e')
        risk_desc = "Borderline motor or vocal deviations detected. Patient displays moderate instability. Close monitoring and longitudinal screening follow-up in 3 months is advised."
    else:
        risk_color = colors.HexColor('#d1fae5')
        risk_text_color = colors.HexColor('#065f46')
        risk_desc = "Acoustic parameters and drawing kinematics reside within healthy baseline ranges. No indications of Parkinsonian motor tremors or vocal fold instabilities detected."

    # Build summary table according to active mode
    if mode == 'voice':
        result_data = [
            [
                Paragraph("Vocal Risk Score", label_style),
                Paragraph("Screening Verdict", label_style)
            ],
            [
                Paragraph(f"{voice_score * 100:.1f}%", ParagraphStyle('VS', parent=value_style, fontName='Helvetica-Bold', fontSize=15)),
                Paragraph(risk_level.upper(), ParagraphStyle('RL', parent=value_style, fontName='Helvetica-Bold', fontSize=15, textColor=risk_text_color))
            ]
        ]
        result_table = Table(result_data, colWidths=[270, 270])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (1,1), (1,1), risk_color), # highlight verdict
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
    elif mode == 'handwriting':
        result_data = [
            [
                Paragraph("Motor Risk Score", label_style),
                Paragraph("Screening Verdict", label_style)
            ],
            [
                Paragraph(f"{motor_score * 100:.1f}%", ParagraphStyle('MS', parent=value_style, fontName='Helvetica-Bold', fontSize=15)),
                Paragraph(risk_level.upper(), ParagraphStyle('RL', parent=value_style, fontName='Helvetica-Bold', fontSize=15, textColor=risk_text_color))
            ]
        ]
        result_table = Table(result_data, colWidths=[270, 270])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (1,1), (1,1), risk_color), # highlight verdict
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
    else:
        result_data = [
            [
                Paragraph("Vocal Risk Score", label_style),
                Paragraph("Motor Risk Score", label_style),
                Paragraph("Combined Risk Score", label_style),
                Paragraph("Screening Verdict", label_style)
            ],
            [
                Paragraph(f"{voice_score * 100:.1f}%", ParagraphStyle('VS', parent=value_style, fontName='Helvetica-Bold', fontSize=14)),
                Paragraph(f"{motor_score * 100:.1f}%", ParagraphStyle('MS', parent=value_style, fontName='Helvetica-Bold', fontSize=14)),
                Paragraph(f"{final_score * 100:.1f}%", ParagraphStyle('FS', parent=value_style, fontName='Helvetica-Bold', fontSize=16, textColor=risk_text_color)),
                Paragraph(risk_level.upper(), ParagraphStyle('RL', parent=value_style, fontName='Helvetica-Bold', fontSize=14, textColor=risk_text_color))
            ]
        ]
        result_table = Table(result_data, colWidths=[135, 135, 135, 135])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (2,1), (3,1), risk_color), # highlight final score & verdict
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
    
    story.append(Paragraph("DIAGNOSTIC EXECUTIVE SUMMARY", section_heading))
    story.append(result_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Clinical Interpretation Note:</b> {risk_desc}", body_style))
    story.append(Spacer(1, 15))
    
    # 3. Handwriting Analysis Section (Omitted in Voice-Only Mode)
    if mode != 'voice':
        story.append(Paragraph("MOTOR ACQUISITION & SKELETONIZATION", section_heading))
        orig_img = Image(original_img_path, width=2.3*inch, height=2.3*inch) if os.path.exists(original_img_path) else Paragraph("Image Missing", body_style)
        thin_img = Image(thinned_img_path, width=2.3*inch, height=2.3*inch) if os.path.exists(thinned_img_path) else Paragraph("Image Missing", body_style)
        
        drawing_data = [
            [orig_img, thin_img],
            [Paragraph("<b>Figure 1a:</b> Original Drawing", ParagraphStyle('F1a', parent=body_style, align='CENTER')), 
             Paragraph("<b>Figure 1b:</b> Zhang-Suen Thinned Output", ParagraphStyle('F1b', parent=body_style, align='CENTER'))]
        ]
        
        drawing_table = Table(drawing_data, colWidths=[270, 270])
        drawing_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(drawing_table)
        story.append(Spacer(1, 15))
    
    # 4. Vocal Analysis Section (Omitted in Handwriting-Only Mode)
    if mode != 'handwriting':
        story.append(Paragraph("VOCAL ACOUSTIC ANALYSIS & EXPLAINABILITY (XAI)", section_heading))
        shap_img = Image(shap_plot_path, width=5.5*inch, height=3.0*inch) if os.path.exists(shap_plot_path) else Paragraph("SHAP Explanation Plot Missing", body_style)
        story.append(shap_img)
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Figure 2:</b> Local SHAP Feature Contribution. Displays the positive/negative pressure vectors of RFE features mapping Jitter, Shimmer, and MFCC spectral elements onto the screening verdict.", ParagraphStyle('F2', parent=body_style, align='CENTER', fontSize=8, textColor=colors.HexColor('#64748b'))))
        story.append(Spacer(1, 15))
    
    # 5. Doctor sign off notes
    story.append(Paragraph("CLINICAL SIGN-OFF & RECOMMENDATIONS", section_heading))
    notes_data = [
        [
            Paragraph("<b>Diagnostic Comments / Observations:</b><br/><br/><br/><br/><br/><br/>", ParagraphStyle('NC', parent=body_style, fontSize=9)),
            Paragraph("<b>Clinician Sign-off:</b><br/><br/>___________________________<br/>M.D. Neurologist Signature<br/><br/>Date: ____________________", ParagraphStyle('CS', parent=body_style, fontSize=9))
        ]
    ]
    notes_table = Table(notes_data, colWidths=[360, 180])
    notes_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(notes_table)
    
    # Compile
    doc.build(story)
    print(f"Report successfully compiled at: {output_pdf_path}")

def datetime_now_str():
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
