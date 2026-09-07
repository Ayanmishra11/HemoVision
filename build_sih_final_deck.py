"""
Build the Master SIH 2026 Winner Pitch Deck: HEMOVISION_SIH_2026_FINAL.pptx
Strictly following the official SIH template structure from HHTH1.pptx.
Incorporates all 15 required content headers across the 6 official slides,
embedding verified model evolution and biomarker audit charts from the project.
"""

import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Color Palette (Clinical & SIH Official Aligned) ──
COLOR_PRIMARY_NAVY = RGBColor(0x1B, 0x36, 0x5D) # Deep Navy for headers & titles
COLOR_ACCENT_TEAL  = RGBColor(0x00, 0x80, 0x80) # Clinical Teal for verified badges & lines
COLOR_DARK_TEXT    = RGBColor(0x1E, 0x29, 0x3B) # Slate 800 for high-contrast body text
COLOR_MUTED_TEXT   = RGBColor(0x64, 0x74, 0x8B) # Slate 500 for secondary details
COLOR_CARD_BG      = RGBColor(0xF8, 0xFA, 0xFC) # Slate 50 for clean card backgrounds
COLOR_CARD_BORDER  = RGBColor(0xCB, 0xD5, 0xE1) # Slate 300 for subtle borders
COLOR_CARD_GREEN   = RGBColor(0xEC, 0xFD, 0xF5) # Emerald 50 for validated / positive cards
COLOR_BORDER_GREEN = RGBColor(0xA7, 0xF3, 0xD0) # Emerald 200
COLOR_TEXT_GREEN   = RGBColor(0x06, 0x5F, 0x46) # Emerald 800
COLOR_CARD_RED     = RGBColor(0xFE, 0xF2, 0xF2) # Rose 50 for warnings / honest limits
COLOR_BORDER_RED   = RGBColor(0xFE, 0xCD, 0xCD) # Rose 200
COLOR_TEXT_RED     = RGBColor(0x99, 0x1B, 0x1B) # Rose 800
COLOR_WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_HIGHLIGHT    = RGBColor(0x02, 0x84, 0xC7) # Sky Blue for key metrics

FONT_FAMILY = "Segoe UI"

def create_deck():
    src_path = r"C:\Users\VICTUS\HemoVision\HHTH1.pptx"
    dst_path = r"C:\Users\VICTUS\HemoVision\HEMOVISION_SIH_2026_FINAL.pptx"
    
    shutil.copyfile(src_path, dst_path)
    prs = Presentation(dst_path)
    print(f"Presentation loaded. Total slides: {len(prs.slides)}")
    
    # Clean old content shapes on all slides
    clean_slide_1(prs.slides[0])
    clean_slide_2(prs.slides[1])
    clean_slide_3(prs.slides[2])
    clean_slide_4(prs.slides[3])
    clean_slide_5(prs.slides[4])
    clean_slide_6(prs.slides[5])
    
    # Populate all slides with calibrated layout and embedded evidence
    build_slide_1(prs.slides[0])
    build_slide_2(prs.slides[1])
    build_slide_3(prs.slides[2])
    build_slide_4(prs.slides[3])
    build_slide_5(prs.slides[4])
    build_slide_6(prs.slides[5])
    
    prs.save(dst_path)
    print(f"Successfully generated: {dst_path}")

def delete_shape_by_name(slide, shape_names):
    """Remove shapes matching given names."""
    to_remove = [s for s in slide.shapes if s.name in shape_names]
    for s in to_remove:
        sp = s._element
        sp.getparent().remove(sp)

def clean_slide_1(slide):
    delete_shape_by_name(slide, ["Title 7", "Subtitle 3", "TextBox 9"])

def clean_slide_2(slide):
    delete_shape_by_name(slide, ["Title 1", "TextBox 8", "TextBox 13", "TextBox 4", "TextBox 7", 
                                 "Straight Arrow Connector 11", "Straight Arrow Connector 14"])

def clean_slide_3(slide):
    delete_shape_by_name(slide, ["Title 1", "TextBox 8", "TextBox 4", "Picture 7", "Straight Arrow Connector 8"])

def clean_slide_4(slide):
    delete_shape_by_name(slide, ["Title 1", "Straight Arrow Connector 2", "Straight Arrow Connector 3", 
                                 "TextBox 4", "TextBox 7", "TextBox 8", "TextBox 12"])

def clean_slide_5(slide):
    delete_shape_by_name(slide, ["Title 1", "TextBox 8", "TextBox 2", "TextBox 3", 
                                 "Straight Arrow Connector 4", "Straight Arrow Connector 7"])

def clean_slide_6(slide):
    delete_shape_by_name(slide, ["Title 1", "TextBox 8"])

# ── Helper Drawing Functions ──

def add_card(slide, left_in, top_in, width_in, height_in, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
    """Add a clean rounded card container."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, 
        Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def set_shape_text(shape, paragraphs_info):
    """Set formatted paragraphs in shape."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.12)
    tf.margin_bottom = Inches(0.12)
    
    for i, p_info in enumerate(paragraphs_info):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = p_info.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(p_info.get("space_after", 3))
        p.space_before = Pt(p_info.get("space_before", 0))
        
        runs = p_info.get("runs", [])
        if runs:
            for r_info in runs:
                run = p.add_run()
                run.text = r_info.get("text", "")
                run.font.name = FONT_FAMILY
                run.font.size = Pt(r_info.get("size", 12))
                run.font.bold = r_info.get("bold", False)
                run.font.color.rgb = r_info.get("color", COLOR_DARK_TEXT)
        else:
            run = p.add_run()
            run.text = p_info.get("text", "")
            run.font.name = FONT_FAMILY
            run.font.size = Pt(p_info.get("size", 12))
            run.font.bold = p_info.get("bold", False)
            run.font.color.rgb = p_info.get("color", COLOR_DARK_TEXT)

def add_header(slide, title_text, sub_text=""):
    """
    Centered header between Team Name oval (left: 0.3-1.8) and SIH Logo (right: 10.7-13.1).
    Width = 8.7 inches, left = 1.9 inches.
    """
    tb = slide.shapes.add_textbox(Inches(1.9), Inches(0.18), Inches(8.7), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = title_text
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(17)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY_NAVY
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(2)
    
    if sub_text:
        p2 = tf.add_paragraph()
        p2.text = sub_text
        p2.font.name = FONT_FAMILY
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_MUTED_TEXT
        p2.alignment = PP_ALIGN.CENTER
        p2.space_after = Pt(0)

def add_footer_citation(slide, citation_text):
    """Add small footer citation cleanly above the bottom blue bar."""
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(6.55), Inches(11.8), Inches(0.35))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = citation_text
    p.font.name = FONT_FAMILY
    p.font.size = Pt(8.5)
    p.font.italic = True
    p.font.color.rgb = COLOR_MUTED_TEXT

# ── SLIDE 1: Title, Metadata & Verified Callouts ─────────────
# Headers Covered: 1 (Title & Team), 4 (Solution & Value), 10 (Prototype Snippet), 11 (Differentiation), 12 (Disclaimer)

def build_slide_1(slide):
    # SIH Top Header
    sih_top = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(8.5), Inches(0.45))
    p_sih = sih_top.text_frame.paragraphs[0]
    p_sih.text = "SMART INDIA HACKATHON 2026"
    p_sih.font.name = FONT_FAMILY
    p_sih.font.size = Pt(22)
    p_sih.font.bold = True
    p_sih.font.color.rgb = COLOR_PRIMARY_NAVY

    # Subtitle / Category
    sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.82), Inches(8.5), Inches(0.35))
    p_sub = sub_box.text_frame.paragraphs[0]
    p_sub.text = "STUDENT INNOVATION  |  SOFTWARE  |  MEDTECH & HEALTHTECH"
    p_sub.font.name = FONT_FAMILY
    p_sub.font.size = Pt(11)
    p_sub.font.bold = True
    p_sub.font.color.rgb = COLOR_ACCENT_TEAL

    # Main Project Title
    main_title = slide.shapes.add_textbox(Inches(0.8), Inches(1.18), Inches(8.5), Inches(1.05))
    tf_m = main_title.text_frame
    tf_m.word_wrap = True
    p1 = tf_m.paragraphs[0]
    p1.text = "HEMOVISION"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(34)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY_NAVY
    
    p2 = tf_m.add_paragraph()
    p2.text = "Contactless Heart-Rate Screening from Facial Video"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_DARK_TEXT

    # One-line Thesis Banner
    thesis_card = add_card(slide, 0.8, 2.30, 7.8, 0.65, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(thesis_card, [{
        "runs": [
            {"text": "Thesis: ", "bold": True, "size": 11.5, "color": COLOR_PRIMARY_NAVY},
            {"text": "A quality-gated heart-rate estimate from a standard camera before a teleconsult, while honestly identifying which health signals RGB video cannot yet measure.", "size": 11, "color": COLOR_DARK_TEXT}
        ],
        "space_after": 0
    }])

    # Left: Official SIH Metadata Box
    meta_card = add_card(slide, 0.8, 3.10, 4.4, 3.25, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    meta_info = [
        {"runs": [{"text": "SIH SUBMISSION METADATA", "bold": True, "size": 11, "color": COLOR_PRIMARY_NAVY}], "space_after": 7},
        {"runs": [{"text": "Problem Statement ID: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "[Registered Portal ID]", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "Problem Statement: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "Contactless Heart-Rate Estimation from Video", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "Theme: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "MedTech / Bio-Informatics", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "Category: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "Software (Student Innovation)", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "Team Name: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "[Your Registered Team Name]", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "Mentor: ", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}, {"text": "[Academic / Clinical Guide]", "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 0}
    ]
    set_shape_text(meta_card, meta_info)

    # Center-Right: 3 Verified Evidence Callouts
    c1 = add_card(slide, 5.4, 3.10, 3.6, 0.95, bg_color=COLOR_CARD_GREEN, border_color=COLOR_BORDER_GREEN)
    set_shape_text(c1, [
        {"runs": [{"text": "✓ Working Real-Time Prototype", "bold": True, "size": 11.5, "color": COLOR_TEXT_GREEN}], "space_after": 2},
        {"runs": [{"text": "Live webcam & video pipeline at 30 FPS with 20s sliding window inference.", "size": 10, "color": COLOR_DARK_TEXT}]}
    ])

    c2 = add_card(slide, 5.4, 4.20, 3.6, 0.95, bg_color=COLOR_CARD_GREEN, border_color=COLOR_BORDER_GREEN)
    set_shape_text(c2, [
        {"runs": [{"text": "✓ 8 Anatomical Skin Regions", "bold": True, "size": 11.5, "color": COLOR_TEXT_GREEN}], "space_after": 2},
        {"runs": [{"text": "MediaPipe 468 mesh extracts 24-channel temporal RGB signal across forehead & cheeks.", "size": 10, "color": COLOR_DARK_TEXT}]}
    ])

    c3 = add_card(slide, 5.4, 5.30, 3.6, 1.05, bg_color=COLOR_CARD_GREEN, border_color=COLOR_BORDER_GREEN)
    set_shape_text(c3, [
        {"runs": [{"text": "✓ 9.19 BPM Held-Out MAE", "bold": True, "size": 11.5, "color": COLOR_TEXT_GREEN}], "space_after": 2},
        {"runs": [{"text": "Subject-disjoint test on 91 individuals (523 clips); 52% error cut vs. POS (19.17 BPM).", "size": 10, "color": COLOR_DARK_TEXT}]}
    ])

    # Discreet Disclaimer at bottom
    disc = slide.shapes.add_textbox(Inches(0.8), Inches(6.52), Inches(11.5), Inches(0.35))
    p_disc = disc.text_frame.paragraphs[0]
    p_disc.text = "Notice: Research prototype developed for academic innovation. Not approved for clinical diagnosis or emergency monitoring."
    p_disc.font.name = FONT_FAMILY
    p_disc.font.size = Pt(9)
    p_disc.font.italic = True
    p_disc.font.color.rgb = COLOR_MUTED_TEXT

# ── SLIDE 2: Problem Statement, User Journey & Solution ─────
# Headers Covered: 2 (Problem & India Need), 3 (Target Users & Use Case), 4 (Solution & Value), 13 (Telehealth Opportunity)

def build_slide_2(slide):
    add_header(slide, "A LOW-FRICTION FIRST STEP BEFORE REMOTE CARE", 
               "Target Use Case: Patient or Community Health Worker Prior to a Telemedicine Consultation")
    
    # Left Card: Current Clinical Friction & India Need
    card_left = add_card(slide, 0.6, 1.35, 4.0, 4.45, bg_color=COLOR_CARD_RED, border_color=COLOR_BORDER_RED)
    left_info = [
        {"runs": [{"text": "INDIA-RELEVANT PROBLEM & NEED", "bold": True, "size": 12, "color": COLOR_TEXT_RED}], "space_after": 8},
        {"runs": [{"text": "• Physical Peripheral Bottleneck:", "bold": True, "size": 11, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Initial vital collection requires finger clips, BP cuffs, or ECG patches (₹5,000–₹50,000) that rural families and remote clinics rarely own.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 7},
        {"runs": [{"text": "• Pre-Consultation Triage Blindspot:", "bold": True, "size": 11, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Remote telemedicine consultations frequently begin without a single objective vital sign, forcing doctors to rely on subjective verbal reports.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 7},
        {"runs": [{"text": "• Hygiene & Point-of-Care Friction:", "bold": True, "size": 11, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Contact sensors require sanitization between patients and cause discomfort in neonatal, burn, or infectious isolation wards.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ]
    set_shape_text(card_left, left_info)

    # Right Card: The 5-Step Quality-Gated Proposed Workflow
    card_right = add_card(slide, 4.8, 1.35, 7.9, 4.45, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    right_info = [
        {"runs": [{"text": "PROPOSED QUALITY-GATED PRE-CONSULT WORKFLOW", "bold": True, "size": 12, "color": COLOR_PRIMARY_NAVY}], "space_after": 8}
    ]
    set_shape_text(card_right, right_info)

    steps = [
        ("1. Video Capture", "Patient records a 15–20s facial video via smartphone or laptop webcam under standard ambient room lighting."),
        ("2. Quality Gate Check", "Automated real-time validation: checks face visibility, lighting adequacy (>40 lux), and rigid head stability."),
        ("3. Quality-Gated Estimation", "Heart rate estimated ONLY if quality gates pass; noisy, corrupted, or shadowed frames are strictly rejected."),
        ("4. Contextual Display", "Displays estimated BPM alongside reconstructed BVP waveform and signal confidence score for clinician context."),
        ("5. Clinical Escalation", "Low-confidence or abnormal readings immediately prompt standard in-person contact reference measurement.")
    ]
    
    step_y = 1.82
    for idx, (stitle, sdesc) in enumerate(steps):
        s_box = add_card(slide, 5.0, step_y, 7.5, 0.68, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
        set_shape_text(s_box, [
            {"runs": [
                {"text": f"{stitle}: ", "bold": True, "size": 10.5, "color": COLOR_PRIMARY_NAVY},
                {"text": sdesc, "size": 10, "color": COLOR_DARK_TEXT}
            ], "space_after": 0}
        ])
        step_y += 0.74

    # Bottom Future Interoperability Strip
    future_strip = add_card(slide, 0.6, 5.92, 12.1, 0.55, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(future_strip, [{
        "runs": [
            {"text": "Future Telehealth Integration Opportunity: ", "bold": True, "size": 10.5, "color": COLOR_PRIMARY_NAVY},
            {"text": "Designed for prospective interoperability with national telemedicine platforms (e.g., eSanjeevani) and ABDM health records following formal clinical trials.", "size": 10, "color": COLOR_MUTED_TEXT}
        ],
        "space_after": 0
    }])

    add_footer_citation(slide, "Sources: MoHFW eSanjeevani Operational Framework (2024); PubMed 37313385 (Contactless PPG Systematic Review). Note: Government integration is a future opportunity, not an existing deployment.")

# ── SLIDE 3: Architecture, Methodology & Live Prototype ─────
# Headers Covered: 5 (End-to-End Architecture), 6 (Methodology & Experiments), 10 (Live Prototype), 11 (Innovation)

def build_slide_3(slide):
    add_header(slide, "FROM FACIAL VIDEO TO A QUALITY-GATED HEART-RATE ESTIMATE", 
               "End-to-End 7-Stage Signal Processing and Temporal Deep Learning Pipeline")

    # Pipeline Steps (Left Column)
    pipeline_card = add_card(slide, 0.6, 1.35, 6.8, 4.55, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    p_header = [
        {"runs": [{"text": "7-STAGE RIGOROUS OPTICAL PIPELINE", "bold": True, "size": 12, "color": COLOR_PRIMARY_NAVY}], "space_after": 6}
    ]
    set_shape_text(pipeline_card, p_header)

    stages = [
        ("Stage 1 — Video Capture", "Standard RGB sensor (FullHDwebcam / USBVideo / IriunWebcam) at 30 FPS."),
        ("Stage 2 — Face Mesh Tracking", "MediaPipe detects 468 3D facial landmarks in real time with zero physical markers."),
        ("Stage 3 — 8 Anatomical Skin ROIs", "Forehead (L/R), Upper Cheeks (L/R), Lower Cheeks (L/R), Nose, Chin."),
        ("Stage 4 — 24-Channel Temporal Signal", "Spatial RGB pooling (8 ROIs × 3 channels) into AC normalized & DC raw representations."),
        ("Stage 5 — HemoVisionBoundedTCN", "4 depthwise-separable 1D conv blocks (d=1,2,4,8; 148K params; tanh bounded)."),
        ("Stage 6 — Butterworth Bandpass", "4th-order zero-phase filter (0.65–3.25 Hz / 39–195 BPM) isolates cardiac frequencies."),
        ("Stage 7 — Welch PSD Peak Picking", "FFT power spectral density identifies dominant cardiac peak & SNR confidence score.")
    ]

    sy = 1.75
    for sname, sdesc in stages:
        s_box = add_card(slide, 0.75, sy, 6.5, 0.52, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
        set_shape_text(s_box, [{
            "runs": [
                {"text": f"{sname}: ", "bold": True, "size": 9.5, "color": COLOR_PRIMARY_NAVY},
                {"text": sdesc, "size": 9, "color": COLOR_DARK_TEXT}
            ], "space_after": 0
        }])
        sy += 0.58

    # Right Top: Core Original Contributions Card
    c_orig = add_card(slide, 7.6, 1.35, 5.1, 2.35, bg_color=COLOR_CARD_GREEN, border_color=COLOR_BORDER_GREEN)
    set_shape_text(c_orig, [
        {"runs": [{"text": "CORE ORIGINAL CONTRIBUTIONS", "bold": True, "size": 12, "color": COLOR_TEXT_GREEN}], "space_after": 6},
        {"runs": [{"text": "1. Multi-ROI Temporal Fusion: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT},
                  {"text": "Replaces single-box face crops with 8 anatomically targeted capillary regions.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 4},
        {"runs": [{"text": "2. Bounded Temporal ConvNet: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT},
                  {"text": "Tanh-bounded receptive field (>20s) prevents gradient explosions on noisy webcams.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 4},
        {"runs": [{"text": "3. Biomarker Feasibility Audit: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT},
                  {"text": "Rigorously demonstrated that RGB video cannot infer blood chemistry without NIR.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ])

    # Right Bottom: Architecture Specs & Live Demo Engine
    c_spec = add_card(slide, 7.6, 3.82, 5.1, 2.08, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(c_spec, [
        {"runs": [{"text": "CONFIRMED IMPLEMENTATION SPECIFICATIONS", "bold": True, "size": 11.5, "color": COLOR_PRIMARY_NAVY}], "space_after": 6},
        {"runs": [{"text": "• Model Parameters: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}, {"text": "148,650 (78% fewer than standard 1D CNN)", "size": 10, "color": COLOR_MUTED_TEXT}], "space_after": 3},
        {"runs": [{"text": "• Cardiac Passband: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}, {"text": "0.65 Hz – 3.25 Hz (39 to 195 BPM)", "size": 10, "color": COLOR_MUTED_TEXT}], "space_after": 3},
        {"runs": [{"text": "• Working Prototype Engine: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}, {"text": "Interactive Streamlit & OpenCV dashboard (`app.py`, `hemovision_demo.py`) at 30 FPS.", "size": 10, "color": COLOR_MUTED_TEXT}], "space_after": 3},
        {"runs": [{"text": "• Classical Baselines: ", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}, {"text": "POS (Wang 2017) & CHROM (de Haan 2013) implemented for direct benchmark comparison.", "size": 10, "color": COLOR_MUTED_TEXT}], "space_after": 0}
    ])

    add_footer_citation(slide, "Source Code: v3_pipeline/cell_10_dataset.py, cell_11_dataset.py, hemovision_demo.py, app.py; Wang et al. (IEEE TBME 2017); de Haan & Jeanne (IEEE TBME 2013).")

# ── SLIDE 4: Models, Hit-and-Trial Progression & Benchmark ──
# Headers Covered: 7 (Models & Hit-and-Trial), 8 (Dataset & Ground Truth), 9 (Validation Results & Benchmarks)

def build_slide_4(slide):
    add_header(slide, "SUBJECT-DISJOINT EVALUATION SHOWS A WORKING HEART-RATE SIGNAL", 
               "Rigorous Benchmark on MCD-rPPG Dataset (598 Unique Subjects, 3,427 Video Clips)")

    # Top Metric Scorecard (3 KPI Cards)
    kpi1 = add_card(slide, 0.6, 1.35, 3.8, 1.05, bg_color=COLOR_CARD_GREEN, border_color=COLOR_BORDER_GREEN)
    set_shape_text(kpi1, [
        {"runs": [{"text": "9.19 BPM", "bold": True, "size": 22, "color": COLOR_TEXT_GREEN}], "align": PP_ALIGN.CENTER, "space_after": 0},
        {"runs": [{"text": "Held-Out Test MAE (r = 0.285)", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "align": PP_ALIGN.CENTER}
    ])

    kpi2 = add_card(slide, 4.75, 1.35, 3.8, 1.05, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(kpi2, [
        {"runs": [{"text": "52.0% Reduction", "bold": True, "size": 22, "color": COLOR_PRIMARY_NAVY}], "align": PP_ALIGN.CENTER, "space_after": 0},
        {"runs": [{"text": "Error Cut vs. POS Baseline (19.17 BPM)", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "align": PP_ALIGN.CENTER}
    ])

    kpi3 = add_card(slide, 8.9, 1.35, 3.8, 1.05, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(kpi3, [
        {"runs": [{"text": "1.33 BPM", "bold": True, "size": 22, "color": COLOR_HIGHLIGHT}], "align": PP_ALIGN.CENTER, "space_after": 0},
        {"runs": [{"text": "High-Quality Demo Clips (r = 0.884)", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "align": PP_ALIGN.CENTER}
    ])

    # Left: Protocol & Benchmark Comparison Table
    table_card = add_card(slide, 0.6, 2.52, 6.8, 2.75, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(table_card, [
        {"runs": [{"text": "HELD-OUT TEST SET PERFORMANCE (91 SUBJECTS, 523 CLIPS)", "bold": True, "size": 11, "color": COLOR_PRIMARY_NAVY}], "space_after": 5},
        {"runs": [{"text": "Model / Algorithm                    Heart Rate MAE    RMSE       Pearson r", "bold": True, "size": 9, "color": COLOR_DARK_TEXT}], "space_after": 3},
        {"runs": [{"text": "CHROM (de Haan et al. 2013)   20.28 BPM         24.61 BPM  0.047", "size": 9, "color": COLOR_MUTED_TEXT}], "space_after": 2},
        {"runs": [{"text": "POS (Wang et al. 2017)             19.17 BPM         23.72 BPM  0.119", "size": 9, "color": COLOR_MUTED_TEXT}], "space_after": 2},
        {"runs": [{"text": "HemoVisionBoundedTCN (Ours) 9.19 BPM         18.35 BPM  0.285  [✓ 52% cut]", "bold": True, "size": 9, "color": COLOR_TEXT_GREEN}], "space_after": 5},
        {"runs": [{"text": "• Strict Subject-Disjoint Protocol: ", "bold": True, "size": 9, "color": COLOR_DARK_TEXT},
                  {"text": "Zero subject overlap between train (418), val (89), and test (91). No identity leakage.", "size": 9, "color": COLOR_MUTED_TEXT}], "space_after": 2},
        {"runs": [{"text": "• Hit-and-Trial Bug Fix: ", "bold": True, "size": 9, "color": COLOR_DARK_TEXT},
                  {"text": "Fixed reference ECG T-wave doubling anomaly via 250ms refractory blanking, eliminating 2x ground-truth artifacts.", "size": 9, "color": COLOR_MUTED_TEXT}], "space_after": 0}
    ])

    # Right: Embedded Actual Model Evolution Chart from Workspace
    img_path = r"C:\Users\VICTUS\HemoVision\ppt_assets\model_evolution.png"
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(7.6), Inches(2.52), Inches(5.1), Inches(2.75))

    # Bottom Callout: What This Does NOT Prove
    not_proven = add_card(slide, 0.6, 5.38, 12.1, 0.95, bg_color=COLOR_CARD_RED, border_color=COLOR_BORDER_RED)
    set_shape_text(not_proven, [
        {"runs": [{"text": "CRITICAL SCIENTIFIC DISTINCTION: WHAT THIS DOES NOT PROVE", "bold": True, "size": 10.5, "color": COLOR_TEXT_RED}], "space_after": 2},
        {"runs": [{"text": "This result proves a functioning optical pulse recovery signal under retrospective benchmark conditions on recorded data. ", "bold": True, "size": 9.5, "color": COLOR_DARK_TEXT},
                  {"text": "It is NOT yet a prospective clinical trial. Real-world performance across motion artifacts, extreme skin tones (Fitzpatrick V-VI), low-light environments, and diverse mobile camera sensors requires formal prospective clinical validation.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ])

    add_footer_citation(slide, "Evaluation Source: HemoVision Vocational Training Report (Table 6); cell_14_eval.py; MCD-rPPG Clinical Dataset (598 Subjects). Ground truth: synchronized ECG lead contact sensor.")

# ── SLIDE 5: Limitations, Safety Boundaries & Biomarker Audit 
# Headers Covered: 11 (Differentiation), 12 (Limitations & Honest Findings), 14 (Prospective Validation Path)

def build_slide_5(slide):
    add_header(slide, "QUALITY GATES AND HONEST LIMITS MAKE THE PROTOTYPE SAFER", 
               "Refusing Unsupported Healthcare Claims & Defining the Roadmap to Safe Clinical Translation")

    # Left Card: The 5 Automated Quality Gates
    c_gates = add_card(slide, 0.6, 1.35, 3.8, 5.0, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(c_gates, [
        {"runs": [{"text": "AUTOMATED QUALITY GATES", "bold": True, "size": 11.5, "color": COLOR_PRIMARY_NAVY}], "space_after": 6},
        {"runs": [{"text": "1. Lighting Adequacy:", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Rejects underexposed (<40 lux) or severely back-lit frames.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "2. Face Mesh Stability:", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Requires MediaPipe landmark detection confidence > 0.75.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "3. Motion Jitter Guard:", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Bounds inter-frame displacement; pauses during sudden yaw/pitch.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "4. Temporal Window Buffer:", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Requires minimum 20s (600 frames at 30 FPS) for cardiac stability.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 5},
        {"runs": [{"text": "5. Spectral SNR Confidence:", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Welch PSD peak-to-noise ratio must exceed threshold before output.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 0}
    ])

    # Center Card: Validated vs. Not Claimed & The Negative Result
    c_limits = add_card(slide, 4.6, 1.35, 4.1, 5.0, bg_color=COLOR_CARD_RED, border_color=COLOR_BORDER_RED)
    set_shape_text(c_limits, [
        {"runs": [{"text": "VALIDATED VS. NOT CLAIMED", "bold": True, "size": 11.5, "color": COLOR_TEXT_RED}], "space_after": 5},
        {"runs": [{"text": "✓ VALIDATED NOW (RGB Video):", "bold": True, "size": 10, "color": COLOR_TEXT_GREEN}], "space_after": 1},
        {"runs": [{"text": "Heart-rate estimation from facial video via optical capillary blood volume pulse (BVP).", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 7},
        {"runs": [{"text": "✕ STRICTLY NOT CLAIMED:", "bold": True, "size": 10, "color": COLOR_TEXT_RED}], "space_after": 1},
        {"runs": [{"text": "SpO2, Blood Pressure, Hemoglobin, Respiratory Rate, Stress, HbA1c, or Disease Diagnosis.", "bold": True, "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 7},
        {"runs": [{"text": "The Biomarker Audit Differentiator:", "bold": True, "size": 10, "color": COLOR_PRIMARY_NAVY}], "space_after": 2},
        {"runs": [{"text": "Tested 10 biomarkers on 598 subjects. Deep models failed to beat demographic baselines (e.g. Hemoglobin video r=0.44 beaten by age/sex/BMI OLS r=0.70). The network learned subject ID shortcuts, not blood chemistry.", "size": 9, "color": COLOR_DARK_TEXT}], "space_after": 5},
        {"runs": [{"text": "Optical Physics Reality: ", "bold": True, "size": 9.5, "color": COLOR_DARK_TEXT},
                  {"text": "Blood chemistry requires narrow-band dual-wavelength NIR (660/940 nm). RGB Bayer filters conflate melanin with absorption.", "size": 9, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ])

    # Right Card: The 4-Step Prospective Validation Path
    c_future = add_card(slide, 8.9, 1.35, 3.8, 5.0, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(c_future, [
        {"runs": [{"text": "FUTURE VALIDATION PATHWAY", "bold": True, "size": 11.5, "color": COLOR_PRIMARY_NAVY}], "space_after": 6},
        {"runs": [{"text": "Step 1: Diverse Cohort Trial", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Prospective multi-center study across skin tones (Fitzpatrick I–VI), age brackets, and lighting environments.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 6},
        {"runs": [{"text": "Step 2: Concurrent Clinical Reference", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Simultaneous validation against hospital-grade ECG and calibrated contact pulse oximeters.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 6},
        {"runs": [{"text": "Step 3: Rigorous Statistical Analysis", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Bland-Altman 95% limits of agreement, mean bias profiling, and comprehensive failure taxonomy.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 6},
        {"runs": [{"text": "Step 4: CDSCO SaMD Assessment", "bold": True, "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 1},
        {"runs": [{"text": "Formal Software as a Medical Device (SaMD) regulatory filing and patient data privacy certification.", "size": 9.5, "color": COLOR_MUTED_TEXT}], "space_after": 0}
    ])

    add_footer_citation(slide, "References: CDSCO Draft Guidance on Medical Device Software (2025); PubMed 37313385; PubMed 42106569; cell16_hemoglobin_feasibility_audit.py (Table 7).")

# ── SLIDE 6: Impact, Scalability, Roadmap & Team Conclusion ──
# Headers Covered: 13 (Impact, Scalability & Telehealth), 14 (Roadmap & Deployment), 15 (References & Conclusion)

def build_slide_6(slide):
    add_header(slide, "A CREDIBLE PATH FROM PROTOTYPE TO TELE-TRIAGE SUPPORT", 
               "Student Innovation Closing: Evidence-Backed Heart-Rate Prototype with Transparent Clinical Boundaries")

    # Top Core Distinction Banner
    dist_card = add_card(slide, 0.6, 1.35, 12.1, 0.75, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(dist_card, [{
        "runs": [
            {"text": "HemoVision's Winning Differentiator: ", "bold": True, "size": 11.5, "color": COLOR_PRIMARY_NAVY},
            {"text": "A working, evidence-backed heart-rate prototype (9.19 BPM held-out MAE) combined with an intellectually honest boundary around what RGB video cannot yet measure.", "size": 11, "color": COLOR_DARK_TEXT}
        ],
        "space_after": 0
    }])

    # Left Box: Impact Hypothesis (Tele-Triage Support)
    card_hypo = add_card(slide, 0.6, 2.22, 5.8, 3.45, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(card_hypo, [
        {"runs": [{"text": "HEALTHCARE IMPACT HYPOTHESIS", "bold": True, "size": 12, "color": COLOR_PRIMARY_NAVY}], "space_after": 8},
        {"runs": [{"text": "• Accessible Pre-Consultation Screening:", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Enables patients to provide an objective, quality-gated heart-rate reading prior to a teleconsult without purchasing dedicated hardware.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 8},
        {"runs": [{"text": "• Rural Primary Health Centre (PHC) Potential:", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Can be explored by community health workers (ASHAs/ANMs) as a low-friction pre-screening tool in remote clinics following prospective validation.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 8},
        {"runs": [{"text": "• Zero Peripheral Hardware Cost:", "bold": True, "size": 10.5, "color": COLOR_DARK_TEXT}], "space_after": 2},
        {"runs": [{"text": "Utilizes ubiquitous commodity smartphone and laptop cameras already available across India.", "size": 10, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ])

    # Right Box: 3-Stage Development Roadmap
    card_road = add_card(slide, 6.7, 2.22, 6.0, 3.45, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    set_shape_text(card_road, [
        {"runs": [{"text": "THREE-STAGE DEVELOPMENT ROADMAP", "bold": True, "size": 12, "color": COLOR_PRIMARY_NAVY}], "space_after": 8},
        {"runs": [{"text": "[COMPLETED] Phase 1: Prototype & Audit", "bold": True, "size": 10.5, "color": COLOR_TEXT_GREEN}], "space_after": 2},
        {"runs": [{"text": "• Real-time webcam pipeline (30 FPS, 8 ROIs, 24 channels).\n• HemoVisionBoundedTCN evaluated on MCD-rPPG (9.19 BPM MAE).\n• Systematic 10-biomarker audit exposing demographic confounds.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 6},
        {"runs": [{"text": "[NEXT MILESTONE] Phase 2: Prospective Validation", "bold": True, "size": 10.5, "color": COLOR_PRIMARY_NAVY}], "space_after": 2},
        {"runs": [{"text": "• Multi-center prospective clinical study across Fitzpatrick skin tones.\n• Hardened automated quality gates & ambient light calibration.\n• Containerized web application for research evaluation.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 6},
        {"runs": [{"text": "[FUTURE RESEARCH] Phase 3: Multi-Wavelength Optics", "bold": True, "size": 10.5, "color": COLOR_MUTED_TEXT}], "space_after": 2},
        {"runs": [{"text": "• Dual-wavelength NIR active illumination hardware research for SpO2.\n• CDSCO Software as a Medical Device (SaMD) regulatory evaluation.", "size": 9.5, "color": COLOR_DARK_TEXT}], "space_after": 0}
    ])

    # Bottom Callout: Compliance & Ethics
    card_eth = add_card(slide, 0.6, 5.82, 12.1, 0.62, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    set_shape_text(card_eth, [{
        "runs": [
            {"text": "Research Disclaimer & Compliance Notice: ", "bold": True, "size": 10, "color": COLOR_PRIMARY_NAVY},
            {"text": "HemoVision is an academic research prototype. It does not provide medical diagnosis or emergency vital tracking. Formal clinical trial verification and CDSCO regulatory clearance are strictly required prior to real-world healthcare deployment.", "size": 9.5, "color": COLOR_MUTED_TEXT}
        ],
        "space_after": 0
    }])

    add_footer_citation(slide, "References: MoHFW eSanjeevani Overview; NHA ABDM Framework; CDSCO SaMD Guidance (2025); HemoVision Project Repository & Technical Report.")

if __name__ == "__main__":
    create_deck()
