import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register system TrueType fonts for native UTF-8 and unicode symbol support
font_dir = Path("C:/Windows/Fonts")
if (font_dir / "times.ttf").exists():
    pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(font_dir / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", str(font_dir / "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", str(font_dir / "timesbi.ttf")))
    FONT_NORMAL = "TimesNewRoman"
    FONT_BOLD = "TimesNewRoman-Bold"
    FONT_ITALIC = "TimesNewRoman-Italic"
    FONT_BOLDITALIC = "TimesNewRoman-BoldItalic"
else:
    FONT_NORMAL = "Times-Roman"
    FONT_BOLD = "Times-Bold"
    FONT_ITALIC = "Times-Italic"
    FONT_BOLDITALIC = "Times-BoldItalic"

class NumberedCanvas(canvas.Canvas):
    """
    Custom canvas that assigns:
    - Cover page (page 1): no page number
    - Preliminary pages (pages 2 to 9): Roman numerals ii, iii, iv, v, vi, vii, viii, ix
    - Main body pages (page 10 onward): Arabic numerals 1, 2, 3, ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        page_num = self._pageNumber
        self.saveState()
        self.setFont(FONT_NORMAL, 10)
        self.setFillColor(colors.HexColor("#1e293b"))
        
        roman_map = {2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi", 7: "vii", 8: "viii", 9: "ix"}
        
        if page_num == 1:
            pass  # Cover page suppresses page number
        elif page_num in roman_map:
            text = roman_map[page_num]
            self.drawCentredString(A4[0] / 2.0, 36, text)
        else:
            arabic_num = page_num - 9
            text = str(arabic_num)
            self.drawCentredString(A4[0] / 2.0, 36, text)
            
        self.restoreState()


def build_pdf(filename="HemoVision_Vocational_Training_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=48,
        bottomMargin=52
    )
    
    styles = getSampleStyleSheet()
    
    # Academic Typography Styles matching the provided university standard
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        fontName=FONT_BOLD,
        fontSize=15,
        leading=20,
        alignment=1, # Centered
        textColor=colors.black,
        spaceAfter=10
    )
    
    style_cover_sub = ParagraphStyle(
        'CoverSub',
        fontName=FONT_NORMAL,
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.black
    )
    
    style_cover_bold = ParagraphStyle(
        'CoverBold',
        fontName=FONT_BOLD,
        fontSize=13,
        leading=18,
        alignment=1,
        textColor=colors.black
    )
    
    style_cover_italic = ParagraphStyle(
        'CoverItalic',
        fontName=FONT_ITALIC,
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.black
    )
    
    style_chapter_heading = ParagraphStyle(
        'ChapterHeading',
        fontName=FONT_BOLD,
        fontSize=14,
        leading=18,
        alignment=1, # Center
        textColor=colors.black,
        spaceAfter=14
    )
    
    style_section_heading = ParagraphStyle(
        'SectionHeading',
        fontName=FONT_BOLD,
        fontSize=12,
        leading=16,
        alignment=0, # Left
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=5
    )
    
    style_sub_heading = ParagraphStyle(
        'SubHeading',
        fontName=FONT_BOLD,
        fontSize=11,
        leading=15,
        alignment=0,
        textColor=colors.black,
        spaceBefore=7,
        spaceAfter=3
    )
    
    style_body = ParagraphStyle(
        'AcademicBody',
        fontName=FONT_NORMAL,
        fontSize=10.5,
        leading=14.5,
        alignment=4, # Justified
        textColor=colors.black,
        spaceAfter=7
    )
    
    style_bullet = ParagraphStyle(
        'AcademicBullet',
        fontName=FONT_NORMAL,
        fontSize=10.5,
        leading=14,
        alignment=4,
        leftIndent=16,
        firstLineIndent=-10,
        textColor=colors.black,
        spaceAfter=3.5
    )
    
    style_caption = ParagraphStyle(
        'FigureCaption',
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        alignment=1, # Center
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=8
    )
    
    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName=FONT_NORMAL,
        fontSize=9.5,
        leading=12.5,
        textColor=colors.black
    )
    
    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName=FONT_BOLD,
        fontSize=9.5,
        leading=12.5,
        textColor=colors.black
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE (Page i)
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("A", style_cover_sub))
    story.append(Paragraph("Industrial Training/Internship Report", style_cover_sub))
    story.append(Paragraph("On", style_cover_sub))
    story.append(Spacer(1, 18))
    story.append(Paragraph("VOCATIONAL TRAINING IN MACHINE LEARNING &amp; COMPUTER VISION", style_cover_title))
    story.append(Paragraph("REPORT", style_cover_title))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Submitted to", style_cover_sub))
    story.append(Paragraph("CHHATTISGARH SWAMI VIVEKANAND TECHNICAL UNIVERSITY", style_cover_bold))
    story.append(Paragraph("BHILAI", style_cover_bold))
    story.append(Spacer(1, 14))
    story.append(Paragraph("<i>in partial fulfilment of requirement for the award of the degree</i>", style_cover_italic))
    story.append(Paragraph("<i>of</i>", style_cover_italic))
    story.append(Paragraph("Bachelor of Technology", style_cover_bold))
    story.append(Paragraph("<i>in</i>", style_cover_italic))
    story.append(Paragraph("Computer Science and Engineering", style_cover_bold))
    story.append(Spacer(1, 22))
    story.append(Paragraph("by", style_cover_sub))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Harsh Patel", style_cover_bold))
    story.append(Paragraph("300111323021", style_cover_sub))
    story.append(Spacer(1, 45))
    
    story.append(Paragraph("<b>BHILAI INSTITUTE OF TECHNOLOGY, DURG (CG)</b>", style_cover_bold))
    story.append(Paragraph("<i>(Seth Balkrishan Memorial) Estd. 1986</i>", style_cover_italic))
    story.append(Paragraph("ALL UG &amp; MBA COURSES | NAAC 'A' GRADE | ISO 9001:2015 | ISO 14001:2015 | NIRF", ParagraphStyle('CoverBadges', fontName=FONT_NORMAL, fontSize=8, leading=11, alignment=1)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CERTIFICATE BY COMPANY / INDUSTRY (Page ii)
    # =========================================================================
    story.append(Paragraph("<u>Certificate by Company / Industry</u>", style_chapter_heading))
    story.append(Paragraph("Certificate of Completion issued by BS Digital Technology", style_cover_sub))
    story.append(Spacer(1, 18))
    
    if Path("pdf_assets/fig1_certificate.png").exists():
        story.append(Image("pdf_assets/fig1_certificate.png", width=460, height=325))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Figure: Company/Industry Certificate of Completion", style_caption))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: DECLARATION BY STUDENT (Page iii)
    # =========================================================================
    story.append(Paragraph("<u>Declaration by Student</u>", style_chapter_heading))
    story.append(Spacer(1, 15))
    
    decl_p1 = (
        "I, Harsh Patel, a student of Bachelor of Technology in Computer Science and Engineering "
        "(Artificial Intelligence), 6th Semester, hereby declare that the Industrial Training / Vocational "
        "Training report entitled <b>“HemoVision – Contactless Vital-Sign Estimation and Biomarker Analysis via "
        "Remote Photoplethysmography (rPPG)”</b> is based on the work carried out by me during my project-based "
        "training at <b>BS Digital Technology</b> from 08 June 2026 to 08 July 2026."
    )
    story.append(Paragraph(decl_p1, style_body))
    story.append(Spacer(1, 8))
    
    decl_p2 = (
        "I declare that the work presented in this report reflects my learning, implementation, analysis, and "
        "project activities during the training period. The report has been prepared for academic submission and "
        "has not been submitted elsewhere for the award of any other degree, diploma, or certificate."
    )
    story.append(Paragraph(decl_p2, style_body))
    story.append(Spacer(1, 8))
    
    decl_p3 = (
        "Wherever information, concepts, tools, documentation, or external references have been used, "
        "appropriate acknowledgement and references have been provided. I have made every effort to present "
        "the project work accurately and in accordance with the prescribed university report format."
    )
    story.append(Paragraph(decl_p3, style_body))
    story.append(Spacer(1, 40))
    
    sig_data = [
        ["", "Signature of Student: ___________________________"],
        ["", "<b>Harsh Patel</b>"],
        ["", "Roll No.: 300111323021"],
        ["", "Enrolment No.: CD8812"]
    ]
    t_sig = Table(sig_data, colWidths=[200, 280])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,-1), FONT_NORMAL),
        ('FONTSIZE', (0,0), (-1,-1), 10.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_sig)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: ACKNOWLEDGEMENT (Page iv)
    # =========================================================================
    story.append(Paragraph("<u>Acknowledgement</u>", style_chapter_heading))
    story.append(Spacer(1, 15))
    
    ack_p1 = (
        "I have great pleasure in the submission of this project report entitled <b>“HemoVision – Contactless "
        "Vital-Sign Estimation and Biomarker Analysis via Remote Photoplethysmography (rPPG)”</b> in partial fulfilment "
        "of Vocational Training. While submitting this project report, I take this opportunity to thank those directly "
        "or indirectly related to project work. I would like to thank my supervisor <b>Mr Kailash Sinha</b> and my vocational "
        "training incharge <b>Dr Sumit Sar</b> who has provided the opportunity and organizing project for me. Without his active "
        "co-operation and guidance, it would have become very difficult to complete the task in time. I would like to express "
        "sincere thanks and gratitude to <b>Dr. Anup Mishra</b>, Principal of the Institution, <b>Dr. (Mrs.) Sunita Soni</b>, "
        "Head of the Department Computer Science &amp; Engineering for their encouragement and cordial support."
    )
    story.append(Paragraph(ack_p1, style_body))
    story.append(Spacer(1, 10))
    
    ack_p2 = (
        "Acknowledgement is due to our parents, family members, friends and all those persons who have helped us "
        "directly or indirectly in the successful completion of the project work."
    )
    story.append(Paragraph(ack_p2, style_body))
    story.append(Spacer(1, 50))
    
    ack_sig = [
        ["", "<b>Name of the Student:</b> Harsh Patel"],
        ["", "<b>Roll No:</b> 300111323021"],
        ["", "<b>Enrollment:</b> CD8812"]
    ]
    t_ack = Table(ack_sig, colWidths=[200, 280])
    t_ack.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,-1), FONT_NORMAL),
        ('FONTSIZE', (0,0), (-1,-1), 10.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_ack)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: ABSTRACT (Page v)
    # =========================================================================
    story.append(Paragraph("<u>Abstract</u>", style_chapter_heading))
    story.append(Spacer(1, 12))
    
    abs_p1 = (
        "<b>HemoVision</b> is a project-based computer vision and deep learning solution developed during vocational "
        "training at BS Digital Technology. The project focuses on contactless physiological vital-sign estimation "
        "and biomarker analysis from standard RGB facial video using Remote Photoplethysmography (rPPG). The solution "
        "combines facial landmark tracking, multi-region of interest (ROI) extraction, signal AC/DC disentanglement, "
        "temporal deep neural networks, and interactive clinical visualization into an end-to-end workflow."
    )
    story.append(Paragraph(abs_p1, style_body))
    story.append(Spacer(1, 8))
    
    abs_p2 = (
        "The project uses the clinical MCD-rPPG dataset comprising 3,427 video clips across 598 unique subjects "
        "captured across three consumer camera types (FullHDwebcam, USBVideo, IriunWebcam) and two conditions (resting "
        "and post-exercise). The data preparation process includes tracking 468 3D landmarks using MediaPipe Face Mesh, "
        "segmenting 8 anatomically independent facial skin ROIs, extracting 24-channel RGB time-series, isolating AC "
        "pulsatile signals via z-score normalization, retaining raw DC luminance representations, and applying 2nd-order "
        "Butterworth bandpass filtering (0.65–3.25 Hz). Multiple temporal models were explored, and a lightweight "
        "<b>Depthwise-Separable Bounded Temporal Convolutional Network (HemoVisionBoundedTCN)</b> was selected as the final "
        "model for pulse waveform reconstruction and heart-rate estimation."
    )
    story.append(Paragraph(abs_p2, style_body))
    story.append(Spacer(1, 8))
    
    abs_p3 = (
        "The trained solution achieved a Mean Absolute Error (MAE) of 9.19 BPM on a strictly subject-disjoint held-out "
        "test set of 91 subjects, with high-quality clips reaching 1.33 BPM MAE, outperforming classical signal processing "
        "baselines (POS MAE 19.17 BPM; CHROM MAE 20.28 BPM). Furthermore, an in-depth Biomarker Feasibility Audit demonstrated "
        "that RGB cameras cannot reliably reconstruct blood chemistry without dual-wavelength NIR hardware. An interactive "
        "Streamlit and OpenCV dashboard was developed to present real-time facial ROI tracking, BVP waveforms, and frequency "
        "spectral peaks. The project provided practical exposure to the complete computer vision and deep learning lifecycle."
    )
    story.append(Paragraph(abs_p3, style_body))
    story.append(Spacer(1, 14))
    
    kw_text = (
        "<b>Keywords:</b> Remote Photoplethysmography (rPPG), Deep Learning, Temporal Convolutional Network (TCN), "
        "Computer Vision, MediaPipe Face Mesh, Vital Signs, Heart Rate Estimation, Biomarker Feasibility, PyTorch, OpenCV, Streamlit."
    )
    story.append(Paragraph(kw_text, style_body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: TABLE OF CONTENTS (Page vi)
    # =========================================================================
    story.append(Paragraph("<u>Table of Contents</u>", style_chapter_heading))
    story.append(Spacer(1, 10))
    
    toc_data = [
        [Paragraph("<b>S.no.</b>", style_table_header), Paragraph("<b>Chapter Name</b>", style_table_header), Paragraph("<b>Page Number</b>", style_table_header)],
        [Paragraph("1", style_table_cell), Paragraph("Cover Page", style_table_cell), Paragraph("i", style_table_cell)],
        [Paragraph("2", style_table_cell), Paragraph("Certificate of Completion", style_table_cell), Paragraph("ii", style_table_cell)],
        [Paragraph("3", style_table_cell), Paragraph("Declaration by the Candidate", style_table_cell), Paragraph("iii", style_table_cell)],
        [Paragraph("4", style_table_cell), Paragraph("Acknowledgment", style_table_cell), Paragraph("iv", style_table_cell)],
        [Paragraph("5", style_table_cell), Paragraph("Abstract", style_table_cell), Paragraph("v", style_table_cell)],
        [Paragraph("6", style_table_cell), Paragraph("Table of Content", style_table_cell), Paragraph("vi", style_table_cell)],
        [Paragraph("7", style_table_cell), Paragraph("List of Tables", style_table_cell), Paragraph("vii", style_table_cell)],
        [Paragraph("8", style_table_cell), Paragraph("List of Figures", style_table_cell), Paragraph("viii", style_table_cell)],
        [Paragraph("9", style_table_cell), Paragraph("Abbreviations and Nomenclature", style_table_cell), Paragraph("ix", style_table_cell)],
        [Paragraph("10", style_table_cell), Paragraph("<b>1. Introduction</b>", style_table_cell), Paragraph("1–2", style_table_cell)],
        [Paragraph("11", style_table_cell), Paragraph("<b>2. Formal Training Provided</b>", style_table_cell), Paragraph("3–4", style_table_cell)],
        [Paragraph("12", style_table_cell), Paragraph("<b>3. Industrial Training</b>", style_table_cell), Paragraph("5–7", style_table_cell)],
        [Paragraph("13", style_table_cell), Paragraph("<b>4. Problem Identification and Case Study</b>", style_table_cell), Paragraph("8–11", style_table_cell)],
        [Paragraph("14", style_table_cell), Paragraph("<b>5. Recommendations</b>", style_table_cell), Paragraph("12–13", style_table_cell)],
        [Paragraph("15", style_table_cell), Paragraph("<b>6. References</b>", style_table_cell), Paragraph("14", style_table_cell)],
        [Paragraph("16", style_table_cell), Paragraph("<b>7. Appendices</b>", style_table_cell), Paragraph("15–16", style_table_cell)],
    ]
    t_toc = Table(toc_data, colWidths=[45, 340, 95])
    t_toc.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: LIST OF TABLES (Page vii)
    # =========================================================================
    story.append(Paragraph("<u>List of Tables</u>", style_chapter_heading))
    story.append(Spacer(1, 15))
    
    lot_items = [
        "1. Table 1. Training profile and project details",
        "2. Table 2. Major tools and technologies",
        "3. Table 3. MCD-rPPG Clinical Dataset Summary",
        "4. Table 4. Ground-Truth Clinical Biomarkers in Dataset",
        "5. Table 5. Facial Regions of Interest (ROIs) and Channel Specification",
        "6. Table 6. Performance Comparison on Held-Out Test Set (91 Subjects, 523 Clips)",
        "7. Table 7. Multi-Biomarker Feasibility Audit & Baseline Comparison",
        "8. Table 8. Training outcomes and skills gained",
        "9. Table 9. Suggested clinical dashboard layout"
    ]
    for item in lot_items:
        story.append(Paragraph(item, style_body))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: LIST OF FIGURES (Page viii)
    # =========================================================================
    story.append(Paragraph("<u>List of Figures</u>", style_chapter_heading))
    story.append(Spacer(1, 15))
    
    lof_items = [
        "1. Figure 1. Company/Industry Certificate of Completion",
        "2. Figure 2. HemoVision End-to-End Workflow",
        "3. Figure 3. HemoVision System Architecture",
        "4. Figure 4. Data-to-Decision Pipeline"
    ]
    for item in lof_items:
        story.append(Paragraph(item, style_body))
        story.append(Spacer(1, 6))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: ABBREVIATIONS AND NOMENCLATURE (Page ix)
    # =========================================================================
    story.append(Paragraph("<u>Abbreviations and Nomenclature</u>", style_chapter_heading))
    story.append(Spacer(1, 10))
    
    abbr_data = [
        [Paragraph("<b>Term</b>", style_table_header), Paragraph("<b>Meaning</b>", style_table_header)],
        [Paragraph("AI", style_table_cell), Paragraph("Artificial Intelligence", style_table_cell)],
        [Paragraph("ML", style_table_cell), Paragraph("Machine Learning", style_table_cell)],
        [Paragraph("DL", style_table_cell), Paragraph("Deep Learning", style_table_cell)],
        [Paragraph("rPPG", style_table_cell), Paragraph("Remote Photoplethysmography", style_table_cell)],
        [Paragraph("PPG", style_table_cell), Paragraph("Photoplethysmography", style_table_cell)],
        [Paragraph("BVP", style_table_cell), Paragraph("Blood Volume Pulse", style_table_cell)],
        [Paragraph("ECG", style_table_cell), Paragraph("Electrocardiogram", style_table_cell)],
        [Paragraph("ROI", style_table_cell), Paragraph("Region of Interest", style_table_cell)],
        [Paragraph("TCN", style_table_cell), Paragraph("Temporal Convolutional Network", style_table_cell)],
        [Paragraph("POS", style_table_cell), Paragraph("Plane-Orthogonal-to-Skin rPPG algorithm", style_table_cell)],
        [Paragraph("CHROM", style_table_cell), Paragraph("Chrominance-based rPPG method", style_table_cell)],
        [Paragraph("FFT", style_table_cell), Paragraph("Fast Fourier Transform", style_table_cell)],
        [Paragraph("PSD", style_table_cell), Paragraph("Power Spectral Density", style_table_cell)],
        [Paragraph("MAE", style_table_cell), Paragraph("Mean Absolute Error", style_table_cell)],
        [Paragraph("RMSE", style_table_cell), Paragraph("Root Mean Squared Error", style_table_cell)],
        [Paragraph("BPM", style_table_cell), Paragraph("Beats Per Minute", style_table_cell)],
        [Paragraph("SpO2", style_table_cell), Paragraph("Peripheral Capillary Oxygen Saturation", style_table_cell)],
        [Paragraph("BP", style_table_cell), Paragraph("Blood Pressure (Systolic / Diastolic)", style_table_cell)],
        [Paragraph("AMP", style_table_cell), Paragraph("Automatic Mixed Precision", style_table_cell)],
        [Paragraph("CCC", style_table_cell), Paragraph("Lin's Concordance Correlation Coefficient", style_table_cell)],
        [Paragraph("Streamlit", style_table_cell), Paragraph("Python-based interactive web deployment framework", style_table_cell)],
        [Paragraph("MediaPipe", style_table_cell), Paragraph("Cross-platform framework for 468 3D Face Mesh perception", style_table_cell)]
    ]
    t_abbr = Table(abbr_data, colWidths=[90, 390])
    t_abbr.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_abbr)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10 (Main Body Page 1): 1. INTRODUCTION
    # =========================================================================
    story.append(Paragraph("<u>1. Introduction</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    intro_p1 = (
        "Industrial and vocational training provides an important bridge between academic learning and "
        "practical application. For a Computer Science and Engineering student specializing in Artificial "
        "Intelligence, exposure to real biological datasets, computer vision pipelines, deep temporal learning models, "
        "and interactive clinical visualization tools is particularly valuable. The present training was completed at "
        "<b>BS Digital Technology</b> through a project-based program from 08 June 2026 to 08 July 2026."
    )
    story.append(Paragraph(intro_p1, style_body))
    
    intro_p2 = (
        "The central project undertaken during the training was <b>HemoVision</b>, a contactless vital-sign estimation "
        "and biomarker analytics solution. The project was designed around a practical healthcare requirement: "
        "conventional vital-sign monitoring relies on intrusive contact sensors (pulse oximeters, blood pressure cuffs, "
        "ECG electrodes) and invasive blood sampling. Remote Photoplethysmography (rPPG) provides a non-invasive alternative "
        "by measuring sub-visual skin color fluctuations caused by cardiovascular blood pulses from standard RGB video cameras."
    )
    story.append(Paragraph(intro_p2, style_body))
    
    intro_p3 = (
        "HemoVision follows an end-to-end workflow. Facial video is captured via consumer webcams and processed with "
        "<b>MediaPipe Face Mesh</b> to extract 8 anatomically stable facial Regions of Interest (ROIs). Python and PyTorch "
        "are then used for signal preprocessing, AC/DC disentanglement, Butterworth bandpass filtering, and temporal deep learning "
        "experimentation. The final pulse wave and heart rate predictions are presented through an interactive Streamlit clinical dashboard."
    )
    story.append(Paragraph(intro_p3, style_body))
    
    intro_p4 = (
        "The project uses the clinical MCD-rPPG dataset covering 598 unique subjects across 3,427 video clips. "
        "Variables such as 468 facial landmark coordinates, 8 skin ROIs, camera sensor characteristics (FullHDwebcam, USBVideo, "
        "IriunWebcam), subject physical state (resting and post-exercise), and synchronized contact ground truth (ECG pulse, SpO2, "
        "blood pressure, respiratory rate, hemoglobin) help transform raw video streams into a structured analytical framework."
    )
    story.append(Paragraph(intro_p4, style_body))
    
    story.append(Paragraph("1.1 Training Profile and Project Details", style_section_heading))
    t1_data = [
        [Paragraph("<b>Item</b>", style_table_header), Paragraph("<b>Details</b>", style_table_header)],
        [Paragraph("Student", style_table_cell), Paragraph("Harsh Patel", style_table_cell)],
        [Paragraph("Program", style_table_cell), Paragraph("B.Tech – Computer Science and Engineering (Artificial Intelligence)", style_table_cell)],
        [Paragraph("Semester", style_table_cell), Paragraph("6th Semester", style_table_cell)],
        [Paragraph("Roll No.", style_table_cell), Paragraph("300111323021", style_table_cell)],
        [Paragraph("Enrolment No.", style_table_cell), Paragraph("CD8812", style_table_cell)],
        [Paragraph("Training Organization", style_table_cell), Paragraph("BS Digital Technology", style_table_cell)],
        [Paragraph("Project", style_table_cell), Paragraph("HemoVision — Contactless Vital-Sign Estimation via rPPG", style_table_cell)],
        [Paragraph("Training Type", style_table_cell), Paragraph("Project-based vocational training", style_table_cell)],
        [Paragraph("Training Period", style_table_cell), Paragraph("08 June 2026 – 08 July 2026", style_table_cell)],
        [Paragraph("Duration", style_table_cell), Paragraph("30 days", style_table_cell)],
        [Paragraph("Primary Technologies", style_table_cell), Paragraph("Python, PyTorch, OpenCV, MediaPipe, Deep Learning", style_table_cell)],
        [Paragraph("Deployment / UI Layer", style_table_cell), Paragraph("Streamlit, SciPy, Matplotlib, Scikit-Learn", style_table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[140, 340])
    t1.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t1)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 11 (Main Body Page 2): 1.2 PROJECT OVERVIEW
    # =========================================================================
    story.append(Paragraph("1.2 Project Overview", style_section_heading))
    story.append(Spacer(1, 10))
    
    if Path("pdf_assets/fig2_workflow.png").exists():
        story.append(Image("pdf_assets/fig2_workflow.png", width=475, height=120))
    story.append(Paragraph("Figure 2: HemoVision End-to-End Workflow", style_caption))
    story.append(Spacer(1, 15))
    
    po_text = (
        "The workflow starts with ambient facial video captures and ends with clinical-grade physiological forecasts. "
        "The modular separation between facial tracking, signal preprocessing, temporal neural modeling, and visualization "
        "makes HemoVision robust, easy to interpret, maintain, and extend into real-time telemedicine applications."
    )
    story.append(Paragraph(po_text, style_body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 12 (Main Body Page 3): 2. FORMAL TRAINING PROVIDED
    # =========================================================================
    story.append(Paragraph("<u>2. Formal Training Provided</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    ft_intro = (
        "The training was project-oriented and focused on connecting computer vision, biological signal processing, "
        "and deep learning concepts into a practical contactless health monitoring system. Rather than treating individual "
        "technologies as isolated topics, the learning process connected them through a complete workflow: video decoding, "
        "facial landmark mesh tracking, multi-ROI spatial signal extraction, digital filtering, PyTorch neural model development, "
        "and interactive web-based deployment."
    )
    story.append(Paragraph(ft_intro, style_body))
    
    story.append(Paragraph("2.1 Python and Computer Vision / Signal Processing", style_section_heading))
    story.append(Paragraph(
        "Python was used as the main programming environment for computer vision and signal processing. "
        "The training strengthened practical proficiency in libraries such as OpenCV, MediaPipe, NumPy, and SciPy for video decoding, "
        "spatial pixel pooling, and 1D digital signal analysis:",
        style_body
    ))
    story.append(Paragraph("• Decoding raw video frames and handling camera stream buffers using OpenCV.", style_bullet))
    story.append(Paragraph("• Extracting 468 3D facial landmarks and dynamic region masking using MediaPipe Face Mesh.", style_bullet))
    story.append(Paragraph("• Implementing 8 facial anatomical ROIs and spatial RGB channel averaging.", style_bullet))
    story.append(Paragraph("• Digital filtering using 2nd-order zero-phase Butterworth bandpass filters (0.65–3.25 Hz).", style_bullet))
    story.append(Paragraph("• Calculating Welch Power Spectral Density (PSD) and zero-padded Fast Fourier Transforms.", style_bullet))
    story.append(Paragraph("• Implementing classical rPPG projection baselines (POS and CHROM algorithms).", style_bullet))
    
    story.append(Paragraph("2.2 Deep Learning and PyTorch Framework", style_section_heading))
    story.append(Paragraph(
        "PyTorch was utilized as the deep learning framework for designing, training, and evaluating custom neural architectures:",
        style_body
    ))
    story.append(Paragraph("• Designing modular 1D convolution and Depthwise-Separable Temporal Convolution layers.", style_bullet))
    story.append(Paragraph("• Implementing exponential dilation schedules (1, 2, 4, 8, 16) to capture long-range receptive fields.", style_bullet))
    story.append(Paragraph("• Formulating custom loss functions including Negative Pearson Correlation and Lin's CCC.", style_bullet))
    story.append(Paragraph("• Utilizing Automatic Mixed Precision (AMP) and gradient accumulation for memory optimization.", style_bullet))
    
    story.append(Paragraph("2.3 Remote Photoplethysmography &amp; Temporal Modeling", style_section_heading))
    story.append(Paragraph(
        "The rPPG modeling component focused on learning pulsatile blood volume patterns under noisy conditions:",
        style_body
    ))
    story.append(Paragraph("• Disentangling AC pulsatile cardiac signals from static DC skin illumination baselines.", style_bullet))
    story.append(Paragraph("• Training multi-task neural networks for joint waveform and multi-biomarker estimation.", style_bullet))
    story.append(Paragraph("• Selecting HemoVisionBoundedTCN as the final pulse wave reconstruction model.", style_bullet))
    story.append(Paragraph("• Conducting rigorous baseline comparisons against demographic shortcut baselines.", style_bullet))
    
    story.append(Paragraph("2.4 Interactive Clinical Visualization and Deployment", style_section_heading))
    story.append(Paragraph(
        "Streamlit and OpenCV were used to transform analytical outputs into responsive, user-friendly clinical views:",
        style_body
    ))
    story.append(Paragraph("• Building a real-time web application for video upload, playback, and live heart rate tracking.", style_bullet))
    story.append(Paragraph("• Rendering dynamic facial ROI overlays, predicted BVP pulse waveforms, and FFT power spectrums.", style_bullet))
    story.append(Paragraph("• Generating clinical agreement charts (Bland-Altman and correlation plots).", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 13 (Main Body Page 4): 2.5 LEARNING OUTCOMES
    # =========================================================================
    story.append(Paragraph("2.5 Learning Outcome of Formal Training", style_section_heading))
    story.append(Spacer(1, 10))
    
    lo_text = (
        "The formal/project-oriented training developed a comprehensive understanding of how computer vision, digital "
        "signal processing, and deep neural networks converge into a complete clinical AI product. The major learning "
        "outcome was not only theoretical knowledge of PyTorch, OpenCV, MediaPipe, and Streamlit separately, but also "
        "the practical ability to move physiological video data through a repeatable pipeline from optical camera capture "
        "to neural pulse wave reconstruction, rigorous error auditing, and real-time visualization."
    )
    story.append(Paragraph(lo_text, style_body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 14 (Main Body Page 5): 3. INDUSTRIAL TRAINING
    # =========================================================================
    story.append(Paragraph("<u>3. Industrial Training</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("3.1 Objectives", style_section_heading))
    story.append(Paragraph("1. To understand the practical lifecycle of a contactless biomedical vision project from video stream to clinical output.", style_bullet))
    story.append(Paragraph("2. To gain hands-on experience in extracting and preprocessing multi-ROI facial rPPG datasets.", style_bullet))
    story.append(Paragraph("3. To design and implement a lightweight, depthwise-separable Bounded Temporal Convolutional Network in PyTorch.", style_bullet))
    story.append(Paragraph("4. To implement and evaluate classical benchmark algorithms (POS and CHROM) against deep learning architectures.", style_bullet))
    story.append(Paragraph("5. To discover, isolate, and debug clinical ground-truth anomalies (e.g., ECG T-wave double counting).", style_bullet))
    story.append(Paragraph("6. To conduct a scientifically rigorous multi-biomarker feasibility audit across 11 physiological variables.", style_bullet))
    story.append(Paragraph("7. To deploy a real-time, interactive clinical demonstration dashboard using Streamlit and OpenCV.", style_bullet))
    story.append(Paragraph("8. To improve problem-solving, documentation, debugging, and technical communication skills.", style_bullet))
    
    story.append(Paragraph("3.2 Tools and Technologies Used", style_section_heading))
    t2_data = [
        [Paragraph("<b>Technology / Tool</b>", style_table_header), Paragraph("<b>Role in HemoVision</b>", style_table_header)],
        [Paragraph("Python", style_table_cell), Paragraph("Data preparation, signal extraction, model training, prediction", style_table_cell)],
        [Paragraph("PyTorch (torch/nn)", style_table_cell), Paragraph("Deep learning framework, custom TCN layers, multi-task loss", style_table_cell)],
        [Paragraph("OpenCV (cv2)", style_table_cell), Paragraph("Video decoding, frame extraction, color space conversion, UI", style_table_cell)],
        [Paragraph("MediaPipe", style_table_cell), Paragraph("468-point 3D Face Mesh tracking and real-time ROI segmentation", style_table_cell)],
        [Paragraph("NumPy &amp; Pandas", style_table_cell), Paragraph("Array processing, dataframes, dataset indexing, and metrics", style_table_cell)],
        [Paragraph("SciPy (signal/fft)", style_table_cell), Paragraph("Butterworth digital filtering, Welch PSD, zero-padded FFT", style_table_cell)],
        [Paragraph("Scikit-learn", style_table_cell), Paragraph("Classical regression baselines, evaluation metrics (MAE, RMSE, r)", style_table_cell)],
        [Paragraph("HemoVisionBoundedTCN", style_table_cell), Paragraph("Final selected deep temporal convolutional neural network", style_table_cell)],
        [Paragraph("Streamlit", style_table_cell), Paragraph("Web dashboard, real-time telemetry, interactive clinical views", style_table_cell)],
        [Paragraph("VS Code / Git", style_table_cell), Paragraph("IDE development, modular organization, and version control", style_table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[140, 340])
    t2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("3.3 Techniques Studied in Different Departments", style_section_heading))
    story.append(Paragraph("3.3.1 Facial Landmark Extraction and Multi-ROI Processing", style_sub_heading))
    story.append(Paragraph(
        "Face detection alone is insufficient for robust rPPG because facial regions exhibit varying capillary density "
        "and motion susceptibility. MediaPipe Face Mesh was used to track 468 3D landmarks and segment 8 independent ROIs "
        "(Forehead Left/Right, Upper Cheek Left/Right, Lower Cheek Left/Right, Nose Bridge, Chin) yielding a 24-channel signal tensor.",
        style_body
    ))
    story.append(Paragraph("3.3.2 Data Preprocessing and AC/DC Disentanglement", style_sub_heading))
    story.append(Paragraph(
        "Raw pixel averages contain a large static DC illumination component (~99%) and a tiny pulsatile AC wave (~0.1–1%). "
        "Channel-wise z-score normalization was applied to isolate AC pulsatile cardiac signals, while raw scaled luminance (I/255.0) "
        "was preserved for skin baseline tone analysis.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 15 (Main Body Page 6): 3.3.3 to 3.4
    # =========================================================================
    story.append(Paragraph("3.3.3 Feature Engineering and Classical Baselines", style_sub_heading))
    story.append(Paragraph(
        "Feature engineering was a major learning component because rPPG signals are sensitive to subject motion and illumination drifts. "
        "The project incorporated spatial ROI signals, temporal filtering, and classical optical projections (POS and CHROM).",
        style_body
    ))
    
    fg_data = [
        [Paragraph("<b>Feature Group</b>", style_table_header), Paragraph("<b>Examples / Purpose</b>", style_table_header)],
        [Paragraph("Facial ROIs (8)", style_table_cell), Paragraph("Forehead L/R, Cheek L/R Upper/Lower, Nose Bridge, Chin", style_table_cell)],
        [Paragraph("Optical Channels", style_table_cell), Paragraph("24 channels (8 ROIs x 3 RGB channels); AC and DC streams", style_table_cell)],
        [Paragraph("Digital Filters", style_table_cell), Paragraph("2nd-order zero-phase Butterworth bandpass (0.65–3.25 Hz)", style_table_cell)],
        [Paragraph("Spectral Features", style_table_cell), Paragraph("Zero-padded Welch PSD (N_fft=4096) for sub-BPM peak picking", style_table_cell)],
        [Paragraph("Classical Projections", style_table_cell), Paragraph("Plane-Orthogonal-to-Skin (POS), Chrominance (CHROM)", style_table_cell)],
        [Paragraph("Metadata Conditioning", style_table_cell), Paragraph("Camera sensor one-hot encoding (FullHDwebcam, USBVideo, Iriun)", style_table_cell)],
        [Paragraph("Biomarker Targets", style_table_cell), Paragraph("Pulse, SpO2, Respiratory Rate, Blood Pressure, Hemoglobin, BMI", style_table_cell)],
    ]
    t_fg = Table(fg_data, colWidths=[140, 340])
    t_fg.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_fg)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("3.3.4 Machine-Learning &amp; Deep Learning Modelling", style_sub_heading))
    story.append(Paragraph(
        "The modelling stage treated BVP pulse wave reconstruction as the primary regression target. Multiple neural architectures "
        "were trained and compared. HemoVisionBoundedTCN was selected as the final model because its dilated depthwise separable "
        "convolutions captured long-range cardiac cycles (>20s) with 78% fewer parameters while bounded Tanh heads prevented numerical explosion.",
        style_body
    ))
    
    story.append(Paragraph("3.3.5 Ground Truth Rectification &amp; Spectral Estimation", style_sub_heading))
    story.append(Paragraph(
        "The final model was used to predict continuous BVP waveforms, followed by zero-padded Welch FFT peak picking within the "
        "physiological range (39–195 BPM). An ECG ground-truth T-wave double-counting bug was discovered and rectified.",
        style_body
    ))
    
    story.append(Paragraph("3.3.6 Clinical Intelligence Dashboard", style_sub_heading))
    story.append(Paragraph(
        "Streamlit was used to make the deep learning outputs clinically intuitive. The interactive interface displays synchronized "
        "facial bounding boxes, live pulse waveforms, power spectrum peaks, and ground-truth validation comparisons.",
        style_body
    ))
    
    story.append(Paragraph("3.4 Software and Tools Used", style_section_heading))
    story.append(Paragraph(
        "The practical development environment consisted of Python/PyTorch for deep learning, OpenCV/MediaPipe for video processing, "
        "SciPy for signal processing, and Streamlit for web visualization. VS Code was used for organizing scripts and project files.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 16 (Main Body Page 7): 3.5 HIGHLIGHTS
    # =========================================================================
    story.append(Paragraph("3.5 Highlights of Training Exposure", style_section_heading))
    story.append(Spacer(1, 10))
    story.append(Paragraph("• Exposure to a complete biological data-to-decision pipeline rather than an isolated model.", style_bullet))
    story.append(Paragraph("• Practical handling of 3,427 clinical video clips across multiple consumer camera sensors.", style_bullet))
    story.append(Paragraph("• Experience with AC/DC signal disentanglement and 8-ROI spatial pooling.", style_bullet))
    story.append(Paragraph("• Hands-on dilated temporal convolution architecture design in PyTorch.", style_bullet))
    story.append(Paragraph("• Discovery and correction of medical ground-truth annotation bugs (ECG T-wave anomaly).", style_bullet))
    story.append(Paragraph("• Exposure to scientific falsifiability, confound analysis, and real-time clinical deployment.", style_bullet))
    story.append(Paragraph("• Improved understanding of how computer vision, DSP, and deep learning layers interact.", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 17 (Main Body Page 8): 4. PROBLEM IDENTIFICATION / CASE STUDY
    # =========================================================================
    story.append(Paragraph("<u>4. Problem Identification/ Case Study</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("4.1 Problem Statement", style_section_heading))
    story.append(Paragraph(
        "Traditional patient monitoring relies on intrusive contact sensors (pulse oximeters, blood pressure cuffs, ECG leads) "
        "and invasive blood draws. These methods cause patient discomfort, risk skin breakdown in neonatal/geriatric care, and present "
        "cross-contamination risks during epidemic outbreaks. Remote Photoplethysmography (rPPG) provides a non-invasive alternative "
        "by measuring blood volume pulse oscillations from facial video. However, existing literature suffers from severe sensor overfitting "
        "and unsubstantiated biomarker claims. The HemoVision case study addresses this problem by combining robust facial landmark "
        "tracking, AC/DC signal disentanglement, dilated TCN modeling, and rigorous confound auditing.",
        style_body
    ))
    
    story.append(Paragraph("4.2 Data Used in the Case Study", style_section_heading))
    t3_data = [
        [Paragraph("<b>Dataset / Split</b>", style_table_header), Paragraph("<b>Period / Scope</b>", style_table_header), Paragraph("<b>Purpose</b>", style_table_header)],
        [Paragraph("MCD-rPPG Dataset", style_table_cell), Paragraph("3,427 clips / 598 subjects", style_table_cell), Paragraph("Multi-camera facial video &amp; synchronized vital ground truth", style_table_cell)],
        [Paragraph("Train Partition", style_table_cell), Paragraph("2,387 clips / 418 subjects", style_table_cell), Paragraph("Subject-disjoint model training &amp; feature learning", style_table_cell)],
        [Paragraph("Validation Partition", style_table_cell), Paragraph("517 clips / 89 subjects", style_table_cell), Paragraph("Hyperparameter tuning &amp; early stopping checkpointing", style_table_cell)],
        [Paragraph("Held-Out Test Partition", style_table_cell), Paragraph("523 clips / 91 subjects", style_table_cell), Paragraph("Strictly leak-free final evaluation &amp; metric benchmarking", style_table_cell)],
        [Paragraph("Clinical Database (db.csv)", style_table_cell), Paragraph("11 Biomarker fields", style_table_cell), Paragraph("ECG pulse, SpO2, blood pressure, hemoglobin, BMI ground truth", style_table_cell)],
    ]
    t3 = Table(t3_data, colWidths=[130, 140, 210])
    t3.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("4.3 Data Preparation", style_section_heading))
    story.append(Paragraph(
        "The raw video clips and reference records were consolidated into an analytical structure. Video frames were processed "
        "using MediaPipe Face Mesh, and spatial channel means across 8 facial regions were integrated:",
        style_body
    ))
    story.append(Paragraph("• Raw video frames were decoded at 30 FPS across 600-frame sliding windows.", style_bullet))
    story.append(Paragraph("• MediaPipe tracked 468 3D landmarks to extract 8 anatomically stable skin ROIs.", style_bullet))
    story.append(Paragraph("• Channel order was verified and auto-corrected to RGB format via skin reflection heuristics.", style_bullet))
    story.append(Paragraph("• Signals were split into AC normalized ($x$) and DC raw scaled ($x_{\\text{raw}}$) streams.", style_bullet))
    story.append(Paragraph("• Synchronized contact BVP pulse waveforms were retained as the training target.", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 18 (Main Body Page 9): 4.4 to 4.6
    # =========================================================================
    story.append(Paragraph("4.4 Machine-Learning Feature Set", style_section_heading))
    
    t4_data = [
        [Paragraph("<b>Category</b>", style_table_header), Paragraph("<b>Representative Features / Specifications</b>", style_table_header)],
        [Paragraph("Facial ROIs", style_table_cell), Paragraph("Forehead L/R, Cheek L/R Upper/Lower, Nose Bridge, Chin", style_table_cell)],
        [Paragraph("Spatial Channels", style_table_cell), Paragraph("24 channels (8 ROIs x 3 RGB channels)", style_table_cell)],
        [Paragraph("AC Normalization", style_table_cell), Paragraph("Temporal z-score normalization (mean=0, std=1)", style_table_cell)],
        [Paragraph("DC Baseline", style_table_cell), Paragraph("Raw luminance scaled to [0, 1] (signal / 255.0)", style_table_cell)],
        [Paragraph("Temporal Window", style_table_cell), Paragraph("600 frames (~20 seconds @ 30 FPS, stride 150 frames)", style_table_cell)],
        [Paragraph("Camera Sensor", style_table_cell), Paragraph("FullHDwebcam, USBVideo, IriunWebcam one-hot vector", style_table_cell)],
        [Paragraph("Subject State", style_table_cell), Paragraph("Resting ('before') and post-exercise ('after')", style_table_cell)],
        [Paragraph("Ground Truth", style_table_cell), Paragraph("Synchronized ECG pulse, SpO2, BP, Hb, BMI, respiratory rate", style_table_cell)],
    ]
    t4 = Table(t4_data, colWidths=[130, 350])
    t4.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t4)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("4.5 System Architecture", style_section_heading))
    if Path("pdf_assets/fig3_architecture.png").exists():
        story.append(Image("pdf_assets/fig3_architecture.png", width=475, height=210))
    story.append(Paragraph("Figure 3: HemoVision System Architecture", style_caption))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("4.6 Methodology", style_section_heading))
    story.append(Paragraph("1. Requirement analysis: identify clinical contactless vital estimation requirements and ethical boundaries.", style_bullet))
    story.append(Paragraph("2. Data collection and organization: assemble 3,427 clips across 598 subjects with synchronized clinical sensors.", style_bullet))
    story.append(Paragraph("3. Data partitioning: enforce strict subject-disjoint splits (418 train, 89 val, 91 test subjects).", style_bullet))
    story.append(Paragraph("4. Feature engineering: extract 24-channel spatial averages across 8 facial regions via MediaPipe.", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 19 (Main Body Page 10): 4.6 (cont) to 4.10
    # =========================================================================
    story.append(Paragraph("5. Signal normalization: construct dual AC (z-score standardized) and DC (raw scaled) temporal tensors.", style_bullet))
    story.append(Paragraph("6. Ground truth validation: inspect and correct reference annotations (resolving ECG T-wave doubling).", style_bullet))
    story.append(Paragraph("7. Model experimentation: train and compare classical baselines (POS/CHROM) and deep neural architectures.", style_bullet))
    story.append(Paragraph("8. Model selection: select HemoVisionBoundedTCN as the final model used in the project.", style_bullet))
    story.append(Paragraph("9. Spectral estimation: extract Heart Rate via zero-padded Welch FFT peak picking (39–195 BPM band).", style_bullet))
    story.append(Paragraph("10. Visualization: deploy interactive clinical telemetry dashboard using Streamlit and OpenCV.", style_bullet))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.7 Data-to-Decision Pipeline", style_section_heading))
    if Path("pdf_assets/fig4_pipeline.png").exists():
        story.append(Image("pdf_assets/fig4_pipeline.png", width=475, height=95))
    story.append(Paragraph("Figure 4: Data-to-Decision Pipeline for HemoVision", style_caption))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.8 Model Development", style_section_heading))
    story.append(Paragraph(
        "The project used a supervised deep learning approach. Candidate architectures were explored, and "
        "HemoVisionBoundedTCN was selected. The model employs depthwise-separable 1D convolutions with residual dilation "
        "blocks (dilations 1, 2, 4, 8, 16) and a bounded Tanh waveform head, optimizing Negative Pearson Correlation loss.",
        style_body
    ))
    
    story.append(Paragraph("4.9 Vital-Sign Evaluation Outputs", style_section_heading))
    t6_data = [
        [Paragraph("<b>Model / Method</b>", style_table_header), Paragraph("<b>Heart Rate MAE</b>", style_table_header), Paragraph("<b>Heart Rate RMSE</b>", style_table_header), Paragraph("<b>Pearson r</b>", style_table_header)],
        [Paragraph("POS (Wang et al. 2017)", style_table_cell), Paragraph("19.17 BPM", style_table_cell), Paragraph("23.72 BPM", style_table_cell), Paragraph("0.119", style_table_cell)],
        [Paragraph("CHROM (de Haan 2013)", style_table_cell), Paragraph("20.28 BPM", style_table_cell), Paragraph("24.61 BPM", style_table_cell), Paragraph("0.047", style_table_cell)],
        [Paragraph("<b>HemoVisionBoundedTCN (Ours)</b>", style_table_cell), Paragraph("<b>9.19 BPM</b>", style_table_cell), Paragraph("<b>18.35 BPM</b>", style_table_cell), Paragraph("<b>0.285</b>", style_table_cell)],
        [Paragraph("<i>HemoVision Demo Clips</i>", style_table_cell), Paragraph("<i>1.33 BPM</i>", style_table_cell), Paragraph("<i>1.82 BPM</i>", style_table_cell), Paragraph("<i>0.884</i>", style_table_cell)],
    ]
    t6 = Table(t6_data, colWidths=[170, 100, 110, 100])
    t6.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t6)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.10 Results and Discussion", style_section_heading))
    story.append(Paragraph(
        "The project successfully completed the deep learning and rPPG vital-sign workflow. The final HemoVisionBoundedTCN "
        "model achieved 9.19 BPM MAE across 523 held-out test clips, cutting the error of classical methods in half. On high-quality "
        "demo clips, the error dropped to 1.33 BPM.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 20 (Main Body Page 11): 4.10 (cont) & 4.11
    # =========================================================================
    story.append(Paragraph(
        "The project also conducted a rigorous <b>Biomarker Feasibility Audit</b> investigating 10 additional physiological targets. "
        "Five independent architectural hypotheses were tested and eliminated (target standardisation, batchnorm freeze, multi-head "
        "attention pooling, raw DC bypass, and CCC loss). For hemoglobin, while the video model achieved an apparent Pearson r of 0.44, "
        "a simple demographic baseline (age, sex, BMI) achieved r=0.70 and MAE=0.95 g/dL. This demonstrated that apparent video "
        "correlations are demographic identity shortcuts rather than true optical sensing, proving that RGB cameras cannot reconstruct "
        "blood chemistry without dual-wavelength NIR hardware.",
        style_body
    ))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("4.11 Skills and Outcomes", style_section_heading))
    t8_data = [
        [Paragraph("<b>Area</b>", style_table_header), Paragraph("<b>Training Outcome</b>", style_table_header)],
        [Paragraph("Computer Vision", style_table_cell), Paragraph("Mastered 468-point 3D facial landmark detection and multi-ROI spatial signal pooling with MediaPipe", style_table_cell)],
        [Paragraph("Digital Signal Processing", style_table_cell), Paragraph("Implemented Butterworth bandpass filters, POS/CHROM projections, and zero-padded Welch PSD", style_table_cell)],
        [Paragraph("Deep Learning Architecture", style_table_cell), Paragraph("Designed and trained Depthwise-Separable Bounded TCNs with residual dilation trunks in PyTorch", style_table_cell)],
        [Paragraph("Loss Engineering", style_table_cell), Paragraph("Formulated Negative Pearson correlation waveform loss and Lin's Concordance Correlation Coefficient", style_table_cell)],
        [Paragraph("Debugging &amp; Data Integrity", style_table_cell), Paragraph("Discovered and resolved medical ground-truth anomalies (ECG T-wave doubling) and FFT quantization", style_table_cell)],
        [Paragraph("Scientific Verification", style_table_cell), Paragraph("Conducted leak-proof subject-disjoint evaluation and exposed demographic confound shortcuts in AI", style_table_cell)],
        [Paragraph("UI &amp; Web Deployment", style_table_cell), Paragraph("Built interactive real-time Streamlit and OpenCV clinical monitoring dashboards", style_table_cell)],
        [Paragraph("Technical Documentation", style_table_cell), Paragraph("Prepared publication-grade biomedical reports, architecture schematics, and comparative audits", style_table_cell)],
    ]
    t8 = Table(t8_data, colWidths=[140, 340])
    t8.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t8)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 21 (Main Body Page 12): 5. RECOMMENDATIONS
    # =========================================================================
    story.append(Paragraph("<u>5. Recommendations</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        "The current HemoVision implementation provides a strong end-to-end foundation. The following recommendations "
        "can improve the system in future iterations without changing its core architecture.",
        style_body
    ))
    
    story.append(Paragraph("5.1 Model Improvements", style_section_heading))
    story.append(Paragraph("• Evaluate self-supervised pre-training on large facial video corpora using masked temporal autoencoders.", style_bullet))
    story.append(Paragraph("• Integrate optical flow vectors into TCN conditioning to dynamically compensate for speaking and head motion.", style_bullet))
    story.append(Paragraph("• Implement Bayesian neural dropout or deep evidential regression to output real-time confidence intervals.", style_bullet))
    story.append(Paragraph("• Study feature attribution and temporal saliency maps to support clinical interpretability.", style_bullet))
    story.append(Paragraph("• Monitor model generalization across diverse skin tone distributions.", style_bullet))
    
    story.append(Paragraph("5.2 Hardware &amp; Multi-Wavelength Optical Improvements", style_section_heading))
    story.append(Paragraph("• Pair software with a dual-wavelength active LED ring (660nm visible red + 940nm NIR) for true non-contact SpO2 and hemoglobin quantification.", style_bullet))
    story.append(Paragraph("• Utilize 60–120 FPS global-shutter cameras to eliminate rolling-shutter artifacts.", style_bullet))
    story.append(Paragraph("• Create scheduled illumination calibration routines for varying ambient lighting environments.", style_bullet))
    
    story.append(Paragraph("5.3 Physiological Metric Extensions (HRV / RMSSD / PWTT Blood Pressure)", style_section_heading))
    story.append(Paragraph("• Extract inter-beat intervals (IBI) from reconstructed BVP waves to compute clinical autonomic stress markers: RMSSD, SDNN, and LF/HF ratios.", style_bullet))
    story.append(Paragraph("• Utilize dual-site multi-ROI waveforms (measuring phase delay between forehead and chin micro-vessels) for Pulse Wave Transit Time (PWTT) blood pressure estimation.", style_bullet))
    story.append(Paragraph("• Add confidence or prediction intervals where the chosen forecasting approach supports them.", style_bullet))
    
    story.append(Paragraph("5.4 Clinical Interface &amp; Deployment Improvements", style_section_heading))
    story.append(Paragraph("• Quantize the PyTorch model to ONNX / TensorRT / INT8 for deployment on low-power edge medical devices (e.g., Raspberry Pi 5, Jetson).", style_bullet))
    story.append(Paragraph("• Implement secure FHIR / HL7 REST APIs for direct electronic health record (EHR) integration.", style_bullet))
    story.append(Paragraph("• Add alerts or exception views for unusually high or low vital sign readings (tachycardia/bradycardia).", style_bullet))
    
    story.append(Paragraph("5.5 Future Scope", style_section_heading))
    story.append(Paragraph("• Develop an automated streaming ETL pipeline connecting camera ingestion, PyTorch inference, and clinical dashboards.", style_bullet))
    story.append(Paragraph("• Deploy the rPPG model as a containerized microservice for telemedicine consultations.", style_bullet))
    story.append(Paragraph("• Implement real-time multi-patient simultaneous tracking for hospital waiting rooms.", style_bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 22 (Main Body Page 13): 5.6 DASHBOARD LAYOUT
    # =========================================================================
    story.append(Paragraph("5.6 Recommended Future Dashboard Layout", style_section_heading))
    story.append(Spacer(1, 8))
    
    t9_data = [
        [Paragraph("<b>Dashboard Page</b>", style_table_header), Paragraph("<b>Suggested Content</b>", style_table_header)],
        [Paragraph("Live Telemetry Overview", style_table_cell), Paragraph("Real-time facial video feed, MediaPipe 8-ROI wireframe, live BVP waveform, and instantaneous BPM", style_table_cell)],
        [Paragraph("Vital Signs &amp; Trend Monitor", style_table_cell), Paragraph("Multi-hour heart-rate history, respiratory rate trends, and physiological confidence gauge", style_table_cell)],
        [Paragraph("Spectral &amp; Diagnostic View", style_table_cell), Paragraph("Welch PSD power spectrum plot, peak frequency selector, and SNR quality indicator", style_table_cell)],
        [Paragraph("Subject &amp; Clinical History", style_table_cell), Paragraph("Patient ID, session notes, comparative baseline metrics, and exportable PDF medical summary", style_table_cell)],
        [Paragraph("Hardware &amp; Calibration", style_table_cell), Paragraph("Camera sensor selection (FPS/resolution), illumination brightness meter, and ROI sensitivity tuner", style_table_cell)],
    ]
    t9 = Table(t9_data, colWidths=[150, 330])
    t9.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t9)
    story.append(Spacer(1, 15))
    
    rec_concl = (
        "The recommended extensions would move HemoVision from an experimental vital-sign forecasting project toward a "
        "complete, hospital-grade contactless clinical decision-support platform."
    )
    story.append(Paragraph(rec_concl, style_body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 23 (Main Body Page 14): 6. REFERENCES
    # =========================================================================
    story.append(Paragraph("<u>6. References</u>", style_chapter_heading))
    story.append(Spacer(1, 15))
    
    refs = [
        "1. Wang, W., den Brinker, A. C., Stuijk, S., &amp; de Haan, G. (2017). Algorithmic principles of remote photoplethysmography. <i>IEEE Transactions on Biomedical Engineering</i>, 64(7), 1479–1491.",
        "2. de Haan, G., &amp; Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. <i>IEEE Transactions on Biomedical Engineering</i>, 60(10), 2878–2886.",
        "3. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. <i>arXiv preprint arXiv:1906.08172</i>.",
        "4. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 32.",
        "5. Bradski, G. (2000). The OpenCV Library. <i>Dr. Dobb's Journal of Software Tools</i>.",
        "6. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. <i>Nature Methods</i>, 17(3), 261–272.",
        "7. BS Digital Technology. Training certificate and project-based training curriculum in Machine Learning and Computer Vision (2026).",
        "8. MCD-rPPG Clinical Dataset, notebooks, Python scripts, and Streamlit work produced during the HemoVision vocational training project."
    ]
    for r in refs:
        story.append(Paragraph(r, style_body))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 24 (Main Body Page 15): 7. APPENDICES (A & B)
    # =========================================================================
    story.append(Paragraph("<u>7. Appendices</u>", style_chapter_heading))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Appendix A – Suggested Project Folder Structure", style_section_heading))
    
    folder_text = """
HemoVision/<br/>
├── app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Interactive Streamlit clinical monitoring dashboard<br/>
├── hemovision_demo.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Standalone OpenCV live inference &amp; testing engine<br/>
├── HemoVision_V3_Pipeline.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Complete end-to-end training and evaluation script<br/>
├── data/<br/>
│&nbsp;&nbsp;&nbsp;└── vitalscan-clinic/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── MCD-rPPG/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── db.csv &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Ground-truth clinical biomarker database<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── manifest.json &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Video clip metadata and subject-disjoint splits<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── video/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Raw RGB video clips (3 camera sensors)<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── ppg_sync/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Synchronized contact reference waveforms<br/>
├── v3/<br/>
│&nbsp;&nbsp;&nbsp;├── config.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Hyperparameters, task specs, and camera configs<br/>
│&nbsp;&nbsp;&nbsp;├── dataset.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# PyTorch Dataset and multi-ROI temporal loader<br/>
│&nbsp;&nbsp;&nbsp;├── model.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# HemoVisionBoundedTCN &amp; Attention Biomarker Head<br/>
│&nbsp;&nbsp;&nbsp;└── loss.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Negative Pearson r and CCC multi-task loss<br/>
└── reports/<br/>
&nbsp;&nbsp;&nbsp;&nbsp;└── HemoVision_Vocational_Training_Report.pdf
    """
    story.append(Paragraph(folder_text, ParagraphStyle('FolderStyle', fontName='Courier', fontSize=8, leading=10)))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Appendix B – Final Feature Groups", style_section_heading))
    t_apb_data = [
        [Paragraph("<b>Feature Type</b>", style_table_header), Paragraph("<b>Fields</b>", style_table_header)],
        [Paragraph("Identifiers / Meta", style_table_cell), Paragraph("subject_id, clip_id, camera_type, condition, split", style_table_cell)],
        [Paragraph("Spatial ROIs (8)", style_table_cell), Paragraph("Forehead L/R, Cheek L/R Upper/Lower, Nose Bridge, Chin", style_table_cell)],
        [Paragraph("Optical Channels", style_table_cell), Paragraph("24 channels (8 ROIs x 3 RGB channels); AC and DC streams", style_table_cell)],
        [Paragraph("Waveform Target", style_table_cell), Paragraph("BVP pulse waveform (600 temporal frames @ 30 FPS)", style_table_cell)],
        [Paragraph("Vital Signs (Tier 1)", style_table_cell), Paragraph("Pulse / Heart Rate (BPM), Respiratory Rate, SpO2", style_table_cell)],
        [Paragraph("Blood Pressure (Tier 2)", style_table_cell), Paragraph("Systolic BP (mmHg), Diastolic BP (mmHg)", style_table_cell)],
        [Paragraph("Blood Chem (Tier 3)", style_table_cell), Paragraph("Hemoglobin, HbA1c, Cholesterol, Rigidity, Stress, BMI", style_table_cell)],
    ]
    t_apb = Table(t_apb_data, colWidths=[140, 340])
    t_apb.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9E1F2")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_apb)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 25 (Main Body Page 16): APPENDIX C
    # =========================================================================
    story.append(Paragraph("Appendix C – rPPG Training &amp; Evaluation Checklist", style_section_heading))
    story.append(Spacer(1, 10))
    story.append(Paragraph("• Verify source video integrity, resolution, and nominal frame rates (30 FPS).", style_bullet))
    story.append(Paragraph("• Validate temporal synchronization between facial video and contact reference waveforms.", style_bullet))
    story.append(Paragraph("• Check for BGR versus RGB color channel ordering using skin reflection ratio heuristics.", style_bullet))
    story.append(Paragraph("• Confirm subject-disjoint isolation across train (418), val (89), and test (91) splits.", style_bullet))
    story.append(Paragraph("• Run AC z-score normalization and DC raw luminance scaling consistently.", style_bullet))
    story.append(Paragraph("• Load the selected HemoVisionBoundedTCN PyTorch model checkpoint.", style_bullet))
    story.append(Paragraph("• Generate continuous BVP pulse waveform predictions on held-out clips.", style_bullet))
    story.append(Paragraph("• Extract heart rate using zero-padded Welch PSD (N_fft=4096) across 39–195 BPM.", style_bullet))
    story.append(Paragraph("• Audit biomarker predictions against demographic baselines before claiming clinical validity.", style_bullet))
    story.append(Paragraph("• Refresh Streamlit clinical dashboard and review real-time telemetry values.", style_bullet))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}!")

if __name__ == "__main__":
    build_pdf()
