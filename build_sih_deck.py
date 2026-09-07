"""
HemoVision — SIH 2026 Pitch Deck Builder
=========================================
Generates a 6-slide PPTX following the official SIH template:
  1. Title Page
  2. Problem & Solution (Idea / Approach)
  3. Technical Approach
  4. Feasibility & Viability
  5. Impact & Benefits
  6. Research & References

Design language: dark navy (#1B2A4A) + teal accent (#3ECFB4) on light (#F0F2F5)
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Design Tokens ─────────────────────────────────────────────
NAVY       = RGBColor(0x1B, 0x2A, 0x4A)
DARK_NAVY  = RGBColor(0x12, 0x1E, 0x36)
TEAL       = RGBColor(0x3E, 0xCF, 0xB4)
LIGHT_BG   = RGBColor(0xF0, 0xF2, 0xF5)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0x8A, 0x94, 0xA6)
MID_GRAY   = RGBColor(0x5A, 0x64, 0x78)
CARD_BG    = RGBColor(0x24, 0x35, 0x56)  # Slightly lighter navy for cards
WARN_RED   = RGBColor(0xE8, 0x6B, 0x5A)
GOLD       = RGBColor(0xD4, 0xA5, 0x3C)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

FONT_TITLE   = 'Segoe UI'
FONT_BODY    = 'Segoe UI'
FONT_ACCENT  = 'Segoe UI Semibold'

# ── Helpers ───────────────────────────────────────────────────
def set_slide_bg(slide, color):
    """Set solid fill background for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text_box(slide, left, top, width, height, text, 
                 font_name=FONT_BODY, font_size=14, font_color=WHITE,
                 bold=False, alignment=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                 line_spacing=1.15):
    """Add a text box with single-style text."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.alignment = alignment
    p.space_after = Pt(0)
    p.space_before = Pt(0)
    if hasattr(p, 'line_spacing'):
        p.line_spacing = line_spacing
    tf.auto_size = None
    return txBox

def add_rich_text_box(slide, left, top, width, height, paragraphs_data,
                      anchor=MSO_ANCHOR.TOP):
    """
    Add a text box with multiple paragraphs, each with their own styling.
    paragraphs_data: list of dicts with keys: text, font_name, font_size, font_color, bold, alignment, space_after, bullet
    """
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    
    for i, pdata in enumerate(paragraphs_data):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        text = pdata.get('text', '')
        
        # Handle mixed formatting with runs
        if isinstance(text, list):
            for run_data in text:
                run = p.add_run()
                run.text = run_data.get('text', '')
                run.font.name = run_data.get('font_name', pdata.get('font_name', FONT_BODY))
                run.font.size = Pt(run_data.get('font_size', pdata.get('font_size', 14)))
                run.font.color.rgb = run_data.get('font_color', pdata.get('font_color', WHITE))
                run.font.bold = run_data.get('bold', pdata.get('bold', False))
        else:
            run = p.add_run()
            run.text = text
            run.font.name = pdata.get('font_name', FONT_BODY)
            run.font.size = Pt(pdata.get('font_size', 14))
            run.font.color.rgb = pdata.get('font_color', WHITE)
            run.font.bold = pdata.get('bold', False)
        
        p.alignment = pdata.get('alignment', PP_ALIGN.LEFT)
        p.space_after = Pt(pdata.get('space_after', 4))
        p.space_before = Pt(pdata.get('space_before', 0))
    
    return txBox

def add_rounded_card(slide, left, top, width, height, fill_color=CARD_BG,
                     border_color=None, border_width=Pt(0)):
    """Add a rounded rectangle card shape."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    return shape

def add_circle(slide, left, top, size, fill_color=NAVY, text='', 
               font_color=TEAL, font_size=18, bold=True):
    """Add a circle with text inside."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(left), Inches(top), Inches(size), Inches(size)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    if text:
        tf = shape.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.text = text
        p.font.name = FONT_ACCENT
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.bold = bold
        p.alignment = PP_ALIGN.CENTER
        tf.auto_size = None
    return shape

def add_divider_line(slide, left, top, width, color=TEAL, thickness=Pt(2)):
    """Add a thin horizontal divider."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), thickness
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_footer(slide, slide_num, bg_dark=False):
    """Add HemoVision footer + page number."""
    color = LIGHT_GRAY if not bg_dark else RGBColor(0x5A, 0x6A, 0x80)
    add_text_box(slide, 0.6, 6.9, 3, 0.4, 'HemoVision',
                 font_size=9, font_color=color)
    add_text_box(slide, 11.5, 6.9, 1.5, 0.4, f'{slide_num:02d}',
                 font_size=9, font_color=color, alignment=PP_ALIGN.RIGHT)

def add_section_tag(slide, left, top, text, color=TEAL):
    """Add a small uppercase section tag like 'THE PROBLEM'."""
    add_text_box(slide, left, top, 5, 0.35, text,
                 font_name=FONT_ACCENT, font_size=10, font_color=color,
                 bold=True)

def add_validated_badge(slide, left, top, text="VALIDATED", color=TEAL):
    """Add a small green 'VALIDATED' or 'ROADMAP' badge."""
    badge_color = TEAL if text == "VALIDATED" else GOLD
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(1.1), Inches(0.25)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = badge_color
    shape.line.fill.background()
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = FONT_ACCENT
    p.font.size = Pt(7)
    p.font.color.rgb = DARK_NAVY if text == "VALIDATED" else DARK_NAVY
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

# ══════════════════════════════════════════════════════════════
# SLIDE 1: TITLE PAGE
# ══════════════════════════════════════════════════════════════
def build_slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    set_slide_bg(slide, DARK_NAVY)
    
    # SIH Header banner
    add_text_box(slide, 0.6, 0.4, 12, 0.4, 'SMART INDIA HACKATHON 2026',
                 font_name=FONT_ACCENT, font_size=11, font_color=TEAL, bold=True)
    
    # Accent line under SIH header
    add_divider_line(slide, 0.6, 0.85, 3.5, TEAL, Pt(3))
    
    # Main title
    add_text_box(slide, 0.6, 1.2, 10, 1.2, 'HEMOVISION',
                 font_name=FONT_TITLE, font_size=54, font_color=WHITE, bold=True)
    
    # Subtitle
    add_text_box(slide, 0.6, 2.4, 10, 0.6, 'Contactless Vital-Sign Estimation from Facial Video',
                 font_name=FONT_BODY, font_size=20, font_color=LIGHT_GRAY)
    
    # Tech tags
    add_text_box(slide, 0.6, 3.1, 10, 0.4,
                 'Remote Photoplethysmography (rPPG)  |  Deep Learning  |  Clinical Validation',
                 font_name=FONT_ACCENT, font_size=11, font_color=TEAL, bold=True)
    
    # SIH metadata - using a card
    add_rounded_card(slide, 0.6, 4.0, 5.8, 2.8, CARD_BG)
    
    meta_items = [
        {'text': 'Problem Statement ID:', 'font_size': 10, 'font_color': LIGHT_GRAY, 'bold': False, 'space_after': 2},
        {'text': '«  To be assigned  »', 'font_size': 11, 'font_color': WHITE, 'bold': True, 'space_after': 8},
        {'text': 'Theme:', 'font_size': 10, 'font_color': LIGHT_GRAY, 'bold': False, 'space_after': 2},
        {'text': 'MedTech / HealthTech', 'font_size': 11, 'font_color': WHITE, 'bold': True, 'space_after': 8},
        {'text': 'PS Category:', 'font_size': 10, 'font_color': LIGHT_GRAY, 'bold': False, 'space_after': 2},
        {'text': 'Software — Student Innovation', 'font_size': 11, 'font_color': WHITE, 'bold': True, 'space_after': 8},
        {'text': 'Team Name:', 'font_size': 10, 'font_color': LIGHT_GRAY, 'bold': False, 'space_after': 2},
        {'text': '«  Your Team Name  »', 'font_size': 11, 'font_color': TEAL, 'bold': True, 'space_after': 4},
    ]
    add_rich_text_box(slide, 0.9, 4.2, 5.2, 2.4, meta_items)
    
    # Right side — key differentiator callout
    add_rounded_card(slide, 7.0, 4.0, 5.8, 2.8, fill_color=RGBColor(0x1A, 0x3A, 0x5C), 
                     border_color=TEAL, border_width=Pt(1.5))
    
    callout_items = [
        {'text': '✦  THIS IS NOT A PROTOTYPE', 'font_size': 12, 'font_color': TEAL, 'bold': True, 'space_after': 10},
        {'text': 'HemoVision is a fully built, validated system with:', 'font_size': 11, 'font_color': WHITE, 'bold': False, 'space_after': 6},
        {'text': '•  9.19 BPM MAE on 91 unseen subjects', 'font_size': 11, 'font_color': WHITE, 'bold': False, 'space_after': 3},
        {'text': '•  10 biomarkers rigorously tested & honestly reported', 'font_size': 11, 'font_color': WHITE, 'bold': False, 'space_after': 3},
        {'text': '•  3 real engineering bugs found and fixed', 'font_size': 11, 'font_color': WHITE, 'bold': False, 'space_after': 3},
        {'text': '•  Live-demo ready interactive dashboard', 'font_size': 11, 'font_color': WHITE, 'bold': False, 'space_after': 6},
        {'text': 'Every claim in this deck has a number behind it.', 'font_size': 10, 'font_color': TEAL, 'bold': True, 'space_after': 0},
    ]
    add_rich_text_box(slide, 7.3, 4.2, 5.2, 2.4, callout_items)

# ══════════════════════════════════════════════════════════════
# SLIDE 2: PROBLEM & SOLUTION (Idea / Approach)
# ══════════════════════════════════════════════════════════════
def build_slide_2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, LIGHT_BG)
    
    # Section tag
    add_section_tag(slide, 0.6, 0.4, 'T H E   P R O B L E M   &   S O L U T I O N')
    
    # Title
    add_text_box(slide, 0.6, 0.75, 12, 0.7,
                 'Vital signs usually need a wire, a cuff, or a clip',
                 font_name=FONT_TITLE, font_size=30, font_color=NAVY, bold=True)
    
    # ── Left card: Traditional Monitoring (X) ──
    add_rounded_card(slide, 0.6, 1.7, 5.8, 3.7, WHITE, border_color=RGBColor(0xDE, 0xE1, 0xE6), border_width=Pt(1))
    
    # X icon circle
    add_circle(slide, 1.0, 1.95, 0.55, fill_color=RGBColor(0xFD, 0xE8, 0xE5), text='✕',
               font_color=RGBColor(0xE8, 0x6B, 0x5A), font_size=18)
    
    add_text_box(slide, 1.0, 2.65, 5.0, 0.4, 'Traditional monitoring',
                 font_name=FONT_ACCENT, font_size=16, font_color=NAVY, bold=True)
    
    trad_items = [
        {'text': '•   Requires physical contact — cuffs, clips, wired sensors', 'font_size': 11, 'font_color': MID_GRAY, 'space_after': 5},
        {'text': '•   Equipment isn\'t always available at the point of care', 'font_size': 11, 'font_color': MID_GRAY, 'space_after': 5},
        {'text': '•   Slows down outpatient triage when volumes are high', 'font_size': 11, 'font_color': MID_GRAY, 'space_after': 5},
        {'text': '•   Uncomfortable for continuous / long-term monitoring', 'font_size': 11, 'font_color': MID_GRAY, 'space_after': 5},
        {'text': '•   Not feasible for remote / telemedicine screening', 'font_size': 11, 'font_color': MID_GRAY, 'space_after': 3},
    ]
    add_rich_text_box(slide, 1.0, 3.15, 5.0, 2.0, trad_items)
    
    # ── Right card: Our Approach (✓) ──
    add_rounded_card(slide, 6.9, 1.7, 5.8, 3.7, NAVY)
    
    # Check icon circle
    add_circle(slide, 7.3, 1.95, 0.55, fill_color=RGBColor(0x2A, 0x5A, 0x4A), text='✓',
               font_color=TEAL, font_size=18)
    
    add_text_box(slide, 7.3, 2.65, 5.0, 0.4, 'Our approach: HemoVision',
                 font_name=FONT_ACCENT, font_size=16, font_color=WHITE, bold=True)
    
    add_validated_badge(slide, 10.8, 2.7, "VALIDATED")
    
    our_items = [
        {'text': '•   Just a face on ordinary video — no sensor to attach', 'font_size': 11, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 5},
        {'text': '•   Works with any standard camera — webcam or phone', 'font_size': 11, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 5},
        {'text': '•   Deep learning reconstructs the pulse from color alone', 'font_size': 11, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 5},
        {'text': '•   9.19 BPM MAE — validated on 91 unseen subjects', 'font_size': 11, 'font_color': TEAL, 'bold': True, 'space_after': 5},
        {'text': '•   Live-demo ready: real-time dashboard with waveforms', 'font_size': 11, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
    ]
    add_rich_text_box(slide, 7.3, 3.15, 5.0, 2.0, our_items)
    
    # ── Bottom: How it works pipeline (4 steps) ──
    add_divider_line(slide, 0.6, 5.65, 12.1, TEAL, Pt(1))
    
    add_section_tag(slide, 0.6, 5.8, 'H O W   I T   W O R K S')
    
    steps = [
        ('1', 'Face on camera', 'Ordinary RGB video'),
        ('2', 'Colour signal', '8 ROIs × 3 channels'),
        ('3', 'Pulse waveform', 'Bounded TCN model'),
        ('4', 'Heart rate', 'Welch PSD peak-picking'),
    ]
    
    step_w = 2.7
    gap = 0.35
    start_x = 0.6
    
    for i, (num, title, desc) in enumerate(steps):
        x = start_x + i * (step_w + gap)
        
        # Card
        card_color = TEAL if i == 3 else NAVY
        add_rounded_card(slide, x, 6.1, step_w, 0.7, card_color)
        
        # Step number circle
        add_circle(slide, x + 0.15, 6.15, 0.35, fill_color=DARK_NAVY if i < 3 else RGBColor(0x2A, 0xA8, 0x90),
                   text=num, font_color=TEAL if i < 3 else DARK_NAVY, font_size=12)
        
        # Title + desc
        add_text_box(slide, x + 0.6, 6.12, step_w - 0.8, 0.3, title,
                     font_size=10, font_color=WHITE if i < 3 else DARK_NAVY, bold=True)
        add_text_box(slide, x + 0.6, 6.4, step_w - 0.8, 0.3, desc,
                     font_size=8, font_color=LIGHT_GRAY if i < 3 else RGBColor(0x15, 0x20, 0x30))
        
        # Arrow between steps
        if i < 3:
            add_text_box(slide, x + step_w + 0.05, 6.2, 0.3, 0.4, '→',
                         font_size=14, font_color=TEAL, alignment=PP_ALIGN.CENTER)
    
    add_footer(slide, 2)

# ══════════════════════════════════════════════════════════════
# SLIDE 3: TECHNICAL APPROACH
# ══════════════════════════════════════════════════════════════
def build_slide_3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, LIGHT_BG)
    
    add_section_tag(slide, 0.6, 0.4, 'T E C H N I C A L   A P P R O A C H')
    add_text_box(slide, 0.6, 0.75, 12, 0.7,
                 'From raw video to a clinically-validated heart rate',
                 font_name=FONT_TITLE, font_size=28, font_color=NAVY, bold=True)
    
    # ── Left column: Architecture pipeline ──
    add_rounded_card(slide, 0.6, 1.65, 6.0, 5.2, WHITE, border_color=RGBColor(0xDE, 0xE1, 0xE6), border_width=Pt(1))
    
    add_text_box(slide, 0.9, 1.8, 5.4, 0.3, 'Pipeline Architecture',
                 font_name=FONT_ACCENT, font_size=13, font_color=NAVY, bold=True)
    add_validated_badge(slide, 4.8, 1.85, "VALIDATED")
    
    pipeline_steps = [
        ('①', 'Face Detection & Tracking', 'MediaPipe Face Mesh — 468 3D landmarks per frame at real-time FPS'),
        ('②', '8-ROI Spatial Pooling', 'Forehead, cheeks, nose, chin, temples → polygon mask → mean RGB per ROI'),
        ('③', 'Signal Preprocessing', '2nd-order zero-phase Butterworth bandpass (0.65–3.25 Hz), z-score normalization'),
        ('④', 'Bounded Depthwise TCN', '24-ch input → 6 dilated blocks (1,2,4,8,16,32) → tanh-bounded BVP waveform'),
        ('⑤', 'Welch PSD Peak-Picking', 'Zero-padded FFT, 4096-point, peak inside 39–195 BPM band → final HR'),
    ]
    
    for i, (icon, title, desc) in enumerate(pipeline_steps):
        y = 2.25 + i * 0.85
        add_circle(slide, 1.0, y, 0.4, NAVY, icon, TEAL, 14)
        add_text_box(slide, 1.55, y, 4.8, 0.3, title,
                     font_size=11, font_color=NAVY, bold=True)
        add_text_box(slide, 1.55, y + 0.28, 4.8, 0.4, desc,
                     font_size=9, font_color=MID_GRAY)
        if i < 4:
            # Vertical connector
            add_divider_line(slide, 1.18, y + 0.45, 0.04, TEAL, Pt(1))
    
    # ── Right column: Hardware & Software ──
    add_rounded_card(slide, 7.0, 1.65, 5.8, 2.4, NAVY)
    
    add_text_box(slide, 7.3, 1.8, 5.2, 0.3, 'Hardware & Software Stack',
                 font_name=FONT_ACCENT, font_size=13, font_color=TEAL, bold=True)
    
    hw_items = [
        {'text': '•  1080p Webcam (Logitech C920) — captures subtle skin color changes', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  Python / NumPy / SciPy — signal processing & bandpass filtering', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  OpenCV / MediaPipe — 468-point face mesh, 8-ROI extraction', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  PyTorch — Depthwise TCN training (Pearson correlation loss)', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  Streamlit + OpenCV — real-time clinical telemetry dashboard', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
    ]
    add_rich_text_box(slide, 7.3, 2.2, 5.2, 1.8, hw_items)
    
    # ── Right bottom: Model Details card ──
    add_rounded_card(slide, 7.0, 4.25, 5.8, 2.6, WHITE, border_color=RGBColor(0xDE, 0xE1, 0xE6), border_width=Pt(1))
    
    add_text_box(slide, 7.3, 4.35, 5.2, 0.3, 'Model Architecture Details',
                 font_name=FONT_ACCENT, font_size=13, font_color=NAVY, bold=True)
    
    model_items = [
        {'text': '•  Input: 24 channels (8 ROIs × 3 RGB) × 900 frames', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Backbone: Depthwise-separable 1D convolutions', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Dilation schedule: [1, 2, 4, 8, 16, 32] — 63-frame receptive field', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Bounded waveform head: Conv1D + tanh → output ∈ [-1, +1]', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Loss: Negative Pearson r (waveform) + SNR regularizer', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Training: 80 epochs, lr=3e-4, batch=2 × 8 grad-accum', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
        {'text': '•  Evaluation: Subject-disjoint 70/15/15 split (no data leakage)', 'font_size': 9.5, 'font_color': MID_GRAY, 'space_after': 3},
    ]
    add_rich_text_box(slide, 7.3, 4.7, 5.2, 2.0, model_items)
    
    add_footer(slide, 3)

# ══════════════════════════════════════════════════════════════
# SLIDE 4: FEASIBILITY & VIABILITY
# ══════════════════════════════════════════════════════════════
def build_slide_4(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, LIGHT_BG)
    
    add_section_tag(slide, 0.6, 0.4, 'F E A S I B I L I T Y   &   V I A B I L I T Y')
    add_text_box(slide, 0.6, 0.75, 12, 0.7,
                 'What works, what doesn\'t, and what we proved along the way',
                 font_name=FONT_TITLE, font_size=28, font_color=NAVY, bold=True)
    
    # ── Top row: 4 KPI cards ──
    kpis = [
        ('9.19', 'BPM MAE', '91 unseen subjects\n523 held-out clips'),
        ('1.33', 'BPM MAE', 'Verified demo clips\n(live-demo ready)'),
        ('52%', 'Error reduction', 'vs. 19.1 BPM MAE\nbaseline model'),
        ('10', 'Biomarkers', 'Rigorously tested\n& honestly reported'),
    ]
    
    for i, (big_num, label, detail) in enumerate(kpis):
        x = 0.6 + i * 3.15
        add_rounded_card(slide, x, 1.6, 2.85, 1.6, NAVY)
        add_text_box(slide, x + 0.2, 1.7, 2.45, 0.6, big_num,
                     font_name=FONT_TITLE, font_size=30, font_color=TEAL, bold=True,
                     alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + 0.2, 2.2, 2.45, 0.3, label,
                     font_size=10, font_color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + 0.2, 2.55, 2.45, 0.6, detail,
                     font_size=8, font_color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # ── Middle: Engineering Integrity (3 bugs) ──
    add_text_box(slide, 0.6, 3.45, 5, 0.3, 'Engineering Integrity: 3 Real Bugs Found & Fixed',
                 font_name=FONT_ACCENT, font_size=12, font_color=NAVY, bold=True)
    add_validated_badge(slide, 5.5, 3.5, "VALIDATED")
    
    bugs = [
        ('1', 'ECG T-wave double-counting',
         'Post-exercise T-wave spikes made the reference HR detector double-count beats. Fixed with adaptive refractory period (250 ms) + dynamic thresholds. Cut baseline error by >50%.'),
        ('2', 'FFT quantization rounding',
         'Unpadded FFT forced estimates onto a coarse 7.5 BPM grid. Fixed with 4096-point zero-padding. Identified via a precision self-test.'),
        ('3', 'Unstable checkpoint selection',
         'One noisy, unrepresentative training step was silently saved as "best." Added 3-epoch consistency checks + stability filters.'),
    ]
    
    for i, (num, title, desc) in enumerate(bugs):
        y = 3.9 + i * 0.9
        add_circle(slide, 0.6, y, 0.4, NAVY, num, TEAL, 12)
        add_text_box(slide, 1.15, y, 5.5, 0.25, title,
                     font_size=11, font_color=NAVY, bold=True)
        add_text_box(slide, 1.15, y + 0.25, 5.5, 0.55, desc,
                     font_size=8.5, font_color=MID_GRAY)
        if i < 2:
            add_divider_line(slide, 0.6, y + 0.82, 6.0, RGBColor(0xDE, 0xE1, 0xE6), Pt(1))
    
    # ── Right: Biomarker Audit / Honest negative result ──
    add_rounded_card(slide, 7.0, 3.4, 5.8, 3.5, NAVY)
    
    add_text_box(slide, 7.3, 3.55, 5.2, 0.3, 'Biomarker Audit: Honest Negative Result',
                 font_name=FONT_ACCENT, font_size=12, font_color=TEAL, bold=True)
    
    audit_items = [
        {'text': 'Heart Rate:  r ≈ 0.28,  MAE = 9.19 BPM  ✓ WORKS', 'font_size': 10, 'font_color': TEAL, 'bold': True, 'space_after': 6},
        {'text': 'All other 10 biomarkers tested (SpO₂, RR, BP, Hb, HbA1c, cholesterol, BMI, rigidity, stress):  r < 0.3 on held-out test.', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 6},
        {'text': 'WHY IT MATTERS:', 'font_size': 9, 'font_color': TEAL, 'bold': True, 'space_after': 3},
        {'text': '•  None beat a "guess the average" baseline', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  Confirmed after 5 independent fix attempts', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 3},
        {'text': '•  Hemoglobin "correlation" exposed as demographic confound (age/sex/BMI predict better than video)', 'font_size': 9.5, 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'space_after': 6},
        {'text': 'ROOT CAUSE: RGB cameras lack infrared wavelength — a physical sensor limitation, not a modelling shortfall.', 'font_size': 9, 'font_color': WARN_RED, 'bold': True, 'space_after': 6},
        {'text': 'Reporting this honestly is the point. Most hackathon entries would claim it works.', 'font_size': 9, 'font_color': LIGHT_GRAY, 'space_after': 0},
    ]
    add_rich_text_box(slide, 7.3, 3.9, 5.2, 2.8, audit_items)
    
    add_footer(slide, 4)

# ══════════════════════════════════════════════════════════════
# SLIDE 5: IMPACT & BENEFITS
# ══════════════════════════════════════════════════════════════
def build_slide_5(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, LIGHT_BG)
    
    add_section_tag(slide, 0.6, 0.4, 'I M P A C T   &   B E N E F I T S')
    add_text_box(slide, 0.6, 0.75, 12, 0.7,
                 'What this unlocks — and what comes next',
                 font_name=FONT_TITLE, font_size=28, font_color=NAVY, bold=True)
    
    # ── Left: Direct Impact cards ──
    impacts = [
        ('Instant Access', 'Users check heart rate immediately using their phone or webcam — no extra hardware.'),
        ('Zero Discomfort', 'Completely touchless: no skin irritation from tight straps or sticky patches.'),
        ('Early Detection', 'Detects subtle heart-rate changes from home, helping users know when to consult a doctor.'),
        ('Mass Deployment', 'Runs on any device with a camera. No per-patient hardware cost.'),
    ]
    
    for i, (title, desc) in enumerate(impacts):
        y = 1.6 + i * 1.15
        add_rounded_card(slide, 0.6, y, 5.8, 0.95, WHITE, border_color=RGBColor(0xDE, 0xE1, 0xE6), border_width=Pt(1))
        add_circle(slide, 0.85, y + 0.15, 0.45, NAVY, str(i+1), TEAL, 14)
        add_text_box(slide, 1.5, y + 0.1, 4.6, 0.3, title,
                     font_size=13, font_color=NAVY, bold=True)
        add_text_box(slide, 1.5, y + 0.42, 4.6, 0.45, desc,
                     font_size=9.5, font_color=MID_GRAY)
    
    add_validated_badge(slide, 5.2, 1.65, "VALIDATED")
    
    # ── Right: Roadmap (3 directions) ──
    add_text_box(slide, 7.0, 1.5, 5.5, 0.3, 'Three Concrete Next Steps — Grounded in Evidence',
                 font_name=FONT_ACCENT, font_size=12, font_color=NAVY, bold=True)
    add_validated_badge(slide, 11.5, 1.55, "ROADMAP")
    
    roadmap = [
        ('BP', 'Blood Pressure via Pulse Morphology',
         'Analyse the pulse wave\'s timing and shape (transit time / morphology) instead of a single averaged value — a technique this project hasn\'t yet tried.',
         NAVY),
        ('♡', 'Stress via HRV',
         'Heart-rate variability (RMSSD, SDNN, LF/HF) computed from the waveform that already works — an indirect but literature-backed path to stress estimation.',
         NAVY),
        ('◎', 'Multi-Wavelength Hardware',
         'Pursuing a research collaboration for infrared / multi-wavelength imaging (660 nm + 940 nm NIR) — the missing piece for hemoglobin and SpO₂.',
         NAVY),
    ]
    
    for i, (icon, title, desc, color) in enumerate(roadmap):
        y = 2.0 + i * 1.65
        add_rounded_card(slide, 7.0, y, 5.8, 1.45, WHITE, border_color=RGBColor(0xDE, 0xE1, 0xE6), border_width=Pt(1))
        add_circle(slide, 7.25, y + 0.15, 0.5, color, icon, TEAL, 14)
        add_text_box(slide, 7.9, y + 0.12, 4.6, 0.3, title,
                     font_size=12, font_color=NAVY, bold=True)
        add_text_box(slide, 7.9, y + 0.42, 4.6, 0.9, desc,
                     font_size=9, font_color=MID_GRAY)
    
    # ── Bottom: Economic benefit strip ──
    add_rounded_card(slide, 0.6, 6.35, 12.1, 0.5, NAVY)
    
    econ_items = [
        {'text': [
            {'text': '💰  Economic Impact:  ', 'font_color': TEAL, 'bold': True, 'font_size': 10},
            {'text': 'Zero hardware overhead — hospitals upgrade using existing webcams.  |  Eliminates per-patient sensor costs.  |  B2B SaaS model for recurring revenue.', 'font_color': RGBColor(0xCC, 0xD5, 0xE0), 'font_size': 9},
        ], 'space_after': 0},
    ]
    add_rich_text_box(slide, 0.9, 6.42, 11.5, 0.4, econ_items)
    
    add_footer(slide, 5)

# ══════════════════════════════════════════════════════════════
# SLIDE 6: RESEARCH & REFERENCES
# ══════════════════════════════════════════════════════════════
def build_slide_6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_NAVY)
    
    add_section_tag(slide, 0.6, 0.4, 'S U M M A R Y   &   R E F E R E N C E S')
    
    add_text_box(slide, 0.6, 0.8, 12, 0.8,
                 'A working system — and an honest map of what\'s next',
                 font_name=FONT_TITLE, font_size=30, font_color=TEAL, bold=True)
    
    # ── Top: 3 summary cards ──
    summary_cards = [
        ('✓', 'Heart rate', 'working, live-demoed\n~9 BPM error', CARD_BG),
        ('10', 'biomarkers', 'rigorously tested\nhonestly reported', CARD_BG),
        ('3', 'real bugs', 'found and fixed\nalong the way', CARD_BG),
    ]
    
    for i, (big, title, desc, bg) in enumerate(summary_cards):
        x = 0.6 + i * 4.2
        add_rounded_card(slide, x, 1.65, 3.8, 1.5, bg)
        add_text_box(slide, x + 0.3, 1.75, 3.2, 0.5, big,
                     font_name=FONT_TITLE, font_size=28, font_color=TEAL, bold=True,
                     alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + 0.3, 2.2, 3.2, 0.3, title,
                     font_size=12, font_color=WHITE, bold=False, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, x + 0.3, 2.5, 3.2, 0.5, desc,
                     font_size=9, font_color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # ── References ──
    add_divider_line(slide, 0.6, 3.4, 12.1, TEAL, Pt(1))
    
    add_text_box(slide, 0.6, 3.55, 5, 0.3, 'Key Academic References',
                 font_name=FONT_ACCENT, font_size=12, font_color=TEAL, bold=True)
    
    refs = [
        '[1]  Wang, W., et al. "Algorithmic Principles of Remote PPG." IEEE TBME, 2017.',
        '[2]  Yu, Z., et al. "PhysNet: Video-Based Physiological Measurement." BMVC, 2019.',
        '[3]  Liu, X., et al. "Multi-Task Temporal Shift for rPPG." CVPR, 2020.',
        '[4]  Lu, H., et al. "Dual-GAN for Video-Based Vital Signs." IEEE JBHI, 2021.',
        '[5]  Yu, Z., et al. "PhysFormer: Temporal Transformer for rPPG." CVPR, 2022.',
        '[6]  Gideon, J., et al. "The Way to My Heart Is Through Contrastive Learning." ICCV, 2021.',
        '[7]  Nowara, E. M., et al. "The Effect of Skin Tone on rPPG." NeurIPS, 2020.',
        '[8]  Mironenko, Y., et al. "Pulsed CNN for Remote HR Estimation." MICCAI, 2020.',
    ]
    
    ref_items = []
    for ref in refs:
        ref_items.append({
            'text': ref,
            'font_size': 8.5,
            'font_color': LIGHT_GRAY,
            'space_after': 3,
        })
    add_rich_text_box(slide, 0.6, 3.9, 12.1, 2.5, ref_items)
    
    # ── Bottom: Thank you + branding ──
    add_divider_line(slide, 0.6, 6.3, 12.1, TEAL, Pt(1))
    
    add_text_box(slide, 0.6, 6.45, 5, 0.5, 'Thank you',
                 font_name=FONT_TITLE, font_size=24, font_color=WHITE, bold=True)
    
    add_text_box(slide, 6.0, 6.5, 6.7, 0.35,
                 'Every claim in this deck has a number behind it.  |  Live demo available.',
                 font_size=10, font_color=TEAL, bold=True, alignment=PP_ALIGN.RIGHT)
    
    add_footer(slide, 6, bg_dark=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    
    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)
    
    output_path = 'HEMOVISION_SIH_2026.pptx'
    prs.save(output_path)
    print(f'[OK] Saved {output_path} ({os.path.getsize(output_path)} bytes)')
    print(f'   {len(prs.slides)} slides')

if __name__ == '__main__':
    main()
