"""
Generate Banasthali Vidyapith Synopsis Report (.docx)
OCR Readiness Evaluation Platform - Session 2026-2027

Formatting Guidelines:
- A4 Page Size (210 x 297 mm)
- Margins: Left=1.5", Right=1.0", Top=1.0", Bottom=1.0"
- Times New Roman throughout
- Body: 12pt, 1.5 line spacing, Justified
- Chapter Headings: 16pt Bold Centered
- Division Headings: 14pt Bold Left
- Sub-division Headings: 12pt Bold Left
- Target Document Length: 11-13 Pages (Max 10-14 Pages)
"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Banasthali_Synopsis_OCR_Readiness_Platform.docx")
PRIMARY_OUTPUT = os.path.join(os.path.dirname(__file__), "Synopsis_OCR_Readiness_Platform.docx")

doc = Document()

# ============================================================
# PAGE SETUP & GLOBAL STYLES
# ============================================================
section = doc.sections[0]
section.page_width = Cm(21.0)   # A4
section.page_height = Cm(29.7)  # A4
section.top_margin = Inches(1.0)
section.bottom_margin = Inches(1.0)
section.left_margin = Inches(1.5)
section.right_margin = Inches(1.0)

# Default style
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.space_before = Pt(0)

# Force Times New Roman for East Asian text
rPr = style.element.get_or_add_rPr()
rFonts = rPr.find(qn('w:rFonts'))
if rFonts is None:
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>')
    rPr.append(rFonts)


def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def add_page_number(doc):
    sec = doc.sections[0]
    footer = sec.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    run._r.append(fldChar1)
    run2 = p.add_run()
    run2.font.name = 'Times New Roman'
    run2.font.size = Pt(10)
    instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
    run2._r.append(instrText)
    run3 = p.add_run()
    run3.font.name = 'Times New Roman'
    run3.font.size = Pt(10)
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    run3._r.append(fldChar2)


def add_title_centered(text, size=14, bold=True, space_before=0, space_after=6, color=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    return p


def add_chapter_heading(chapter_num, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"CHAPTER {chapter_num}\n{title}")
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    return p


def add_division_heading(number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"{number} {title}")
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    return p


def add_subdivision_heading(number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"{number} {title}")
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p


def add_body(text, space_after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p


def add_numbered_item(number, text, indent=0.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(indent)
    run = p.add_run(f"{number}. {text}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p


def add_bullet(text, indent=0.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(indent)
    run = p.add_run(f"\u2022  {text}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p


def add_table_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.italic = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    return p


def add_figure_placeholder(caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run("[ Figure / Architectural Diagram Placeholder ]")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = RGBColor(100, 100, 100)

    pc = doc.add_paragraph()
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_before = Pt(1)
    pc.paragraph_format.space_after = Pt(4)
    pc.paragraph_format.line_spacing = 1.15
    rc = pc.add_run(caption)
    rc.bold = True
    rc.italic = True
    rc.font.name = 'Times New Roman'
    rc.font.size = Pt(10)


def make_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(hdr_cells[i], "1A2B4A")
        set_cell_margins(hdr_cells[i], top=40, bottom=40, left=80, right=80)

    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(str(val))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
            set_cell_margins(row_cells[c_idx], top=30, bottom=30, left=60, right=60)
        if r_idx % 2 == 1:
            for c_idx in range(len(headers)):
                set_cell_shading(row_cells[c_idx], "F5F7FA")

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    return table


# ============================================================
# TITLE PAGE (Page 1)
# ============================================================
add_title_centered("Synopsis on", size=16, bold=True, space_before=18, space_after=14)

add_title_centered(
    "OCR READINESS EVALUATION PLATFORM:\nA MULTI-FACTOR QUALITY ASSESSMENT, DIAGNOSTIC,\nAND SAFE REFINEMENT SYSTEM FOR BILINGUAL\nENGLISH AND DEVANAGARI DOCUMENTS",
    size=13, bold=True, space_after=18
)

add_title_centered("Submitted by", size=11, bold=False, space_before=10, space_after=4)
add_title_centered("Yash (Team Lead), Vivek, Mansi, Krish, Tanusha", size=12, bold=True, space_after=14)

add_title_centered("For the award of the degree of", size=11, bold=False, space_before=4, space_after=4)
add_title_centered("Bachelor of Technology (Computer Science and Engineering)", size=12, bold=False, space_after=14)

add_title_centered("Under the Supervision of", size=11, bold=False, space_before=6, space_after=4)
add_title_centered("Department Faculty / Project Supervisor", size=12, bold=True, space_after=20)

add_title_centered("Department of Computer Science", size=13, bold=True, space_after=2)
add_title_centered("Banasthali Vidyapith", size=13, bold=True, space_after=2)
add_title_centered("Banasthali \u2013 304022, Rajasthan, India", size=12, bold=False, space_after=2)
add_title_centered("Academic Session: 2026\u20132027", size=12, bold=True, space_after=0)

doc.add_page_break()

# ============================================================
# TABLE OF CONTENTS (Page 2)
# ============================================================
add_title_centered("Table of Contents (Software Development Project)", size=13, bold=True, space_before=0, space_after=10)

toc_data = [
    ("1.", "Introduction", "3\u20134"),
    ("", "   1.1 Background", "3"),
    ("", "   1.2 Problem Statement", "3"),
    ("", "   1.3 Motivation", "3"),
    ("", "   1.4 Objectives", "3"),
    ("", "   1.5 Scope of the Project", "4"),
    ("", "   1.6 Significance of the Project", "4"),
    ("", "   1.7 Overview of the Proposed Solution", "4"),
    ("", "   1.8 Organization of the Synopsis", "4"),
    ("2.", "System Study", "5\u20136"),
    ("", "   2.1 Organization & Context", "5"),
    ("", "   2.2 Existing System", "5"),
    ("", "   2.3 Limitations of Existing System", "5"),
    ("", "   2.4 Proposed System", "5"),
    ("", "   2.5 Advantages of Proposed System", "6"),
    ("", "   2.6 Comparison of Existing and Proposed System", "6"),
    ("3.", "Requirement Analysis", "7\u20138"),
    ("", "   3.1 Requirement Specification (Functional, Non-Functional)", "7"),
    ("", "   3.2 Hardware and Software Requirements", "7"),
    ("", "   3.3 Technology Stack", "7"),
    ("", "   3.4 System Architecture", "8"),
    ("", "   3.5 System Workflow & Factor Weight Distribution", "8"),
    ("4.", "Feasibility Study", "9"),
    ("", "   4.1 Technical, Economic, Operational, Schedule Feasibility", "9"),
    ("5.", "Work Plan", "10\u201311"),
    ("", "   5.1 Methodology & Project Phases", "10"),
    ("", "   5.2 Development Timeline & Work Breakdown Matrix", "10"),
    ("", "   5.3 Testing, Validation, Current Status & Future Work", "11"),
    ("6.", "References", "12\u201313"),
]

toc_table = doc.add_table(rows=len(toc_data), cols=3)
toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (sno, title, pno) in enumerate(toc_data):
    for ci, val in enumerate([sno, title, pno]):
        cell = toc_table.rows[i].cells[ci]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 1 else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(val)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        if sno and ci <= 1:
            run.bold = True
        set_cell_margins(cell, top=20, bottom=20, left=40, right=40)
    toc_table.rows[i].cells[0].width = Inches(0.5)
    toc_table.rows[i].cells[1].width = Inches(4.7)
    toc_table.rows[i].cells[2].width = Inches(0.8)

doc.add_page_break()

# ============================================================
# CHAPTER 1: INTRODUCTION
# ============================================================
add_chapter_heading("1", "INTRODUCTION")

add_division_heading("1.1", "Background")
add_body(
    "Optical Character Recognition (OCR) forms the foundational backbone of contemporary document digitization "
    "pipelines, enabling automated indexing, archival retrieval, administrative governance, and natural language "
    "processing. In Indian institutional repositories, document collections encompass bilingual printed corpora "
    "combining English (Latin script) and Hindi (Devanagari script), frequently scanned from legacy books, court records, "
    "and administrative forms. While modern OCR engines such as Tesseract 5.x exhibit high accuracy on pristine 300 DPI "
    "scans, their performance degrades sharply on real-world degraded inputs containing defocus blur, uneven illumination, "
    "sensor noise, rotational skew, low resolution, and typographical stroke erosion."
)

add_division_heading("1.2", "Problem Statement")
add_body(
    "In conventional workflows, OCR engines are deployed as unguided black boxes. Document images are ingested "
    "indiscriminately without verifying typographical legibility or script structural integrity, leading to three major "
    "failures: (1) lack of pre-ingestion quality screening, leading to heavy computational waste on unreadable scans; "
    "(2) total failure on complex Indic/Devanagari scripts where headline (Shirorekha) breaks and modifier (Matra) "
    "erosions corrupt word segmentation; and (3) absence of explainable visual diagnostics and safe image refinement, "
    "leaving operators without actionable guidance on scan recapture."
)

add_division_heading("1.3", "Motivation")
add_body(
    "The primary motivation of this project is to shift document digitization from reactive post-OCR error correction "
    "to proactive, explainable quality governance. In high-throughput digitization environments, human proofreading is "
    "cost-prohibitive. By screening images at capture, predicting OCR accuracy via an objective Readiness Score (0\u2013100), "
    "providing intuitive visual diagnostic heatmaps, and applying mathematically safeguarded image refinement, "
    "organizations can drastically reduce recognition failures and streamline digitization workflows."
)

add_division_heading("1.4", "Objectives")
add_body("The core objectives of the OCR Readiness Evaluation Platform are:")
add_numbered_item(1, "Formulate a 10-factor analytical quality suite covering generic optical properties (Blur, Noise, Resolution, Contrast, Stroke Width, Text Density, Connected Component Stability, Skew) and Devanagari script features (Matra Continuity, Zone Integrity).")
add_numbered_item(2, "Synthesize an empirically weighted composite OCR Readiness Score (0\u2013100) classified into standard quality tiers (Poor, Average, Good, Excellent) that strongly correlates with OCR accuracy.")
add_numbered_item(3, "Engineer six pixel-level visual diagnostic overlays explaining defect causes to scanning operators.")
add_numbered_item(4, "Build a non-destructive safe image refinement engine enforcing invariant safety rules to prevent document degradation.")
add_numbered_item(5, "Implement a fault-tolerant distributed microservice architecture with instant local fallback for zero downtime.")
add_numbered_item(6, "Provide live Tesseract 5.x OCR bilingual cross-validation, word confidence tracking, CSV logging, and automated A4 PDF report generation.")

add_division_heading("1.5", "Scope of the Project")
add_body(
    "The project supports single and multi-page scanned document images (PNG, JPEG, TIFF, BMP, WebP) in English and "
    "Hindi. The software provides single-image deep diagnostics, interactive sub-region cropping, batch evaluation "
    "dashboards, and automated PDF export. It serves as an intelligent pre-processing and quality assurance layer rather "
    "than replacing the downstream recognition engine."
)

add_division_heading("1.6", "Significance of the Project")
add_body(
    "Unlike generic no-reference quality metrics (PSNR, BRISQUE) designed for photographic scenes, this platform is "
    "typographically tuned and Indic-script-aware. Furthermore, its invariant-based refinement policy prevents the "
    "destructive artifacts typical of blind global image filters, ensuring only provably safe enhancements are applied."
)

add_division_heading("1.7", "Overview of the Proposed Solution")
add_body(
    "The system operates as an interactive Streamlit web dashboard backed by distributed microservices and a local fallback "
    "computation suite. Ingested images are evaluated across 10 factors, synthesized into a readiness score, visually "
    "diagnosed via overlays, safely refined if requested, and cross-validated against bilingual Tesseract OCR."
)
add_figure_placeholder("Figure 1.1: High-Level System Workflow of OCR Readiness Evaluation Platform")

add_division_heading("1.8", "Organization of the Synopsis")
add_body(
    "The remainder of this synopsis is organized as follows: Chapter 2 presents the System Study comparing existing and "
    "proposed approaches. Chapter 3 provides the Requirement Analysis, system architecture, and workflows. Chapter 4 details "
    "the Feasibility Study. Chapter 5 outlines the Work Plan, testing, and current status. Chapter 6 provides formal References."
)

# ============================================================
# CHAPTER 2: SYSTEM STUDY
# ============================================================
add_chapter_heading("2", "SYSTEM STUDY")

add_division_heading("2.1", "Organization & Context")
add_body(
    "This project is developed in the Department of Computer Science, Banasthali Vidyapith, Rajasthan, for the academic "
    "session 2026\u20132027 under the Speech and Natural Language Processing (SNLP) research domain by Yash (Team Lead & "
    "Architect), Vivek, Mansi, Krish, and Tanusha."
)

add_division_heading("2.2", "Existing System")
add_body(
    "In current document processing pipelines, OCR execution is performed reactively without pre-ingestion screening. "
    "Current practices rely on: (1) blind ingestion directly into OCR models; (2) generic photographic metrics (BRISQUE/NIQE) "
    "that ignore typography; or (3) brute-force post-OCR confidence thresholding, which discards documents only after full inference."
)

add_division_heading("2.3", "Limitations of Existing System")
add_bullet("High Computational Waste: Running deep LSTM OCR models on unreadable documents wastes heavy GPU/CPU cycles.")
add_bullet("No Actionable Feedback: Post-OCR rejection offers no explanation of optical defects to guide recapture.")
add_bullet("Destructive Pre-processing: Blind global thresholding frequently erodes thin strokes and breaks characters.")
add_bullet("Devanagari Incompatibility: Conventional metrics overlook broken Shirorekha headlines and eroded Matras.")

add_division_heading("2.4", "Proposed System")
add_body(
    "The OCR Readiness Evaluation Platform introduces a script-aware quality evaluation, diagnostic, and safe refinement "
    "framework. Key components include: (1) 10-Factor Analytical Engine including Matra Continuity and Zone Integrity; "
    "(2) Visual Diagnostic Overlays (6 interactive heatmaps); (3) Safe Refinement Engine with invariant verification; "
    "(4) Distributed Microservice Architecture with local fallback; and (5) Dual Tesseract Validation and PDF Analytics."
)

add_division_heading("2.5", "Advantages of Proposed System")
add_bullet("Proactive Defect Screening: Rejects or flags unreadable images prior to expensive recognition inference.")
add_bullet("Explainable Root-Cause Analysis: Highlights specific defects (blur, skew, noise) with visual heatmaps.")
add_bullet("Guaranteed Non-Destructive Refinement: Refuses transformations that degrade non-target quality factors.")
add_bullet("Native Indic Script Protection: Accurately assesses Devanagari headlines and vertical modifier zones.")
add_bullet("Operational Resilience: Seamless fallback guarantees 100% availability even if network services fail.")

add_division_heading("2.6", "Comparison of Existing and Proposed System")
add_table_caption("Table 2.1: Comparison Between Existing Systems and Proposed OCR Readiness Platform")
make_table(
    ["Dimension", "Conventional Workflow", "Generic Quality Tools", "Proposed OCR Readiness Platform"],
    [
        ["Evaluation Timing", "Post-OCR or None", "Pre-OCR (Natural Scenes)", "Pre-OCR (Typographic & Script-Calibrated)"],
        ["Assessment Factors", "0 (Blind)", "1 (Generic Index)", "10 Calibrated Dimensional Scores (0-100)"],
        ["Devanagari Awareness", "None", "None", "Native Matra Continuity & 4-Zone Integrity"],
        ["Visual Diagnostics", "None", "None", "6 Interactive Diagnostic Heatmaps & Vectors"],
        ["Image Refinement", "Blind global filters", "Fixed presets", "Safe Multi-Candidate Search + Invariant Rules"],
        ["Degradation Safeguard", "None", "None", "Strict Safety Policy (Non-target drop \u2264 5.0 pts)"],
        ["OCR Validation", "Post-hoc text review", "None", "Live Dual Tesseract 5.x + Spatial Error Heatmap"],
        ["System Architecture", "Monolithic", "Standalone script", "Distributed HTTP Microservices + 100% Fallback"],
        ["Reporting & Storage", "Console output", "None", "Single/Batch A4 PDF Reports + CSV Analytics"],
    ],
    col_widths=[1.3, 1.4, 1.4, 1.9]
)

# ============================================================
# CHAPTER 3: REQUIREMENT ANALYSIS
# ============================================================
add_chapter_heading("3", "REQUIREMENT ANALYSIS")

add_division_heading("3.1", "Requirement Specification")

add_subdivision_heading("3.1.1", "Functional Requirements")
add_table_caption("Table 3.1: Key Functional Requirements Specification")
make_table(
    ["Req. ID", "Module", "Functional Description"],
    [
        ["FR-01", "Ingestion & Crop", "Upload single/batch images (PNG, JPEG, TIFF, BMP, WebP) with interactive cropping."],
        ["FR-02", "Optical Quality", "Compute Blur (Laplacian var), Noise (MAD sigma), Resolution (MP curve), Contrast (Otsu)."],
        ["FR-03", "Structural Quality", "Evaluate Stroke Width (Distance Transform), Text Density (Gaussian), Skew (Hough Lines)."],
        ["FR-04", "Indic Script Engine", "Analyze Devanagari Matra Continuity (Shirorekha SCS/MVS) and 4-Zone Structural Integrity."],
        ["FR-05", "Readiness Synthesis", "Calculate weighted composite score (0\u2013100) and map to 4 standardized quality tiers."],
        ["FR-06", "Visual Diagnostics", "Render 6 interactive overlays: Blur Heatmap, Noise Map, Skew Lines, Density, Stroke, CC."],
        ["FR-07", "Safe Refinement", "Generate candidate transformations; apply only if safety policy invariants are satisfied."],
        ["FR-08", "OCR Cross-Validation", "Run bilingual Tesseract 5.x, extract word confidence, language mix, spatial error maps."],
        ["FR-09", "Microservice Sync", "Query distributed team APIs via HTTP with automatic 10-second local fallback switch."],
        ["FR-10", "Storage & Reporting", "Append session records to results.csv and export multi-page A4 PDF reports via ReportLab."],
    ],
    col_widths=[0.7, 1.3, 4.0]
)

add_subdivision_heading("3.1.2", "Non-Functional & System Requirements")
add_bullet("Performance: 10-factor evaluation completes in < 1.5 seconds on standard CPU hardware for 2.0 MP images.")
add_bullet("Reliability: Network timeouts or remote microservice downtime trigger seamless local execution fallback.")
add_bullet("Safety Invariant: Candidate image rejected if target factor gain < +0.5 or non-target factor drop > 5.0.")
add_bullet("Usability: Interactive Streamlit web interface with radar charts, formula explanations, and PDF export.")

add_division_heading("3.2", "Hardware and Software Requirements")
add_table_caption("Table 3.2: Minimum and Recommended System Specifications")
make_table(
    ["Component", "Minimum Requirement", "Recommended Specification"],
    [
        ["Processor", "Intel Core i3 / AMD Ryzen 3 (2.0 GHz)", "Intel Core i5 / AMD Ryzen 5 or higher"],
        ["System RAM", "4 GB Physical RAM", "8 GB RAM or higher"],
        ["Disk Storage", "500 MB free disk space", "1 GB (including virtualenv, tessdata, test images)"],
        ["Operating System", "Windows 10/11 (64-bit), Linux, macOS", "Windows 11 (64-bit) / Ubuntu 22.04 LTS"],
        ["Python Runtime", "Python 3.10.x", "Python 3.11.x (64-bit)"],
        ["OCR Engine", "Tesseract OCR v5.0.0+", "Tesseract OCR v5.3.0+ (eng + hin traineddata)"],
    ],
    col_widths=[1.5, 2.2, 2.3]
)

add_division_heading("3.3", "Technology Stack")
add_table_caption("Table 3.3: Technology Stack Specification")
make_table(
    ["Layer / Role", "Technology & Libraries", "Purpose & Architectural Role"],
    [
        ["Programming Language", "Python 3.10+", "Algorithmic computation, backend pipeline, microservice coordination"],
        ["Web Dashboard", "Streamlit", "Reactive multi-page user interface, interactive cropping canvas"],
        ["Computer Vision", "OpenCV (opencv-python)", "Morphological filters, distance transforms, Canny, Otsu, Hough Lines"],
        ["Numerical Computation", "NumPy & SciPy", "Matrix operations, projection profiles, MAD sigma, Gaussian modeling"],
        ["OCR Engine", "PyTesseract & Tesseract 5.x", "Bilingual text extraction, word-level confidence and bounding boxes"],
        ["Visualization", "Plotly & Matplotlib", "10-factor radar charts, histogram distributions, spatial error maps"],
        ["Reporting & Storage", "ReportLab & Pandas", "Automated A4 PDF report generation, CSV historical logging"],
    ],
    col_widths=[1.4, 1.8, 2.8]
)

add_division_heading("3.4", "System Architecture")
add_body(
    "The platform utilizes a 4-tier distributed microservice architecture with local fallback: (1) Presentation Tier "
    "(Streamlit UI); (2) Orchestration Tier (dynamic IP routing via config.json, microservice client with 10s timeout, "
    "fallback supervisor, safe refinement manager); (3) Computation Tier (Team microservices: Mansi on :8000 for Blur/Contrast, "
    "Vivek on :8001 for Stroke/Density, Krish on :8002 for Matra/Zone, Tanusha on :9001 for CC/Skew, Yash for Noise/Resolution, "
    "with 100% local fallback in factors.py); and (4) Storage Tier (results.csv and ReportLab PDF compiler)."
)
add_figure_placeholder("Figure 3.1: Distributed System Architecture and Fallback Topology")

add_division_heading("3.5", "System Workflow & Factor Weight Distribution")
add_body(
    "The workflow follows 6 phases: (1) Ingestion & Interactive Crop; (2) Factor Dispatch to APIs / Fallback; "
    "(3) 10-Factor Algorithmic Evaluation; (4) Weighted Synthesis; (5) Visual Diagnostics & Safe Refinement; and "
    "(6) Tesseract Cross-Validation and PDF Export. The aggregate OCR Readiness Score is synthesized as:"
)
add_body("       Readiness Score = \u2211 (Score_i \u00d7 Weight_i)   for i = 1 to 10")

add_table_caption("Table 3.4: OCR Readiness Factor Weight Formulation")
make_table(
    ["Factor Name", "Owner", "Weight", "Algorithmic Method & Core Mathematical Basis"],
    [
        ["Blur Score", "Mansi", "15%", "Modified Laplacian variance on smoothed grayscale with log-sigmoidal map"],
        ["Noise Score", "Yash", "12%", "Otsu background isolation, Median Absolute Deviation (\u03c3_MAD), impulse fraction"],
        ["Resolution Score", "Yash", "12%", "2D Megapixel rational polynomial saturation curve calibrated for 300 DPI"],
        ["Contrast Score", "Mansi", "12%", "Otsu inter-class variance and normalized luminance spread"],
        ["CC Stability Score", "Tanusha", "10%", "Connected component area Coefficient of Variation and fragment ratio"],
        ["Skew Penalty Score", "Tanusha", "10%", "Probabilistic Hough Line Transform and minimum area bounding box angle"],
        ["Stroke Width Score", "Vivek", "8%", "Euclidean Distance Transform along Medial Axis Skeleton vs. char height"],
        ["Text Density Score", "Vivek", "8%", "Morphological gradient ink coverage against Gaussian optimal band (20%)"],
        ["Matra Continuity", "Krish", "8%", "Devanagari Shirorekha Segment Continuity (SCS) and Matra Visibility (MVS)"],
        ["Zone Integrity", "Krish", "5%", "4-Zone vertical projection profile and upper/lower modifier structural health"],
    ],
    col_widths=[1.3, 0.8, 0.6, 3.3]
)

# ============================================================
# CHAPTER 4: FEASIBILITY STUDY
# ============================================================
add_chapter_heading("4", "FEASIBILITY STUDY")

add_division_heading("4.1", "Technical Feasibility")
add_body(
    "The project is highly technically feasible. The 10 factor algorithms are based on established mathematical principles "
    "(spatial derivatives, distance transforms, projection profiles, Hough transforms) and operate deterministically without "
    "requiring GPU hardware. The underlying libraries (OpenCV, NumPy, Streamlit) are mature and stable. Dual-execution design "
    "(microservices + local fallback) ensures complete technical resilience, with full evaluation completing in under 1.5 seconds."
)

add_division_heading("4.2", "Economic Feasibility")
add_body(
    "The platform is 100% economically feasible. It is built entirely on open-source software under permissive licenses "
    "(MIT, BSD, Apache 2.0), incurring zero software licensing fees. It runs on commodity hardware without cloud GPU expenses. "
    "By preventing failed OCR runs and guiding scanning operators, it delivers substantial operational cost savings."
)

add_division_heading("4.3", "Operational Feasibility")
add_body(
    "The platform provides an intuitive web interface requiring no programming skills. Scanning operators receive instant visual "
    "heatmaps and plain-language guidance on capture corrections. Standardized CSV logs and PDF reports integrate seamlessly "
    "into institutional archival workflows."
)

add_division_heading("4.4", "Schedule Feasibility")
add_body(
    "The project was executed across a 16-week academic schedule. Modular separation of factor algorithms, microservices, "
    "refinement logic, and UI enabled parallel development across all five team members, completing all deliverables on time."
)

add_division_heading("4.5", "Overall Feasibility Summary")
add_table_caption("Table 4.1: Overall Feasibility Assessment Matrix")
make_table(
    ["Feasibility Dimension", "Evaluation Criteria", "Outcome & Assessment"],
    [
        ["Technical", "Algorithm complexity, runtime speed, dependency maturity", "Fully Feasible (Sub-1.5s execution, 100% fallback)"],
        ["Economic", "Development and deployment software/infrastructure costs", "Fully Feasible (100% Open-source, zero license cost)"],
        ["Operational", "User adoption, operator workflow integration, reporting", "Fully Feasible (No-code UI, visual heatmaps, PDF export)"],
        ["Schedule", "Milestone delivery within 16-week university timeline", "Fully Feasible (All phases completed on schedule)"],
    ],
    col_widths=[1.5, 2.2, 2.3]
)

# ============================================================
# CHAPTER 5: WORK PLAN
# ============================================================
add_chapter_heading("5", "WORK PLAN")

add_division_heading("5.1", "Methodology & Project Phases")
add_body(
    "The project adopted an Iterative Component-Based Agile Methodology. The 16-week lifecycle progressed through six phases: "
    "Phase I (Research & Formulation), Phase II (Algorithmic Implementation), Phase III (Microservices & Fallback Engine), "
    "Phase IV (Diagnostic Overlays & Safe Refinement), Phase V (Frontend UI & Tesseract OCR), and Phase VI (Validation & Reporting)."
)

add_division_heading("5.2", "Development Timeline & Work Breakdown")
add_table_caption("Table 5.1: Project Development Timeline (Academic Session 2026\u20132027)")
make_table(
    ["Phase", "Timeframe", "Key Deliverables & Milestones", "Status"],
    [
        ["Phase I: Formulation", "Weeks 1\u20133", "Mathematical factor definitions, scoring curves, weight matrix", "Completed"],
        ["Phase II: Core Algorithms", "Weeks 4\u20136", "10 calibrated factor algorithms in factors.py with 33 test images", "Completed"],
        ["Phase III: Microservices", "Weeks 7\u20138", "Distributed HTTP team endpoints, config manager, fallback engine", "Completed"],
        ["Phase IV: Refinement & UI", "Weeks 9\u201311", "6 diagnostic overlays in overlay.py, safe refinement in refinement.py", "Completed"],
        ["Phase V: OCR & Dashboard", "Weeks 12\u201313", "Streamlit UI, interactive cropping, bilingual Tesseract validation", "Completed"],
        ["Phase VI: Validation & Docs", "Weeks 14\u201316", "ReportLab PDF export, CSV logging, correlation analysis, synopsis", "Completed"],
    ],
    col_widths=[1.4, 1.0, 2.6, 1.0]
)

add_table_caption("Table 5.2: Work Breakdown and Module Ownership Matrix")
make_table(
    ["Team Member", "Role", "Assigned Modules", "Technical Responsibilities"],
    [
        ["Yash (Team Lead)", "Architect & Integrator", "app.py, run.py, storage.py, report.py, refinement.py, overlay.py", "UI, orchestration, safe refinement engine, overlays, PDF/CSV engines, Noise & Resolution factors"],
        ["Vivek", "Core Developer", "factors.py, Team API", "Stroke Width Regularity and Text Density layout algorithms"],
        ["Mansi", "Core Developer", "factors.py, Team API", "Blur / Sharpness and Contrast / Dynamic Range algorithms"],
        ["Krish", "Core Developer", "factors.py, Team API", "Indic Matra Continuity and 4-Zone Structural Integrity algorithms"],
        ["Tanusha", "Core Developer", "factors.py, Team API", "Connected Component Stability and Hough Skew algorithms"],
    ],
    col_widths=[1.3, 1.2, 1.5, 2.0]
)

add_division_heading("5.3", "Testing and Validation")
add_body(
    "The platform underwent four levels of validation: (1) Unit Testing of Factor Algorithms across 33 dedicated test "
    "images covering all 10 degradation modes; (2) Safety Invariant Testing evaluating > 200 refinement candidates against "
    "the safety policy; (3) Fault-Tolerance Stress Testing verifying 100% seamless fallback during API disconnects; and "
    "(4) Empirical OCR Correlation Validation confirming strong positive Pearson correlation between the Readiness Score and "
    "actual Tesseract word recognition confidence."
)

add_division_heading("5.4", "Current Status & Completed Deliverables")
add_body(
    "All core modules are fully implemented, calibrated, and operational: (1) 10-Factor Engine in factors.py; (2) Distributed "
    "APIs and Fallback in api_integration.py and config_manager.py; (3) Safe Refinement Engine in refinement.py; (4) 6 Visual "
    "Overlays in overlay.py; (5) Bilingual Tesseract Integration with Spatial Error Maps; (6) PDF and CSV Reporting in report.py "
    "and storage.py; and (7) Multi-page Web UI in app.py."
)

add_division_heading("5.5", "Future Enhancements")
add_body(
    "Post-synopsis roadmap items include: (1) Vision Transformer (ViT) lightweight neural quality backend; (2) Multi-engine "
    "consensus validation (EasyOCR, PaddleOCR); (3) Expansion to other Indic scripts (Bengali, Gurmukhi, Gujarati); (4) Docker "
    "containerization; and (5) Real-time mobile camera pre-capture quality evaluation."
)

# ============================================================
# CHAPTER 6: REFERENCES
# ============================================================
add_chapter_heading("6", "REFERENCES")

references = [
    "Bansal V. and Sinha R.M.K. (2002) \u2018Integrating knowledge sources in Devanagari text recognition\u2019, IEEE Transactions on Systems, Man, and Cybernetics, Part A: Systems and Humans, Vol. 32, No. 4, pp. 443\u2013454.",
    "Bhowmik S., Sarkar R., Das N. and Subhash C. (2018) \u2018Devanagari text recognition: A survey of the state of the art\u2019, Artificial Intelligence Review, Vol. 50, No. 4, pp. 579\u2013619.",
    "Bradski G. (2000) \u2018The OpenCV Library\u2019, Dr. Dobb\u2019s Journal of Software Tools, Vol. 25, No. 11, pp. 120\u2013125.",
    "Haralick R.M., Shanmugam K. and Dinstein I. (1973) \u2018Textural features for image classification\u2019, IEEE Transactions on Systems, Man, and Cybernetics, Vol. SMC-3, No. 6, pp. 610\u2013621.",
    "Hough P.V.C. (1962) \u2018Method and means for recognizing complex patterns\u2019, U.S. Patent 3,069,654.",
    "Mittal A., Moorthy A.K. and Bovik A.C. (2012) \u2018No-reference image quality assessment in the spatial domain\u2019, IEEE Transactions on Image Processing, Vol. 21, No. 12, pp. 4695\u20134708.",
    "Otsu N. (1979) \u2018A threshold selection method from gray-level histograms\u2019, IEEE Transactions on Systems, Man, and Cybernetics, Vol. 9, No. 1, pp. 62\u201366.",
    "Pal U. and Chaudhuri B.B. (2004) \u2018Indian script character recognition: a survey\u2019, Pattern Recognition, Vol. 37, No. 9, pp. 1887\u20131899.",
    "Sauvola J. and Pietik\u00e4inen M. (2000) \u2018Adaptive document image binarization\u2019, Pattern Recognition, Vol. 33, No. 2, pp. 225\u2013236.",
    "Smith R. (2007) \u2018An Overview of the Tesseract OCR Engine\u2019, Proc. Ninth International Conference on Document Analysis and Recognition (ICDAR), Curitiba, Brazil, pp. 629\u2013633.",
    "Tsai C.M. and Lee H.J. (2002) \u2018Binarization of low-contrast document images through dynamic thresholding\u2019, Pattern Recognition Letters, Vol. 23, No. 4, pp. 433\u2013444.",
    "Wang Z., Bovik A.C., Sheikh H.R. and Simoncelli E.P. (2004) \u2018Image quality assessment: from error visibility to structural similarity\u2019, IEEE Transactions on Image Processing, Vol. 13, No. 4, pp. 600\u2013612.",
    "Google Tesseract OCR (2024) \u2018Tesseract OCR Open Source Engine Documentation\u2019, Available at: https://github.com/tesseract-ocr/tesseract [Accessed: September 2026].",
    "Streamlit Inc. (2024) \u2018Streamlit Framework Documentation\u2019, Available at: https://docs.streamlit.io/ [Accessed: September 2026].",
]

for i, ref in enumerate(references, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)
    run = p.add_run(f"[{i}]  {ref}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

# ============================================================
# PAGE NUMBERING
# ============================================================
add_page_number(doc)

# ============================================================
# SAVE
# ============================================================
doc.save(OUTPUT_PATH)
print(f"Document generated successfully: {OUTPUT_PATH}")
try:
    doc.save(PRIMARY_OUTPUT)
    print(f"Also saved to: {PRIMARY_OUTPUT}")
except Exception as e:
    print(f"Note: Could not overwrite {PRIMARY_OUTPUT} directly (currently open/locked). Saved to {OUTPUT_PATH}.")

