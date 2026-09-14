"""
PDF report generator for OCR Readiness Platform.
Uses reportlab to produce a clean single-page report.
"""

import io
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from factors import DISPLAY_NAMES, WEIGHTS


# Brand colours
NAVY    = colors.HexColor("#1A2B4A")
TEAL    = colors.HexColor("#00C4B4")
LIGHT   = colors.HexColor("#F5F7FA")
WARN    = colors.HexColor("#F59E0B")
GOOD    = colors.HexColor("#10B981")
BAD     = colors.HexColor("#EF4444")
MID     = colors.HexColor("#F59E0B")


def _status_color(score: float) -> colors.Color:
    if score >= 81:
        return GOOD
    elif score >= 61:
        return TEAL
    elif score >= 41:
        return WARN
    return BAD


def generate_pdf_report(
    image_name: str,
    factor_results: Dict[str, Any],
    ocr_readiness: float,
    ocr_confidence: Optional[float],
    recommendations: List[str],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=20,
        textColor=NAVY,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=12,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        spaceAfter=4,
    )

    story = []

    # ── Header ──────────────────────────────────
    story.append(Paragraph("OCR Readiness Evaluation Report", title_style))
    story.append(Paragraph(
        f"Image: <b>{image_name}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; "
        f"SNLP Department",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=10))

    # ── OCR Readiness Score ──────────────────────
    status = factor_results.get("ocr_readiness_status", "—")
    story.append(Paragraph("OCR Readiness Score", section_style))

    score_data = [
        [
            Paragraph(f"<font size=28 color='#{NAVY.hexval()[2:]}' name='Helvetica-Bold'>"
                      f"{ocr_readiness}/100</font>", body_style),
            Paragraph(
                f"<font size=14 color='#{_status_color(ocr_readiness).hexval()[2:]}' "
                f"name='Helvetica-Bold'>{status}</font>",
                body_style,
            ),
        ]
    ]
    score_table = Table(score_data, colWidths=["50%", "50%"])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT]),
        ("BOX", (0, 0), (-1, -1), 1, TEAL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))

    if ocr_confidence is not None:
        diff = abs(ocr_readiness - ocr_confidence)
        story.append(Paragraph(
            f"Tesseract OCR Confidence: <b>{ocr_confidence:.1f}%</b> &nbsp;|&nbsp; "
            f"Predicted vs Actual gap: <b>{diff:.1f}%</b>",
            body_style,
        ))

    # ── Factor Scores Table ──────────────────────
    story.append(Paragraph("Factor Scores", section_style))

    header = ["Factor", "Score", "Status", "Weight", "Description"]
    table_data = [header]
    for key, display in DISPLAY_NAMES.items():
        r = factor_results.get(key, {})
        sc = r.get("score", "—")
        st = r.get("status", "—")
        wt = f"{int(WEIGHTS[key]*100)}%"
        desc = r.get("description", "")[:60] + ("…" if len(r.get("description","")) > 60 else "")
        table_data.append([display, f"{sc}", st, wt, desc])

    col_widths = [3.2*cm, 1.5*cm, 2.0*cm, 1.5*cm, 8.0*cm]
    factor_table = Table(table_data, colWidths=col_widths)

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (3, -1), "CENTER"),
    ])

    # Colour-code score column
    for i, key in enumerate(DISPLAY_NAMES.keys(), start=1):
        sc = factor_results.get(key, {}).get("score", 50)
        c = _status_color(float(sc))
        ts.add("TEXTCOLOR", (1, i), (1, i), c)
        ts.add("FONTNAME",  (1, i), (1, i), "Helvetica-Bold")
        ts.add("TEXTCOLOR", (2, i), (2, i), c)

    factor_table.setStyle(ts)
    story.append(factor_table)

    # ── Recommendations ──────────────────────────
    story.append(Paragraph("Recommendations", section_style))
    for rec in recommendations:
        # strip markdown bold markers for PDF
        clean = rec.replace("**", "").replace("🔧", "▸").replace("✅", "✓")
        story.append(Paragraph(clean, body_style))

    # ── Footer ───────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "OCR Readiness Evaluation Platform · SNLP Department · "
        "Team: Yash (Lead), Vivek, Mansi, Krish",
        ParagraphStyle("Footer", parent=body_style, textColor=colors.grey,
                       alignment=TA_CENTER, fontSize=7),
    ))

    doc.build(story)
    return buf.getvalue()


def generate_batch_pdf_report(
    batch_data: List[Dict[str, Any]],
) -> bytes:
    """
    Generates a multi-page PDF report summarizing a batch of analyzed images.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=18,
        textColor=NAVY,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=11,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#333333"),
        spaceAfter=3,
    )

    story = []

    # ── Header ──────────────────────────────────
    story.append(Paragraph("OCR Readiness Batch Analysis Report", title_style))
    story.append(Paragraph(
        f"Total Images Analyzed: <b>{len(batch_data)}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; "
        f"SNLP Department",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=8))

    # ── Executive Comparison Table ──────────────
    story.append(Paragraph("Batch Comparison Matrix", section_style))

    headers = ["Image Name", "Overall Score", "Status", "Tesseract Conf"] + [DISPLAY_NAMES[k] for k in DISPLAY_NAMES.keys()]
    table_rows = [headers]

    for item in batch_data:
        name = item.get("image_name", "Image")[:16]
        res = item.get("results", {})
        sc = res.get("ocr_readiness_score", 0)
        st = res.get("ocr_readiness_status", "—")
        conf = item.get("ocr_conf")
        conf_str = f"{conf:.1f}%" if conf is not None else "N/A"
        
        row = [name, f"{sc:.1f}", st, conf_str]
        for k in DISPLAY_NAMES.keys():
            factor_sc = res.get(k, {}).get("score", 0)
            row.append(f"{factor_sc}")
        table_rows.append(row)

    col_w = [2.5*cm, 1.3*cm, 1.4*cm, 1.5*cm] + [1.1*cm]*10
    batch_table = Table(table_rows, colWidths=col_w)

    t_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    batch_table.setStyle(t_style)
    story.append(batch_table)
    story.append(Spacer(1, 10))

    # ── Per-Image Details ────────────────────────
    story.append(Paragraph("Detailed Per-Image Results", section_style))

    for idx, item in enumerate(batch_data, start=1):
        name = item.get("image_name", f"Image {idx}")
        res = item.get("results", {})
        sc = res.get("ocr_readiness_score", 0)
        st = res.get("ocr_readiness_status", "—")
        recs = item.get("recs", [])

        story.append(Paragraph(f"<b>{idx}. {name}</b> — Score: <font color='#{_status_color(sc).hexval()[2:]}'><b>{sc}/100 ({st})</b></font>", section_style))
        
        # Recommendations summary
        top_recs = [r.replace("**", "").replace("🔧", "▸").replace("✅", "✓") for r in recs[:2]]
        for r in top_recs:
            story.append(Paragraph(r, body_style))
        story.append(Spacer(1, 4))

    # ── Footer ───────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "OCR Readiness Evaluation Platform · Batch Report · SNLP Department",
        ParagraphStyle("Footer", parent=body_style, textColor=colors.grey,
                       alignment=TA_CENTER, fontSize=7),
    ))

    doc.build(story)
    return buf.getvalue()
