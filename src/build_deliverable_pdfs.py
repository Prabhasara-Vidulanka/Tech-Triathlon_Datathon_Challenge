"""Build print-ready PDFs for the written Datathon deliverables."""

from __future__ import annotations

import re
import subprocess
import tempfile
from html import escape
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    XPreformatted,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "final_package" / "reports"
NAVY = colors.HexColor("#172B46")
TEAL = colors.HexColor("#087E83")
MUTED = colors.HexColor("#587080")
PALE = colors.HexColor("#D8E7EB")

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

STYLES = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=9),
    "deck": ParagraphStyle("deck", fontName="Arial", fontSize=9, leading=13, textColor=MUTED, spaceAfter=16),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=15, spaceAfter=6, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.3, leading=14, textColor=NAVY, spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.3, leading=14, textColor=NAVY, leftIndent=11, firstLineIndent=-8, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Arial", fontSize=9, leading=14, textColor=NAVY, leftIndent=12, spaceBefore=5, spaceAfter=10),
}


def inline(value: str) -> str:
    value = escape(value)
    value = re.sub(r"`([^`]+)`", r'<font color="#087E83">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(PALE)
    canvas.line(47, 44, width - 47, 44)
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(47, 30, "WAYPOINT GROUP  /  TECH-TRIATHLON DATATHON")
    canvas.drawRightString(width - 47, 30, f"{doc.page}")
    canvas.restoreState()


def md_story(source: Path, subtitle: str):
    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    buffer = []
    bullets = []
    code = []
    in_code = False

    def flush_para():
        if buffer:
            story.append(Paragraph(inline(" ".join(buffer)), STYLES["body"]))
            buffer.clear()

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(inline(x), STYLES["bullet"])) for x in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=16, bulletFontName="Arial", bulletFontSize=7))
            story.append(Spacer(1, 5))
            bullets.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            flush_bullets()
            if in_code:
                story.append(XPreformatted(escape("\n".join(code)), STYLES["code"]))
                code.clear()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
        elif stripped.startswith("# "):
            flush_para()
            title = stripped[2:]
            story.extend([Paragraph(escape(title), STYLES["title"]), Paragraph(escape(subtitle), STYLES["deck"]), HRFlowable(width="100%", thickness=1.2, color=TEAL), Spacer(1, 8)])
        elif stripped.startswith("## "):
            flush_para()
            flush_bullets()
            story.append(Paragraph(escape(stripped[3:]), STYLES["h2"]))
        elif stripped.startswith("- "):
            flush_para()
            bullets.append(stripped[2:])
        elif not stripped:
            flush_para()
            flush_bullets()
        else:
            buffer.append(stripped)
    flush_para()
    flush_bullets()
    return story


def write_text_pdf(stem: str, subtitle: str):
    source = REPORTS / f"{stem}.md"
    output = REPORTS / f"{stem}.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=47, leftMargin=47, topMargin=48, bottomMargin=58,
                            title=source.read_text(encoding="utf-8").splitlines()[0][2:], author="Waypoint Group")
    doc.build(md_story(source, subtitle), onFirstPage=footer, onLaterPages=footer)
    return output


DIAGRAMS = [
    (
        "TASK 1  /  SERVICE TIME + LATENESS",
        "Planning inputs are checked and transformed before chronological validation and two saved predictors.",
        """digraph G {
          rankdir=TB; nodesep=.22; ranksep=.38; graph [bgcolor="transparent", pad=.15];
          node [shape=box, style="rounded,filled", color="#B4C9D4", fillcolor="#EFF5F8", fontname="Arial", fontsize=10, margin=".12,.08"];
          edge [color="#6F91A1", arrowsize=.6];
          orders [label="Orders + route plans"]; refs [label="Outlet, vehicle +\ndistrict references"]; context [label="Calendar, traffic +\nroad conditions"];
          joins [label="Schema + one-to-one\njoin checks", fillcolor="#DDEFF0"];
          labels [label="Training-only\nlabel construction"]; features [label="Plan-time\nfeature pipeline"];
          folds [label="41-day chronological\nvalidation folds", fillcolor="#DDEFF0"];
          service [label="GPU XGBoost\nservice regressor", fillcolor="#E9E6F7"];
          late [label="GPU XGBoost\nlateness classifier", fillcolor="#E9E6F7"];
          serviceout [label="Nonnegative service\nminutes"]; lateout [label="Bounded late\nprobability"];
          csv [label="submission_task1.csv", fillcolor="#DFF1E8", color="#9ACAB1"];
          {rank=same; orders; refs; context}; {rank=same; labels; features}; {rank=same; service; late}; {rank=same; serviceout; lateout};
          orders->joins; refs->joins; context->joins; joins->labels; joins->features; labels->folds; features->folds;
          folds->service; folds->late; service->serviceout; late->lateout; serviceout->csv; lateout->csv;
        }""",
    ),
    (
        "TASK 2A  /  WEEKLY DEPOT DEMAND",
        "The forecast uses requested-week demand, future-known calendar data, and recursive ten-week backtests.",
        """digraph G {
          rankdir=TB; nodesep=.22; ranksep=.34; graph [bgcolor="transparent", pad=.15];
          node [shape=box, style="rounded,filled", color="#B4C9D4", fillcolor="#EFF5F8", fontname="Arial", fontsize=10, margin=".12,.08"];
          edge [color="#6F91A1", arrowsize=.6];
          train [label="Training orders"]; test [label="Task 1 test orders"]; calendar [label="Future-known calendar"];
          combine [label="Deduplicate + combine\norder history", fillcolor="#DDEFF0"];
          week [label="Requested ISO week\nby depot + brand"]; calfeat [label="Weekly calendar\nfeatures"];
          lags [label="Shifted lags, rolling\nstats + trends"];
          backtest [label="Three rolling ten-week\nbacktests", fillcolor="#DDEFF0"];
          total [label="Total volume\nboosting model", fillcolor="#E9E6F7"];
          chilled [label="Fresh chilled\nboosting model", fillcolor="#E9E6F7"];
          recurse [label="Recursive ten-week\nforecast"]; constraints [label="Nonnegative + chilled\nvolume constraints"];
          csv [label="submission_task2a.csv", fillcolor="#DFF1E8", color="#9ACAB1"];
          {rank=same; train; test; calendar}; {rank=same; week; calfeat}; {rank=same; total; chilled};
          train->combine; test->combine; combine->week; calendar->calfeat; week->lags; calfeat->lags; lags->backtest;
          backtest->total; backtest->chilled; total->recurse; chilled->recurse; recurse->constraints; constraints->csv;
        }""",
    ),
    (
        "TASK 2B  /  ALLOCATION + DEPLOYMENT CONCEPT",
        "Allocation is independently validated. Saved predictive models could later support planning decisions.",
        """digraph G {
          rankdir=TB; nodesep=.28; ranksep=.42; graph [bgcolor="transparent", pad=.15];
          node [shape=box, style="rounded,filled", color="#B4C9D4", fillcolor="#EFF5F8", fontname="Arial", fontsize=10, margin=".12,.08"];
          edge [color="#6F91A1", arrowsize=.6];
          orders [label="Peak-day orders"]; fleet [label="Available fleet"]; standards [label="Travel + handling\nstandards"];
          prep [label="Constraint\npreparation", fillcolor="#DDEFF0"];
          cold [label="Exact refrigerated\nMILP", fillcolor="#E9E6F7"];
          ambient [label="Ambient capacity +\ntime packing", fillcolor="#E9E6F7"];
          policy [label="Fairness-first policy\ncomparison"]; check [label="Official feasibility\nvalidator"];
          csv [label="submission_task2b.csv\n+ written policy", fillcolor="#DFF1E8", color="#9ACAB1"];
          saved [label="Saved Task 1 + 2A\nmodels"]; service [label="Proposed planning\nservice"]; dispatcher [label="Dispatcher estimates +\ncapacity planning", fillcolor="#DFF1E8", color="#9ACAB1"];
          {rank=same; orders; fleet; standards; saved}; {rank=same; cold; ambient; service};
          orders->prep; fleet->prep; standards->prep; prep->cold; prep->ambient; cold->policy; ambient->policy;
          policy->check; check->csv; saved->service; service->dispatcher; csv->dispatcher [style=dashed, label="optional decision support", fontsize=8];
        }""",
    ),
]


def write_architecture_pdf():
    output = REPORTS / "architecture_diagrams.pdf"
    width, height = A4
    result = pymupdf.open()
    with tempfile.TemporaryDirectory() as scratch:
        for index, (title, note, dot) in enumerate(DIAGRAMS, 1):
            dot_path = Path(scratch) / f"diagram_{index}.dot"
            pdf_path = Path(scratch) / f"diagram_{index}.pdf"
            dot_path.write_text(dot, encoding="utf-8")
            subprocess.run(["dot", "-Tpdf", str(dot_path), "-o", str(pdf_path)], check=True)
            diagram = pymupdf.open(pdf_path)
            page = result.new_page(width=width, height=height)
            page.draw_rect(pymupdf.Rect(0, 0, width, height), color=None, fill=(1, 1, 1))
            page.insert_text((45, 54), "WAYPOINT GROUP  /  DATATHON ARCHITECTURE", fontname="helv", fontsize=9, color=(.34, .44, .50))
            page.insert_text((45, 91), title, fontname="hebo", fontsize=20, color=(.09, .17, .27))
            page.draw_line((45, 108), (width - 45, 108), color=(.03, .49, .51), width=1.5)
            page.insert_textbox(pymupdf.Rect(45, 119, width - 45, 151), note, fontname="helv", fontsize=10, color=(.25, .35, .42))
            box = pymupdf.Rect(38, 154, width - 38, height - 76)
            page.show_pdf_page(box, diagram, 0, keep_proportion=True)
            page.draw_line((45, height - 47), (width - 45, height - 47), color=(.85, .91, .93), width=.8)
            footer_note = "Proposed deployment only; Datathon integration with the Hackathon app is not required." if index == 3 else "Source: final_package/reports/architecture_diagrams.md"
            page.insert_text((45, height - 30), footer_note, fontname="helv", fontsize=8, color=(.34, .44, .50))
            page.insert_text((width - 56, height - 30), str(index), fontname="helv", fontsize=8, color=(.34, .44, .50))
            diagram.close()
        result.set_metadata({"title": "Datathon Architecture Diagrams", "author": "Waypoint Group"})
        result.save(output, garbage=4, deflate=True)
        result.close()
    return output


if __name__ == "__main__":
    outputs = [
        write_text_pdf("data_preprocessing", "Data preparation, label construction, features, validation and limitations"),
        write_architecture_pdf(),
        write_text_pdf("task2b_prioritization_policy", "Peak-day fleet allocation  /  Scenario S1"),
        write_text_pdf("ai_tool_disclosure", "Transparent account of AI assistance and local model development"),
    ]
    for item in outputs:
        print(item)
