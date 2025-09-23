# streamlit run app.py
# -*- coding: utf-8 -*-
"""
Conversation Report (P.2 IEP) — Streamlit App
- อัปโหลด Excel/CSV → map คอลัมน์ → สร้างรายงานเหมือนต้นฉบับ 100%
- ส่งออกได้: PDF (รายคนต่อหน้า), Canva Bulk CSV, PPLX (layout JSON)

วิธีรัน:
1) pip install -r requirements.txt
2) streamlit run app.py

หมายเหตุเรื่องฟอนต์ภาษาไทย:
- ถ้าต้องการให้ PDF แสดงภาษาไทยสมบูรณ์ ให้เตรียมไฟล์ฟอนต์ .ttf (เช่น NotoSansThai-Regular.ttf)
- อัปโหลดฟอนต์ในแอป (Sidebar) ระบบจะลงทะเบียนฟอนต์และใช้เรนเดอร์อัตโนมัติ
"""
from __future__ import annotations
import io
import json
from typing import Dict, List, Any, Optional

import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm

# --------------------------- UI CONFIG ---------------------------
st.set_page_config(page_title="Conversation Report (P.2 IEP)", layout="wide")

# --------------------------- Helpers -----------------------------
CANONICAL_FIELDS = [
    "no", "studentId", "name", "semester1", "semester2", "total", "grade",
    "idea", "pronunciation", "preparedness", "confidence"
]

AUTO_MAP = {
    # order
    "no": "no", "ลำดับ": "no", "เลขที่": "no",
    # id
    "student id": "studentId", "sid": "studentId", "รหัสนักเรียน": "studentId", "เลขประจำตัว": "studentId",
    # name
    "name": "name", "ชื่อ": "name", "ชื่อนักเรียน": "name",
    # sem1/sem2
    "semester1": "semester1", "semester 1": "semester1", "sem1": "semester1", "เทอม1": "semester1",
    "semester2": "semester2", "semester 2": "semester2", "sem2": "semester2", "เทอม2": "semester2",
    # total/grade
    "total": "total", "รวม": "total",
    "grade": "grade", "เกรด": "grade",
    # detailed skills
    "idea": "idea", "ความคิด": "idea",
    "pronunciation": "pronunciation", "ออกเสียง": "pronunciation",
    "preparedness": "preparedness", "ความพร้อม": "preparedness",
    "confidence": "confidence", "ความมั่นใจ": "confidence",
}

def auto_header_to_canonical(h: str) -> Optional[str]:
    return AUTO_MAP.get(h.strip().lower())

def to_number(v):
    if v is None or v == "":
        return None
    try:
        s = str(v).replace(",", "").strip()
        return float(s)
    except Exception:
        return None

def compute_total_and_grade(row: Dict[str, Any], S_min: float) -> Dict[str, Any]:
    s1 = to_number(row.get("semester1")) or 0.0
    s2 = to_number(row.get("semester2")) or 0.0
    total = to_number(row.get("total"))
    if total is None:
        total = s1 + s2
    grade = row.get("grade")
    if not grade:
        grade = "S" if total >= S_min else "U"
    row.update({"semester1": s1, "semester2": s2, "total": total, "grade": grade})
    return row

def fill_sem_from_skills(row: Dict[str, Any]) -> Dict[str, Any]:
    skills = ["idea", "pronunciation", "preparedness", "confidence"]
    if any(to_number(row.get(k)) is not None for k in skills):
        s = sum(to_number(row.get(k)) or 0 for k in skills)
        # สมมติสกิลรวมคิดเป็นเทอมเดียว (สูงสุด 50)
        row["semester1"] = row.get("semester1") or min(max(s, 0), 50)
    return row

# --------------------------- Sidebar Options ---------------------
st.sidebar.header("ตัวเลือกเลย์เอาต์ & เกณฑ์")
title = st.sidebar.text_input("Title", "Record of Test Result for Conversation")
subtitle = st.sidebar.text_input("Subtitle", "Semester 1, 2025  Grade 2/5  Intensive English Program (IEP)")
school = st.sidebar.text_input("School", "Anuban Sriprachanukool School")
year = st.sidebar.text_input("Year", "2025")

examiner = st.sidebar.text_input("Examiner", "Mrs. Pantiwa Akkasin")
registrar = st.sidebar.text_input("Registrar", "Mrs. Yanaya Duangmani")
grade_key = st.sidebar.text_input("Key to Grade", "Key to Grade : S - Satisfactory : U - Unsatisfactory")

S_min = st.sidebar.number_input("เกณฑ์ได้ S (min total)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("ฟอนต์ภาษาไทย (อัปโหลด .ttf)")
font_file = st.sidebar.file_uploader("อัปโหลดฟอนต์ .ttf (เช่น NotoSansThai-Regular.ttf)", type=["ttf"])
font_name = "Helvetica"
use_thai_font = False
if font_file is not None:
    font_bytes = font_file.read()
    font_buf = io.BytesIO(font_bytes)
    try:
        # ลงทะเบียนฟอนต์
        font_name = "CustomThai"
        pdfmetrics.registerFont(TTFont(font_name, font_buf))
        use_thai_font = True
        st.sidebar.success("ลงทะเบียนฟอนต์สำเร็จ ✓")
    except Exception as e:
        st.sidebar.error(f"ลงทะเบียนฟอนต์ไม่สำเร็จ: {e}")
else:
    st.sidebar.info("ไม่อัปโหลดก็ได้ ระบบจะใช้ Helvetica (อังกฤษได้ ไทยอาจไม่สมบูรณ์)")

# --------------------------- Upload Data -------------------------
st.title("Conversation Report (P.2 IEP) — Streamlit")
st.caption("โฟกัส: เรนเดอร์รายงานรายคนให้เหมือนต้นฉบับ 100% + ส่งออก PDF/Canva CSV/PPLX")

file = st.file_uploader("อัปโหลด Excel (.xlsx) หรือ CSV", type=["xlsx", "csv"])

if file is not None:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception:
        # รองรับไฟล์ CSV encoding แปลก
        file.seek(0)
        df = pd.read_csv(file, encoding="utf-8", errors="ignore")
else:
    st.info("ยังไม่อัปโหลดไฟล์ — จะโชว์ตัวอย่างเล็ก ๆ ให้ก่อน")
    df = pd.DataFrame({
        "No": [1, 2],
        "Student ID": ["13700", "13701"],
        "Name": ["Phumchai Promwatee", "Nattawat Srilachai"],
        "Semester 1": [42, 42],
        "Semester 2": [41, 41],
        # "Total": จะคำนวณเองได้
        # "Grade": จะคำนวณเองได้
    })

st.subheader("พรีวิวข้อมูล")
st.dataframe(df.head(20))

# --------------------------- Mapping UI --------------------------
st.markdown("### จับคู่คอลัมน์ (Mapping)")
cols = list(df.columns)
mapping: Dict[str, Optional[str]] = {}

# สร้าง default mapping ตาม AUTO_MAP
auto_guess = {}
for c in cols:
    canon = auto_header_to_canonical(str(c))
    if canon and canon not in auto_guess.values():
        auto_guess[c] = canon

# วาดตัวเลือกให้ผู้ใช้ปรับ
map_cols = st.columns(3)
canonical_targets = ["no", "studentId", "name", "semester1", "semester2", "total", "grade",
                     "idea", "pronunciation", "preparedness", "confidence"]
for idx, target in enumerate(canonical_targets):
    with map_cols[idx % 3]:
        default_choice = None
        for c, t in auto_guess.items():
            if t == target:
                default_choice = c
                break
        mapping[target] = st.selectbox(f"{target} ←", options=[None] + cols, index=( [None] + cols ).index(default_choice) if default_choice in cols else 0, key=f"map_{target}")

# --------------------------- Normalize Records -------------------
def normalize_records(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for target, src_col in mapping.items():
            if src_col and src_col in df.columns:
                rec[target] = row[src_col]
        # fallback no
        if rec.get("no") in (None, "", float("nan")):
            rec["no"] = i + 1
        rec = fill_sem_from_skills(rec)
        rec = compute_total_and_grade(rec, S_min)
        out.append(rec)
    return out

records = normalize_records(df, mapping)

st.markdown("### ตัวอย่างระเบียนหลัง Normalize")
st.write(pd.DataFrame(records).head(10))

# --------------------------- Layout & PDF ------------------------
PAGE_W, PAGE_H = A4  # pt

def draw_center(c: rl_canvas.Canvas, text: str, x_center: float, y: float, size=12, bold=False):
    c.setFont(font_name, size)
    w = c.stringWidth(text, font_name, size)
    c.drawString(x_center - w/2.0, y, text)

def draw_text(c: rl_canvas.Canvas, text: str, x: float, y: float, size=11, bold=False):
    c.setFont(font_name, size)
    c.drawString(x, y, text)

def generate_pdf_bytes(records: List[Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    for r in records:
        # --- Header ---
        draw_center(c, title, PAGE_W/2, PAGE_H - 40, size=16, bold=True)
        draw_center(c, subtitle, PAGE_W/2, PAGE_H - 60, size=11)
        draw_center(c, school, PAGE_W/2, PAGE_H - 78, size=11)

        # --- Meta Row ---
        y_meta = PAGE_H - 120
        draw_text(c, "Conversation Test Result No. ", 40, y_meta, size=11)
        draw_text(c, str(int(r.get("no", 0))), 210, y_meta, size=11)
        draw_text(c, "Student ID ", 250, y_meta, size=11)
        draw_text(c, str(r.get("studentId", "")), 330, y_meta, size=11)
        draw_text(c, "Grade ", 450, y_meta, size=11)
        draw_text(c, "2/5", 492, y_meta, size=11)

        # Name Row
        draw_text(c, "Name ", 40, PAGE_H - 140, size=11)
        draw_text(c, str(r.get("name","")), 80, PAGE_H - 140, size=12)

        # --- Table ---
        # outer box
        x0, y0 = 40, PAGE_H - 240
        table_w = PAGE_W - 80
        table_h = 80
        c.rect(x0, y0, table_w, table_h)

        # columns: Course (40%), 1(50) 15%, 2(50) 15%, Total 15%, Grade 15%
        col_w = [0.40, 0.15, 0.15, 0.15, 0.15]
        xs = [x0]
        for frac in col_w[:-1]:
            xs.append(xs[-1] + table_w * frac)
        xs.append(x0 + table_w)

        # vertical lines
        for x in xs[1:-1]:
            c.line(x, y0, x, y0 + table_h)

        # headers
        draw_center(c, "Course", (xs[0]+xs[1])/2, y0 + table_h - 18, size=11, bold=True)
        draw_center(c, "1 (50)", (xs[1]+xs[2])/2, y0 + table_h - 18, size=11, bold=True)
        draw_center(c, "2 (50)", (xs[2]+xs[3])/2, y0 + table_h - 18, size=11, bold=True)
        draw_center(c, "Total",  (xs[3]+xs[4])/2, y0 + table_h - 18, size=11, bold=True)
        draw_center(c, "Grade",  (xs[4]+xs[5])/2, y0 + table_h - 18, size=11, bold=True)

        # values
        draw_center(c, "Conversation", (xs[0]+xs[1])/2, y0 + 22, size=11)
        draw_center(c, str(int(r.get("semester1") if r.get("semester1") is not None else 0)), (xs[1]+xs[2])/2, y0 + 22, size=11)
        draw_center(c, str(int(r.get("semester2") if r.get("semester2") is not None else 0)), (xs[2]+xs[3])/2, y0 + 22, size=11)
        draw_center(c, str(int(r.get("total") if r.get("total") is not None else 0)), (xs[3]+xs[4])/2, y0 + 22, size=11)
        draw_center(c, str(r.get("grade","")), (xs[4]+xs[5])/2, y0 + 22, size=12)

        # --- Signatures ---
        y_sign = y0 - 80
        # left
        draw_center(c, ".................................................", PAGE_W*0.25, y_sign+40, size=11)
        draw_center(c, f"({examiner})", PAGE_W*0.25, y_sign+22, size=11)
        draw_center(c, "Examiner", PAGE_W*0.25, y_sign+6, size=11)
        # right
        draw_center(c, ".................................................", PAGE_W*0.75, y_sign+40, size=11)
        draw_center(c, f"({registrar})", PAGE_W*0.75, y_sign+22, size=11)
        draw_center(c, "Registrar of Intensive English Program", PAGE_W*0.75, y_sign+6, size=11)

        # Key to Grade
        draw_text(c, grade_key, 40, y_sign - 30, size=10.5)
        draw_text(c, year, 40, y_sign - 50, size=10.5)

        c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

def generate_canva_csv_bytes(records: List[Dict[str, Any]]) -> bytes:
    # Default placeholders; ปรับชื่อหัวได้ตามต้องการ
    rows = []
    for r in records:
        rows.append({
            "No": r.get("no"),
            "ID": r.get("studentId"),
            "Name": r.get("name"),
            "Semester1": int(r.get("semester1") if r.get("semester1") is not None else 0),
            "Semester2": int(r.get("semester2") if r.get("semester2") is not None else 0),
            "Total": int(r.get("total") if r.get("total") is not None else 0),
            "Grade": r.get("grade"),
            "Examiner": examiner,
            "Registrar": registrar,
        })
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
    return csv_bytes

def generate_pplx_bytes(records: List[Dict[str, Any]]) -> bytes:
    layout = {
        "version": "pplx-1.0",
        "pageSize": {"width": PAGE_W, "height": PAGE_H},
        "fontFamily": font_name,
        "context": {
            "header": {"title": title, "subtitle": subtitle, "school": school, "year": year},
            "footer": {"examiner": examiner, "registrar": registrar, "gradeKey": grade_key},
        },
        "pages": [
            {
                "fields": {
                    "no": r.get("no"), "studentId": r.get("studentId"), "name": r.get("name"),
                    "semester1": r.get("semester1"), "semester2": r.get("semester2"),
                    "total": r.get("total"), "grade": r.get("grade"),
                }
            } for r in records
        ],
    }
    return json.dumps(layout, ensure_ascii=False, indent=2).encode("utf-8")

# --------------------------- Actions -----------------------------
st.markdown("---")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    gen_pdf = st.button("🔧 สร้าง PDF")
with col_b:
    gen_csv = st.button("🔧 สร้าง Canva Bulk CSV")
with col_c:
    gen_pplx = st.button("🔧 สร้าง PPLX (JSON)")

out_pdf = out_csv = out_pplx = None
if gen_pdf or gen_csv or gen_pplx:
    if not records:
        st.error("ไม่มีข้อมูลหลัง normalize")
    else:
        if gen_pdf:
            out_pdf = generate_pdf_bytes(records)
        if gen_csv:
            out_csv = generate_canva_csv_bytes(records)
        if gen_pplx:
            out_pplx = generate_pplx_bytes(records)

# --------------------------- Downloads ---------------------------
dl_cols = st.columns(3)
if out_pdf:
    with dl_cols[0]:
        st.download_button("⬇️ ดาวน์โหลด PDF", data=out_pdf, file_name="conversation_report.pdf", mime="application/pdf")
if out_csv:
    with dl_cols[1]:
        st.download_button("⬇️ ดาวน์โหลด Canva CSV", data=out_csv, file_name="canva_bulk.csv", mime="text/csv")
if out_pplx:
    with dl_cols[2]:
        st.download_button("⬇️ ดาวน์โหลด PPLX", data=out_pplx, file_name="report.pplx", mime="application/json")

st.markdown("---")
st.caption("ทิป: ถ้าฟอนต์ไทยเพี้ยน ให้หาไฟล์ .ttf แล้วอัปโหลดใน Sidebar — รายงานจะตรงกับต้นฉบับมากขึ้น")
