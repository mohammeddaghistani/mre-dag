import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# PDF + Arabic
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

# QR + Images
import qrcode
from PIL import Image
from io import BytesIO
import base64

# Utils
import math
import uuid
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
CITY_LOCK = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262
FONT_PATH = "Tajawal-Regular.ttf"  # الملف الثالث المطلوب بجانب app.py

# الأعمدة المطلوبة
REQUIRED_COLS = ["latitude", "longitude", "القيمة السنوية", "المساحة", "المدينة"]

# =============================================================================
# EMBEDDED LOGO (Base64) - شعارك مضمّن داخل الملف (لا تحتاج logo.png)
# =============================================================================
LOGO_B64 = """iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAYAAADL1t+KAAEAAElEQVR4nOzdd5gcxZkw8Leqc/fktDlH
...SNIP...
"""  # تم تقصير العرض هنا في الرسالة

# ⚠️ مهم:
# أنا قصّيت السطر في العرض هنا. في نسختك النهائية لازم يكون LOGO_B64 كامل.
# إذا تبي أرسله لك كامل بدون قص: قل لي "أرسل LOGO_B64 كامل".
# (أطول من 150 ألف حرف – لذلك بعض واجهات الدردشة تقصه.)

# =============================================================================
# UI / CSS
# =============================================================================
st.set_page_config(page_title="إستدامة | منصة التقييم العقاري - مكة", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family:'Tajawal', sans-serif; direction: rtl; text-align: right; }

:root {
  --bg:#f6f7fb;
  --card:#ffffff;
  --ink:#101114;
  --muted:#6b7280;
  --gold:#c5a059;
  --dark:#111827;
  --border: rgba(17,24,39,0.08);
}

.main { background: var(--bg); }
.block-container { padding-top: 1.4rem; padding-bottom: 1.8rem; max-width: 1300px; }

.hero {
  background: radial-gradient(1200px 500px at 70% -20%, rgba(197,160,89,0.22), transparent),
              linear-gradient(135deg, #0b0f19 0%, #121826 70%, #101827 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px;
  padding: 26px 26px 18px 26px;
  color: #fff;
  box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: 0.2px; }
.hero p { margin: 8px 0 0 0; color: rgba(255,255,255,0.78); line-height: 1.8; }

.badges { margin-top: 14px; display:flex; gap:10px; flex-wrap: wrap; }
.badge {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 7px 10px;
  border-radius: 999px;
  color: rgba(255,255,255,0.88);
  font-size: 0.92rem;
}

.grid {
  margin-top: 16px;
  display:grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 14px;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
  box-shadow: 0 12px 30px rgba(17,24,39,0.05);
}

.card h3 { margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800; color: var(--ink); }
.card p { margin: 0; color: var(--muted); line-height: 1.8; }

.kpi {
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 12px;
}
.kpi .k {
  background: linear-gradient(180deg, rgba(197,160,89,0.12), rgba(197,160,89,0.04));
  border: 1px solid rgba(197,160,89,0.25);
  border-radius: 16px;
  padding: 12px;
}
.kpi .k .t { color: var(--muted); font-size: 0.9rem; }
.kpi .k .v { color: var(--ink); font-size: 1.22rem; font-weight: 800; margin-top: 4px; }

.sep { height: 1px; background: rgba(17,24,39,0.08); margin: 12px 0; }

.stButton > button {
  width: 100%;
  background: linear-gradient(135deg, #0b0f19 0%, #1b2235 100%);
  border: 1px solid rgba(197,160,89,0.65);
  color: var(--gold);
  border-radius: 14px;
  height: 3.1rem;
  font-size: 1.05rem;
  font-weight: 800;
  transition: 0.2s;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(197,160,89,0.20);
  color: #fff;
}

.small { font-size: 0.92rem; color: var(--muted); }
.warn {
  background: rgba(245,158,11,0.10);
  border: 1px solid rgba(245,158,11,0.25);
  padding: 10px 12px;
  border-radius: 14px;
  color: #92400e;
}
.good {
  background: rgba(16,185,129,0.10);
  border: 1px solid rgba(16,185,129,0.25);
  padding: 10px 12px;
  border-radius: 14px;
  color: #065f46;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# ARABIC HELPERS
# =============================================================================
def ar(s: str) -> str:
    return get_display(arabic_reshaper.reshape(str(s)))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def make_report_id():
    return f"MK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

def confidence_label(similar_count: int, r2v: float):
    score = min(60, similar_count * 3) + int(np.clip(r2v, 0, 1) * 40)
    if score >= 75: return "High", score
    if score >= 50: return "Medium", score
    return "Low", score

def decode_logo():
    # لو LOGO_B64 ناقص، نتجاوز الشعار بدون كسر التطبيق
    try:
        raw = base64.b64decode(LOGO_B64.encode("utf-8"))
        return Image.open(BytesIO(raw)).convert("RGBA")
    except Exception:
        return None

# =============================================================================
# TEMPLATE GENERATOR
# =============================================================================
def build_template_excel() -> bytes:
    # نموذج فارغ يساعد المستخدم
    df = pd.DataFrame({
        "latitude": [21.3891],
        "longitude": [39.8579],
        "القيمة السنوية": [150000],
        "المساحة": [500],
        "المدينة": [CITY_LOCK],
        "اسم المشروع": ["مثال - حي العزيزية"]
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Deals_DB")
    buf.seek(0)
    return buf.getvalue()

# =============================================================================
# DATA PIPELINE
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة الناقصة: {missing}")

    x = df.copy()

    x["lat"] = pd.to_numeric(x["latitude"], errors="coerce")
    x["lon"] = pd.to_numeric(x["longitude"], errors="coerce")
    x["price"] = pd.to_numeric(x["القيمة السنوية"], errors="coerce")
    x["area"] = pd.to_numeric(x["المساحة"], errors="coerce")

    x = x.dropna(subset=["lat", "lon", "price", "area", "المدينة"]).copy()
    x = x[(x["price"] > 0) & (x["area"] > 0)].copy()

    # قفل مكة
    x = x[x["المدينة"] == CITY_LOCK].copy()

    # مسافة الحرم
    if not x.empty:
        x["dist_haram_km"] = x.apply(lambda r: haversine_km(r["lat"], r["lon"], HARAM_LAT, HARAM_LON), axis=1)

        # معدل المتر
        x["sqm_rate"] = x["price"] / x["area"]

        # قصّ outliers بشكل حكومي محافظ (IQR) لتقليل التشويش
        q1, q3 = x["sqm_rate"].quantile(0.25), x["sqm_rate"].quantile(0.75)
        iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        x["is_outlier"] = ~x["sqm_rate"].between(lo, hi)

    return x

# =============================================================================
# ML TRAINING
# =============================================================================
@st.cache_resource
def train_model(df: pd.DataFrame):
    # نعتمد على بيانات غير شاذة أساسًا إن وجدت
    d = df.copy()
    if "is_outlier" in d.columns and d["is_outlier"].any():
        core = d[~d["is_outlier"]].copy()
        if len(core) >= max(25, int(len(d) * 0.6)):
            d = core

    X = d[["area", "lat", "lon", "dist_haram_km"]]
    y = d["price"]

    # لو البيانات قليلة جدًا: ندرّب نموذج بسيط
    if len(d) < 25:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        return model, 0.0, pd.Series({"area": np.nan, "lat": np.nan, "lon": np.nan, "dist_haram_km": np.nan})

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=320,
        learning_rate=0.04,
        max_depth=4,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    r2v = float(r2_score(yte, model.predict(Xte)))

    # Permutation importance للتفسير
    try:
        imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=42)
        fi = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    except Exception:
        fi = pd.Series({"area": np.nan, "lat": np.nan, "lon": np.nan, "dist_haram_km": np.nan})

    return model, r2v, fi

# =============================================================================
# PDF BUILDER (Gov)
# =============================================================================
class GovPDF(FPDF):
    pass

def pdf_report(report: dict, model_card: dict, verify_url: str, logo_img: Image.Image | None) -> bytes:
    # QR
    qr = qrcode.make(verify_url)
    qr_buf = BytesIO()
    qr.save(qr_buf, format="PNG")
    qr_buf.seek(0)

    # PDF
    pdf = GovPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    pdf.add_font("Tajawal", "", FONT_PATH, uni=True)
    pdf.set_font("Tajawal", size=16)

    # Watermark
    pdf.set_text_color(210, 210, 210)
    pdf.set_font("Tajawal", size=46)
    pdf.text(30, 140, ar("نسخة أولية"))
    pdf.set_text_color(20, 20, 20)

    # Header area
    pdf.set_font("Tajawal", size=18)
    pdf.cell(0, 10, ar("منصة إستدامة للتقييم العقاري الذكي"), ln=True, align="R")
    pdf.set_font("Tajawal", size=12)
    pdf.cell(0, 8, ar(f"مدينة {CITY_LOCK} | تقرير تقديري آلي للاستخدام الرسمي"), ln=True, align="R")

    # Logo
    if logo_img is not None:
        # فPDF يفضّل path، فنعمل تحويل مؤقت في الذاكرة
        lbuf = BytesIO()
        logo_img.save(lbuf, format="PNG")
        lbuf.seek(0)
        # fpdf2 يدعم BytesIO مباشرة
        pdf.image(lbuf, x=165, y=10, w=30)

    pdf.ln(4)
    pdf.set_font("Tajawal", size=12)

    # Body
    for k, v in report.items():
        pdf.multi_cell(0, 8, ar(f"{k}: {v}"), align="R")

    pdf.ln(2)
    pdf.set_font("Tajawal", size=10)
    pdf.multi_cell(0, 7, ar("تنبيه: هذا التقرير تقديري آلي لدعم القرار ولا يعد اعتمادًا نهائيًا. يخضع للتحقق والمراجعة الرسمية."), align="R")

    # QR
    pdf.image(qr_buf, x=165, y=250, w=35, h=35)
    pdf.set_xy(10, 288)
    pdf.set_font("Tajawal", size=9)
    pdf.cell(0, 6, ar(f"QR للتحقق: {verify_url}"), align="R")

    # Page 2: Model Card
    pdf.add_page()
    pdf.add_font("Tajawal", "", FONT_PATH, uni=True)
    pdf.set_font("Tajawal", size=16)
    pdf.cell(0, 10, ar("Model Card – منهجية النموذج"), ln=True, align="R")
    pdf.ln(2)

    pdf.set_font("Tajawal", size=12)
    for k, v in model_card.items():
        pdf.multi_cell(0, 8, ar(f"{k}: {v}"), align="R")

    out = pdf.output(dest="S").encode("latin-1")
    return out

# =============================================================================
# HEADER / HERO
# =============================================================================
logo = decode_logo()

st.markdown(
    f"""
<div class="hero">
  <h1>🏛️ منصة <span style="color:var(--gold)">إستدامة</span> للتقييم العقاري</h1>
  <p>
    نسخة حكومية عالية الاعتمادية تعمل سحابيًا بدون Google Sheets. 
    تعتمد على نموذج تعلم آلي وتولد تقريرًا رسميًا شاملًا مع رقم تقرير وQR وختم نسخة أولية.
  </p>
  <div class="badges">
    <div class="badge">قفل المدينة: {CITY_LOCK}</div>
    <div class="badge">AI Model: Gradient Boosting</div>
    <div class="badge">Gov PDF + Model Card</div>
    <div class="badge">QR Verification</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR (حكومي)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ إعدادات تشغيل")
    st.caption("هذه الإعدادات لا تظهر في التقرير، فقط لتشغيل المنصة.")

    strict_lock = st.toggle("قفل صارم لمكة فقط", value=True)
    outlier_filter = st.toggle("تصفية القيم الشاذة (IQR)", value=True)
    show_debug = st.toggle("إظهار مؤشرات جودة البيانات", value=True)

    st.markdown("---")
    st.markdown("### 📄 قالب البيانات")
    st.download_button(
        "⬇️ تنزيل قالب Excel جاهز",
        data=build_template_excel(),
        file_name="Estidama_Template_Makkah.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("املأ القالب ثم ارفعه في الصفحة الرئيسية.")

# =============================================================================
# DATA UPLOAD
# =============================================================================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 رفع ملف البيانات")
    up = st.file_uploader("ارفع Excel أو CSV", type=["xlsx", "xls", "csv"])
    st.caption("المطلوب: latitude, longitude, القيمة السنوية, المساحة, المدينة")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 نطاق المنصة")
    st.write("هذه النسخة مهيأة لمدينة **مكة المكرمة** مع عامل **قرب الحرم** داخل ML.")
    st.write("بعد رفع الملف ستظهر: التحليلات، الخريطة، والتقارير الرسمية.")
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">ملاحظة: لا يتم حفظ ملفاتك؛ المعالجة تتم داخل الجلسة فقط.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not up:
    st.info("⬆️ ارفع ملف البيانات للبدء. يمكنك تنزيل قالب Excel من الشريط الجانبي.")
    st.stop()

# Read file
try:
    if up.name.lower().endswith(".csv"):
        raw = pd.read_csv(up)
    else:
        raw = pd.read_excel(up)
except Exception as e:
    st.error(f"تعذر قراءة الملف: {e}")
    st.stop()

# Clean + lock
try:
    df = clean_data(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.error("لا توجد بيانات مناسبة بعد التنظيف. تأكد أن المدينة = مكة المكرمة وأن الإحداثيات والقيم صحيحة.")
    st.stop()

if strict_lock and (df["المدينة"].nunique() != 1 or df["المدينة"].iloc[0] != CITY_LOCK):
    st.error("قفل صارم مفعل: يجب أن تكون كل البيانات لمكة المكرمة فقط.")
    st.stop()

if not outlier_filter and "is_outlier" in df.columns:
    df["is_outlier"] = False

# =============================================================================
# KPIs + QUALITY
# =============================================================================
count_all = len(df)
count_outliers = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0
avg_sqm = float(df["sqm_rate"].mean()) if "sqm_rate" in df.columns else float("nan")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📌 مؤشرات بيانات مكة")
st.markdown(
    f"""
<div class="kpi">
  <div class="k"><div class="t">عدد الصفقات</div><div class="v">{count_all:,}</div></div>
  <div class="k"><div class="t">متوسط سعر/م²</div><div class="v">{avg_sqm:,.0f} ريال</div></div>
  <div class="k"><div class="t">قيم شاذة (IQR)</div><div class="v">{count_outliers:,}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

if show_debug:
    nulls = raw[REQUIRED_COLS].isna().sum().to_dict() if all(c in raw.columns for c in REQUIRED_COLS) else {}
    if nulls:
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("**مؤشرات جودة:**")
        st.write({k: int(v) for k, v in nulls.items()})
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Train model
# =============================================================================
model, model_r2, feat_imp = train_model(df)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 محرك التقييم", "🗺️ الخريطة التفاعلية", "📊 بنك البيانات", "📄 التقارير/المنهجية"])

with tab1:
    c1, c2 = st.columns([1.0, 1.25], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 مواصفات العقار المستهدف")

        target_area = st.number_input("المساحة (م²)", min_value=1, value=500)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("**📍 إحداثيات العقار (اختياري لرفع الدقة)**")
        use_custom = st.checkbox("سأدخل إحداثيات العقار", value=False)
        if use_custom:
            tlat = st.number_input("Latitude", value=float(df["lat"].mean()), format="%.6f")
            tlon = st.number_input("Longitude", value=float(df["lon"].mean()), format="%.6f")
        else:
            tlat = float(df["lat"].mean())
            tlon = float(df["lon"].mean())

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("**📈 معايير الجودة النوعية (للمعلومية)**")
        q_loc = st.select_slider("قوة الموقع", options=[1,2,3,4,5], value=3)
        q_spec = st.select_slider("المواصفات", options=[1,2,3,4,5], value=3)
        q_age = st.select_slider("العمر والحالة", options=[1,2,3,4,5], value=3)

        issue = st.checkbox("تمييز التقرير كـ (حالة/مراجعة)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 النتائج (Gov + AI)")

        # Similar deals
        similar = df[df["area"].between(target_area*0.8, target_area*1.2)]
        similar_count = int(len(similar))

        # ML prediction
        tdist = float(haversine_km(tlat, tlon, HARAM_LAT, HARAM_LON))
        pred = float(model.predict([[target_area, tlat, tlon, tdist]])[0])

        conf_label, conf_score = confidence_label(similar_count, model_r2)

        st.metric("القيمة السنوية المتوقعة (ML)", f"{pred:,.0f} ريال")
        st.metric("قرب الحرم", f"{tdist:.2f} كم")
        st.metric("مستوى الثقة الحكومي", conf_label)
        st.progress(conf_score / 100)

        st.caption(f"درجة الثقة: {conf_score}/100 | R²: {model_r2:.2f} | صفقات مشابهة: {similar_count}")

        # Explainability
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("### 🧠 تفسير مبسط (Feature Importance)")
        if feat_imp.notna().any():
            ex = feat_imp.rename({
                "area": "المساحة",
                "lat": "Latitude",
                "lon": "Longitude",
                "dist_haram_km": "قرب الحرم (كم)"
            })
            st.bar_chart(ex)
        else:
            st.info("لم تتوفر أهمية الخصائص (قد تكون البيانات قليلة).")

        # Report
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        if st.button("🧾 إصدار تقرير PDF رسمي"):
            rid = make_report_id()
            verify_url = f"https://verify.estidama.sa/{rid}"  # رابط تحقق صوري (يمكن ربطه لاحقًا)

            report = {
                "رقم التقرير": rid,
                "المدينة": CITY_LOCK,
                "تاريخ الإصدار": datetime.now().strftime("%Y-%m-%d"),
                "حالة التقرير": "نسخة أولية – للاستخدام الرسمي" + (" (قيد المراجعة)" if issue else ""),
                "المساحة": f"{target_area} م²",
                "القيمة السنوية المتوقعة (ML)": f"{pred:,.0f} ريال",
                "قرب الحرم": f"{tdist:.2f} كم",
                "مستوى الثقة الحكومي": f"{conf_label} ({conf_score}/100)",
                "دقة النموذج R²": f"{model_r2:.2f}",
                "عدد صفقات مكة": f"{len(df):,}",
                "صفقات مشابهة (مساحة)": f"{similar_count:,}",
                "معايير الجودة (للمعلومية)": f"موقع={q_loc} | مواصفات={q_spec} | عمر={q_age}",
            }

            model_card = {
                "نوع النموذج": "Gradient Boosting Regressor",
                "نطاق العمل": f"مدينة {CITY_LOCK} فقط",
                "المدخلات": "المساحة + الإحداثيات + المسافة للحرم",
                "المخرجات": "القيمة السنوية المتوقعة",
                "التقييم": f"R² = {model_r2:.2f}",
                "منهجية الثقة": "عدد الصفقات المشابهة + جودة النموذج (R²) إلى 100",
                "حدود الاستخدام": "تقدير آلي لدعم القرار وليس اعتمادًا نهائيًا",
            }

            pdf_bytes = pdf_report(report, model_card, verify_url, logo)

            st.success("تم إنشاء التقرير.")
            st.download_button(
                "⬇️ تحميل التقرير PDF",
                data=pdf_bytes,
                file_name=f"{rid}.pdf",
                mime="application/pdf",
            )

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ خريطة الصفقات داخل مكة")

    view = pdk.ViewState(
        latitude=float(df["lat"].mean()),
        longitude=float(df["lon"].mean()),
        zoom=12,
        pitch=45,
    )

    # لون حسب السعر/م² (تطبيع)
    rates = df["sqm_rate"].to_numpy()
    rmin, rmax = float(np.nanmin(rates)), float(np.nanmax(rates))
    denom = (rmax - rmin) if (rmax - rmin) != 0 else 1.0

    map_df = df.copy()
    map_df["rate_norm"] = (map_df["sqm_rate"] - rmin) / denom
    map_df["elev"] = (map_df["price"] / 120).clip(0, 20000)

    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_elevation="elev",
        radius=100,
        get_fill_color="[255, 255*(1-rate_norm), 0, 150]",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {"text": "القيمة السنوية: {القيمة السنوية}\nالمساحة: {المساحة}\nسعر/م²: {sqm_rate}\nقرب الحرم (كم): {dist_haram_km}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 بنك بيانات مكة (بعد التنظيف)")
    show_cols = [c for c in df.columns if c not in ["price", "area", "lat", "lon"]]
    st.dataframe(df[show_cols], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 التقارير والمنهجية")
    st.markdown("**ماذا تحتوي التقارير؟**")
    st.write("- رقم تقرير تلقائي + تاريخ الإصدار")
    st.write("- QR للتحقق (قابل للربط لاحقًا بصفحة تحقق رسمية)")
    st.write("- ختم (نسخة أولية – للاستخدام الرسمي)")
    st.write("- مؤشرات الثقة الحكومية + جودة النموذج R²")
    st.write("- صفحة Model Card توضح منهجية النموذج وحدود الاستخدام")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("**سياسة قفل مكة**")
    st.write("أي صفقة ليست (مكة المكرمة) يتم استبعادها تلقائيًا. ويمكن تفعيل القفل الصارم من الشريط الجانبي.")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("**جاهزية حكومية (اقتراحات تطوير لاحقًا)**")
    st.write("1) صفحة تحقق رسمية تستقبل رقم التقرير وتعرض نسخة مختصرة.")
    st.write("2) صلاحيات مستخدمين (Viewer/Analyst/Admin).")
    st.write("3) سجل تدقيق Audit Log (من أصدر تقرير؟ متى؟ ما المعلمات؟).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr><center>إستدامة | تطوير: محمد داغستاني © 2026</center>", unsafe_allow_html=True)
