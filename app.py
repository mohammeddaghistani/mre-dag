import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import math
import uuid
from datetime import datetime
from io import BytesIO
import qrcode

from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

# =========================
# CONFIG
# =========================
CITY_NAME = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262
FONT_PATH = "Tajawal-Regular.ttf"

# أعمدة ملفك (حسب Excel المرفوع)
REQ_COLS = [
    "latitude", "longitude", "القيمة السنوية",
    "اسم الحي", "النشاط الرئيسي", "النشاط الفرعي",
    "المدة", "التجهيز", "اسم المشروع", "رقم العقد"
]

# =========================
# UI / CSS (GovTech)
# =========================
st.set_page_config(page_title="إستدامة | منصة حكومية للتقييم العقاري – مكة", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family:'Tajawal', sans-serif; direction: rtl; text-align: right; }
:root{
 --bg:#f6f7fb; --card:#ffffff; --ink:#0f172a; --muted:#64748b;
 --gold:#c5a059; --dark:#0b0f19; --border: rgba(15,23,42,0.08);
}
.main{background:var(--bg);}
.block-container{max-width:1300px; padding-top:1.2rem; padding-bottom:1.6rem;}
.hero{
 background: radial-gradient(1200px 520px at 75% -10%, rgba(197,160,89,0.25), transparent),
             linear-gradient(135deg, #0b0f19 0%, #121a2b 70%, #0b1222 100%);
 border:1px solid rgba(255,255,255,0.10); border-radius:22px;
 padding:24px; color:#fff; box-shadow:0 22px 60px rgba(0,0,0,0.25);
}
.hero h1{margin:0; font-size:2.0rem; font-weight:800;}
.hero p{margin:8px 0 0 0; color:rgba(255,255,255,0.78); line-height:1.8;}
.badges{margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;}
.badge{background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10);
 padding:7px 10px; border-radius:999px; color:rgba(255,255,255,0.88); font-size:0.92rem;}
.card{background:var(--card); border:1px solid var(--border); border-radius:18px;
 padding:18px; box-shadow:0 12px 30px rgba(15,23,42,0.05);}
.sep{height:1px; background: rgba(15,23,42,0.08); margin:12px 0;}
.kpi{display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:12px;}
.k{background: linear-gradient(180deg, rgba(197,160,89,0.12), rgba(197,160,89,0.04));
 border:1px solid rgba(197,160,89,0.25); border-radius:16px; padding:12px;}
.k .t{color:var(--muted); font-size:0.9rem;}
.k .v{color:var(--ink); font-size:1.22rem; font-weight:800; margin-top:4px;}
.stButton>button{
 width:100%; background:linear-gradient(135deg, #0b0f19 0%, #1a2338 100%);
 border:1px solid rgba(197,160,89,0.65); color: var(--gold);
 border-radius:14px; height:3.1rem; font-size:1.05rem; font-weight:800; transition:0.2s;
}
.stButton>button:hover{transform:translateY(-2px); box-shadow:0 10px 20px rgba(197,160,89,0.20); color:#fff;}
.small{font-size:0.92rem; color:var(--muted);}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
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

def gov_confidence(similar_count: int, r2v: float) -> tuple[str, int]:
    score = min(60, similar_count * 3) + int(np.clip(r2v, 0, 1) * 40)
    if score >= 75: return "High", score
    if score >= 50: return "Medium", score
    return "Low", score

# =========================
# PDF Generator (Arabic + Gov)
# =========================
class GovPDF(FPDF):
    pass

def build_pdf(report: dict, model_card: dict, verify_url: str) -> bytes:
    # QR
    qr = qrcode.make(verify_url)
    qr_buf = BytesIO()
    qr.save(qr_buf, format="PNG")
    qr_buf.seek(0)

    pdf = GovPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    pdf.add_font("Tajawal", "", FONT_PATH, uni=True)
    pdf.set_font("Tajawal", size=16)

    # Watermark (نسخة أولية)
    pdf.set_text_color(210, 210, 210)
    pdf.set_font("Tajawal", size=46)
    pdf.text(20, 150, ar("نسخة أولية"))
    pdf.set_text_color(15, 23, 42)

    # Header
    pdf.set_font("Tajawal", size=18)
    pdf.cell(0, 10, ar("منصة إستدامة للتقييم العقاري الذكي"), ln=True, align="R")
    pdf.set_font("Tajawal", size=12)
    pdf.cell(0, 8, ar(f"مدينة {CITY_NAME} | تقرير تقديري آلي للاستخدام الرسمي"), ln=True, align="R")
    pdf.ln(4)

    pdf.set_font("Tajawal", size=12)
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

    return pdf.output(dest="S").encode("latin-1")

# =========================
# Data Load & Clean (your schema)
# =========================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة الناقصة في الملف: {missing}")

    x = df.copy()
    x["lat"] = pd.to_numeric(x["latitude"], errors="coerce")
    x["lon"] = pd.to_numeric(x["longitude"], errors="coerce")
    x["annual"] = pd.to_numeric(x["القيمة السنوية"], errors="coerce")
    x["duration"] = pd.to_numeric(x["المدة"], errors="coerce")  # قد تكون NaN
    x["equip"] = x["التجهيز"].astype(str).fillna("غير محدد")

    x = x.dropna(subset=["lat", "lon", "annual"]).copy()
    x = x[x["annual"] > 0].copy()

    # Feature: distance to Haram
    x["dist_haram_km"] = x.apply(lambda r: haversine_km(r["lat"], r["lon"], HARAM_LAT, HARAM_LON), axis=1)

    # تنظيف نصوص
    for col in ["اسم الحي", "النشاط الرئيسي", "النشاط الفرعي", "اسم المشروع"]:
        x[col] = x[col].astype(str).fillna("غير محدد").str.strip()

    return x

@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    # Features from your dataset
    numeric_features = ["lat", "lon", "dist_haram_km", "duration"]
    categorical_features = ["اسم الحي", "النشاط الرئيسي", "النشاط الفرعي", "equip"]

    X = df[numeric_features + categorical_features].copy()
    y = df["annual"].copy()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=320,
        learning_rate=0.04,
        max_depth=4,
        random_state=42
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # تقييم
    if len(df) < 30:
        pipe.fit(X, y)
        return pipe, 0.0, np.nan

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    r2v = float(r2_score(yte, pred))
    mape = float(mean_absolute_percentage_error(yte, np.maximum(pred, 1)))
    return pipe, r2v, mape

# =========================
# Header
# =========================
st.markdown(f"""
<div class="hero">
  <h1>🏛️ منصة <span style="color:var(--gold)">إستدامة</span> للتقييم العقاري</h1>
  <p>
    نسخة حكومية احترافية مخصصة لمدينة مكة المكرمة، تعمل عبر رفع ملف البيانات (Excel/CSV) دون Google Sheets،
    مع نموذج تعلم آلي لتقدير القيمة السنوية وإصدار تقرير رسمي شامل.
  </p>
  <div class="badges">
    <div class="badge">قفل المدينة: مكة المكرمة</div>
    <div class="badge">AI Model: Gradient Boosting</div>
    <div class="badge">Gov PDF + Model Card</div>
    <div class="badge">QR + رقم تقرير</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Upload
# =========================
colL, colR = st.columns([1.15, 0.85], gap="large")

with colL:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 رفع ملف البيانات")
    up = st.file_uploader("ارفع ملف Excel أو CSV (هيكل Estidama_System_DB)", type=["xlsx", "xls", "csv"])
    st.caption("الأعمدة المطلوبة: latitude, longitude, القيمة السنوية, اسم الحي, النشاط الرئيسي, النشاط الفرعي, المدة, التجهيز, اسم المشروع, رقم العقد")
    st.markdown("</div>", unsafe_allow_html=True)

with colR:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ ملاحظات تشغيل")
    st.write("هذه النسخة مبنية على ملفك الفعلي (بدون المساحة). التقييم يتم للقيمة السنوية مباشرة.")
    st.write("يمكن إدخال موقع العقار والنشاط والحي والمدة والتجهيز لإصدار تقرير رسمي.")
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">لا يتم حفظ ملفاتك؛ التحليل داخل الجلسة فقط.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not up:
    st.info("ارفع ملف البيانات للبدء.")
    st.stop()

# Read
try:
    if up.name.lower().endswith(".csv"):
        raw = pd.read_csv(up)
    else:
        raw = pd.read_excel(up)
except Exception as e:
    st.error(f"تعذر قراءة الملف: {e}")
    st.stop()

try:
    data = clean_df(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

if data.empty:
    st.error("بعد التنظيف لم يتم العثور على صفقات صالحة (القيمة السنوية > 0 + إحداثيات صحيحة).")
    st.stop()

# Train model once
pipe, r2v, mape = train_pipeline(data)

# KPIs
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📌 مؤشرات بيانات مكة")
st.markdown(f"""
<div class="kpi">
  <div class="k"><div class="t">عدد الصفقات الصالحة</div><div class="v">{len(data):,}</div></div>
  <div class="k"><div class="t">متوسط القيمة السنوية</div><div class="v">{data["annual"].mean():,.0f} ريال</div></div>
  <div class="k"><div class="t">جودة النموذج</div><div class="v">{(r2v if not np.isnan(r2v) else 0):.2f} R²</div></div>
</div>
""", unsafe_allow_html=True)
if not np.isnan(mape):
    st.caption(f"MAPE (تقريبي): {mape*100:.1f}%")
st.markdown("</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 محرك التقييم", "🗺️ الخريطة التفاعلية", "📊 بنك البيانات"])

with tab1:
    c1, c2 = st.columns([1.0, 1.25], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧾 بيانات العقار المستهدف")

        use_custom_loc = st.checkbox("سأدخل إحداثيات العقار المستهدف", value=False)
        if use_custom_loc:
            tlat = st.number_input("Latitude", value=float(data["lat"].mean()), format="%.6f")
            tlon = st.number_input("Longitude", value=float(data["lon"].mean()), format="%.6f")
        else:
            tlat = float(data["lat"].mean())
            tlon = float(data["lon"].mean())

        neighborhoods = sorted(data["اسم الحي"].dropna().unique().tolist())
        main_acts = sorted(data["النشاط الرئيسي"].dropna().unique().tolist())

        حي = st.selectbox("اسم الحي", neighborhoods)
        نشاط_رئيسي = st.selectbox("النشاط الرئيسي", main_acts)

        # فلترة نشاط فرعي بناء على الرئيسي إن أمكن
        sub_candidates = data.loc[data["النشاط الرئيسي"] == نشاط_رئيسي, "النشاط الفرعي"].dropna().unique().tolist()
        sub_candidates = sorted(sub_candidates) if sub_candidates else sorted(data["النشاط الفرعي"].dropna().unique().tolist())
        نشاط_فرعي = st.selectbox("النشاط الفرعي", sub_candidates)

        تجهيز = st.selectbox("التجهيز", sorted(data["equip"].dropna().unique().tolist()))
        مدة = st.number_input("المدة (سنوات/حسب بياناتك)", min_value=0.0, value=float(np.nanmedian(data["duration"])), step=0.5)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("**📝 معايير جودة (تدخل في التقرير كمعلومات فقط)**")
        q_loc = st.select_slider("قوة الموقع", options=[1,2,3,4,5], value=3)
        q_spec = st.select_slider("المواصفات", options=[1,2,3,4,5], value=3)
        q_age = st.select_slider("العمر والحالة", options=[1,2,3,4,5], value=3)

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 نتائج التقييم")

        tdist = haversine_km(tlat, tlon, HARAM_LAT, HARAM_LON)

        Xq = pd.DataFrame([{
            "lat": tlat,
            "lon": tlon,
            "dist_haram_km": tdist,
            "duration": مدة,
            "اسم الحي": حي,
            "النشاط الرئيسي": نشاط_رئيسي,
            "النشاط الفرعي": نشاط_فرعي,
            "equip": تجهيز,
        }])

        pred = float(pipe.predict(Xq)[0])

        # صفقات مشابهة (حي + نشاط رئيسي) + قرب مكاني 3 كم تقريبًا
        sim = data[
            (data["اسم الحي"] == حي) &
            (data["النشاط الرئيسي"] == نشاط_رئيسي) &
            (np.abs(data["dist_haram_km"] - tdist) <= 3.0)
        ]
        sim_count = int(len(sim))

        conf, conf_score = gov_confidence(sim_count, r2v if not np.isnan(r2v) else 0.0)

        st.metric("القيمة السنوية المتوقعة (ML)", f"{pred:,.0f} ريال")
        st.metric("قرب الحرم", f"{tdist:.2f} كم")
        st.metric("مستوى الثقة الحكومي", conf)
        st.progress(conf_score/100)
        st.caption(f"درجة الثقة: {conf_score}/100 | صفقات مشابهة: {sim_count} | R²: {(r2v if not np.isnan(r2v) else 0):.2f}")

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        if st.button("🧾 إصدار تقرير PDF رسمي"):
            rid = make_report_id()
            verify_url = f"https://verify.estidama.sa/{rid}"  # رابط تحقق صوري (قابل للربط لاحقًا)

            report = {
                "رقم التقرير": rid,
                "المدينة": CITY_NAME,
                "تاريخ الإصدار": datetime.now().strftime("%Y-%m-%d"),
                "حالة التقرير": "نسخة أولية – للاستخدام الرسمي",
                "اسم الحي": حي,
                "النشاط": f"{نشاط_رئيسي} / {نشاط_فرعي}",
                "التجهيز": تجهيز,
                "المدة": مدة,
                "القيمة السنوية المتوقعة (ML)": f"{pred:,.0f} ريال",
                "قرب الحرم": f"{tdist:.2f} كم",
                "مستوى الثقة الحكومي": f"{conf} ({conf_score}/100)",
                "جودة النموذج R²": f"{(r2v if not np.isnan(r2v) else 0):.2f}",
                "عدد صفقات مكة": f"{len(data):,}",
                "عدد الصفقات المشابهة": f"{sim_count:,}",
                "مؤشرات جودة (معلومية)": f"موقع={q_loc} | مواصفات={q_spec} | عمر={q_age}",
            }

            model_card = {
                "نوع النموذج": "Gradient Boosting Regressor + OneHotEncoder",
                "النطاق": "مكة المكرمة (بيانات بلدية/استثمار)",
                "الهدف": "التنبؤ بالقيمة السنوية للعقود/الفرص الاستثمارية",
                "المدخلات": "الموقع + الحي + النشاط + المدة + التجهيز + قرب الحرم",
                "المخرجات": "القيمة السنوية المتوقعة",
                "التقييم": f"R² = {(r2v if not np.isnan(r2v) else 0):.2f} | MAPE ≈ {(mape*100 if not np.isnan(mape) else 0):.1f}%",
                "منهجية الثقة": "تجمع بين عدد الصفقات المشابهة ومؤشر جودة النموذج حتى 100",
                "القيود": "تقدير آلي لدعم القرار وليس اعتمادًا نهائيًا",
            }

            pdf_bytes = build_pdf(report, model_card, verify_url)
            st.success("تم إنشاء التقرير.")
            st.download_button("⬇️ تحميل التقرير PDF", data=pdf_bytes, file_name=f"{rid}.pdf", mime="application/pdf")

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ خريطة الصفقات داخل مكة")
    view = pdk.ViewState(latitude=float(data["lat"].mean()), longitude=float(data["lon"].mean()), zoom=12, pitch=45)

    map_df = data.copy()
    # تطبيع القيم السنوية للون
    vmin, vmax = float(map_df["annual"].min()), float(map_df["annual"].max())
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    map_df["vnorm"] = (map_df["annual"] - vmin) / denom
    map_df["elev"] = (map_df["annual"] / 80).clip(0, 20000)

    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_elevation="elev",
        radius=100,
        get_fill_color="[255, 255*(1-vnorm), 0, 150]",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {"text": "المشروع: {اسم المشروع}\nالعقد: {رقم العقد}\nالقيمة السنوية: {القيمة السنوية}\nالحي: {اسم الحي}\nقرب الحرم (كم): {dist_haram_km}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 بنك البيانات (بعد التنظيف)")
    show = data.copy()
    # عرض أعمدة مفيدة
    cols = ["رقم العقد","اسم المشروع","القيمة السنوية","اسم الحي","النشاط الرئيسي","النشاط الفرعي","المدة","التجهيز","dist_haram_km","latitude","longitude"]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr><center>إستدامة | تطوير: محمد داغستاني © 2026</center>", unsafe_allow_html=True)
