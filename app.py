import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Utils
import math
import uuid
from datetime import datetime
from io import BytesIO
from tempfile import NamedTemporaryFile

# PDF + Arabic
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

# QR
import qrcode
from PIL import Image

# =========================
# CONFIG
# =========================
CITY_NAME = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262
FONT_PATH = "Tajawal-Regular.ttf"  # لازم يكون موجود بجانب app.py

REQUIRED_COLS = ["latitude", "longitude", "القيمة السنوية", "المساحة", "المدينة"]

# =========================
# UI SETUP
# =========================
st.set_page_config(
    page_title="إستدامة | التقييم العقاري – مكة",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"] { text-align: right; }
.gold { color:#c5a059; font-weight:bold; }
.card { background:#fff; padding:18px; border-radius:14px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
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
    if score >= 75:
        return "High", score
    if score >= 50:
        return "Medium", score
    return "Low", score

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة الناقصة في الملف: {missing}")

    df = df.copy()
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["price"] = pd.to_numeric(df["القيمة السنوية"], errors="coerce")
    df["area"] = pd.to_numeric(df["المساحة"], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "price", "area", "المدينة"]).copy()
    df = df[(df["price"] > 0) & (df["area"] > 0)].copy()

    # قفل مكة
    df = df[df["المدينة"] == CITY_NAME].copy()

    if df.empty:
        return df

    # dist to Haram
    df["dist_haram_km"] = np.sqrt(0)  # placeholder
    df["dist_haram_km"] = df.apply(
        lambda r: haversine_km(r["lat"], r["lon"], HARAM_LAT, HARAM_LON),
        axis=1,
    )
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[["area", "lat", "lon", "dist_haram_km"]]
    y = df["price"]

    # إذا البيانات قليلة جدًا
    if len(df) < 25:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        return model, 0.0

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    r2v = r2_score(yte, model.predict(Xte))
    return model, float(r2v)

class GovPDF(FPDF):
    pass

def build_pdf_report(report: dict, verify_url: str) -> bytes:
    # QR image -> temp file (fpdf2 يحب المسار)
    qr_img = qrcode.make(verify_url)
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    qr_img.save(tmp.name)

    pdf = GovPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    # Arabic font
    pdf.add_font("Tajawal", "", FONT_PATH, uni=True)
    pdf.set_font("Tajawal", size=14)

    # Watermark (نسخة أولية)
    pdf.set_text_color(200, 200, 200)
    pdf.set_font("Tajawal", size=42)
    pdf.rotate(25, x=60, y=180)
    pdf.text(25, 160, ar("نسخة أولية"))
    pdf.rotate(0)

    # Header
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Tajawal", size=16)
    pdf.cell(0, 10, ar("منصة إستدامة للتقييم العقاري الذكي – مكة المكرمة"), ln=True, align="R")

    pdf.set_font("Tajawal", size=11)
    pdf.cell(0, 8, ar("تقرير تقديري آلي (للاستخدام الرسمي)"), ln=True, align="R")

    pdf.ln(4)

    # Body fields
    pdf.set_font("Tajawal", size=12)
    for k, v in report.items():
        pdf.multi_cell(0, 8, ar(f"{k}: {v}"), align="R")

    # QR
    pdf.image(tmp.name, x=165, y=250, w=35, h=35)
    pdf.set_font("Tajawal", size=9)
    pdf.set_xy(10, 288)
    pdf.cell(0, 6, ar("QR للتحقق: " + verify_url), align="R")

    # Page 2: Model Card
    pdf.add_page()
    pdf.add_font("Tajawal", "", FONT_PATH, uni=True)
    pdf.set_font("Tajawal", size=16)
    pdf.cell(0, 10, ar("Model Card – منهجية النموذج"), ln=True, align="R")
    pdf.ln(2)
    pdf.set_font("Tajawal", size=12)

    model_card_lines = [
        "نوع النموذج: Gradient Boosting Regressor",
        "النطاق: مدينة مكة المكرمة فقط",
        "الهدف: تقدير القيمة السنوية للعقار بناءً على بيانات الصفقات",
        "المدخلات (Features): المساحة، خط العرض، خط الطول، المسافة للحرم (كم)",
        "المخرجات: القيمة السنوية المتوقعة (ريال)",
        "التقييم: R² لقياس جودة التنبؤ (كلما اقترب من 1 كان أفضل)",
        "حدود الاستخدام:",
        "- التقرير تقديري آلي ولا يُعد اعتمادًا نهائيًا",
        "- يتأثر بجودة البيانات وعدد الصفقات المشابهة",
        "- يُستخدم كأداة دعم قرار وليس بديلًا عن التقييم الرسمي",
    ]
    for line in model_card_lines:
        pdf.multi_cell(0, 8, ar(line), align="R")

    out = pdf.output(dest="S").encode("latin-1")
    return out

# =========================
# APP
# =========================
st.markdown(f"""
<h1 style="text-align:center">
🏛️ منصة <span class="gold">إستدامة</span><br>
مدينة مكة المكرمة
</h1>
<p style="text-align:center; color:#666">
نسخة حكومية – تشغيل عبر رفع ملف البيانات (بدون Google Sheets)
</p>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 رفع ملف البيانات")
    up = st.file_uploader("ارفع ملف Excel أو CSV يحتوي على بيانات الصفقات", type=["xlsx", "xls", "csv"])

    st.caption("المطلوب: latitude, longitude, القيمة السنوية, المساحة, المدينة")

    if not up:
        st.info("ارفع ملف البيانات للبدء.")
        st.stop()

    try:
        if up.name.lower().endswith(".csv"):
            raw = pd.read_csv(up)
        else:
            raw = pd.read_excel(up)

        data = clean_df(raw)

        if data.empty:
            st.error("لا توجد صفقات لمكة المكرمة في الملف أو البيانات ناقصة/غير صحيحة.")
            st.stop()

    except Exception as e:
        st.error(f"تعذر قراءة الملف أو تنظيفه: {e}")
        st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

# Train once
model, model_r2 = train_model(data)

tab1, tab2, tab3 = st.tabs(["🎯 محرك التقييم", "🗺️ الخريطة", "📊 بنك البيانات"])

with tab1:
    c1, c2 = st.columns([1, 1.2], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 مواصفات العقار المستهدف")
        target_area = st.number_input("المساحة (م²)", min_value=1, value=500)

        st.write("📌 (اختياري) إدخال إحداثيات العقار لزيادة دقة قرب الحرم")
        use_loc = st.checkbox("سأدخل الإحداثيات", value=False)
        if use_loc:
            target_lat = st.number_input("Latitude", value=float(data["lat"].mean()), format="%.6f")
            target_lon = st.number_input("Longitude", value=float(data["lon"].mean()), format="%.6f")
        else:
            target_lat = float(data["lat"].mean())
            target_lon = float(data["lon"].mean())

        if st.button("إصدار تقرير حكومي"):
            target_dist = float(haversine_km(target_lat, target_lon, HARAM_LAT, HARAM_LON))

            pred = float(model.predict([[target_area, target_lat, target_lon, target_dist]])[0])

            similar = data[data["area"].between(target_area * 0.8, target_area * 1.2)]
            conf_label, conf_score = gov_confidence(int(len(similar)), model_r2)

            rid = make_report_id()
            verify_url = f"https://verify.estidama.sa/{rid}"

            report = {
                "رقم التقرير": rid,
                "المدينة": CITY_NAME,
                "تاريخ الإصدار": datetime.now().strftime("%Y-%m-%d"),
                "المساحة": f"{target_area} م²",
                "القيمة السنوية المتوقعة (ML)": f"{pred:,.0f} ريال",
                "قرب الحرم": f"{target_dist:.2f} كم",
                "مستوى الثقة الحكومي": f"{conf_label} ({conf_score}/100)",
                "دقة النموذج R²": f"{model_r2:.2f}",
                "عدد الصفقات (مكة)": int(len(data)),
                "عدد الصفقات المشابهة (مساحة)": int(len(similar)),
            }

            pdf_bytes = build_pdf_report(report, verify_url)

            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("✅ نتائج فورية")
                st.metric("القيمة السنوية المتوقعة (ML)", f"{pred:,.0f} ريال")
                st.metric("قرب الحرم", f"{target_dist:.2f} كم")
                st.metric("مستوى الثقة الحكومي", conf_label)
                st.progress(conf_score / 100)
                st.caption(f"R²: {model_r2:.2f} | صفقات مشابهة: {len(similar)}")
                st.download_button(
                    "⬇️ تحميل التقرير PDF",
                    data=pdf_bytes,
                    file_name=f"{rid}.pdf",
                    mime="application/pdf",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🗺️ خريطة الصفقات داخل مكة")
    view = pdk.ViewState(
        latitude=float(data["lat"].mean()),
        longitude=float(data["lon"].mean()),
        zoom=12,
        pitch=45,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color=[255, 180, 0, 140],
        pickable=True,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "القيمة: {القيمة السنوية}\nالمساحة: {المساحة}"}))

with tab3:
    st.subheader("📊 بنك بيانات مكة")
    st.dataframe(data.drop(columns=["lat", "lon", "price", "area"], errors="ignore"), use_container_width=True)

st.markdown("<hr><center>إستدامة | تطوير: محمد داغستاني © 2026</center>", unsafe_allow_html=True)
