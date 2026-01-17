import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from streamlit_gsheets import GSheetsConnection

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from io import BytesIO
from datetime import datetime
import uuid
import math
import qrcode
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ======================================================
# CONFIG
# ======================================================
CITY_NAME = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262

FONT_PATH = "Tajawal-Regular.ttf"
LOGO_PATH = "logo.png"

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(
    page_title="إستدامة | منصة التقييم العقاري – مكة",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Tajawal', sans-serif;
    text-align: right;
}
.gold { color:#c5a059; font-weight:bold; }
.watermark {
    position: fixed;
    top: 40%;
    left: 30%;
    opacity: 0.05;
    font-size: 80px;
    transform: rotate(-30deg);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# UTILITIES
# ======================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def generate_report_id():
    return f"MK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

def generate_qr(data: str) -> Image.Image:
    qr = qrcode.make(data)
    return qr.resize((120, 120))

def gov_confidence(similar, r2):
    score = min(60, similar * 3) + int(max(0, min(r2,1)) * 40)
    if score >= 75: return "High", score
    if score >= 50: return "Medium", score
    return "Low", score

# ======================================================
# DATA
# ======================================================
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=60)
def load_data():
    df = conn.read(worksheet="Deals_DB")
    df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['القيمة السنوية'], errors='coerce')
    df['area'] = pd.to_numeric(df['المساحة'], errors='coerce')
    df = df.dropna()
    df['dist_haram'] = df.apply(
        lambda r: haversine(r['lat'], r['lon'], HARAM_LAT, HARAM_LON), axis=1
    )
    return df[df['المدينة'] == CITY_NAME]

# ======================================================
# ML MODEL
# ======================================================
@st.cache_resource
def train_model(df):
    X = df[['area', 'lat', 'lon', 'dist_haram']]
    y = df['price']
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    model.fit(Xtr, ytr)
    return model, r2_score(yte, model.predict(Xte))

# ======================================================
# PDF REPORT
# ======================================================
def build_pdf(report):
    buffer = BytesIO()
    pdfmetrics.registerFont(TTFont("Tajawal", FONT_PATH))
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    # Watermark
    c.setFont("Tajawal", 60)
    c.setFillGray(0.9)
    c.drawCentredString(w/2, h/2, "نسخة أولية")

    c.setFillGray(0)
    c.setFont("Tajawal", 14)
    c.drawString(50, h-50, "منصة إستدامة – التقييم العقاري الذكي")
    c.drawString(50, h-75, "مدينة مكة المكرمة")

    if Image.open(LOGO_PATH):
        c.drawImage(LOGO_PATH, w-150, h-120, width=90, height=90)

    y = h-130
    c.setFont("Tajawal", 11)

    for k,v in report.items():
        c.drawString(50, y, f"{k} : {v}")
        y -= 18

    c.showPage()

    # =====================
    # MODEL CARD PAGE
    # =====================
    c.setFont("Tajawal", 14)
    c.drawString(50, h-50, "Model Card – منهجية النموذج")

    c.setFont("Tajawal", 11)
    text = c.beginText(50, h-90)
    text.textLine("نوع النموذج: Gradient Boosting Regressor")
    text.textLine("الغرض: تقدير القيمة السنوية للعقارات داخل مكة المكرمة")
    text.textLine("المدخلات:")
    text.textLine("- المساحة")
    text.textLine("- خط العرض والطول")
    text.textLine("- المسافة إلى الحرم المكي")
    text.textLine("المخرجات:")
    text.textLine("- القيمة السنوية المتوقعة")
    text.textLine("القيود:")
    text.textLine("- التقدير آلي ولا يُعد اعتمادًا نهائيًا")
    text.textLine("- يعتمد على جودة البيانات المدخلة")
    c.drawText(text)

    c.save()
    buffer.seek(0)
    return buffer

# ======================================================
# APP
# ======================================================
st.markdown("<div class='watermark'>ESTIDAMA</div>", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center'>
🏛️ منصة <span class='gold'>إستدامة</span><br>
مدينة مكة المكرمة
</h1>
""", unsafe_allow_html=True)

data = load_data()
model, r2 = train_model(data)

area = st.number_input("المساحة (م2)", value=500)

if st.button("إصدار تقرير حكومي"):
    lat, lon = data['lat'].mean(), data['lon'].mean()
    dist = haversine(lat, lon, HARAM_LAT, HARAM_LON)
    prediction = model.predict([[area, lat, lon, dist]])[0]

    similar = data[data['area'].between(area*0.8, area*1.2)]
    conf, score = gov_confidence(len(similar), r2)

    report_id = generate_report_id()
    verify_url = f"https://verify.estidama.gov/{report_id}"

    report = {
        "رقم التقرير": report_id,
        "المدينة": CITY_NAME,
        "المساحة": area,
        "القيمة السنوية المتوقعة": f"{prediction:,.0f} ريال",
        "قرب الحرم (كم)": round(dist,2),
        "مستوى الثقة": f"{conf} ({score}/100)",
        "دقة النموذج R²": round(r2,2),
        "تاريخ الإصدار": datetime.now().strftime("%Y-%m-%d"),
    }

    pdf = build_pdf(report)
    st.success("تم إنشاء التقرير بنجاح")
    st.download_button("⬇️ تحميل التقرير PDF", pdf, file_name=f"{report_id}.pdf")
