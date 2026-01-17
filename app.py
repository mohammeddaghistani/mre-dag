import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Utils
from datetime import datetime
import uuid
import math
import qrcode
from io import BytesIO
from PIL import Image

# PDF (خفيف ومستقر)
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

# =============================
# CONFIG
# =============================
CITY_NAME = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262

# =============================
# PAGE SETUP
# =============================
st.set_page_config(
    page_title="إستدامة | التقييم العقاري – مكة",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'sans-serif';
    text-align: right;
}
.gold { color:#c5a059; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# =============================
# HELPERS
# =============================
def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def report_id():
    return f"MK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

def gov_confidence(similar, r2):
    score = min(60, similar * 3) + int(max(0, min(r2,1)) * 40)
    if score >= 75: return "High", score
    if score >= 50: return "Medium", score
    return "Low", score

# =============================
# LOAD DATA FROM GSHEETS
# =============================
@st.cache_data(ttl=120)
def load_data():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    client = gspread.authorize(creds)

    sh = client.open_by_key(st.secrets["gsheets"]["spreadsheet_id"])
    ws = sh.worksheet(st.secrets["gsheets"]["worksheet"])
    df = pd.DataFrame(ws.get_all_records())

    # تنظيف
    df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['القيمة السنوية'], errors='coerce')
    df['area'] = pd.to_numeric(df['المساحة'], errors='coerce')

    df = df.dropna(subset=['lat','lon','price','area','المدينة'])
    df = df[(df['price']>0) & (df['area']>0)]
    df = df[df['المدينة'] == CITY_NAME].copy()

    df['dist_haram'] = df.apply(
        lambda r: haversine(r['lat'], r['lon'], HARAM_LAT, HARAM_LON),
        axis=1
    )
    return df

# =============================
# TRAIN MODEL
# =============================
@st.cache_resource
def train_model(df):
    X = df[['area','lat','lon','dist_haram']]
    y = df['price']
    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    r2 = r2_score(yte, model.predict(Xte))
    return model, r2

# =============================
# PDF REPORT
# =============================
def build_pdf(data: dict, qr_img: Image.Image):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", "", "", uni=True)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, ar("منصة إستدامة – تقرير تقييم عقاري (نسخة أولية)"), ln=True)
    pdf.ln(5)

    for k,v in data.items():
        pdf.cell(0, 8, ar(f"{k} : {v}"), ln=True)

    # QR
    qr_buffer = BytesIO()
    qr_img.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    pdf.image(qr_buffer, x=150, y=240, w=40)

    return pdf.output(dest="S").encode("latin-1")

# =============================
# APP UI
# =============================
st.markdown("""
<h1 style="text-align:center">
🏛️ منصة <span class="gold">إستدامة</span><br>
مدينة مكة المكرمة
</h1>
""", unsafe_allow_html=True)

data = load_data()
if data.empty:
    st.error("لا توجد بيانات لمكة المكرمة")
    st.stop()

model, r2 = train_model(data)

area = st.number_input("المساحة (م²)", min_value=1, value=500)

if st.button("إصدار تقرير حكومي"):
    lat, lon = data['lat'].mean(), data['lon'].mean()
    dist = haversine(lat, lon, HARAM_LAT, HARAM_LON)
    prediction = model.predict([[area, lat, lon, dist]])[0]

    similar = data[data['area'].between(area*0.8, area*1.2)]
    conf, score = gov_confidence(len(similar), r2)

    rid = report_id()
    verify_url = f"https://verify.estidama.sa/{rid}"
    qr = qrcode.make(verify_url)

    report = {
        "رقم التقرير": rid,
        "المدينة": CITY_NAME,
        "المساحة": f"{area} م²",
        "القيمة السنوية المتوقعة": f"{prediction:,.0f} ريال",
        "قرب الحرم": f"{dist:.2f} كم",
        "مستوى الثقة": f"{conf} ({score}/100)",
        "دقة النموذج": round(r2,2),
        "تاريخ الإصدار": datetime.now().strftime("%Y-%m-%d"),
    }

    pdf_bytes = build_pdf(report, qr)

    st.success("تم إنشاء التقرير بنجاح")
    st.download_button(
        "⬇️ تحميل التقرير PDF",
        data=pdf_bytes,
        file_name=f"{rid}.pdf",
        mime="application/pdf"
    )

# =============================
# MAP
# =============================
st.subheader("🗺️ خريطة الصفقات داخل مكة")
view = pdk.ViewState(
    latitude=data['lat'].mean(),
    longitude=data['lon'].mean(),
    zoom=12,
    pitch=45
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position='[lon, lat]',
    get_radius=80,
    get_fill_color=[255, 180, 0, 140],
    pickable=True
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))

st.markdown("<hr><center>إستدامة | تطوير: محمد داغستاني © 2026</center>", unsafe_allow_html=True)
