import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from streamlit_gsheets import GSheetsConnection

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# 0) ثوابت مكة + أدوات
# =========================
CITY_NAME = "مكة المكرمة"
HARAM_LAT = 21.4225
HARAM_LON = 39.8262

def haversine_km_np(lat, lon, lat2, lon2):
    """Vectorized Haversine distance in KM (lat/lon arrays)."""
    R = 6371.0
    lat = np.radians(lat.astype(float))
    lon = np.radians(lon.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat
    dlon = lon2 - lon
    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def gov_confidence(similar_count: int, r2v: float) -> tuple[str, int]:
    # نقاط على 100: (60 للبيانات المشابهة) + (40 لدقة النموذج)
    count_score = min(60, similar_count * 3)      # 20 صفقة ~ 60
    r2_pts = int(np.clip(r2v, 0, 1) * 40)         # 0..40
    score = int(count_score + r2_pts)
    if score >= 75:
        return "High", score
    elif score >= 50:
        return "Medium", score
    return "Low", score

def build_pdf_report(payload: dict) -> bytes:
    """PDF بسيط رسمي (إنجليزي/أرقام) لضمان التوافق بدون خطوط عربية."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Estidama | Smart Real Estate Valuation")
    y -= 25
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Makkah Al-Mukarramah - Preliminary AI Valuation Report (2026)")
    y -= 30

    c.setFont("Helvetica", 11)
    for k, v in payload.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 11)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Disclaimer: Automated estimate based on available transactions; subject to official review.")
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.getvalue()

@st.cache_resource
def train_makkah_model(df: pd.DataFrame):
    X = df[['area', 'lat', 'lon', 'dist_haram_km']]
    y = df['price']

    # حماية: إن كانت البيانات قليلة جدًا
    if len(df) < 20:
        # نموذج بسيط جدًا لتفادي انهيار التدريب
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        return model, 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    return model, float(score)

# =========================
# 1) إعداد الصفحة + CSS
# =========================
st.set_page_config(
    page_title="إستدامة | التقييم العقاري الذكي - مكة",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; text-align: right; }
.main { background-color: #f4f7f6; }
.stMetric { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid #c5a059; }
.report-card { background: white; padding: 25px; border-radius: 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); margin-bottom: 25px; border-right: 10px solid #1a1a1a; }
.gold-text { color: #c5a059; font-weight: bold; }
.stButton>button { background: linear-gradient(135deg, #1a1a1a 0%, #333 100%); color: #c5a059; border: 1px solid #c5a059; border-radius: 12px; height: 3.5rem; font-size: 1.1rem; transition: 0.3s; }
.stButton>button:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(197, 160, 89, 0.3); color: white; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2) جلب البيانات وتنظيفها
# =========================
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=60)
def get_clean_data():
    df = conn.read(worksheet="Deals_DB")

    # تأكيد الأعمدة الأساسية
    required_cols = ['latitude', 'longitude', 'القيمة السنوية', 'المساحة', 'المدينة']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة التالية غير موجودة في الشيت: {missing}")

    df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['القيمة السنوية'], errors='coerce')
    df['area'] = pd.to_numeric(df['المساحة'], errors='coerce')

    df = df.dropna(subset=['lat', 'lon', 'price', 'area', 'المدينة']).copy()

    # فلترة قيم غير منطقية (اختياري حكومي)
    df = df[(df['area'] > 0) & (df['price'] > 0)].copy()

    # مسافة الحرم (كم) - Vectorized
    df['dist_haram_km'] = haversine_km_np(df['lat'].to_numpy(), df['lon'].to_numpy(), HARAM_LAT, HARAM_LON)

    return df

try:
    data = get_clean_data()

    # =========================
    # 3) قفل مكة المكرمة
    # =========================
    city_data = data[data['المدينة'] == CITY_NAME].copy()
    if city_data.empty:
        st.error("لا توجد بيانات كافية لمدينة مكة المكرمة داخل الشيت (Deals_DB).")
        st.stop()

    # تدريب ML على مكة
    model, model_score = train_makkah_model(city_data)

    # واجهة رئيسية
    st.markdown(f"""
        <h1 style='text-align: center; color: #1a1a1a;'>
        🏛️ منصة <span class='gold-text'>إستدامة</span> للتقييم العقاري
        <br><span style='font-size:0.85em; color:#444;'>مدينة مكة المكرمة</span>
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; color: #666;'>
        نظام ذكاء اصطناعي لتحليل وتقدير القيم العقارية بناءً على الصفقات الفعلية ومعايير 2026
        </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 محرك التقييم (ML)", "🗺️ الخريطة التفاعلية", "📊 بنك بيانات مكة"])

    # =========================
    # TAB 1: محرك التقييم
    # =========================
    with tab1:
        col_input, col_res = st.columns([1, 1.2], gap="large")

        with col_input:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.subheader("📝 مواصفات العقار المستهدف (مكة)")

            target_area = st.number_input("المساحة الإجمالية (م2)", min_value=1, value=500)

            st.markdown("---")
            st.write("📈 **معايير الجودة النوعية (1-5)**")
            q_loc = st.select_slider("قوة الموقع الاستراتيجي", options=[1, 2, 3, 4, 5], value=3)
            q_spec = st.select_slider("المواصفات الفنية والإنشائية", options=[1, 2, 3, 4, 5], value=3)
            q_age = st.select_slider("عمر العقار وحالته", options=[1, 2, 3, 4, 5], value=3)

            # اختياري: السماح بإدخال موقع تقريبي داخل مكة (لرفع دقة قرب الحرم)
            st.markdown("---")
            st.write("📍 **إحداثيات العقار (اختياري لتحسين قرب الحرم)**")
            use_custom_loc = st.checkbox("سأدخل إحداثيات العقار", value=False)
            if use_custom_loc:
                target_lat = st.number_input("Latitude", value=float(city_data['lat'].mean()), format="%.6f")
                target_lon = st.number_input("Longitude", value=float(city_data['lon'].mean()), format="%.6f")
            else:
                target_lat = float(city_data['lat'].mean())
                target_lon = float(city_data['lon'].mean())

            if st.button("إصدار التقرير التقديري (مكة)"):
                # صفقات مشابهة بالمساحة (للثقة الحكومية)
                similar = city_data[city_data['area'].between(target_area * 0.8, target_area * 1.2)]
                similar_count = int(len(similar))

                # متوسط سعري مرجعي (للمقارنة فقط)
                sqm_rate = city_data['price'] / city_data['area']
                base_avg = float(sqm_rate.mean())

                # تعديل جودة (مقيد)
                adj = (
                    (q_loc - 3) * 0.04 +
                    (q_spec - 3) * 0.035 +
                    (q_age - 3) * 0.025
                )
                adj = float(np.clip(adj, -0.12, 0.12))
                quality_rate = base_avg * (1 + adj)

                # ML: قرب الحرم كـ Feature
                target_dist = float(haversine_km_np(np.array([target_lat]), np.array([target_lon]), HARAM_LAT, HARAM_LON)[0])

                ml_prediction = float(model.predict([[
                    target_area,
                    target_lat,
                    target_lon,
                    target_dist
                ]])[0])

                # ثقة حكومية
                conf_label, conf_score = gov_confidence(similar_count, model_score)

                with col_res:
                    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
                    st.markdown("### 📊 نتائج التحليل الذكي (مكة)")

                    r1, r2 = st.columns(2)
                    r1.metric("تقدير مرجعي (سعر/م بعد الجودة)", f"{quality_rate:,.2f} ريال/م²")
                    r2.metric("تقدير سنوي مرجعي", f"{(quality_rate * target_area):,.0f} ريال")

                    st.markdown("---")
                    r3, r4 = st.columns(2)
                    r3.metric("القيمة السنوية المتوقعة (ML)", f"{ml_prediction:,.0f} ريال")
                    r4.metric("قرب الحرم", f"{target_dist:.2f} كم")

                    st.markdown("---")
                    st.metric("مستوى الثقة الحكومي", conf_label)
                    st.progress(conf_score / 100)
                    st.caption(f"درجة الثقة: {conf_score}/100 | صفقات مشابهة بالمساحة: {similar_count} | دقة النموذج R²: {round(model_score, 2)}")

                    st.markdown("---")
                    st.write(f"🧾 تم الحساب بناءً على بيانات **{len(city_data)}** صفقة داخل **مكة المكرمة**.")

                    # PDF
                    report_payload = {
                        "City": "Makkah Al-Mukarramah",
                        "Target Area (sqm)": target_area,
                        "AI Predicted Annual Value (SAR)": f"{ml_prediction:,.0f}",
                        "Reference Annual Value (SAR)": f"{(quality_rate * target_area):,.0f}",
                        "Distance to Haram (km)": round(target_dist, 2),
                        "Gov Confidence": f"{conf_label} ({conf_score}/100)",
                        "Model R2": round(model_score, 2),
                        "Comparable Deals (Area Similar)": similar_count,
                        "Deals Count (Makkah)": len(city_data),
                        "Quality Location (1-5)": q_loc,
                        "Quality Specs (1-5)": q_spec,
                        "Quality Age (1-5)": q_age,
                    }

                    pdf_bytes = build_pdf_report(report_payload)
                    st.download_button(
                        "⬇️ تحميل التقرير PDF",
                        data=pdf_bytes,
                        file_name="Estidama_Makkah_AI_Report.pdf",
                        mime="application/pdf"
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # TAB 2: خريطة تفاعلية
    # =========================
    with tab2:
        st.subheader("📍 التوزيع الجغرافي للصفقات المعتمدة داخل مكة المكرمة")

        view_state = pdk.ViewState(
            latitude=float(city_data['lat'].mean()),
            longitude=float(city_data['lon'].mean()),
            zoom=12,
            pitch=45
        )

        # لون آمن بدون قيم سالبة: نعتمد تطبيع بسيط على نطاق الأسعار
        p_min = float(city_data['price'].min())
        p_max = float(city_data['price'].max())
        denom = (p_max - p_min) if (p_max - p_min) != 0 else 1.0

        city_data = city_data.copy()
        city_data['price_norm'] = (city_data['price'] - p_min) / denom

        layer = pdk.Layer(
            "ColumnLayer",
            data=city_data,
            get_position="[lon, lat]",
            get_elevation="price / 100",
            radius=100,
            get_fill_color="[255, 255*(1-price_norm), 0, 140]",
            pickable=True,
            auto_highlight=True,
        )

        tooltip_text = "القيمة السنوية: {القيمة السنوية}\nالمساحة: {المساحة}\nقرب الحرم (كم): {dist_haram_km}"
        if 'اسم المشروع' in city_data.columns:
            tooltip_text = "المشروع: {اسم المشروع}\n" + tooltip_text

        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": tooltip_text})
        )

    # =========================
    # TAB 3: بنك البيانات
    # =========================
    with tab3:
        st.subheader("📊 سجل بيانات مكة المكرمة")
        drop_cols = [c for c in ['lat', 'lon', 'price_norm'] if c in city_data.columns]
        st.dataframe(city_data.drop(drop_cols, axis=1), use_container_width=True)

except Exception as e:
    st.error(f"حدث خطأ. يرجى التأكد من الأعمدة والإحداثيات في الشيت (latitude/longitude) وباقي الحقول: {e}")

st.markdown("<br><hr><center>إستدامة | تطوير: محمد داغستاني 2026 ©</center>", unsafe_allow_html=True)
