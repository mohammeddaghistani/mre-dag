# Stage 1 Prototype – Municipal Rental Valuation System
# Internal Use – Decision Support & Strategic Planning

import streamlit as st
from streamlit_folium import st_folium
import folium

# ------------------ App Config ------------------
st.set_page_config(
    page_title="Municipal Rental Valuation – Prototype",
    page_icon="📊",
    layout="wide",
)

# ------------------ Styling ------------------
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'Tajawal';
        src: url('Tajawal-Regular.ttf');
    }
    html, body, [class*="css"]  {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Header ------------------
col1, col2 = st.columns([1,4])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.markdown("## نظام دعم قرار تقييم القيمة الإيجارية")
    st.markdown("### نموذج أولي – مرحلة أولى (استخدام داخلي)")

st.divider()

# ------------------ Sidebar ------------------
st.sidebar.header("بيانات التقييم")

activity = st.sidebar.selectbox(
    "نوع النشاط",
    [
        "تجاري", "صناعي", "صحي", "تعليمي", "رياضي وترفيهي",
        "سياحي", "زراعي وحيواني", "بيئي", "اجتماعي", "نقل",
        "مركبات", "صيانة وتعليم وتركيب", "تشييد وإدارة عقارات",
        "خدمات عامة", "ملبوسات ومنسوجات", "مرافق عامة", "مالي"
    ]
)

city = st.sidebar.text_input("المدينة")
district = st.sidebar.text_input("الحي")
area = st.sidebar.number_input("المساحة (م²)", min_value=0.0)
contract_years = st.sidebar.number_input("مدة العقد (سنة)", min_value=1, value=10)

st.sidebar.divider()
st.sidebar.info("القيم الصفرية يمكن إدخالها لاحقًا دون التأثير على التقييم")

# ------------------ Map Section ------------------
st.markdown("## تحديد موقع الأرض")

m = folium.Map(location=[24.7136, 46.6753], zoom_start=6)

map_data = st_folium(m, height=450, width=None)

lat, lon = None, None
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"تم اختيار الموقع: خط العرض {lat:.5f} ، خط الطول {lon:.5f}")

# ------------------ Evaluation Logic (Prototype) ------------------
st.divider()
st.markdown("## نتيجة التقييم المبدئي")

if st.button("تنفيذ التقييم"):
    if area > 0 and lat and lon:
        base_rate = 50  # قيمة استرشادية مؤقتة
        recommended = area * base_rate
        st.metric("القيمة الإيجارية السنوية المقترحة", f"{recommended:,.0f} ريال")
        st.write("**المنهج المستخدم:** أسلوب الدخل (تجريبي)")
        st.write("**مستوى الثقة:** متوسط – مرحلة أولى")
    else:
        st.warning("يرجى إدخال المساحة وتحديد الموقع على الخريطة")

# ------------------ Footer ------------------
st.divider()
st.caption("© نموذج أولي – دعم قرارات لجان المنافسات والتخطيط الاستراتيجي")
