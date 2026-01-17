import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from streamlit_gsheets import GSheetsConnection

# 1. تهيئة الصفحة وتنسيق الـ CSS المتقدم
st.set_page_config(page_title="إستدامة | التقييم العقاري الذكي", layout="wide", initial_sidebar_state="collapsed")

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

# 2. جلب ومعالجة البيانات
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=60)
def get_clean_data():
    df = conn.read(worksheet="Deals_DB")
    # تحويل الإحداثيات والبيانات المالية لأرقام
    df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['القيمة السنوية'], errors='coerce')
    df['area'] = pd.to_numeric(df['المساحة'], errors='coerce')
    return df.dropna(subset=['lat', 'lon', 'price', 'area'])

try:
    data = get_clean_data()
    
    # 3. واجهة المستخدم الرئيسية
    st.markdown("<h1 style='text-align: center; color: #1a1a1a;'>🏛️ منصة <span class='gold-text'>إستدامة</span> للتقييم العقاري</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>نظام الذكاء الاصطناعي لتحليل عقارات البلدية وفق معايير 2026</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 محرك التقييم", "🗺️ الخريطة التفاعلية", "📊 بنك البيانات"])

    with tab1:
        col_input, col_res = st.columns([1, 1.2], gap="large")
        
        with col_input:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.subheader("📝 مواصفات العقار المستهدف")
            target_area = st.number_input("المساحة الإجمالية (م2)", min_value=1, value=500)
            
            st.markdown("---")
            st.write("📈 **معايير الجودة النوعية (1-5)**")
            q_loc = st.select_slider("قوة الموقع الاستراتيجي", options=[1,2,3,4,5], value=3)
            q_spec = st.select_slider("المواصفات الفنية والإنشائية", options=[1,2,3,4,5], value=3)
            q_age = st.select_slider("عمر العقار وحالته", options=[1,2,3,4,5], value=3)
            
            if st.button("إصدار التقرير التقديري"):
                # محرك التقييم
                data['sqm_rate'] = data['price'] / data['area']
                base_avg = data['sqm_rate'].mean()
                
                # مصفوفة التعديل (40% موقع، 35% مواصفات، 25% عمر)
                adj = ((q_loc - 3) * 0.40 * 0.1) + ((q_spec - 3) * 0.35 * 0.1) + ((q_age - 3) * 0.25 * 0.1)
                final_rate = base_avg * (1 + adj)
                
                with col_res:
                    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
                    st.markdown("### 📊 نتائج التحليل الذكي")
                    res_c1, res_c2 = st.columns(2)
                    res_c1.metric("سعر المتر التقديري", f"{round(final_rate, 2)} ريال")
                    res_c2.metric("الإيجار السنوي المتوقع", f"{round(final_rate * target_area, 0):,} ريال")
                    
                    st.markdown("---")
                    st.write("⚠️ **ملاحظة:** تم الحساب بناءً على مقارنة بـ **{} صفقة** مماثلة في المنطقة.".format(len(data)))
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("📍 التوزيع الجغرافي للصفقات المعتمدة")
        # إعداد الخريطة باستخدام pydeck
        view_state = pdk.ViewState(latitude=data['lat'].mean(), longitude=data['lon'].mean(), zoom=11, pitch=45)
        layer = pdk.Layer(
            "ColumnLayer",
            data=data,
            get_position="[lon, lat]",
            get_elevation="price / 100",
            radius=100,
            get_fill_color="[255, (1 - price/500000) * 255, 0, 140]",
            pickable=True,
            auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "المشروع: {اسم المشروع}\nالقيمة: {القيمة السنوية}"}))

    with tab3:
        st.subheader("📊 سجل البيانات التاريخية")
        st.dataframe(data.drop(['lat', 'lon'], axis=1), use_container_width=True)

except Exception as e:
    st.error(f"يرجى التأكد من تعبئة الإحداثيات في ملف الإكسيل (latitude/longitude): {e}")

st.markdown("<br><hr><center>إستدامة | تطوير: محمد داغستاني 2026 ©</center>", unsafe_allow_html=True)
