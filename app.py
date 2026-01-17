import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import base64

# --- إعدادات الصفحة والجوال ---
st.set_page_config(
    page_title="منصة إستدامة | Estidama Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- نظام اللغات والترجمة ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'Arabic'

def switch_lang():
    st.session_state.lang = 'English' if st.session_state.lang == 'Arabic' else 'Arabic'

st.sidebar.button("🌐 Switch Language / تغيير اللغة", on_click=switch_lang)

# مصفوفة الترجمة الشاملة
if st.session_state.lang == 'Arabic':
    t = {
        "dir": "rtl",
        "title": "🏛️ منصة إستدامة لتنبؤ القيمة الإيجارية",
        "subtitle": "نظام التقييم الذكي للأراضي الاستثمارية (المادة 26)",
        "act_label": "نوع النشاط الاستثماري (17 نشاطاً مسموحاً)",
        "activities": [
            "التجارية", "الصناعية", "الصحية", "التعليمية", "الرياضية والترفيهية",
            "السياحية", "الزراعية والحيوانية", "البيئية", "الاجتماعية", "النقل",
            "المركبات", "الصيانة والتعليم والتركيب", "التشييد وإدارة العقارات",
            "الخدمات العامة", "الملبوسات والمنسوجات", "المرافق العامة", "المالية"
        ],
        "params": "⚙️ محاور التقييم الهامة",
        "dist": "المسافة عن الحرم المكي (كم)",
        "area": "مساحة الأرض (م2)",
        "base": "متوسط سعر الحي (ريال/م2)",
        "fronts": "عدد الواجهات",
        "topo": "طبيعة الأرض",
        "topo_opt": ["مستوية", "منحدرة", "جبلية / مجرى سيل"],
        "map_btn": "📍 تحديد الموقع (مستكشف بلدي الجغرافي)",
        "calc_btn": "إصدار التقرير والتحليل المالي",
        "results": "📊 نتائج التنبؤ والتحليل",
        "sens_title": "📉 تحليل الحساسية للعائد المستهدف (Sensitivity Analysis)",
        "yield_label": "العائد الاستثماري المستهدف (%)",
        "method_label": "أسلوب التقييم المفترض بناءً على النشاط:",
        "final_val": "القيمة الإيجارية السنوية المقدرة:"
    }
else:
    t = {
        "dir": "ltr",
        "title": "🏛️ Estidama Rental Prediction Platform",
        "subtitle": "Smart Valuation System for Investment Lands (Article 26)",
        "act_label": "Investment Activity Type (17 Approved Activities)",
        "activities": [
            "Commercial", "Industrial", "Health", "Educational", "Sports & Leisure",
            "Tourism", "Agricultural", "Environmental", "Social", "Transport",
            "Vehicles", "Maintenance & Installation", "Construction & Property Mgmt",
            "Public Services", "Apparel & Textiles", "Public Utilities", "Financial"
        ],
        "params": "⚙️ Key Valuation Pillars",
        "dist": "Distance to Haram (km)",
        "area": "Land Area (sqm)",
        "base": "District Avg Price (SAR/sqm)",
        "fronts": "Number of Frontages",
        "topo": "Topography",
        "topo_opt": ["Flat", "Sloped", "Mountainous / Flood Path"],
        "map_btn": "📍 Locate via Balady Geo-Explorer",
        "calc_btn": "Generate Report & Financial Analysis",
        "results": "📊 Prediction & Analysis Results",
        "sens_title": "📉 Yield Sensitivity Analysis",
        "yield_label": "Target Yield (%)",
        "method_label": "Assumed Valuation Method based on Activity:",
        "final_val": "Estimated Annual Rental Value:"
    }

# --- تنسيق الواجهة بناءً على اللغة ---
st.markdown(f"""<div style='text-align: center;'> <h1 style='color: #1a3a5a;'>{t['title']}</h1> <p>{t['subtitle']}</p> </div>""", unsafe_allow_html=True)

# --- المدخلات الرئيسية ---
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(t["params"])
        activity = st.selectbox(t["act_label"], t["activities"])
        
        c_inner1, c_inner2 = st.columns(2)
        with c_inner1:
            dist_haram = st.number_input(t["dist"], min_value=0.1, value=5.0, step=0.1)
            land_area = st.number_input(t["area"], min_value=1.0, value=1000.0)
        with c_inner2:
            base_price = st.number_input(t["base"], min_value=1.0, value=500.0)
            frontages = st.slider(t["fronts"], 1, 4, 1)
            
        topography = st.selectbox(t["topo"], t["topo_opt"])
        
    with col2:
        st.subheader("🔗 Links & Maps")
        st.info(f"[{t['map_btn']}](https://umaps.balady.gov.sa/)")
        lat = st.text_input("Latitude (خط العرض)", "21.4225")
        lng = st.text_input("Longitude (خط الطول)", "39.8262")

# --- منطق الحساب (الخوارزمية المرجحة) ---
def run_valuation_engine():
    # 1. تحديد الأسلوب
    income_acts = ["التجارية", "السياحية", "المالية", "الرياضية والترفيهية", "Commercial", "Tourism", "Financial", "Sports & Leisure"]
    method = "أسلوب الدخل (Income Approach)" if activity in income_acts else "أسلوب المقارنة (Market Approach)"
    
    # 2. معاملات التعديل (Adjustments)
    dist_impact = 1.6 if dist_haram < 2 else (1.3 if dist_haram < 5 else 1.0)
    front_impact = 1 + (frontages * 0.05)
    topo_impact = 0.85 if "جبلية" in topography or "Mountainous" in topography else 1.0
    premium = 1.2 if activity in income_acts else 1.0
    
    final_unit_rent = base_price * dist_impact * front_impact * topo_impact * premium
    return round(final_unit_rent, 2), method

# --- عرض النتائج ---
if st.button(t["calc_btn"], type="primary", use_container_width=True):
    predicted_rent, val_method = run_valuation_engine()
    
    st.divider()
    st.subheader(t["results"])
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric(t["final_val"], f"{predicted_rent:,.2f} SAR/m2")
    res_col2.metric("Total Annual Rent", f"{predicted_rent * land_area:,.2f} SAR")
    res_col3.info(f"{t['method_label']} \n **{val_method}**")

    # --- مصفوفة الحساسية (Sensitivity Analysis) ---
    st.subheader(t["sens_title"])
    yields = [i for i in range(5, 13)]
    sens_data = []
    for y in yields:
        # حساب القيمة الرأسمالية المفترضة بناءً على العائد
        capital_value = (predicted_rent * land_area) / (y/100)
        sens_data.append({"Yield %": f"{y}%", "Annual Rent": predicted_rent * land_area, "Capital Value (Est)": capital_value})
    
    df_sens = pd.DataFrame(sens_data)
    
    fig = px.bar(df_sens, x="Yield %", y="Capital Value (Est)", 
                 title="تغير قيمة العقار الرأسمالية حسب العائد المستهدف",
                 color_discrete_sequence=['#1a3a5a'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.table(df_sens)

    # --- زر تحميل التقرير (محاكاة) ---
    st.download_button("📩 Download Full PDF Report", data="Report Content", file_name="Estidama_Report.pdf")

# --- تذييل الصفحة للجوال ---
st.sidebar.markdown("---")
st.sidebar.write("📱 المتصفح يدعم الجوال تماماً (iOS / Android)")
st.sidebar.write("✅ متوافق مع معايير 'تقييم' والمادة 26")
