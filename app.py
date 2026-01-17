import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# 1. إعدادات الصفحة
st.set_page_config(page_title="إستدامة | نسخة الاختبار المباشر", layout="wide")

st.markdown("""
    <style>
    .main-title { color: #1a1a1a; text-align: center; border-bottom: 3px solid #c5a059; padding-bottom: 10px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid #c5a059; margin-bottom: 15px; }
    .stButton>button { background-color: #c5a059; color: white; font-weight: bold; width: 100%; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. الاتصال السحابي (تأكد أن الرابط في Secrets صحيح)
conn = st.connection("gsheets", type=GSheetsConnection)

st.markdown("<h1 class='main-title'>🏛️ منصة إستدامة - نسخة الاختبار</h1>", unsafe_allow_html=True)

# 3. محرك التقييم المباشر (بدون تسجيل دخول)
tab1, tab2 = st.tabs(["🎯 محاكي التقييم العقاري", "📊 بنك الصفقات"])

with tab1:
    st.subheader("تحليل القيمة الإيجارية")
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input("مساحة العقار (م2)", value=100)
            loc_score = st.select_slider("جودة الموقع", options=[1, 2, 3, 4, 5], value=3)
        with c2:
            spec_score = st.select_slider("المواصفات الفنية", options=[1, 2, 3, 4, 5], value=3)
            age_score = st.select_slider("الحالة/العمر", options=[1, 2, 3, 4, 5], value=3)
        
        if st.button("احسب القيمة الآن"):
            try:
                # محاولة قراءة بيانات الصفقات فقط
                deals = conn.read(worksheet="Deals_DB", ttl="1m")
                
                if not deals.empty:
                    # تحويل البيانات وحساب المتوسط
                    deals['price'] = pd.to_numeric(deals['القيمة السنوية'], errors='coerce')
                    deals['size'] = pd.to_numeric(deals['المساحة'], errors='coerce')
                    deals['rate'] = deals['price'] / deals['size']
                    avg_base = deals['rate'].mean()
                    
                    # مصفوفة التعديل
                    adj = ((loc_score - 3) * 0.40 * 0.1) + \
                          ((spec_score - 3) * 0.35 * 0.1) + \
                          ((age_score - 3) * 0.25 * 0.1)
                    
                    final_rate = avg_base * (1 + adj)
                    
                    st.divider()
                    st.metric("سعر المتر التقديري", f"{round(final_rate, 2)} ريال")
                    st.metric("الإيجار السنوي المقدر", f"{round(final_rate * area, 2)} ريال")
                else:
                    st.warning("جدول الصفقات فارغ حالياً.")
            except Exception as e:
                st.error(f"خطأ في جلب البيانات: {e}")
                st.info("تأكد من إعدادات المشاركة (Share) في ملف جوجل شيت.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("البيانات المصدرية (Deals_DB)")
    try:
        deals_view = conn.read(worksheet="Deals_DB", ttl="1m")
        st.dataframe(deals_view, use_container_width=True)
    except:
        st.error("تعذر عرض الجدول. يرجى مراجعة صلاحيات الملف.")

st.markdown("<center>إستدامة - نسخة الاختبار التقني 2026 ©</center>", unsafe_allow_html=True)
