import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- 1. الإعدادات البصرية (تعديل لتجنب أخطاء Markdown) ---
st.set_page_config(page_title="إستدامة", layout="wide")

# استخدام ستايل بسيط ومباشر لتجنب تعارض النسخ
st.write("""
    <style>
    .main-title { color: #1a1a1a; text-align: center; border-bottom: 2px solid #c5a059; }
    .stButton>button { background-color: #c5a059; color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_index=True)

# --- 2. الاتصال ببيانات Google Sheets ---
# ملاحظة: تأكد أن الرابط موجود في Secrets كما اتفقنا
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    st.error("جاري الاتصال بالسحابة... يرجى التأكد من الـ Secrets")

def load_sheet(name):
    try:
        data = conn.read(worksheet=name, ttl="1m")
        return data.dropna(how='all')
    except:
        return pd.DataFrame()

# --- 3. بوابة الدخول ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h2 class='main-title'>🏛️ منصة إستدامة العقارية</h2>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login"):
            u_email = st.text_input("البريد الإلكتروني")
            u_pass = st.text_input("كلمة المرور", type="password")
            if st.form_submit_button("دخول"):
                users = load_sheet("Users_DB")
                # مطابقة دقيقة مع أسماء أعمدة ملفك
                check = users[(users['البريد الإلكتروني (Email)'] == u_email) & 
                              (users['كلمة المرور (Password)'].astype(str) == u_pass)]
                if not check.empty:
                    st.session_state.auth = True
                    st.session_state.user = check.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("البيانات غير صحيحة")
else:
    # --- 4. واجهة التطبيق الرئيسية ---
    st.sidebar.title("إستدامة")
    st.sidebar.write(f"مرحباً: {st.session_state.user['الاسم (Name)']}")
    if st.sidebar.button("خروج"):
        st.session_state.auth = False
        st.rerun()

    t1, t2 = st.tabs(["🎯 التقييم", "📊 البيانات"])
    
    with t1:
        st.subheader("محاكي التقييم العقاري")
        c_a, c_b = st.columns(2)
        with c_a:
            area = st.number_input("المساحة (م2)", value=100)
            loc_val = st.select_slider("جودة الموقع", options=[1, 2, 3, 4, 5], value=3)
        with c_b:
            spec_val = st.select_slider("المواصفات", options=[1, 2, 3, 4, 5], value=3)
            age_val = st.select_slider("الحالة", options=[1, 2, 3, 4, 5], value=3)

        if st.button("احسب القيمة"):
            deals = load_sheet("Deals_DB")
            if not deals.empty:
                # تحويل البيانات لأرقام لضمان الحساب
                deals['sqm'] = pd.to_numeric(deals['القيمة السنوية']) / pd.to_numeric(deals['المساحة'])
                
                # مصفوفة التعديل (الموقع 40%، المواصفات 35%، العمر 25%)
                adj = ((loc_val - 3) * 0.40) + ((spec_val - 3) * 0.35) + ((age_val - 3) * 0.25)
                final_rate = deals['sqm'].mean() * (1 + adj)
                
                st.info(f"سعر المتر التقديري: {round(final_rate, 2)} ريال")
                st.success(f"إجمالي الإيجار السنوي: {round(final_rate * area, 2)} ريال")

    with t2:
        st.subheader("بنك الصفقات الحالي")
        st.dataframe(load_sheet("Deals_DB"))
