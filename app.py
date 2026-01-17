import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- إعدادات الهوية البصرية (الأسود والذهبي) ---
st.set_page_config(page_title="إستدامة | التقييم العقاري", layout="wide")

st.markdown("""
    <style>
    :root { --primary: #1a1a1a; --gold: #c5a059; }
    .stApp { background-color: #ffffff; }
    .main-title { color: var(--primary); text-align: center; border-bottom: 3px solid var(--gold); padding-bottom: 10px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid var(--gold); margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { background-color: var(--gold); color: white; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; width: 100%; }
    </style>
    """, unsafe_allow_index=True)

# --- الاتصال ببنك المعلومات الحية ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data(worksheet_name):
    # جلب البيانات من Google Sheets مباشرة
    return conn.read(worksheet=worksheet_name, ttl="1m")

# --- محرك التقييم (مصفوفة التعديل) ---
def run_valuation(subject, deals):
    # الأوزان المتفق عليها: الموقع 40%، المواصفات 35%، العمر 25%
    weights = {'loc': 0.40, 'spec': 0.35, 'age': 0.25}
    
    # حساب سعر المتر الفعلي لكل صفقة
    deals['sqm_rate'] = pd.to_numeric(deals['القيمة السنوية']) / pd.to_numeric(deals['المساحة'])
    
    adjusted_rates = []
    for _, row in deals.iterrows():
        # معادلة التسويات النوعية (دليل 2023)
        adj = ((subject['loc'] - 3) * weights['loc']) + \
              ((subject['spec'] - 3) * weights['spec']) + \
              ((subject['age'] - 3) * weights['age'])
        adjusted_rates.append(row['sqm_rate'] * (1 + adj))
    
    return np.mean(adjusted_rates)

# --- نظام الدخول والأمان ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>🏛️ بوابة إستدامة الرقمية</h1>", unsafe_allow_index=True)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        if st.button("دخول النظام"):
            users = get_data("Users_DB")
            user_row = users[(users['البريد الإلكتروني (Email)'] == email) & (users['كلمة المرور (Password)'].astype(str) == password)]
            if not user_row.empty:
                st.session_state.logged_in = True
                st.session_state.user = user_row.iloc[0].to_dict()
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة")
        st.markdown("</div>", unsafe_allow_index=True)
else:
    # --- الواجهة الرئيسية للمستخدم ---
    st.sidebar.image("https://mdaghistani.com/wp-content/uploads/2022/04/logo-new.png", width=150)
    st.sidebar.success(f"مرحباً: {st.session_state.user['الاسم (Name)']}")
    
    tab1, tab2, tab3 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات", "⚙️ الإدارة"])
    
    with tab1:
        st.subheader("إجراء تقييم جديد")
        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input("مساحة العقار (م2)", value=1000)
            loc = st.slider("جودة الموقع", 1, 5, 3)
        with c2:
            spec = st.slider("جودة المواصفات", 1, 5, 3)
            age = st.slider("الحالة التشغيلية", 1, 5, 3)
            
        if st.button("بدء التحليل"):
            deals = get_data("Deals_DB")
            result = run_valuation({'loc': loc, 'spec': spec, 'age': age}, deals)
            st.markdown(f"<div class='card'><h3>سعر المتر التقديري: {round(result, 2)} ريال</h3>"
                        f"<h3>الإيجار السنوي المتوقع: {round(result * area, 2)} ريال</h3></div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("سجل الصفقات الحية")
        st.dataframe(get_data("Deals_DB"), use_container_width=True)

    with tab3:
        if st.session_state.user['الدور (Role)'] == 'Admin':
            st.write("إدارة النظام متاحة لك كمسؤول.")
            st.link_button("تعديل البيانات في Google Sheets", "https://docs.google.com/spreadsheets/d/12WCV2C3iiIF8sxpiKplypNA9pRYz5un4GwJMdsssGXA/edit")
        else:
            st.warning("هذه الصفحة للمدير فقط.")
