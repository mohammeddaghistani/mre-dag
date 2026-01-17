import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- 1. الإعدادات البصرية (الهوية الاستراتيجية) ---
st.set_page_config(page_title="إستدامة | التقييم العقاري الذكي", layout="wide")

st.markdown("""
    <style>
    :root { --primary: #1a1a1a; --gold: #c5a059; }
    .stApp { background-color: #ffffff; }
    .main-title { color: var(--primary); text-align: center; border-bottom: 3px solid var(--gold); padding-bottom: 10px; margin-bottom: 20px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid var(--gold); margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { background-color: var(--gold); color: white; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; width: 100%; }
    .stButton>button:hover { background-color: #a6854a; }
    /* تحسين واجهة الجوال */
    @media (max-width: 600px) { .stMetric { font-size: 14px; } .card { padding: 15px; } }
    </style>
    """, unsafe_allow_index=True)

# --- 2. محرك الاتصال ببنك المعلومات السحابي ---
def load_connection():
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception as e:
        st.error(f"خطأ في الاتصال بالسحابة: {e}")
        return None

conn = load_connection()

def get_data(worksheet_name):
    try:
        # قراءة البيانات مع ضمان استقرارها
        df = conn.read(worksheet=worksheet_name, ttl="1m")
        return df.dropna(how='all') # حذف الصفوف الفارغة
    except Exception as e:
        st.error(f"فشل جلب ورقة {worksheet_name}. تأكد من إعدادات Secrets.")
        return pd.DataFrame()

# --- 3. محرك التقييم الذكي (مصفوفة التعديل) ---
def valuation_engine(subject_metrics, deals_df):
    # الأوزان النسبية حسب دليل التقييم 2023: الموقع 40%، المواصفات 35%، العمر 25%
    weights = {'loc': 0.40, 'spec': 0.35, 'age': 0.25}
    
    # تنظيف البيانات الحسابية
    deals_df['sqm_rate'] = pd.to_numeric(deals_df['القيمة السنوية']) / pd.to_numeric(deals_df['المساحة'])
    
    adjusted_rates = []
    for _, row in deals_df.iterrows():
        # معادلة التسويات النوعية (المرجع هو الدرجة 3)
        adj = ((subject_metrics['loc'] - 3) * weights['loc']) + \
              ((subject_metrics['spec'] - 3) * weights['spec']) + \
              ((subject_metrics['age'] - 3) * weights['age'])
        adjusted_rates.append(row['sqm_rate'] * (1 + adj))
    
    return np.mean(adjusted_rates)

# --- 4. بوابة الدخول الآمن ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>🏛️ منصة إستدامة الرقمية</h1>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        email_input = st.text_input("البريد الإلكتروني")
        pass_input = st.text_input("كلمة المرور", type="password")
        if st.button("دخول النظام"):
            users_df = get_data("Users_DB")
            # التحقق من البيانات بناءً على ملفك المرفوع
            user = users_df[(users_df['البريد الإلكتروني (Email)'] == email_input) & 
                            (users_df['كلمة المرور (Password)'].astype(str) == pass_input)]
            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.user_info = user.iloc[0].to_dict()
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة")
        st.markdown("</div>", unsafe_allow_index=True)
else:
    # --- 5. واجهة النظام الرئيسية ---
    st.sidebar.image("https://mdaghistani.com/wp-content/uploads/2022/04/logo-new.png", width=120)
    st.sidebar.markdown(f"**المستخدم:** {st.session_state.user_info['الاسم (Name)']}")
    st.sidebar.markdown(f"**الدور:** {st.session_state.user_info['الدور (Role)']}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.logged_in = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات", "⚙️ الإدارة"])

    with tab1:
        st.subheader("تحليل القيمة الإيجارية العادلة")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_index=True)
            col_a, col_b = st.columns(2)
            with col_a:
                s_area = st.number_input("مساحة العقار (م2)", value=500, step=10)
                s_loc = st.slider("جودة الموقع", 1, 5, 3)
            with col_b:
                s_spec = st.slider("المواصفات الفنية", 1, 5, 3)
                s_age = st.slider("الحالة والعمر", 1, 5, 3)
            
            if st.button("بدء المعالجة الذكية"):
                deals = get_data("Deals_DB")
                if not deals.empty:
                    estimated_sqm = valuation_engine({'loc': s_loc, 'spec': s_spec, 'age': s_age}, deals)
                    st.markdown("---")
                    res1, res2 = st.columns(2)
                    res1.metric("سعر المتر التقديري", f"{round(estimated_sqm, 2)} ريال")
                    res2.metric("إجمالي الإيجار السنوي", f"{round(estimated_sqm * s_area, 2)} ريال")
                else:
                    st.warning("بنك الصفقات فارغ، يرجى تحديث البيانات.")
            st.markdown("</div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("سجل الصفقات المعتمدة")
        st.dataframe(get_data("Deals_DB"), use_container_width=True, hide_index=True)

    with tab3:
        if st.session_state.user_info['الدور (Role)'] == 'Admin':
            st.subheader("لوحة تحكم المسؤول")
            st.info("إدارة قاعدة البيانات السحابية متاحة لك حصراً.")
            st.link_button("تحديث بنك المعلومات في Google Sheets", "https://docs.google.com/spreadsheets/d/12WCV2C3iiIF8sxpiKplypNA9pRYz5un4GwJMdsssGXA/edit")
        else:
            st.warning("هذه الصلاحية مخصصة لمدير النظام فقط.")

st.markdown("<br><hr><center>منصة إستدامة | تطوير: محمد داغستاني 2026 ©</center>", unsafe_allow_index=True)
