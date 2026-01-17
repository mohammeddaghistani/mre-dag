import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- 1. الإعدادات البصرية (هوية mdaghistani.com الفاخرة) ---
st.set_page_config(page_title="إستدامة | المنصة الاستراتيجية", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    :root { --primary: #1a1a1a; --gold: #c5a059; }
    .stApp { background-color: #ffffff; }
    /* تنسيق العناوين والخطوط */
    h1, h2, h3 { color: var(--primary); text-align: center; font-family: 'Arial'; }
    .main-title { border-bottom: 3px solid var(--gold); padding-bottom: 10px; margin-bottom: 25px; }
    /* تنسيق الكروت والحاويات */
    .card { background: #f9f9f9; padding: 25px; border-radius: 15px; border-right: 10px solid var(--gold); 
            margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    /* تنسيق الأزرار */
    .stButton>button { background-color: var(--gold); color: white; width: 100%; border-radius: 10px; 
                       font-weight: bold; border: none; height: 3.5em; transition: 0.3s; }
    .stButton>button:hover { background-color: var(--primary); }
    /* تحسين واجهة الجوال */
    @media (max-width: 600px) { .stMetric { font-size: 14px; } .card { padding: 15px; } }
    </style>
    """, unsafe_allow_index=True)

# --- 2. محرك الاتصال ببنك المعلومات (Google Sheets) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data(sheet_name):
    # جلب البيانات حياً من الرابط المرتبط في Secrets 
    return conn.read(worksheet=sheet_name, ttl="1m")

# --- 3. محرك التقييم الذكي (مصفوفة التعديل المعتمدة) ---
def valuation_engine(subject, bank):
    # الأوزان النسبية: الموقع 40%، المواصفات 35%، العمر 25% [cite: 925]
    weights = {'loc': 0.40, 'spec': 0.35, 'age': 0.25}
    
    # تنظيف البيانات وحساب السعر المرجعي للمتر
    bank = bank.copy()
    bank['rent_sqm'] = bank['القيمة السنوية'] / bank['المساحة']
    
    adjusted_rates = []
    for _, row in bank.iterrows():
        # معادلة التسويات النوعية [cite: 1049, 1513]
        adj = ((subject['loc'] - row['الموقع']) / subject['loc'] * weights['loc']) + \
              ((subject['spec'] - row['المواصفات']) / subject['spec'] * weights['spec']) + \
              ((subject['age'] - row['العمر']) / subject['age'] * weights['age'])
        adjusted_rates.append(row['rent_sqm'] * (1 + adj))
    
    return np.mean(adjusted_rates)

# --- 4. إدارة نظام الدخول والأمان (Session State) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# واجهة تسجيل الدخول
if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>🏛️ منصة إستدامة العقارية</h1>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        st.subheader("بوابة الدخول الآمن")
        email_input = st.text_input("البريد الإلكتروني")
        pass_input = st.text_input("كلمة المرور", type="password")
        if st.button("دخول النظام"):
            users_df = get_data("Users_DB")
            user_row = users_df[users_df['البريد الإلكتروني'] == email_input]
            if not user_row.empty and str(user_row.iloc[0]['كلمة المرور']) == pass_input:
                st.session_state.logged_in = True
                st.session_state.user_info = user_row.iloc[0].to_dict()
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة أو الحساب غير نشط")
        st.markdown("</div>", unsafe_allow_index=True)
else:
    # --- 5. واجهة التطبيق الرئيسية بعد الدخول ---
    st.sidebar.image("https://mdaghistani.com/wp-content/uploads/2022/04/logo-new.png")
    st.sidebar.success(f"مرحباً: {st.session_state.user_info['الاسم']}")
    st.sidebar.info(f"الصلاحية: {st.session_state.user_info['الدور']}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown(f"<h1 class='main-title'>🏛️ نظام التقييم والتحليل الاستراتيجي</h1>", unsafe_allow_index=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات", "📤 إضافة صفقات", "⚙️ الإدارة"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        st.subheader("إجراء تقييم جديد (طريقة مقارنة المبيعات)")
        c1, c2 = st.columns([2, 1])
        with c1:
            subject_area = st.number_input("مساحة العقار المطلوب تقييمه (م2)", value=1000)
            st.write("**تحديد درجات الجودة (1-5):**")
            sl1, sl2, sl3 = st.columns(3)
            s_loc = sl1.slider("الموقع", 1, 5, 3)
            s_spec = sl2.slider("المواصفات", 1, 5, 3)
            s_age = sl3.slider("الحالة/العمر", 1, 5, 3)
        with c2:
            st.info("يتم الحساب بناءً على متوسط الصفقات في بنك المعلومات مع تطبيق نسب التعديل النظامية[cite: 1049].")
        
        if st.button("توليد التقدير الإيجاري"):
            deals_df = get_data("Deals_DB")
            final_sqm = valuation_engine({'loc': s_loc, 'spec': s_spec, 'age': s_age}, deals_df)
            st.markdown("---")
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("إيجار المتر التقديري", f"{round(final_sqm, 2)} ريال")
            res_c2.metric("إجمالي الإيجار السنوي", f"{round(final_sqm * subject_area, 2)} ريال")
        st.markdown("</div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("سجل الصفقات المعتمدة - تحديث حي")
        st.dataframe(get_data("Deals_DB"), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("رفع صفقات جديدة")
        st.write("يمكنك رفع ملف صفقات جديد لمراجعته من قبل الإدارة.")
        st.file_uploader("اختر ملف Excel أو CSV", type=['xlsx', 'csv'])
        st.button("إرسال للمراجعة")

    with tab4:
        if st.session_state.user_info['الدور'] == "Admin":
            st.subheader("إدارة المنظومة")
            st.write("التحكم الكامل في بنك المعلومات والمستخدمين عبر السحابة.")
            st.link_button("فتح قاعدة البيانات في Google Sheets", "https://docs.google.com/spreadsheets/d/12WCV2C3iiIF8sxpiKplypNA9pRYz5un4GwJMdsssGXA/edit")
        else:
            st.warning("هذه الصلاحية متاحة لمدير النظام فقط.")

st.markdown("<br><hr><center>منصة إستدامة | تطوير وتصميم: محمد داغستاني 2026 ©</center>", unsafe_allow_index=True)
