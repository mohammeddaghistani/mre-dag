import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- إعدادات الهوية البصرية الفاخرة ---
st.set_page_config(page_title="إستدامة | المنصة الاستراتيجية", layout="wide")

# CSS مخصص ليتناسب مع mdaghistani.com والجوال
st.markdown("""
    <style>
    :root { --primary: #1a1a1a; --gold: #c5a059; }
    .stApp { background-color: #ffffff; }
    .main-title { color: var(--primary); text-align: center; border-bottom: 3px solid var(--gold); padding-bottom: 10px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid var(--gold); margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { background-color: var(--gold); color: white; width: 100%; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; }
    </style>
    """, unsafe_allow_index=True)

# --- محرك الاتصال ببنك المعلومات الحية (Google Sheets) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_deals():
    # استدعاء بيانات الصفقات من الورقة المحددة في Secrets
    return conn.read(worksheet="Deals_DB", ttl="5m")

def get_users():
    # استدعاء بيانات المستخدمين للتحقق من الصلاحيات
    return conn.read(worksheet="Users_DB", ttl="10m")

# --- محرك التقييم الذكي (مصفوفة التعديل - دليل 2023) ---
def valuation_engine(subject, bank):
    weights = {'loc': 0.40, 'spec': 0.35, 'age': 0.25}
    bank['rent_sqm'] = bank['الإيجار_السنوي'] / bank['المساحة']
    
    adjusted_rates = []
    for _, row in bank.iterrows():
        # مصفوفة التسويات النوعية
        adj = ((subject['loc'] - row['الموقع']) / subject['loc'] * weights['loc']) + \
              ((subject['spec'] - row['المواصفات']) / subject['spec'] * weights['spec']) + \
              ((subject['age'] - row['العمر']) / subject['age'] * weights['age'])
        adjusted_rates.append(row['rent_sqm'] * (1 + adj))
    
    return np.mean(adjusted_rates)

# --- واجهة تسجيل الدخول للأمان السحابي ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>🏛️ منصة إستدامة العقارية</h1>", unsafe_allow_index=True)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        user_input = st.text_input("اسم المستخدم")
        pass_input = st.text_input("كلمة المرور", type="password")
        if st.button("دخول النظام"):
            users_df = get_users()
            user_row = users_df[(users_df['Username'] == user_input) & (users_df['Password'] == pass_input)]
            if not user_row.empty:
                st.session_state.logged_in = True
                st.session_state.user_role = user_row.iloc[0]['Role']
                st.session_state.user_name = user_row.iloc[0]['FullName']
                st.rerun()
            else:
                st.error("بيانات الدخول غير صحيحة")
        st.markdown("</div>", unsafe_allow_index=True)
else:
    # --- واجهة التطبيق الرئيسية بعد الدخول ---
    st.sidebar.image("https://mdaghistani.com/wp-content/uploads/2022/04/logo-new.png")
    st.sidebar.success(f"مرحباً: {st.session_state.user_name}")
    st.sidebar.info(f"صلاحية الحساب: {st.session_state.user_role}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.logged_in = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات", "⚙️ الإدارة"])

    with tab1:
        st.subheader("إجراء تقييم جديد (مقارنة المبيعات)")
        col1, col2 = st.columns([2, 1])
        with col1:
            s_area = st.number_input("المساحة الإجمالية (م2)", value=1000)
            st.write("**درجات الجودة (1-5):**")
            c_a, c_b, c_c = st.columns(3)
            s_loc = c_a.slider("الموقع", 1, 5, 3)
            s_spec = c_b.slider("المواصفات", 1, 5, 3)
            s_age = c_c.slider("الحالة/العمر", 1, 5, 3)
            
        if st.button("توليد التقرير السعري"):
            deals_df = get_deals()
            final_sqm = valuation_engine({'loc': s_loc, 'spec': s_spec, 'age': s_age}, deals_df)
            
            st.markdown("<br>", unsafe_allow_index=True)
            res1, res2 = st.columns(2)
            with res1:
                st.markdown(f"<div class='card'><h4>سعر المتر التقديري</h4><h2>{round(final_sqm, 2)} ريال</h2></div>", unsafe_allow_index=True)
            with res2:
                st.markdown(f"<div class='card'><h4>إجمالي الإيجار السنوي</h4><h2>{round(final_sqm * s_area, 2)} ريال</h2></div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("بنك الصفقات المعتمدة")
        st.dataframe(get_deals(), use_container_width=True, hide_index=True)

    with tab3:
        if st.session_state.user_role == "Admin":
            st.subheader("لوحة تحكم المدير")
            st.write("يمكنك إضافة صفقات جديدة مباشرة عبر Google Sheets وسيتم تحديثها هنا.")
            st.link_button("فتح ملف البيانات السحابي", "https://docs.google.com/spreadsheets/d/your_id")
        else:
            st.warning("هذه الصفحة مخصصة لمدير النظام فقط.")
