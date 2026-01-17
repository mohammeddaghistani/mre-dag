import streamlit as st
import pandas as pd
import requests
from streamlit_gsheets import GSheetsConnection

# 1. إعدادات الهوية البصرية والصفحة
st.set_page_config(page_title="إستدامة | نظام التقييم الذكي", layout="wide")

st.markdown("""
    <style>
    .main-title { color: #1a1a1a; text-align: center; border-bottom: 3px solid #c5a059; padding-bottom: 10px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid #c5a059; margin-bottom: 15px; }
    .stButton>button { background-color: #c5a059; color: white; font-weight: bold; width: 100%; border-radius: 10px; }
    </style>
    """, unsafe_allow_index=True)

# 2. الاتصال ببيانات Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# 3. إدارة حالة الجلسة (Session State)
if 'auth_success' not in st.session_state:
    st.session_state.auth_success = False
    st.session_state.otp_sent = False
    st.session_state.correct_otp = None
    st.session_state.user_email = ""
    st.session_state.user_details = None

# --- بوابة الدخول (OTP) ---
if not st.session_state.auth_success:
    st.markdown("<h1 class='main-title'>🏛️ دخول منصة إستدامة</h1>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        if not st.session_state.otp_sent:
            email_input = st.text_input("أدخل البريد الإلكتروني المسجل")
            if st.button("إرسال رمز الدخول"):
                # جلب المستخدمين للتحقق
                users_df = conn.read(worksheet="Users_DB", ttl="0")
                if email_input.strip() in users_df['البريد الإلكتروني (Email)'].values:
                    # طلب الرمز من Google Script
                    script_url = st.secrets["auth"]["script_url"]
                    try:
                        response = requests.get(f"{script_url}?email={email_input.strip()}")
                        st.session_state.correct_otp = response.text.strip()
                        st.session_state.otp_sent = True
                        st.session_state.user_email = email_input.strip()
                        st.session_state.user_details = users_df[users_df['البريد الإلكتروني (Email)'] == email_input.strip()].iloc[0]
                        st.success(f"تم إرسال الرمز إلى بريدك الإلكتروني")
                        st.rerun()
                    except:
                        st.error("خطأ في الاتصال بمحرك الإرسال")
                else:
                    st.error("عذراً، هذا البريد غير مسجل.")
        else:
            st.info(f"الرمز أُرسل إلى: {st.session_state.user_email}")
            otp_input = st.text_input("أدخل الرمز المكون من 6 أرقام")
            if st.button("تحقق ودخول"):
                if otp_input.strip() == st.session_state.correct_otp:
                    st.session_state.auth_success = True
                    st.rerun()
                else:
                    st.error("الرمز غير صحيح")
            if st.button("تغيير البريد"):
                st.session_state.otp_sent = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_index=True)

# --- الواجهة الرئيسية (بعد نجاح الدخول) ---
else:
    st.sidebar.markdown(f"### مرحباً بك\n**{st.session_state.user_details['الاسم (Name)']}**")
    st.sidebar.info(f"الدور: {st.session_state.user_details['الدور (Role)']}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth_success = False
        st.session_state.otp_sent = False
        st.rerun()

    tab1, tab2 = st.tabs(["🎯 محاكي التقييم العقاري", "📊 بنك الصفقات"])

    with tab1:
        st.subheader("إجراء تقييم جديد")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_index=True)
            c1, c2 = st.columns(2)
            with c1:
                area = st.number_input("مساحة العقار (م2)", value=100)
                loc_score = st.select_slider("جودة الموقع", options=[1, 2, 3, 4, 5], value=3)
            with c2:
                spec_score = st.select_slider("المواصفات الفنية", options=[1, 2, 3, 4, 5], value=3)
                age_score = st.select_slider("الحالة/العمر", options=[1, 2, 3, 4, 5], value=3)
            
            if st.button("بدء عملية التقييم"):
                try:
                    deals = conn.read(worksheet="Deals_DB", ttl="1m")
                    # تحويل البيانات وحساب متوسط سعر المتر
                    deals['price'] = pd.to_numeric(deals['القيمة السنوية'], errors='coerce')
                    deals['size'] = pd.to_numeric(deals['المساحة'], errors='coerce')
                    deals['rate'] = deals['price'] / deals['size']
                    avg_base = deals['rate'].mean()
                    
                    # مصفوفة التعديل (الموقع 40%، المواصفات 35%، العمر 25%)
                    adj = ((loc_score - 3) * 0.40 * 0.1) + \
                          ((spec_score - 3) * 0.35 * 0.1) + \
                          ((age_score - 3) * 0.25 * 0.1)
                    
                    final_rate = avg_base * (1 + adj)
                    
                    st.divider()
                    st.metric("سعر المتر التقديري", f"{round(final_rate, 2)} ريال")
                    st.metric("الإيجار السنوي المقدر", f"{round(final_rate * area, 2)} ريال")
                except Exception as e:
                    st.error(f"خطأ في الحساب: {e}")
            st.markdown("</div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("سجل الصفقات المعتمدة")
        deals_view = conn.read(worksheet="Deals_DB", ttl="1m")
        st.dataframe(deals_view, use_container_width=True)

st.markdown("<center>منصة إستدامة | جميع الحقوق محفوظة 2026 ©</center>", unsafe_allow_index=True)
