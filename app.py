import streamlit as st
import pandas as pd
import requests
from streamlit_gsheets import GSheetsConnection

# 1. إعدادات الهوية البصرية (تبسيط لتجنب TypeError)
st.set_page_config(page_title="إستدامة | نظام التقييم الذكي", layout="wide")

st.write("""
    <style>
    .main-title { color: #1a1a1a; text-align: center; border-bottom: 2px solid #c5a059; padding-bottom: 10px; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 12px; border-right: 6px solid #c5a059; margin-bottom: 15px; }
    .stButton>button { background-color: #c5a059; color: white; font-weight: bold; width: 100%; border-radius: 8px; }
    </style>
    """, unsafe_allow_index=True)

# 2. الاتصال ببنك المعلومات
conn = st.connection("gsheets", type=GSheetsConnection)

# 3. إدارة الجلسة (Login State)
if 'auth_active' not in st.session_state:
    st.session_state.update({
        'auth_active': False,
        'otp_sent': False,
        'correct_otp': None,
        'user_email': "",
        'user_info': None
    })

# --- المرحلة الأولى: بوابة الدخول بالرمز المؤقت (OTP) ---
if not st.session_state.auth_active:
    st.markdown("<h1 class='main-title'>🏛️ دخول منصة إستدامة</h1>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        if not st.session_state.otp_sent:
            email_in = st.text_input("البريد الإلكتروني المسجل")
            if st.button("إرسال رمز الدخول"):
                users_df = conn.read(worksheet="Users_DB", ttl="0")
                if email_in.strip() in users_df['البريد الإلكتروني (Email)'].values:
                    # استدعاء رابط الـ Script الذي زودتني به
                    script_url = st.secrets["auth"]["script_url"]
                    try:
                        res = requests.get(f"{script_url}?email={email_in.strip()}")
                        st.session_state.correct_otp = res.text.strip()
                        st.session_state.otp_sent = True
                        st.session_state.user_email = email_in.strip()
                        st.session_state.user_info = users_df[users_df['البريد الإلكتروني (Email)'] == email_in.strip()].iloc[0]
                        st.success("تم إرسال الرمز بنجاح")
                        st.rerun()
                    except:
                        st.error("فشل الاتصال بمحرك الإرسال")
                else:
                    st.error("هذا البريد غير مسجل.")
        else:
            st.info(f"أُرسل الرمز إلى: {st.session_state.user_email}")
            otp_in = st.text_input("أدخل الرمز المستلم")
            if st.button("تحقق ودخول"):
                if otp_in.strip() == st.session_state.correct_otp:
                    st.session_state.auth_active = True
                    st.rerun()
                else:
                    st.error("الرمز غير صحيح")
            if st.button("رجوع"):
                st.session_state.otp_sent = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_index=True)

# --- المرحلة الثانية: واجهة التقييم (تظهر بعد الدخول بنجاح) ---
else:
    st.sidebar.markdown(f"### مرحباً بك\n**{st.session_state.user_info['الاسم (Name)']}**")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth_active = False
        st.session_state.otp_sent = False
        st.rerun()

    tab1, tab2 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات"])

    with tab1:
        st.subheader("إجراء تقييم جديد")
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input("المساحة (م2)", value=100)
            loc = st.select_slider("جودة الموقع", options=[1, 2, 3, 4, 5], value=3)
        with c2:
            spec = st.select_slider("المواصفات الفنية", options=[1, 2, 3, 4, 5], value=3)
            age = st.select_slider("الحالة التشغيلية", options=[1, 2, 3, 4, 5], value=3)
        
        if st.button("بدء عملية التحليل الحية"):
            try:
                deals = conn.read(worksheet="Deals_DB", ttl="1m")
                # معالجة البيانات رقمياً (تجنب TypeError)
                deals['price'] = pd.to_numeric(deals['القيمة السنوية'], errors='coerce')
                deals['size'] = pd.to_numeric(deals['المساحة'], errors='coerce')
                deals['rate'] = deals['price'] / deals['size']
                base_rate = deals['rate'].mean()
                
                # مصفوفة التعديل الرسمية (الموقع 40%، المواصفات 35%، العمر 25%)
                adjustment = ((loc - 3) * 0.40 * 0.1) + ((spec - 3) * 0.35 * 0.1) + ((age - 3) * 0.25 * 0.1)
                final_val = base_rate * (1 + adjustment)
                
                st.divider()
                r1, r2 = st.columns(2)
                r1.metric("سعر المتر التقديري", f"{round(final_val, 2)} ريال")
                r2.metric("الإيجار السنوي المقدر", f"{round(final_val * area, 2)} ريال")
            except Exception as e:
                st.error(f"خطأ في معالجة البيانات: {e}")
        st.markdown("</div>", unsafe_allow_index=True)

    with tab2:
        st.subheader("سجل الصفقات المعتمدة")
        st.dataframe(conn.read(worksheet="Deals_DB", ttl="1m"), use_container_width=True)

st.markdown("<center>إستدامة | تطوير: محمد داغستاني 2026 ©</center>", unsafe_allow_index=True)
