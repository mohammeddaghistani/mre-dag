import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# إعداد الصفحة بدون أي أكواد CSS معقدة
st.set_page_config(page_title="منصة إستدامة", layout="centered")

# الربط بـ Google Sheets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("خطأ في الاتصال: يرجى التأكد من Secrets")

# دالة جلب البيانات
def load_data(sheet_name):
    return conn.read(worksheet=sheet_name, ttl="1m")

# نظام الدخول
if 'login_status' not in st.session_state:
    st.session_state.login_status = False

if not st.session_state.login_status:
    st.header("🏛️ دخول منصة إستدامة")
    
    with st.form("login_gate"):
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        if st.form_submit_button("دخول"):
            # جلب جدول المستخدمين
            users = load_data("Users_DB")
            # التحقق (مطابقة مع ملفك: البريد الإلكتروني (Email))
            match = users[(users['البريد الإلكتروني (Email)'] == email) & 
                          (users['كلمة المرور (Password)'].astype(str) == password)]
            
            if not match.empty:
                st.session_state.login_status = True
                st.session_state.user_name = match.iloc[0]['الاسم (Name)']
                st.rerun()
            else:
                st.error("خطأ في البيانات")
else:
    # واجهة التطبيق بعد الدخول
    st.success(f"مرحباً بك: {st.session_state.user_name}")
    if st.button("خروج"):
        st.session_state.login_status = False
        st.rerun()

    # محرك التقييم البسيط
    st.divider()
    st.subheader("🎯 محاكي التقييم السريع")
    
    area = st.number_input("المساحة (م2)", value=100)
    # اختيار الجودة (1-5)
    loc = st.selectbox("جودة الموقع", [1, 2, 3, 4, 5], index=2)
    spec = st.selectbox("المواصفات", [1, 2, 3, 4, 5], index=2)
    
    if st.button("احسب"):
        deals = load_data("Deals_DB")
        # حساب متوسط سعر المتر
        deals['sqm_price'] = pd.to_numeric(deals['القيمة السنوية']) / pd.to_numeric(deals['المساحة'])
        avg_base = deals['sqm_price'].mean()
        
        # معادلة بسيطة: كل درجة فوق أو تحت الـ 3 تزيد أو تنقص 5%
        adjustment = ((loc - 3) * 0.05) + ((spec - 3) * 0.05)
        final_price = avg_base * (1 + adjustment)
        
        st.metric("سعر المتر التقديري", f"{round(final_price, 2)} ريال")
        st.metric("الإيجار السنوي المتوقع", f"{round(final_price * area, 2)} ريال")

    st.divider()
    if st.checkbox("عرض بنك الصفقات"):
        st.dataframe(load_data("Deals_DB"))
