import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- 1. الإعدادات البصرية ---
st.set_page_config(page_title="إستدامة | المنصة الاستراتيجية", layout="wide")

# --- 2. محرك الاتصال ببنك المعلومات ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"خطأ في إعداد الاتصال بالسحابة: {e}")

def get_sheet_data(sheet_name):
    try:
        # قراءة البيانات مع ضمان تحويل كافة البيانات لنصوص لتجنب أخطاء المقارنة
        df = conn.read(worksheet=sheet_name, ttl="1m")
        return df.astype(str) 
    except Exception as e:
        st.error(f"فشل جلب البيانات من ورقة {sheet_name}. تأكد من الأسماء في جوجل شيت.")
        return pd.DataFrame()

# --- 3. واجهة تسجيل الدخول ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏛️ منصة إستدامة العقارية")
    with st.form("login_form"):
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        submit = st.form_submit_button("دخول")
        
        if submit:
            users_df = get_sheet_data("Users_DB")
            if not users_df.empty:
                # التحقق من المستخدم بمرونة (بدون الحساسية للفراغات)
                user = users_df[(users_df['البريد الإلكتروني'].str.strip() == email.strip()) & 
                                (users_df['كلمة المرور'].str.strip() == password.strip())]
                
                if not user.empty:
                    st.session_state.logged_in = True
                    st.session_state.user_name = user.iloc[0]['الاسم']
                    st.session_state.role = user.iloc[0]['الدور']
                    st.rerun()
                else:
                    st.error("البريد الإلكتروني أو كلمة المرور غير صحيحة")
else:
    st.sidebar.success(f"مرحباً: {st.session_state.user_name}")
    if st.sidebar.button("خروج"):
        st.session_state.logged_in = False
        st.rerun()
    
    # هنا تضع كود محرك التقييم الذي اتفقنا عليه سابقاً
    st.write("تم الدخول بنجاح إلى النظام السحابي.")
