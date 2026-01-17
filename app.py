import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# إعداد الصفحة الأساسي (بدون CSS معقد لتجنب الأخطاء)
st.set_page_config(page_title="إستدامة", layout="centered")

# الربط ببنك المعلومات السحابي
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    st.error("جاري الاتصال بالسحابة...")

def load_data(name):
    # قراءة البيانات مع تحويلها لنصوص لضمان الاستقرار
    df = conn.read(worksheet=name, ttl="1m")
    return df.astype(str)

# نظام الدخول
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🏛️ دخول منصة إستدامة")
    with st.form("login"):
        u_email = st.text_input("البريد الإلكتروني")
        u_pass = st.text_input("كلمة المرور", type="password")
        if st.form_submit_button("دخول"):
            try:
                users = load_data("Users_DB")
                # مطابقة البيانات مع ملفك (البريد الإلكتروني (Email))
                check = users[(users['البريد الإلكتروني (Email)'].str.strip() == u_email.strip()) & 
                              (users['كلمة المرور (Password)'].str.strip() == u_pass.strip())]
                if not check.empty:
                    st.session_state.auth = True
                    st.session_state.name = check.iloc[0]['الاسم (Name)']
                    st.rerun()
                else:
                    st.error("البيانات غير صحيحة")
            except:
                st.error("خطأ في الوصول لجدول المستخدمين")
else:
    st.success(f"مرحباً بك: {st.session_state.name}")
    if st.button("خروج"):
        st.session_state.auth = False
        st.rerun()

    st.divider()
    st.subheader("🎯 محاكي التقييم العقاري")
    
    # مدخلات التقييم
    area = st.number_input("المساحة (م2)", value=100)
    loc = st.selectbox("جودة الموقع", [1, 2, 3, 4, 5], index=2)
    spec = st.selectbox("المواصفات الفنية", [1, 2, 3, 4, 5], index=2)
    
    if st.button("احسب"):
        try:
            deals = load_data("Deals_DB")
            # تحويل القيم المالية لأرقام للحساب
            deals['sqm_rate'] = pd.to_numeric(deals['القيمة السنوية']) / pd.to_numeric(deals['المساحة'])
            base_avg = deals['sqm_rate'].mean()
            
            # مصفوفة تعديل بسيطة (5% لكل درجة فرق)
            adj = ((loc - 3) * 0.05) + ((spec - 3) * 0.05)
            final_p = base_avg * (1 + adj)
            
            st.metric("سعر المتر التقديري", f"{round(final_p, 2)} ريال")
            st.metric("الإيجار السنوي المتوقع", f"{round(final_p * area, 2)} ريال")
        except:
            st.error("خطأ في حساب بيانات بنك الصفقات")

    if st.checkbox("عرض بنك الصفقات"):
        st.dataframe(load_data("Deals_DB"))
