import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# إعداد الصفحة
st.set_page_config(page_title="إستدامة", layout="centered")

# الاتصال السحابي
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    st.error("جاري تهيئة الاتصال السحابي...")

# دالة جلب البيانات مع تحويل صريح للأنواع لتجنب TypeError
def load_data(name):
    df = conn.read(worksheet=name, ttl="1m")
    return df.astype(str)

if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.header("🏛️ منصة إستدامة - الدخول")
    with st.form("login"):
        u = st.text_input("البريد الإلكتروني")
        p = st.text_input("كلمة المرور", type="password")
        if st.form_submit_button("دخول"):
            try:
                users = load_data("Users_DB")
                # مطابقة البيانات مع ضمان حذف المسافات الزائدة
                check = users[(users['البريد الإلكتروني (Email)'].str.strip() == u.strip()) & 
                              (users['كلمة المرور (Password)'].str.strip() == p.strip())]
                if not check.empty:
                    st.session_state.auth = True
                    st.session_state.user = check.iloc[0]['الاسم (Name)']
                    st.rerun()
                else:
                    st.error("البيانات غير صحيحة")
            except:
                st.error("خطأ في قراءة قاعدة البيانات. تأكد من إعدادات Secrets.")
else:
    st.success(f"مرحباً: {st.session_state.user}")
    if st.sidebar.button("خروج"):
        st.session_state.auth = False
        st.rerun()

    st.divider()
    st.subheader("🎯 محاكي التقييم العقاري")
    
    # مدخلات التقييم
    m_area = st.number_input("المساحة (م2)", value=100)
    m_loc = st.selectbox("جودة الموقع", [1, 2, 3, 4, 5], index=2)
    
    if st.button("احسب"):
        try:
            deals = load_data("Deals_DB")
            # تحويل القيم لأرقام لضمان الحساب (تطابق مع أسماء أعمدتك)
            deals['sqm_rate'] = pd.to_numeric(deals['القيمة السنوية']) / pd.to_numeric(deals['المساحة'])
            base = deals['sqm_rate'].mean()
            
            # مصفوفة التعديل
            final = base * (1 + ((m_loc - 3) * 0.05))
            
            st.metric("سعر المتر التقديري", f"{round(final, 2)} ريال")
            st.metric("الإيجار السنوي المتوقع", f"{round(final * m_area, 2)} ريال")
        except:
            st.error("تأكد من وجود صفقات في ورقة Deals_DB")

    if st.checkbox("عرض بنك الصفقات"):
        st.dataframe(load_data("Deals_DB"))
