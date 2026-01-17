import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# 1. إعداد الصفحة (تبسيط كامل لتجنب أخطاء التنسيق)
st.set_page_config(page_title="إستدامة", layout="centered")

# 2. الربط السحابي بـ Google Sheets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("فشل الاتصال: يرجى التحقق من إعدادات Secrets")

def load_data(name):
    # قراءة البيانات مع تحويلها لنصوص لضمان عدم حدوث Type Error
    df = conn.read(worksheet=name, ttl="1m")
    return df.astype(str)

# 3. نظام إدارة الجلسة (الدخول)
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.header("🏛️ منصة إستدامة - الدخول")
    
    with st.form("login_gate"):
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        if st.form_submit_button("دخول"):
            try:
                users = load_data("Users_DB")
                # مطابقة البيانات مع ملفك المرفوع (تأكد من اسم العمود في شيت جوجل)
                check = users[(users['البريد الإلكتروني (Email)'].str.strip() == email.strip()) & 
                              (users['كلمة المرور (Password)'].str.strip() == password.strip())]
                
                if not check.empty:
                    st.session_state.auth = True
                    st.session_state.user_name = check.iloc[0]['الاسم (Name)']
                    st.rerun()
                else:
                    st.error("بيانات الدخول غير صحيحة")
            except:
                st.error("خطأ في الوصول لجدول المستخدمين")
else:
    # 4. واجهة التطبيق الرئيسية
    st.subheader(f"مرحباً: {st.session_state.user_name}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth = False
        st.rerun()

    st.divider()
    
    # محرك التقييم العقاري
    st.write("### 🎯 محاكي التقييم العقاري")
    
    col1, col2 = st.columns(2)
    with col1:
        # تأكد من أن المستخدم يدخل أرقاماً
        area_input = st.number_input("المساحة (م2)", value=100)
        loc_score = st.selectbox("جودة الموقع", [1, 2, 3, 4, 5], index=2)
    with col2:
        spec_score = st.selectbox("المواصفات", [1, 2, 3, 4, 5], index=2)
        age_score = st.selectbox("الحالة/العمر", [1, 2, 3, 4, 5], index=2)

    if st.button("احسب القيمة الإيجارية"):
        try:
            deals = load_data("Deals_DB")
            # تحويل القيم المالية لأرقام للحساب لضمان عدم حدوث TypeError
            deals['price_annual'] = pd.to_numeric(deals['القيمة السنوية'], errors='coerce')
            deals['area_size'] = pd.to_numeric(deals['المساحة'], errors='coerce')
            
            # حذف أي صفوف بها بيانات غير رقمية
            deals = deals.dropna(subset=['price_annual', 'area_size'])
            
            deals['sqm_rate'] = deals['price_annual'] / deals['area_size']
            base_avg = deals['sqm_rate'].mean()
            
            # مصفوفة التعديل (وزن الموقع 40%، المواصفات 35%، العمر 25%)
            adj = ((loc_score - 3) * 0.40 * 0.1) + \
                  ((spec_score - 3) * 0.35 * 0.1) + \
                  ((age_score - 3) * 0.25 * 0.1)
            
            final_sqm = base_avg * (1 + adj)
            
            st.info(f"سعر المتر التقديري: {round(final_sqm, 2)} ريال")
            st.success(f"الإيجار السنوي المقدر: {round(final_sqm * area_input, 2)} ريال")
        except Exception as e:
            st.error(f"حدث خطأ أثناء الحساب: {e}")

    if st.checkbox("إظهار بنك الصفقات"):
        st.dataframe(load_data("Deals_DB"))
