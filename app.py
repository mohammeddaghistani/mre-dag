import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# 1. إعداد الصفحة الأساسي (بدون CSS)
st.set_page_config(page_title="إستدامة", layout="centered")

# 2. الربط بـ Google Sheets
# تأكد أن الرابط في Secrets تحت اسم [connections.gsheets]
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("فشل الاتصال بالقاعدة السحابية")

def get_table(sheet_name):
    # قراءة البيانات مع تحويلها لنصوص لتجنب أخطاء النوع
    df = conn.read(worksheet=sheet_name, ttl="1m")
    return df.astype(str)

# 3. نظام الجلسة والدخول
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🏛️ دخول منصة إستدامة")
    
    with st.form("login_form"):
        user_email = st.text_input("البريد الإلكتروني")
        user_pass = st.text_input("كلمة المرور", type="password")
        submit_button = st.form_submit_button("تسجيل الدخول")
        
        if submit_button:
            try:
                users_df = get_table("Users_DB")
                # مطابقة البيانات (حسب ملفك المرفوع)
                user_match = users_df[(users_df['البريد الإلكتروني (Email)'] == user_email) & 
                                     (users_df['كلمة المرور (Password)'] == user_pass)]
                
                if not user_match.empty:
                    st.session_state.authenticated = True
                    st.session_state.user_full_name = user_match.iloc[0]['الاسم (Name)']
                    st.rerun()
                else:
                    st.error("بيانات الدخول غير صحيحة")
            except Exception as ex:
                st.error(f"خطأ في قراءة جدول المستخدمين: {ex}")
else:
    # 4. واجهة التطبيق الرئيسية
    st.write(f"مرحباً بك: **{st.session_state.user_full_name}**")
    if st.button("تسجيل الخروج"):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()
    st.subheader("🎯 محاكي التقييم العقاري")
    
    # مدخلات التقييم
    val_area = st.number_input("المساحة (م2)", value=100)
    val_loc = st.selectbox("جودة الموقع", [1, 2, 3, 4, 5], index=2)
    val_spec = st.selectbox("المواصفات", [1, 2, 3, 4, 5], index=2)
    
    if st.button("حساب القيمة"):
        try:
            deals_df = get_table("Deals_DB")
            # تحويل القيم المالية لأرقام للحساب
            deals_df['price_annual'] = pd.to_numeric(deals_df['القيمة السنوية'], errors='coerce')
            deals_df['area_size'] = pd.to_numeric(deals_df['المساحة'], errors='coerce')
            
            # حساب متوسط المتر
            deals_df['meter_rate'] = deals_df['price_annual'] / deals_df['area_size']
            avg_meter = deals_df['meter_rate'].mean()
            
            # مصفوفة تعديل بسيطة (5% لكل درجة)
            adj_factor = ((val_loc - 3) * 0.05) + ((val_spec - 3) * 0.05)
            final_rate = avg_meter * (1 + adj_factor)
            
            st.metric("سعر المتر التقديري", f"{round(final_rate, 2)} ريال")
            st.metric("الإيجار السنوي المتوقع", f"{round(final_rate * val_area, 2)} ريال")
        except Exception as ex_calc:
            st.error(f"خطأ في الحساب: {ex_calc}")

    if st.checkbox("عرض بنك الصفقات"):
        st.dataframe(get_table("Deals_DB"))
