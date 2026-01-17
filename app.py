import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# 1. إعدادات الصفحة والهوية البصرية
st.set_page_config(page_title="إستدامة | التقييم العقاري", layout="wide")

st.markdown("""
    <style>
    .main-title { color: #1a1a1a; text-align: center; border-bottom: 3px solid #c5a059; padding-bottom: 10px; }
    .stButton>button { background-color: #c5a059; color: white; border-radius: 8px; font-weight: bold; width: 100%; }
    .card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-right: 8px solid #c5a059; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_index=True)

# 2. الربط السحابي
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("يرجى التأكد من إعدادات Secrets في لوحة التحكم")

def load_data(sheet_name):
    # جلب البيانات وتحويلها لنصوص لضمان عدم حدوث تعارض في الأنواع
    df = conn.read(worksheet=sheet_name, ttl="1m")
    return df.astype(str)

# 3. نظام الدخول
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h1 class='main-title'>🏛️ منصة إستدامة الرقمية</h1>", unsafe_allow_index=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_index=True)
        u_email = st.text_input("البريد الإلكتروني")
        u_pass = st.text_input("كلمة المرور", type="password")
        if st.button("دخول النظام"):
            try:
                users = load_data("Users_DB")
                check = users[(users['البريد الإلكتروني (Email)'].str.strip() == u_email.strip()) & 
                              (users['كلمة المرور (Password)'].str.strip() == u_pass.strip())]
                if not check.empty:
                    st.session_state.auth = True
                    st.session_state.u_name = check.iloc[0]['الاسم (Name)']
                    st.rerun()
                else:
                    st.error("بيانات الدخول غير صحيحة")
            except:
                st.error("تعذر الوصول لبيانات المستخدمين")
        st.markdown("</div>", unsafe_allow_index=True)
else:
    # 4. الواجهة الرئيسية بعد الدخول
    st.sidebar.title("إستدامة")
    st.sidebar.success(f"مرحباً: {st.session_state.u_name}")
    if st.sidebar.button("خروج"):
        st.session_state.auth = False
        st.rerun()

    tab1, tab2 = st.tabs(["🎯 محاكي التقييم", "📊 بنك الصفقات"])

    with tab1:
        st.markdown("### إجراء تقييم جديد")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_index=True)
            c1, c2 = st.columns(2)
            with c1:
                area = st.number_input("المساحة (م2)", value=100)
                loc = st.select_slider("جودة الموقع", options=[1, 2, 3, 4, 5], value=3)
            with c2:
                spec = st.select_slider("المواصفات", options=[1, 2, 3, 4, 5], value=3)
                age = st.select_slider("الحالة/العمر", options=[1, 2, 3, 4, 5], value=3)
            
            if st.button("بدء المعالجة"):
                try:
                    deals = load_data("Deals_DB")
                    # تحويل البيانات لأرقام للحساب
                    deals['price'] = pd.to_numeric(deals['القيمة السنوية'], errors='coerce')
                    deals['size'] = pd.to_numeric(deals['المساحة'], errors='coerce')
                    deals['rate'] = deals['price'] / deals['size']
                    
                    avg_base = deals['rate'].mean()
                    
                    # مصفوفة التعديل (الموقع 40%، المواصفات 35%، العمر 25%)
                    # التعديل بنسبة 10% لكل درجة فرق عن المتوسط 3
                    adj = ((loc - 3) * 0.40 * 0.1) + ((spec - 3) * 0.35 * 0.1) + ((age - 3) * 0.25 * 0.1)
                    result_rate = avg_base * (1 + adj)
                    
                    st.divider()
                    st.metric("سعر المتر التقديري", f"{round(result_rate, 2)} ريال")
                    st.metric("إجمالي الإيجار السنوي", f"{round(result_rate * area, 2)} ريال")
                except:
                    st.error("حدث خطأ في جلب البيانات أو الحساب")
            st.markdown("</div>", unsafe_allow_index=True)

    with tab2:
        st.dataframe(load_data("Deals_DB"), use_container_width=True)
