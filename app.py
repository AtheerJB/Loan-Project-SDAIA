import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(
    page_title="نظام الذكاء الاصطناعي للموافقة على القروض البنكية",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. تنسيقات CSS كاملة بدعم العربية اليمين RTL والنصوص الكبيرة الواضحة
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');

    html, body, [class*="st-"], div, p, span, label, input, button, select {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .stApp {
        background-color: #0b132b;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #1c2541 !important;
        border-left: 2px solid #5bc0be !important;
        direction: rtl !important;
    }
    
    section[data-testid="stSidebar"] * {
        text-align: right !important;
        direction: rtl !important;
    }

    .header-box {
        background: linear-gradient(135deg, #1c2541 0%, #3a506b 100%);
        border: 2px solid #5bc0be;
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    
    .header-box h1 {
        color: #6fffe9;
        font-size: 2.6rem;
        font-weight: 900;
        margin: 0;
        text-align: center !important;
    }

    .header-box p {
        color: #ffffff;
        font-size: 1.25rem;
        margin-top: 10px;
        font-weight: 500;
        text-align: center !important;
    }

    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #6fffe9 !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }

    input, select, div[data-baseweb="select"] {
        background-color: #1c2541 !important;
        color: #ffffff !important;
        border: 2px solid #5bc0be !important;
        border-radius: 10px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }

    .stat-card {
        background-color: #1c2541;
        border: 2px solid #3a506b;
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .stat-title {
        color: #a3cef1;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .stat-value {
        color: #6fffe9;
        font-size: 2rem;
        font-weight: 900;
    }

    .status-approved {
        background-color: #132a13;
        border: 3px solid #52b788;
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center !important;
        color: #d8f3dc;
        box-shadow: 0 0 25px rgba(82, 183, 136, 0.5);
    }

    .status-rejected {
        background-color: #3d0808;
        border: 3px solid #f72585;
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center !important;
        color: #ffccd5;
        box-shadow: 0 0 25px rgba(247, 37, 133, 0.5);
    }

    .stButton > button {
        background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%) !important;
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 900 !important;
        padding: 0.9rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 5px 20px rgba(0, 180, 216, 0.5) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #03045e 0%, #0077b6 100%) !important;
        box-shadow: 0 8px 30px rgba(0, 180, 216, 0.8) !important;
    }

    button[data-baseweb="tab"] {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #a3cef1 !important;
    }
    button[aria-selected="true"] {
        color: #6fffe9 !important;
        border-bottom-color: #6fffe9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. تحميل وتجهيز البيانات والنماذج والتقييم التفصيلي
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            df[col] = df[col].astype(str).str.strip()
            
    df_raw = df.copy()
    
    encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype == 'string':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            
    y = df_encoded['loan_status']
    X = df_encoded.drop(columns=['loan_id', 'loan_status'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # نموذج شجرة القرار Decision Tree
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    
    tree_train_acc = tree_clf.score(X_train, y_train)
    tree_test_acc = accuracy_score(y_test, y_pred_tree)
    tree_precision = precision_score(y_test, y_pred_tree, pos_label=0)
    tree_recall = recall_score(y_test, y_pred_tree, pos_label=0)
    tree_f1 = f1_score(y_test, y_pred_tree, pos_label=0)
    tree_cm = confusion_matrix(y_test, y_pred_tree)

    # نموذج الانحدار اللوجستي Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    
    log_test_acc = accuracy_score(y_test, y_pred_log)
    log_precision = precision_score(y_test, y_pred_log, pos_label=0)
    log_recall = recall_score(y_test, y_pred_log, pos_label=0)
    log_f1 = f1_score(y_test, y_pred_log, pos_label=0)
    log_cm = confusion_matrix(y_test, y_pred_log)

    metrics_tree = {
        'train_acc': tree_train_acc,
        'test_acc': tree_test_acc,
        'precision': tree_precision,
        'recall': tree_recall,
        'f1': tree_f1,
        'cm': tree_cm
    }

    metrics_log = {
        'test_acc': log_test_acc,
        'precision': log_precision,
        'recall': log_recall,
        'f1': log_f1,
        'cm': log_cm
    }

    return df_raw, X, y, encoders, tree_clf, log_reg, metrics_tree, metrics_log

try:
    df_raw, X, y, encoders, tree_clf, log_reg, metrics_tree, metrics_log = load_data()
except Exception as e:
    st.error(f"حدث خطأ في قراءة ملف البيانات: {e}")
    st.stop()


# ---------------------------------------------------------
# 4. رأس الصفحة بالعربية (Header)
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>🏛️ نظام الذكاء الاصطناعي للتنبؤ بطلبات القروض البنكية</h1>
    <p>منصة تفاعلية باللغة العربية لتحليل وتدقيق طلبات القروض في ثوانٍ معدودة</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 5. القائمة الجانبية (Sidebar)
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ إعدادات النموذج")
model_choice = st.sidebar.radio(
    "اختر خوارزمية الذكاء الاصطناعي:",
    ["شجرة القرار (Decision Tree - دقة 97.8%)", "الإنحدار اللوجستي (Logistic Regression - دقة 79.9%)"]
)

active_model = tree_clf if "Decision Tree" in model_choice else log_reg

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 ملخص أداء Decision Tree")
st.sidebar.success(f"🏋️ **دقة التدريب (Train):** {metrics_tree['train_acc']*100:.2f}%")
st.sidebar.success(f"🧪 **دقة الاختبار (Test):** {metrics_tree['test_acc']*100:.2f}%")
st.sidebar.info(f"🎯 **Precision:** {metrics_tree['precision']*100:.2f}%")
st.sidebar.info(f"🔄 **Recall:** {metrics_tree['recall']*100:.2f}%")
st.sidebar.info(f"⚡ **F1-Score:** {metrics_tree['f1']*100:.2f}%")


# ---------------------------------------------------------
# 6. التبويبات الرئيسية (Tabs)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔮 حاسبة التنبؤ الفورية",
    "📈 تحليلات وتقييم النماذج (Precision & Recall)",
    "📁 استعراض جدول البيانات"
])


# =========================================================
# التبويب الأول: الحاسبة التفاعلية باللغة العربية
# =========================================================
with tab1:
    st.markdown("### 📋 أدخل بيانات طالب القرض للحصول على النتيجة فوراً:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 البيانات الشخصية")
        dependents = st.number_input("عدد المعالين في الأسرة:", min_value=0, max_value=10, value=2)
        edu_str = st.selectbox("المؤهل الدراسي:", ["جامعي (Graduate)", "غير جامعي (Not Graduate)"])
        emp_str = st.selectbox("هل يعمل لحسابه الخاص (عمل حر)؟:", ["لا (No)", "نعم (Yes)"])
        cibil = st.slider("الدرجة الائتمانية CIBIL:", min_value=300, max_value=900, value=750,
                          help="الدرجة الائتمانية من 300 إلى 900 (أعلى من 650 ممتاز)")

    with col2:
        st.markdown("#### 💵 الراتب ومبلغ القرض")
        income = st.number_input("الراتب السنوي الإجمالي (بالريال):", min_value=100000, max_value=100000000, value=6000000, step=100000)
        loan_amt = st.number_input("مبلغ القرض المطلوب (بالريال):", min_value=100000, max_value=200000000, value=15000000, step=500000)
        loan_term = st.slider("مدة سداد القرض (بالسنوات):", min_value=2, max_value=20, value=10)

    with col3:
        st.markdown("#### 🏠 الأصول والضمانات المالية")
        res_assets = st.number_input("قيمة العقارات السكنية:", min_value=0, max_value=100000000, value=10000000, step=500000)
        com_assets = st.number_input("قيمة العقارات التجارية:", min_value=0, max_value=100000000, value=5000000, step=500000)
        lux_assets = st.number_input("قيمة السيارات والمقتنيات الثمينة:", min_value=0, max_value=100000000, value=15000000, step=500000)
        bank_assets = st.number_input("إجمالي الأرصدة والودائع البنكية:", min_value=0, max_value=100000000, value=4000000, step=500000)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("⚡ تحليل طلب القرض وإصدار القرار الآن")
    
    if btn_predict:
        edu_val = 0 if "جامعي" in edu_str else 1
        emp_val = 1 if "نعم" in emp_str else 0
        
        sample_df = pd.DataFrame([{
            'no_of_dependents': dependents,
            'education': edu_val,
            'self_employed': emp_val,
            'income_annum': income,
            'loan_amount': loan_amt,
            'loan_term': loan_term,
            'cibil_score': cibil,
            'residential_assets_value': res_assets,
            'commercial_assets_value': com_assets,
            'luxury_assets_value': lux_assets,
            'bank_asset_value': bank_assets
        }])
        
        pred_raw = active_model.predict(sample_df)[0]
        result_str = encoders['loan_status'].inverse_transform([pred_raw])[0].strip()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if result_str == "Approved":
            st.markdown("""
            <div class="status-approved">
                <h1 style="color: #6fffe9; font-size: 2.6rem; margin: 0; text-align: center !important;">✅ تم القبول (Approved)</h1>
                <p style="font-size: 1.35rem; margin-top: 15px; text-align: center !important;">تهانينا! وفقاً لتحليلات نموذج الذكاء الاصطناعي، يمتلك العميل مؤشرات ائتمانية ممتازة وضمانات كافية للموافقة على القرض.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
            <div class="status-rejected">
                <h1 style="color: #ff4d4d; font-size: 2.6rem; margin: 0; text-align: center !important;">❌ تم الرفض (Rejected)</h1>
                <p style="font-size: 1.35rem; margin-top: 15px; text-align: center !important;">عذراً، تشير تحليلات المخاطر إلى عدم استيفاء الشروط الائتمانية الكافية (انخفاض درجة CIBIL أو التغطية المالية للمبلغ المطلوب).</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# التبويب الثاني: التحليلات ومقاييس الأداء الشاملة (Precision, Recall, F1, CM)
# =========================================================
with tab2:
    st.markdown("### 📊 مقاييس التقييم التفصيلية (Evaluation Metrics)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### 🌲 نموذج شجرة القرار (Decision Tree Classifier)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">دقة التدريب</div>
            <div class="stat-value">{metrics_tree['train_acc']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">دقة الاختبار</div>
            <div class="stat-value">{metrics_tree['test_acc']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Precision</div>
            <div class="stat-value">{metrics_tree['precision']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Recall</div>
            <div class="stat-value">{metrics_tree['recall']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">F1-Score</div>
            <div class="stat-value">{metrics_tree['f1']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        st.markdown("#### 🎯 مصفوفة الارتباك (Confusion Matrix - Decision Tree)")
        cm_df_tree = pd.DataFrame(
            metrics_tree['cm'],
            index=['مقبول فعلي (Approved)', 'مرفوض فعلي (Rejected)'],
            columns=['متنبأ مقبول (Approved)', 'متنبأ مرفوض (Rejected)']
        )
        st.dataframe(cm_df_tree)
        
    with cm_col2:
        st.markdown("#### 📈 أهم العوامل المؤثرة في القرار (Feature Importance)")
        importances = pd.Series(tree_clf.feature_importances_, index=[
            "عدد المعالين", "المؤهل الدراسي", "عمل حر", "الراتب السنوي",
            "مبلغ القرض", "مدة القرض", "درجة CIBIL الائتمانية",
            "عقارات سكنية", "عقارات تجارية", "مقتنيات ثمينة", "أرصدة بنكية"
        ]).sort_values(ascending=True)
        st.bar_chart(importances)


# =========================================================
# التبويب الثالث: جدول البيانات بالعربية
# =========================================================
with tab3:
    st.markdown("### 📁 جدول بيانات القروض الأصلي:")
    st.dataframe(df_raw)
