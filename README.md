# 💳 نظام التنبؤ بالموافقة على القروض البنكية باستخدام الذكاء الاصطناعي
### 🏛️ Loan Approval Prediction System using Machine Learning & Streamlit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 نبذة عن المشروع (Project Description)

هذا المشروع عبارة عن نظام ذكاء اصطناعي متكامل للتعلم الآلي (**Machine Learning**) يقوم بحل مشكلة **التصنيف الثنائي (Binary Classification)** للتنبؤ بما إذا كان طلب القرض سيُقبل (`Approved`) أم سيُرفض (`Rejected`) بناءً على البيانات المالية والائتمانية الشخصية للمتقدم المستخرجة من قاعدة البيانات `loan_approval_dataset.csv`.

يتضمن المشروع:
1. **سكريبت معالجة البيانات وتدريب النماذج (`loan_classification.py`)**: يقدم تحليلاً دقيقاً وشاملاً باستخدام خوارزميات التعلم الآلي وتقييم النتائج بالمقاييس العلمية.
2. **تطبيق واجهة تفاعلية فخمة (`app.py`)**: واجهة موقع إلكتروني مكتوبة بلغة البايثون باستخدام إطار عمل **Streamlit** وتدعم اللغة العربية الكاملة والاتجاه من اليمين إلى اليسار (RTL).

---

## 🎓 الإشارة إلى البرنامج التدريبي (Training Program Acknowledgement)

تم تنفيذ وتطوير هذا المشروع ضمن متطلبات **البرنامج التدريبي المقدم من أكاديمية سدايا (SDAIA Academy)** للهيئة السعودية للذكاء الاصطناعي، وذلك بهدف تطبيق أفضل الممارسات في مجال الذكاء الاصطناعي وتطوير الحلول البرمجية الذكية.

🔗 **رابط حساب أكاديمية سدايا على GitHub:**  
👉 [SDAIA Academy GitHub Repository](https://github.com/SDAIAAcademy)

---

## 🚀 ميزات النظام الأساسية (Key Features)

- **تنظيف البيانات الآلي**: معالجة المسافات الزائدة في الأعمدة والبيانات النصية وتحويلها رقمياً باستخدام `LabelEncoder`.
- **تقسيم موثوق**: تقسيم البيانات بنسبة 80% للتدريب و 20% للاختبار (`train_test_split`).
- **نماذج متعددة**:
  - **شجرة القرار (Decision Tree Classifier)**: دقة تصل إلى **97.78%**.
  - **الانحدار اللوجستي (Logistic Regression)**: دقة **79.86%**.
- **تقييم متكامل للأداء**:
  - حساب **دقة التدريب ودقة الاختبار (Training vs Testing Accuracy)**.
  - حساب مقاييس **Precision**, **Recall**, و **F1-Score**.
  - استخراج **مصفوفة الارتباك (Confusion Matrix)** وتقرير التصنيف الشامل (`Classification Report`).
- **واجهة موقع تفاعلية وشاملة (Streamlit App)**:
  - دعم كامل للغة العربية والاتجاه من اليمين لليسار (RTL).
  - حاسبة تفاعلية لإدخال البيانات وحساب نتيجة القرض فوريّاً مع بالونات واحتفالية بالقبول.
  - لوحة تحليلات ورسومات بيانية لأهم العوامل المؤثرة في اتخاذ القرار (`Feature Importances`).

---

## 📊 التوثيق الفني والنتائج (Technical Documentation & Results)

### 1. بيانات المدخلات (Input Features - $X$):
- `no_of_dependents`: عدد المعالين في الأسرة.
- `education`: المستوى التعليمي (`Graduate` / `Not Graduate`).
- `self_employed`: العمل الحر (`Yes` / `No`).
- `income_annum`: الدخل السنوي الإجمالي.
- `loan_amount`: مبلغ القرض المطلوب.
- `loan_term`: مدة سداد القرض بالسنوات.
- `cibil_score`: الدرجة الائتمانية (CIBIL Score).
- `residential_assets_value`: قيمة العقارات السكنية.
- `commercial_assets_value`: قيمة العقارات التجارية.
- `luxury_assets_value`: قيمة السيارات والمقتنيات الثمينة.
- `bank_asset_value`: الأرصدة والودائع البنكية.

### 2. المتغير الهدف (Target Variable - $y$):
- `loan_status`: حالة القرض (`Approved` مقبول | `Rejected` مرفوض).

### 3. جدول ملخص الأداء (Model Evaluation Summary):

| المقياس (Metric) | شجرة القرار (Decision Tree) | الانحدار اللوجستي (Logistic Regression) |
| :--- | :---: | :---: |
| **دقة التدريب (Train Accuracy)** | **100.00%** | - |
| **دقة الاختبار (Test Accuracy)** | **97.78%** | **79.86%** |
| **Precision Score** | **98.14%** | **79.74%** |
| **Recall Score** | **98.32%** | **91.04%** |
| **F1 Score** | **98.23%** | **85.02%** |

---

## 🛠️ آلية التثبيت والتشغيل (Setup & Execution Guide)

### 1. استنساخ المستودع (Clone Repository):
```bash
git clone https://github.com/AtheerJB/Loan-Project-SDAIA.git
cd Loan-Project-SDAIA
```

### 2. تثبيت الحزم والمكتبات المطلوبة (Install Requirements):
```bash
pip install -r requirements.txt
```

### 3. تشغيل سكريبت تقييم نموذج التعلم الآلي:
```bash
python loan_classification.py
```

### 4. تشغيل واجهة الموقع التفاعلية (Streamlit Dashboard):
```bash
streamlit run app.py
```

تفتح الواجهة التفاعلية فوراً في المتصفح عبر الرابط: `http://localhost:8501`

---

## 📂 هيكلة المشروع (Project Structure)

```text
📁 Loan-Project-SDAIA/
├── 📄 loan_approval_dataset.csv   # قاعدة بيانات القروض البنكية
├── 📄 loan_classification.py      # كود معالجة وتدريب وتقييم نموذج التعلم الآلي
├── 📄 app.py                      # كود واجهة الموقع العربي التفاعلي (Streamlit)
├── 📄 requirements.txt            # قائمة المكتبات المطلوبة
├── 📄 .gitignore                  # ملف استبعاد الملفات المؤقتة
└── 📄 README.md                   # التوثيق الفني الشامل للمشروع
```

---

## 🤝 شكر وتقدير (Acknowledgements)

تقدم خالص الشكر والتقدير لـ **[أكاديمية سدايا (SDAIA Academy)](https://github.com/SDAIAAcademy)** على دعمها المستمر والبرامج التدريبية المتميزة في بناء القدرات الوطنية في مجالات الذكاء الاصطناعي وتنعيم البيانات.
