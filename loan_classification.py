import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# الخطوة 1: قراءة الملف وعرض أول 5 صفوف
# ==========================================
print("--- Step 1: Read Dataset ---")
df = pd.read_csv("loan_approval_dataset.csv")
print(df.head())

# ==========================================
# الخطوة 2: تنظيف أسماء الأعمدة وفصل X و y
# ==========================================
print("\n--- Step 2: Clean Column Names & Split X, y ---")

# إزالة المسافات الزائدة من أسماء الأعمدة (مثل ' loan_status' -> 'loan_status')
df.columns = df.columns.str.strip()

# تنظيف المسافات الزائدة من القيم النصية داخل الجدول إن وجدت
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        df[col] = df[col].astype(str).str.strip()

# فصل البيانات (y: الهدف | X: المدخلات)
y = df['loan_status']
X = df.drop(columns=['loan_id', 'loan_status'])

print("Cleaned Columns:", list(df.columns))
print("Input Features (X):", list(X.columns))
print("Target Feature (y): loan_status")

# ==========================================
# الخطوة 3: تحويل القيم النصية إلى أرقام (Preprocessing)
# ==========================================
print("\n--- Step 3: Encoding Categorical Data ---")

# تحويل المتغير الهدف y
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# تحويل الأعمدة النصية في X
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object' or X_encoded[col].dtype == 'string':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

print("\nFirst 5 rows of X after Encoding:")
print(X_encoded.head())

# ==========================================
# الخطوة 4: تقسيم البيانات (80% تدريب / 20% اختبار)
# ==========================================
print("\n--- Step 4: Train / Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

print(f"Training samples (80%): {X_train.shape[0]}")
print(f"Testing samples  (20%): {X_test.shape[0]}")

# ==========================================
# الخطوة 5: بناء النماذج، التدريب، وتقييم المقاييس (Evaluation Metrics)
# ==========================================
print("\n--- Step 5: Model Training & Comprehensive Evaluation ---")

# --- نموذج شجرة القرار (Decision Tree Classifier) ---
print("\n==========================================")
print("Decision Tree Classifier Metrics:")
print("==========================================")
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)

train_acc_tree = tree_clf.score(X_train, y_train)
test_acc_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, pos_label=0)
recall_tree = recall_score(y_test, y_pred_tree, pos_label=0)
f1_tree = f1_score(y_test, y_pred_tree, pos_label=0)
cm_tree = confusion_matrix(y_test, y_pred_tree)

print(f"Training Accuracy : {train_acc_tree * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc_tree * 100:.2f}%")
print(f"Precision Score   : {precision_tree * 100:.2f}%")
print(f"Recall Score      : {recall_tree * 100:.2f}%")
print(f"F1 Score          : {f1_tree * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm_tree)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_tree, target_names=le_y.classes_))


# --- نموذج الانحدار اللوجستي (Logistic Regression) ---
print("\n==========================================")
print("Logistic Regression Metrics:")
print("==========================================")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)

test_acc_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log, pos_label=0)
recall_log = recall_score(y_test, y_pred_log, pos_label=0)
f1_log = f1_score(y_test, y_pred_log, pos_label=0)
cm_log = confusion_matrix(y_test, y_pred_log)

print(f"Testing Accuracy  : {test_acc_log * 100:.2f}%")
print(f"Precision Score   : {precision_log * 100:.2f}%")
print(f"Recall Score      : {recall_log * 100:.2f}%")
print(f"F1 Score          : {f1_log * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm_log)