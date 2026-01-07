import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Exam Score Prediction", layout="centered")
st.title("📘 Exam Score Prediction App")
st.write("Predict student exam scores using **KNN Regression**")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Exam_Score_Prediction.pkl")

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# -------------------------------
# Preprocessing
# -------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔧 Model Settings")
k = st.sidebar.slider("Select K (Neighbors)", 1, 15, 5)

# -------------------------------
# Model Pipeline
# -------------------------------
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("knn", KNeighborsRegressor(n_neighbors=k))
])

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.success(f"✅ Model trained successfully!")
st.write(f"📈 R² Score: **{score:.2f}**")

# -------------------------------
# User Input Form
# -------------------------------
st.subheader("📝 Enter Student Details")

input_data = {}
for col in X.columns:
    if col in cat_cols:
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([input_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🎯 Predict Exam Score"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Exam Score: **{prediction[0]:.2f}**")
