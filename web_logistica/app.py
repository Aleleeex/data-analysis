
import streamlit as st
import joblib
import os


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_regresion_logistica.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

st.set_page_config(page_title="Predicción de Compra - Regresión Logística", page_icon="🛒")
st.markdown("""
<style>
body { background: #1a1d21; color: #ececf1; }
.stApp { background: #1a1d21; }
.stButton>button { background: #10a37f; color: #fff; border-radius: 8px; }
.stButton>button:hover { background: #0e8c6c; }
</style>
""", unsafe_allow_html=True)

st.title("Predicción de Compra (Edad + Salario)")
st.write("El modelo fue entrenado con las columnas Edad y Salario estimado (escaladas).")

edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
salario = st.number_input("Salario estimado", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

if st.button("Predecir"):
    try:
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        modelo = joblib.load(MODEL_PATH)
        X = [[edad, salario]]
        X_scaled = scaler.transform(X) if scaler is not None else X
        pred = int(modelo.predict(X_scaled)[0])
        prob_txt = ""
        try:
            proba = float(modelo.predict_proba(X_scaled)[0][1])
            prob_txt = f" con probabilidad {proba*100:.1f}%"
        except Exception:
            pass
        if pred == 1:
            st.success(f"Resultado: Compraría{prob_txt}")
        else:
            st.error(f"Resultado: No compraría{prob_txt}")
    except Exception as e:
        st.error(f"Error: {e}")
