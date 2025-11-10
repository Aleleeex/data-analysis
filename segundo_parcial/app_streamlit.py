import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Cáncer de Pulmón",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para interfaz ultra compacta
st.markdown("""
<style>
    /* Reducir espacios generales */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Radio buttons horizontales y compactos */
    .stRadio > div {
        flex-direction: row;
        gap: 0.5rem;
    }
    .stRadio > div > label {
        margin-bottom: 0.1rem;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Inputs más pequeños */
    .stNumberInput > div > div > input {
        height: 2rem;
    }
    
    /* Reducir espacios entre elementos */
    .stSubheader {
        padding-top: 0.25rem;
        padding-bottom: 0.1rem;
        font-size: 1.1rem;
    }
    .element-container {
        margin-bottom: 0.25rem;
    }
    .stMarkdown {
        margin-bottom: 0.25rem;
    }
    
    /* Header más compacto */
    .stMarkdown h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Métricas más pequeñas */
    .metric-container {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo y preprocesadores
@st.cache_resource
def load_model_and_preprocessors():
    """
    Carga el modelo entrenado y los preprocesadores guardados
    """
    try:
        # Cargar modelo
        model = joblib.load('models/best_lung_cancer_model.joblib')
        
        # Cargar preprocesadores
        scaler = joblib.load('models/scaler.joblib')
        label_encoders = joblib.load('models/label_encoders.joblib')
        model_info = joblib.load('models/model_info.joblib')
        
        return model, scaler, label_encoders, model_info
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos del modelo. {e}")
        st.info("Asegúrate de ejecutar primero el notebook 'lung_cancer_ml_analysis.ipynb' para generar los modelos.")
        return None, None, None, None

# Función para hacer predicciones
def predict_lung_cancer(model, scaler, input_data):
    """
    Realiza la predicción usando el modelo cargado
    """
    # Aplicar escalado
    input_scaled = scaler.transform(input_data)
    
    # Hacer predicción
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability

# Cargar modelo y preprocesadores
model, scaler, label_encoders, model_info = load_model_and_preprocessors()

if model is not None:
    # Header compacto con información del modelo en una línea
    st.markdown(f"**Predictor de Cáncer de Pulmón** | Modelo: {model_info['model_name']} | Accuracy: {model_info['accuracy']:.3f} | F1: {model_info['f1_score']:.3f}")
    
    # Formulario de entrada sin separadores innecesarios
    
    # Dividir en tres columnas para mejor organización y menos scroll
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Datos Personales**")
        
        gender = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
        gender_encoded = 0 if gender == "Femenino" else 1
        
        age = st.number_input("Edad", min_value=18, max_value=100, value=50)
        
        
    with col2:
        st.markdown("**Hábitos:**")
        
        # Usar radio buttons horizontales para preguntas Si/No
        smoking = st.radio("¿Fuma?", ["No", "Sí"], horizontal=True)
        smoking_encoded = 1 if smoking == "No" else 2
        
        yellow_fingers = st.radio("¿Dedos Amarillos?", ["No", "Sí"], horizontal=True)
        yellow_fingers_encoded = 1 if yellow_fingers == "No" else 2
        
        anxiety = st.radio("¿Ansiedad?", ["No", "Sí"], horizontal=True)
        anxiety_encoded = 1 if anxiety == "No" else 2
        
        peer_pressure = st.radio("¿Presión de Grupo?", ["No", "Sí"], horizontal=True)
        peer_pressure_encoded = 1 if peer_pressure == "No" else 2
        
        chronic_disease = st.radio("¿Enfermedad Crónica?", ["No", "Sí"], horizontal=True)
        chronic_disease_encoded = 1 if chronic_disease == "No" else 2
    
    with col3:
        st.markdown("**Síntomas Físicos**")
        
        fatigue = st.radio("¿Fatiga?", ["No", "Sí"], horizontal=True)
        fatigue_encoded = 1 if fatigue == "No" else 2
        
        allergy = st.radio("¿Alergias?", ["No", "Sí"], horizontal=True)
        allergy_encoded = 1 if allergy == "No" else 2
        
        wheezing = st.radio("¿Silbido al Respirar?", ["No", "Sí"], horizontal=True)
        wheezing_encoded = 1 if wheezing == "No" else 2
        
        alcohol_consuming = st.radio("¿Consume Alcohol?", ["No", "Sí"], horizontal=True)
        alcohol_consuming_encoded = 1 if alcohol_consuming == "No" else 2
        
        coughing = st.radio("¿Tos?", ["No", "Sí"], horizontal=True)
        coughing_encoded = 1 if coughing == "No" else 2
    
    with col4:
        st.markdown("**Síntomas Respiratorios**")
        
        shortness_of_breath = st.radio("¿Dificultad para Respirar?", ["No", "Sí"], horizontal=True)
        shortness_of_breath_encoded = 1 if shortness_of_breath == "No" else 2
        
        swallowing_difficulty = st.radio("¿Dificultad para Tragar?", ["No", "Sí"], horizontal=True)
        swallowing_difficulty_encoded = 1 if swallowing_difficulty == "No" else 2
        
        chest_pain = st.radio("¿Dolor de Pecho?", ["No", "Sí"], horizontal=True)
        chest_pain_encoded = 1 if chest_pain == "No" else 2
    
    # Botón de predicción sin separador
    if st.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
        # Crear array con los datos de entrada
        input_data = np.array([[
            gender_encoded, age, smoking_encoded, yellow_fingers_encoded, 
            anxiety_encoded, peer_pressure_encoded, chronic_disease_encoded, 
            fatigue_encoded, allergy_encoded, wheezing_encoded, 
            alcohol_consuming_encoded, coughing_encoded, shortness_of_breath_encoded, 
            swallowing_difficulty_encoded, chest_pain_encoded
        ]])
        
        # Realizar predicción
        prediction, probability = predict_lung_cancer(model, scaler, input_data)
        
        # Mostrar resultados en formato compacto
        st.markdown("---")
        
        # Crear tres columnas para mostrar resultados más compactos
        result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
        
        prob_no_cancer = probability[0] * 100
        prob_cancer = probability[1] * 100
        
        with result_col1:
            if prediction == 1:  # Predicción positiva (cáncer)
                st.error("⚠️ **RIESGO ALTO** - Consulte con un oncólogo")
            else:  # Predicción negativa (no cáncer)
                st.success("✅ **RIESGO BAJO** - Mantenga controles regulares")
        
        with result_col2:
            st.metric("Sin Cáncer", f"{prob_no_cancer:.1f}%")
        
        with result_col3:
            st.metric("Con Cáncer", f"{prob_cancer:.1f}%")
        
        # Gráfico más pequeño
        prob_df = pd.DataFrame({
            'Resultado': ['Sin Cáncer', 'Con Cáncer'],
            'Probabilidad': [prob_no_cancer, prob_cancer]
        })
        
        st.bar_chart(prob_df.set_index('Resultado'), height=150)

else:
    st.error("No se pudieron cargar los modelos. Por favor, ejecute primero el análisis de Machine Learning.")

# Sidebar más compacto
st.sidebar.header("Info del Sistema")
st.sidebar.markdown("""
**Modelo ML:**
- 15 características
- Selección automática
- Métricas validadas

**Instrucciones:**
1. Complete el formulario
2. Haga clic en "Realizar Predicción"
3. Revise resultados
""")