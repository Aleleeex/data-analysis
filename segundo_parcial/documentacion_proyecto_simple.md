# PROYECTO PREDICCIÓN DE CÁNCER DE PULMÓN
**Estudiante:** Alexander Parra  
**Fecha:** 9 de noviembre de 2025  
**Curso:** Machine Learning  

---

## RESUMEN DEL PROYECTO

Este proyecto desarrolla un sistema de Machine Learning para predecir el riesgo de cáncer de pulmón basado en síntomas y hábitos del paciente. El sistema incluye análisis de datos, entrenamiento de modelos y una aplicación web para hacer predicciones en tiempo real.

---

## TECNOLOGÍAS UTILIZADAS

### **Lenguajes y Librerías:**
- **Python 3.8+** - Lenguaje principal
- **pandas, numpy** - Manipulación de datos
- **scikit-learn** - Machine Learning
- **matplotlib, seaborn** - Visualización
- **Streamlit** - Aplicación web
- **Jupyter Notebook** - Desarrollo y análisis

### **Herramientas de Desarrollo:**
- **Jupyter Notebook** - Análisis interactivo
- **VS Code** - Editor de código
- **Git** - Control de versiones

---

## DATASET UTILIZADO

### **Información General:**
- **Nombre:** Survey Lung Cancer Dataset
- **Fuente:** Kaggle
- **Tamaño:** 311 registros, 16 variables
- **Tipo:** Clasificación binaria (SI/NO tiene cáncer)

### **Variables Principales:**
- **Demográficas:** Género, Edad
- **Hábitos:** Fumar, Consumo de alcohol
- **Síntomas físicos:** Dedos amarillos, Fatiga, Ansiedad
- **Síntomas respiratorios:** Tos, Dificultad respiratoria, Silbidos, Dolor de pecho
- **Otros:** Alergias, Enfermedad crónica, Presión de grupo

---

## MODELOS CREADOS

Se implementaron **4 modelos de Machine Learning**:

### **1. Regresión Logística**
- Modelo base de clasificación
- Fácil interpretación
- Buen rendimiento general

### **2. K-Vecinos Más Cercanos (k-NN)**
- Optimización del parámetro k (3 a 19)
- Selección automática del mejor k
- Basado en similitud de casos

### **3. Random Forest**
- Ensemble de árboles de decisión
- Análisis de importancia de variables
- Resistente al sobreajuste

### **4. Regresión Logística Optimizada**
- Optimización con GridSearchCV
- Búsqueda de mejores hiperparámetros
- Validación cruzada de 5 pliegues

### **Selección del Modelo:**
- **Criterio:** F1-Score (balance entre precisión y recall)
- **Selección automática:** El modelo con mejor F1-Score se guarda automáticamente
- **Métricas evaluadas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

---

## IMPLEMENTACIÓN WEB

### **Framework:** Streamlit
La aplicación web permite a usuarios hacer predicciones de riesgo de cáncer de pulmón de forma interactiva.

### **Características de la Aplicación:**
- **Interfaz intuitiva** con formulario organizado en 3 columnas
- **Entrada de datos** mediante botones de radio y campos numéricos
- **Predicción en tiempo real** al completar el formulario
- **Visualización de resultados** con probabilidades y recomendaciones
- **Diseño compacto** optimizado para no requerir scroll

### **Funcionalidad:**
1. **Carga del modelo:** Sistema automático que carga el mejor modelo entrenado
2. **Procesamiento de datos:** Codificación y escalado automático de las respuestas
3. **Predicción:** Cálculo de probabilidades de riesgo alto/bajo
4. **Resultados:** Muestra el riesgo con colores (verde=bajo, rojo=alto) y porcentajes

### **Estructura del Formulario:**
- **Columna 1:** Datos personales (edad, género) y hábitos (fumar, alcohol)
- **Columna 2:** Síntomas físicos (fatiga, ansiedad, alergias, dolor de pecho)
- **Columna 3:** Síntomas respiratorios (tos, dificultad respiratoria, silbidos)

---

## PROCESO DE DESARROLLO

### **Parte 1: Análisis de Machine Learning**
1. **Carga y exploración** del dataset
2. **Análisis exploratorio** de datos (EDA)
3. **Preprocesamiento** (codificación de variables categóricas, escalado)
4. **División** de datos (80% entrenamiento, 20% prueba)
5. **Entrenamiento** de 4 modelos diferentes
6. **Evaluación** con múltiples métricas
7. **Selección automática** del mejor modelo
8. **Guardado** del modelo y preprocessors

### **Parte 2: Aplicación Web**
1. **Diseño de interfaz** con Streamlit
2. **Carga del modelo** entrenado
3. **Creación del formulario** de entrada
4. **Implementación de predicción** en tiempo real
5. **Optimización de la interfaz** para máxima usabilidad
6. **Despliegue local** de la aplicación

---

## RESULTADOS

### **Rendimiento de Modelos:**
Todos los modelos lograron un rendimiento superior al 85% en las métricas principales, con el modelo seleccionado automáticamente alcanzando el mejor balance entre precisión y recall.

### **Aplicación Web:**
- **Tiempo de respuesta:** Menos de 2 segundos
- **Usabilidad:** Interfaz clara y sin necesidad de conocimientos técnicos
- **Funcionalidad:** Predicciones precisas basadas en el mejor modelo

---

## ARCHIVOS DEL PROYECTO

```
segundo_parcial/
├── survey_lung_cancer.csv          # Dataset original
├── lung_cancer_ml_analysis.ipynb   # Notebook con análisis completo
├── app_streamlit.py                # Aplicación web
├── models/                         # Modelos entrenados
│   ├── best_lung_cancer_model.joblib
│   ├── scaler.joblib
│   ├── label_encoders.joblib
│   └── model_info.joblib
└── documentacion_proyecto_simple.md # Esta documentación
```

---

## CONCLUSIONES

1. **Desarrollo exitoso** de un sistema completo de Machine Learning para predicción médica
2. **Implementación web funcional** que permite uso práctico del modelo entrenado
3. **Selección automática** del mejor modelo basada en métricas objetivas
4. **Interfaz optimizada** para usuarios no técnicos
5. **Proyecto integral** que demuestra el ciclo completo desde datos hasta aplicación

---

## COMANDOS PARA EJECUTAR

### Análisis de ML:
```bash
jupyter notebook lung_cancer_ml_analysis.ipynb
```

### Aplicación Web:
```bash
streamlit run app_streamlit.py
```

---