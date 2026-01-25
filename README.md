# Deteccion de Neumonia en Rayos X

Proyecto Final - Modulo 7 | Bootcamp de Ciencia de Datos UDD

## Descripcion del Proyecto

Este proyecto implementa un sistema de clasificacion automatica de imagenes de rayos X de torax para detectar neumonia utilizando Deep Learning. El objetivo es proporcionar una herramienta de apoyo al diagnostico medico que pueda identificar patrones en radiografias.

### Problema a Resolver

La neumonia es una infeccion pulmonar que afecta a millones de personas anualmente. El diagnostico temprano es crucial para un tratamiento efectivo. Este modelo analiza radiografias de torax y clasifica si el paciente presenta signos de neumonia o si esta sano.

### Solucion Implementada

Se utiliza **Transfer Learning** con la arquitectura **ResNet18** preentrenada en ImageNet. Esta tecnica aprovecha el conocimiento previo del modelo en reconocimiento de imagenes y lo adapta para nuestro problema especifico de clasificacion binaria (Normal vs Neumonia).

**Caracteristicas tecnicas:**
- Arquitectura: ResNet18 con capa final modificada (Dropout 0.5 + Linear 512->2)
- Entrenamiento optimizado con Mixed Precision (AMP) para GPU
- Data Augmentation: flip horizontal, rotacion, variacion de color
- Balanceo de clases con WeightedRandomSampler
- API REST con FastAPI para predicciones en tiempo real

## Analisis del Dataset

El dataset contiene 5,856 imagenes de rayos X de torax divididas en:

| Conjunto | Normal | Neumonia | Total | Ratio |
|----------|--------|----------|-------|-------|
| Train | 1,341 | 3,875 | 5,216 | 2.89:1 |
| Test | 234 | 390 | 624 | 1.67:1 |
| Val | 8 | 8 | 16 | 1:1 |

**Observacion:** El dataset presenta un desbalance significativo (2.7x mas casos de neumonia). Para mitigar esto se implemento WeightedRandomSampler durante el entrenamiento.

## Analisis de Graficos

### 1. Distribucion de Clases (Grafico de Barras)
El notebook genera un grafico de barras que muestra la cantidad de imagenes por clase en los conjuntos de Entrenamiento y Prueba. Se observa un desbalance significativo: en entrenamiento hay 3,875 imagenes de neumonia contra solo 1,341 normales (ratio 2.89:1).

**Implicaciones:** Sin correccion, el modelo tenderia a predecir siempre "neumonia" para maximizar accuracy. Para solucionar esto se implemento:
- **WeightedRandomSampler:** Muestrea las clases de forma balanceada durante entrenamiento
- **CrossEntropyLoss con pesos:** Penaliza mas los errores en la clase minoritaria

### 2. Muestras de Rayos X (Grid de Imagenes)
El notebook muestra una cuadricula de 2x4 con ejemplos de radiografias de cada clase. Esto permite apreciar visualmente las diferencias:

- **Radiografias Normales:** Pulmones con areas oscuras uniformes (aire), contornos claros del corazon y diafragma, sin manchas anormales.

- **Radiografias con Neumonia:** Areas blanquecinas difusas (opacidades) que indican acumulacion de liquido o pus en los alveolos pulmonares.

### 3. Distribucion de Dimensiones (Histogramas)
El notebook genera dos histogramas:
- **Histograma de dimensiones:** Muestra la distribucion de ancho y alto de las imagenes originales
- **Histograma de aspect ratio:** Muestra la relacion ancho/alto, con linea de referencia en 1.0

**Preprocesamiento aplicado:** Todas las imagenes se redimensionan a 224x224 pixeles para compatibilidad con ResNet18.

### 4. Curvas de Entrenamiento (Loss y Accuracy)
El notebook genera dos graficos que muestran la evolucion del modelo durante 10 epocas de entrenamiento:

| Metrica | Epoca 1 | Epoca 10 |
|---------|---------|----------|
| Loss Train | 0.1258 | 0.0092 |
| Loss Val | 0.4432 | 0.0273 |
| Accuracy Train | 93.69% | 99.54% |
| Accuracy Val | 56.25% | 100.00% |

**Interpretacion:**
- El loss de entrenamiento disminuye consistentemente de 0.12 a 0.009, indicando aprendizaje efectivo
- El accuracy de entrenamiento mejora de 93.69% a 99.54%
- La variabilidad en validacion se debe al tamano pequeno del conjunto (solo 16 imagenes)
- No se observa overfitting severo: el modelo generaliza bien en test

### 5. Matriz de Confusion y Curva ROC
El notebook genera estos dos graficos juntos, mostrando el rendimiento del modelo en el conjunto de prueba (624 imagenes):

**Matriz de Confusion:**

|  | Pred: Normal | Pred: Neumonia |
|--|--------------|----------------|
| **Real: Normal** | 163 (VN) | 71 (FP) |
| **Real: Neumonia** | 2 (FN) | 388 (VP) |

- **Verdaderos Negativos (163):** Pacientes sanos correctamente identificados
- **Falsos Positivos (71):** Pacientes sanos clasificados como neumonia
- **Falsos Negativos (2):** Casos de neumonia no detectados
- **Verdaderos Positivos (388):** Casos de neumonia correctamente detectados

**Curva ROC:**
- **AUC-ROC: 0.9395** - Indica excelente capacidad discriminativa
- El modelo detecta el 99.5% de los casos de neumonia (Recall)
- Solo 2 casos de neumonia no fueron detectados

**Relevancia clinica:** En medicina es preferible minimizar los falsos negativos (no perder casos de enfermedad) aunque esto genere mas falsos positivos que pueden verificarse con estudios adicionales.

## Metricas de Rendimiento

| Metrica | Valor |
|---------|-------|
| Accuracy | 88.30% |
| Precision | 84.53% |
| Recall | 99.49% |
| F1-Score | 91.40% |

## Instalacion

```bash
# Clonar repositorio
git clone https://github.com/Eledun/proyecto-final-neumonia
cd proyecto-final-neumonia

# Instalar dependencias
pip install -r requirements.txt
```

## Dataset

Descargar de [Kaggle - Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) y colocar en carpeta `chest_xray/`.

## Uso

### Entrenar modelo
```bash
jupyter notebook UDD_Proyecto_M7.ipynb
```

### Ejecutar API
```bash
uvicorn main:app --reload --port 8001
```

La documentacion Swagger estara disponible en `http://localhost:8001/docs`

### Exponer en internet con ngrok
```bash
ngrok http 8001
```

### Hacer prediccion
```bash
curl -X POST "http://localhost:8001/predict" -F "file=@imagen.jpg"
```

## Estructura del Proyecto
```
proyecto-final/
├── UDD_Proyecto_M7.ipynb   # Notebook (EDA, entrenamiento, graficos)
├── main.py                 # API REST con FastAPI
├── pneumonia_model.pth     # Modelo entrenado
├── requirements.txt        # Dependencias
└── chest_xray/             # Dataset (descargar de Kaggle)
```

## Autor

Eduardo Herrera Martinez - Cohort 12
