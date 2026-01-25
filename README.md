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
| Loss Train | 0.1400 | 0.0095 |
| Loss Val | 0.2910 | 0.0536 |
| Accuracy Train | 93.21% | 99.54% |
| Accuracy Val | 93.75% | 100.00% |

**Interpretacion:**
- El loss de entrenamiento disminuye consistentemente de 0.14 a 0.0095, indicando aprendizaje efectivo
- El accuracy de entrenamiento mejora de 93.21% a 99.54%
- La variabilidad en validacion se debe al tamano pequeno del conjunto (solo 16 imagenes)
- No se observa overfitting severo: el modelo generaliza bien en test

### 5. Matriz de Confusion y Curva ROC
El notebook genera estos dos graficos juntos, mostrando el rendimiento del modelo en el conjunto de prueba (624 imagenes):

**Matriz de Confusion:**

|  | Pred: Normal | Pred: Neumonia |
|--|--------------|----------------|
| **Real: Normal** | 145 (VN) | 89 (FP) |
| **Real: Neumonia** | 1 (FN) | 389 (VP) |

- **Verdaderos Negativos (145):** Pacientes sanos correctamente identificados
- **Falsos Positivos (89):** Pacientes sanos clasificados como neumonia
- **Falsos Negativos (1):** Casos de neumonia no detectados
- **Verdaderos Positivos (389):** Casos de neumonia correctamente detectados

**Curva ROC:**
- **AUC-ROC: 0.9395** - Indica excelente capacidad discriminativa
- El modelo detecta el 99.74% de los casos de neumonia (Recall)
- Solo 1 caso de neumonia no fue detectado

**Relevancia clinica:** En medicina es preferible minimizar los falsos negativos (no perder casos de enfermedad) aunque esto genere mas falsos positivos que pueden verificarse con estudios adicionales.

## Metricas de Rendimiento

| Metrica | Valor |
|---------|-------|
| Accuracy | 85.42% |
| Precision | 81.21% |
| Recall | 99.74% |
| F1-Score | 89.53% |

## Ajuste de Hiperparametros (Tuning)

Se realizaron 5 experimentos variando learning rate, batch size y dropout:

| Exp | Learning Rate | Batch Size | Dropout | Accuracy | Recall | F1-Score |
|-----|---------------|------------|---------|----------|--------|----------|
| 1 | 0.01 | 64 | 0.3 | 82.37% | 91.28% | 0.8521 |
| 2 | 0.001 | 128 | 0.5 | 88.30% | 99.49% | 0.9140 |
| 3 | 0.0001 | 128 | 0.5 | 85.42% | 97.18% | 0.8892 |
| 4 | 0.001 | 32 | 0.5 | 86.54% | 98.21% | 0.9012 |
| 5 | 0.001 | 128 | 0.7 | 84.78% | 95.64% | 0.8756 |

**Configuracion optima seleccionada:** lr=0.001, batch_size=128, dropout=0.5

## Comparacion de Arquitecturas

Se evaluaron 5 arquitecturas CNN con la misma configuracion:

| Arquitectura | Parametros | Tamanio | Accuracy | Recall | AUC-ROC |
|--------------|------------|---------|----------|--------|---------|
| ResNet18 | 11.2M | 45 MB | 88.30% | 99.49% | 0.9395 |
| ResNet34 | 21.3M | 85 MB | 87.82% | 98.97% | 0.9312 |
| DenseNet121 | 7.0M | 28 MB | 86.54% | 97.69% | 0.9187 |
| VGG16 | 138.4M | 528 MB | 85.26% | 96.41% | 0.9054 |
| MobileNetV2 | 2.2M | 9 MB | 84.13% | 95.38% | 0.8923 |

**Modelo seleccionado:** ResNet18 ofrece el mejor balance entre rendimiento y eficiencia.

## Analisis de Ensemble

Se evaluo si combinar modelos mejora el rendimiento:

| Estrategia | Accuracy | Recall | F1-Score |
|------------|----------|--------|----------|
| ResNet18 (Individual) | 88.30% | 99.49% | 0.9140 |
| Voting: ResNet18 + ResNet34 | 88.14% | 99.23% | 0.9098 |
| Voting: ResNet18 + DenseNet121 | 87.66% | 98.72% | 0.9021 |
| Voting: Top 3 | 87.34% | 98.46% | 0.8987 |

**Conclusion:** El modelo individual ResNet18 supera a todos los ensembles, con menor complejidad y tiempo de inferencia.

## Instalacion

```bash
# Clonar repositorio
git clone https://github.com/Eledun/proyecto-final-neumonia.git
cd proyecto-final-neumonia

# Instalar dependencias
pip install -r requirements.txt
```

## Dataset

Descargar de [Kaggle - Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) y colocar en carpeta `chest_xray/`.

## Modelo Pre-entrenado

El modelo entrenado (`pneumonia_model.pth`) pesa **45 MB** y puede ser descargado directamente desde:

[Descargar modelo desde Google Drive](https://drive.google.com/file/d/1OKixGGrA6RVRTX-EUnTOKwar4bRytqBA/view?usp=drive_link)

Una vez descargado, colocar el archivo en la raiz del proyecto.

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
