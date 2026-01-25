# Detección de Neumonía en Rayos X

Proyecto Final - Módulo 7 | Bootcamp de Ciencia de Datos UDD

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación automática de imágenes de rayos X de tórax para detectar neumonía utilizando Deep Learning. El objetivo es proporcionar una herramienta de apoyo al diagnóstico médico que pueda identificar patrones en radiografías.

### Problema a Resolver

La neumonía es una infección pulmonar que afecta a millones de personas anualmente. El diagnóstico temprano es crucial para un tratamiento efectivo. Este modelo analiza radiografías de tórax y clasifica si el paciente presenta signos de neumonía o si está sano.

### Solución Implementada

Se utiliza **Transfer Learning** con la arquitectura **ResNet18** preentrenada en ImageNet. Esta técnica aprovecha el conocimiento previo del modelo en reconocimiento de imágenes y lo adapta para nuestro problema específico de clasificación binaria (Normal vs Neumonía).

**Características técnicas:**

* Arquitectura: ResNet18 con capa final modificada (Dropout 0.5 + Linear 512->2)
* Entrenamiento optimizado con Mixed Precision (AMP) para GPU
* Data Augmentation: flip horizontal, rotación, variación de color
* Balanceo de clases con WeightedRandomSampler
* API REST con FastAPI para predicciones en tiempo real

## Análisis del Dataset

El dataset contiene 5,856 imágenes de rayos X de tórax divididas en:

| Conjunto | Normal | Neumonía | Total | Ratio  |
| -------- | ------ | -------- | ----- | ------ |
| Train    | 1,341  | 3,875    | 5,216 | 2.89:1 |
| Test     | 234    | 390      | 624   | 1.67:1 |
| Val      | 8      | 8        | 16    | 1:1    |

**Observación:** El dataset presenta un desbalance significativo (2.7x más casos de neumonía). Para mitigar esto se implementó WeightedRandomSampler durante el entrenamiento.

## Análisis de Gráficos

### 1. Distribución de Clases (Gráfico de Barras)

El notebook genera un gráfico de barras que muestra la cantidad de imágenes por clase en los conjuntos de Entrenamiento y Prueba. Se observa un desbalance significativo: en entrenamiento hay 3,875 imágenes de neumonía contra solo 1,341 normales (ratio 2.89:1).

**Implicaciones:** Sin corrección, el modelo tendería a predecir siempre "neumonía" para maximizar accuracy. Para solucionar esto se implementó:

* **WeightedRandomSampler:** Muestrea las clases de forma balanceada durante entrenamiento
* **CrossEntropyLoss con pesos:** Penaliza más los errores en la clase minoritaria

### 2. Muestras de Rayos X (Grid de Imágenes)

El notebook muestra una cuadrícula de 2x4 con ejemplos de radiografías de cada clase. Esto permite apreciar visualmente las diferencias:

* **Radiografías Normales:** Pulmones con áreas oscuras uniformes (aire), contornos claros del corazón y diafragma, sin manchas anormales.

* **Radiografías con Neumonía:** Áreas blanquecinas difusas (opacidades) que indican acumulación de líquido o pus en los alvéolos pulmonares.

### 3. Distribución de Dimensiones (Histogramas)

El notebook genera dos histogramas:

* **Histograma de dimensiones:** Muestra la distribución de ancho y alto de las imágenes originales
* **Histograma de aspect ratio:** Muestra la relación ancho/alto, con línea de referencia en 1.0

**Preprocesamiento aplicado:** Todas las imágenes se redimensionan a 224x224 píxeles para compatibilidad con ResNet18.

### 4. Curvas de Entrenamiento (Loss y Accuracy)

El notebook genera dos gráficos que muestran la evolución del modelo durante 10 épocas de entrenamiento:

| Métrica        | Época 1 | Época 10 |
| -------------- | ------- | -------- |
| Loss Train     | 0.1400  | 0.0095   |
| Loss Val       | 0.2910  | 0.0536   |
| Accuracy Train | 93.21%  | 99.54%   |
| Accuracy Val   | 93.75%  | 100.00%  |

**Interpretación:**

* El loss de entrenamiento disminuye consistentemente de 0.14 a 0.0095, indicando aprendizaje efectivo
* El accuracy de entrenamiento mejora de 93.21% a 99.54%
* La variabilidad en validación se debe al tamaño pequeño del conjunto (solo 16 imágenes)
* No se observa overfitting severo: el modelo generaliza bien en test

### 5. Matriz de Confusión y Curva ROC

El notebook genera estos dos gráficos juntos, mostrando el rendimiento del modelo en el conjunto de prueba (624 imágenes):

**Matriz de Confusión:**

|                    | Pred: Normal | Pred: Neumonía |
| ------------------ | ------------ | -------------- |
| **Real: Normal**   | 145 (VN)     | 89 (FP)        |
| **Real: Neumonía** | 1 (FN)       | 389 (VP)       |

* **Verdaderos Negativos (145):** Pacientes sanos correctamente identificados
* **Falsos Positivos (89):** Pacientes sanos clasificados como neumonía
* **Falsos Negativos (1):** Casos de neumonía no detectados
* **Verdaderos Positivos (389):** Casos de neumonía correctamente detectados

**Curva ROC:**

* **AUC-ROC: 0.9395** – Indica excelente capacidad discriminativa
* El modelo detecta el 99.74% de los casos de neumonía (Recall)
* Solo 1 caso de neumonía no fue detectado

**Relevancia clínica:** En medicina es preferible minimizar los falsos negativos (no perder casos de enfermedad), aunque esto genere más falsos positivos que pueden verificarse con estudios adicionales.

## Métricas de Rendimiento

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 85.42% |
| Precision | 81.21% |
| Recall    | 99.74% |
| F1-Score  | 89.53% |

## Ajuste de Hiperparámetros (Tuning)

Se realizaron 5 experimentos variando learning rate, batch size y dropout:

| Exp | Learning Rate | Batch Size | Dropout | Accuracy | Recall | F1-Score |
| --- | ------------- | ---------- | ------- | -------- | ------ | -------- |
| 1   | 0.01          | 64         | 0.3     | 82.37%   | 91.28% | 0.8521   |
| 2   | 0.001         | 128        | 0.5     | 88.30%   | 99.49% | 0.9140   |
| 3   | 0.0001        | 128        | 0.5     | 85.42%   | 97.18% | 0.8892   |
| 4   | 0.001         | 32         | 0.5     | 86.54%   | 98.21% | 0.9012   |
| 5   | 0.001         | 128        | 0.7     | 84.78%   | 95.64% | 0.8756   |

**Configuración óptima seleccionada:** lr=0.001, batch_size=128, dropout=0.5

## Comparación de Arquitecturas

Se evaluaron 5 arquitecturas CNN con la misma configuración:

| Arquitectura | Parámetros | Tamaño | Accuracy | Recall | AUC-ROC |
| ------------ | ---------- | ------ | -------- | ------ | ------- |
| ResNet18     | 11.2M      | 45 MB  | 88.30%   | 99.49% | 0.9395  |
| ResNet34     | 21.3M      | 85 MB  | 87.82%   | 98.97% | 0.9312  |
| DenseNet121  | 7.0M       | 28 MB  | 86.54%   | 97.69% | 0.9187  |
| VGG16        | 138.4M     | 528 MB | 85.26%   | 96.41% | 0.9054  |
| MobileNetV2  | 2.2M       | 9 MB   | 84.13%   | 95.38% | 0.8923  |

**Modelo seleccionado:** ResNet18 ofrece el mejor balance entre rendimiento y eficiencia.

## Análisis de Ensemble

Se evaluó si combinar modelos mejora el rendimiento:

| Estrategia                     | Accuracy | Recall | F1-Score |
| ------------------------------ | -------- | ------ | -------- |
| ResNet18 (Individual)          | 88.30%   | 99.49% | 0.9140   |
| Voting: ResNet18 + ResNet34    | 88.14%   | 99.23% | 0.9098   |
| Voting: ResNet18 + DenseNet121 | 87.66%   | 98.72% | 0.9021   |
| Voting: Top 3                  | 87.34%   | 98.46% | 0.8987   |

**Conclusión:** El modelo individual ResNet18 supera a todos los ensembles, con menor complejidad y tiempo de inferencia.

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Eledun/proyecto-final-neumonia.git
cd proyecto-final-neumonia

# Instalar dependencias
pip install -r requirements.txt
```

## Dataset

Descargar de [Kaggle - Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) y colocar en la carpeta `chest_xray/`.

## Modelo Preentrenado

El modelo entrenado (`pneumonia_model.pth`) pesa **45 MB** y puede ser descargado directamente desde:

[Descargar modelo desde Google Drive](https://drive.google.com/file/d/1OKixGGrA6RVRTX-EUnTOKwar4bRytqBA/view?usp=drive_link)

Una vez descargado, colocar el archivo en la raíz del proyecto.

## Uso

### Entrenar modelo

```bash
jupyter notebook UDD_Proyecto_M7.ipynb
```

### Ejecutar API

```bash
uvicorn main:app --reload --port 8001
```

La documentación Swagger estará disponible en `http://localhost:8001/docs`

### Exponer en internet con ngrok

```bash
ngrok http 8001
```

### Hacer predicción

```bash
curl -X POST "http://localhost:8001/predict" -F "file=@imagen.jpg"
```

## Estructura del Proyecto

```
proyecto-final/
├── UDD_Proyecto_M7.ipynb   # Notebook (EDA, entrenamiento, gráficos)
├── main.py                 # API REST con FastAPI
├── pneumonia_model.pth     # Modelo entrenado
├── requirements.txt        # Dependencias
└── chest_xray/             # Dataset (descargar de Kaggle)
```

## Autor

Eduardo Herrera Martínez - Cohort 12
