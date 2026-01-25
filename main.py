from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi.responses import RedirectResponse


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cpu"

app = FastAPI(
    title="API para la detección de neumonía",
    description="Soporte imagenologico NO CONSTITUYE DIAGNOSTICO",
    version="1.0"
)

# =====================================================
# DEFINICIÓN DEL MODELO (IDÉNTICA AL NOTEBOOK)
# =====================================================

def create_model(num_classes=2):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    # IMPORTANTE: esta definición debe coincidir EXACTAMENTE
    # con la usada durante el entrenamiento
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model

# Cargar modelo y pesos
model = create_model(num_classes=2)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================================================
# TRANSFORMACIONES (VALIDACIÓN / TEST)
# =====================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# =====================================================
# ENDPOINT DE PREDICCIÓN
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        prob_pneumonia = probs[0, 1].item()
        pred_class = int(torch.argmax(probs, dim=1).item())

    return {
        "prediction": pred_class,
        "probability_pneumonia": round(prob_pneumonia, 4),
        "label": "Neumonía" if pred_class == 1 else "Normal",
        "note": "Resultado de apoyo al screening clínico. No constituye diagnóstico médico."
    }

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


