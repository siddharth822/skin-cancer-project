from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
from pathlib import Path

from .inference import Predictor
from .image_quality import validate_photo
from .guidance import GUIDANCE, stage_info

app = FastAPI(
    title="SkinSight AI",
    description="Educational skin-lesion screening using an own-trained MobileNetV3-Small model.",
    version="0.1.0",
)

APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
predictor = Predictor()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_ready": predictor.ready, "model_meta": predictor.meta},
    )

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": predictor.ready,
        "model": predictor.meta if predictor.ready else None,
    }

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="The model has not been trained yet. Run ml/train.py first.",
        )

    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG, or WEBP image.")

    raw = await image.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        validate_photo(Image.open(io.BytesIO(raw)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = predictor.predict_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    label = result["label"]
    result["guidance"] = GUIDANCE[label]
    result["stage"] = stage_info(label)
    result["medical_disclaimer"] = (
        "This is an educational AI screening result, not a medical diagnosis. "
        "Model scores are not calibrated probabilities of disease. A clinician must assess concerning lesions."
    )
    return JSONResponse(result)
