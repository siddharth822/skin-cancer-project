from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
from pathlib import Path

from .inference import Predictor
from .image_quality import validate_photo
from . import auth
import secrets
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
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    csrf = request.cookies.get("skinsight_csrf") or secrets.token_urlsafe(32)
    response = templates.TemplateResponse(
        "index.html",
        {"request": request, "model_ready": predictor.ready, "model_meta": predictor.meta, "username": user, "csrf": csrf},
    )

    response.set_cookie("skinsight_csrf", csrf, httponly=True, secure=auth.SECURE_COOKIE, samesite="strict")
    response.headers["Cache-Control"] = "no-store"
    return response

def auth_page(request, register=False, error=None, status=200):
    csrf = request.cookies.get("skinsight_csrf") or secrets.token_urlsafe(32)
    response = templates.TemplateResponse("login.html", {"request": request, "register": register, "error": error, "csrf": csrf}, status_code=status)
    response.set_cookie("skinsight_csrf", csrf, httponly=True, secure=auth.SECURE_COOKIE, samesite="strict")
    response.headers["Cache-Control"] = "no-store"
    return response

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return auth_page(request)

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return auth_page(request, register=True)

@app.post("/register")
def register_account(request: Request, username: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    auth.check_csrf(request, csrf)
    try:
        auth.register(username, password)
    except ValueError as exc:
        return auth_page(request, register=True, error=str(exc), status=400)
    return RedirectResponse("/login", status_code=303)

@app.post("/login")
def login_account(request: Request, username: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    auth.check_csrf(request, csrf)
    try:
        token = auth.login(username, password, request.client.host if request.client else "unknown")
    except ValueError as exc:
        return auth_page(request, error=str(exc), status=400)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("skinsight_session", token, max_age=auth.SESSION_SECONDS, httponly=True, secure=auth.SECURE_COOKIE, samesite="strict")
    return response

@app.post("/logout")
def logout_account(request: Request, csrf: str = Form(...)):
    auth.check_csrf(request, csrf)
    auth.logout(request)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("skinsight_session")
    return response

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": predictor.ready,
        "model": predictor.meta if predictor.ready else None,
    }

@app.post("/api/predict")
async def predict(request: Request, image: UploadFile = File(...), user: str = Depends(auth.require_user)):
    auth.check_csrf(request, request.headers.get("X-CSRF-Token"))
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
