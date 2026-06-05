import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our custom Predictor Engine
from ml.prediction import DeepfakePredictor

predictor_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_engine
    print("\n--- Booting TrueLens Backend ---")
    print("Loading AI Model into memory. This may take a few seconds...")
    predictor_engine = DeepfakePredictor()
    print("TrueLens API is live and ready for requests!\n")
    yield
    print("Shutting down TrueLens API...")

app = FastAPI(title="TrueLens Deepfake Detector API", lifespan=lifespan)

# --- THE UNLOCK: Allows Vercel frontend to talk to your laptop ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "TrueLens API is online. Engine Ready."}

# --- THE FIX: Changed route to "/predict" to perfectly match your frontend ---
# Kept it synchronous (no async) to protect TensorFlow's threading.
@app.post("/predict")
def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. Save the incoming image to temporary memory
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Pass the image to your AI brain
        result = predictor_engine.predict(temp_file_path)
        
        # 3. Clean up the file so your laptop doesn't run out of storage
        os.remove(temp_file_path)
        
        # 4. Fire the AI's verdict back to the Vercel frontend
        return result
        
    except Exception as e:
        # Print the exact crash to the terminal so you are never blind
        print("\n[ SERVER CRASH TRACEBACK ]")
        traceback.print_exc()
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")