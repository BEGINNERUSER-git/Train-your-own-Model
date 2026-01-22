import joblib
import os

MODEL_DIR="saved_models"
os.makedirs(MODEL_DIR,exist_ok=True)

def save_model(model,model_name):
    path=os.path.join(MODEL_DIR,f"{model_name}.pkl")
    joblib.dump(model,path)
    return path

def load_model(model_name):
    path=os.path.join(MODEL_DIR,f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")
    return joblib.load(path)

def list_saved_model():
    return [f.replace(".pkl","") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
