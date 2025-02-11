from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify frontend domain)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Database setup (SQLite for simplicity)
DATABASE_URL = "sqlite:///./pipelines.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# Global variables for dataset and model
X_train, X_test, y_train, y_test = None, None, None, None
model = None


# ------------------- Database Model -------------------
class Pipeline(Base):
    __tablename__ = "pipelines"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    nodes = Column(Text)
    edges = Column(Text)

Base.metadata.create_all(bind=engine)


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working with CORS!"}

# ------------------- Data Upload -------------------
@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    global X_train, X_test, y_train, y_test

    try:
        df = pd.read_csv(file.file)
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least two columns (features + target).")

        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return {"message": "Data uploaded and split successfully.", "rows": df.shape[0], "columns": df.shape[1]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ------------------- Data Preprocessing -------------------
class PreprocessingRequest(BaseModel):
    method: str

@app.post("/preprocess/")
async def preprocess_data(request: PreprocessingRequest):
    global X_train, X_test

    if X_train is None:
        raise HTTPException(status_code=400, detail="No dataset found. Upload data first.")

    try:
        if request.method == "StandardScaler":
            scaler = StandardScaler()
        elif request.method == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            raise HTTPException(status_code=400, detail="Invalid preprocessing method.")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train[:] = X_train_scaled
        X_test[:] = X_test_scaled

        return {"message": f"Data preprocessed using {request.method}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


# ------------------- Model Training -------------------
class TrainRequest(BaseModel):
    model_type: str

@app.post("/train-model/")
async def train_model(request: TrainRequest):
    global model, X_train, y_train

    if X_train is None:
        raise HTTPException(status_code=400, detail="No dataset found. Upload data first.")

    try:
        if request.model_type == "RandomForest":
            model = RandomForestClassifier()
        elif request.model_type == "SVM":
            model = SVC()
        elif request.model_type == "NeuralNetwork":
            model = MLPClassifier()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")

        model.fit(X_train, y_train)
        return {"message": f"{request.model_type} model trained successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


# ------------------- Model Evaluation -------------------
@app.get("/evaluate-model/")
async def evaluate_model():
    global model, X_test, y_test

    if model is None:
        raise HTTPException(status_code=400, detail="Model training not completed. Train a model first.")
    if X_test is None or y_test is None:
        raise HTTPException(status_code=400, detail="Dataset missing. Ensure data is uploaded and preprocessed before training.")

    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        return {"accuracy": acc * 100, "f1": f1}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ------------------- Save Pipeline -------------------
class PipelineSchema(BaseModel):
    name: str
    nodes: list
    edges: list

@app.post("/save-pipeline/")
async def save_pipeline(pipeline: PipelineSchema):
    db = SessionLocal()
    
    try:
        existing_pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline.name).first()
        if existing_pipeline:
            db.delete(existing_pipeline)
            db.commit()

        db_pipeline = Pipeline(
            name=pipeline.name,
            nodes=json.dumps(pipeline.nodes),
            edges=json.dumps(pipeline.edges),
        )
        db.add(db_pipeline)
        db.commit()

        return {"message": "Pipeline saved successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")


# ------------------- Load Pipeline -------------------
@app.get("/load-pipeline/{name}")
async def load_pipeline(name: str):
    db = SessionLocal()
    pipeline = db.query(Pipeline).filter(Pipeline.name == name).first()

    if pipeline:
        return {
            "name": pipeline.name,
            "nodes": json.loads(pipeline.nodes),
            "edges": json.loads(pipeline.edges),
        }
    
    raise HTTPException(status_code=404, detail="Pipeline not found")
