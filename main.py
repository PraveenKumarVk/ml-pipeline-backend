from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import json

app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./pipelines.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# Pipeline Model
class Pipeline(Base):
    __tablename__ = "pipelines"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    nodes = Column(Text)
    edges = Column(Text)

Base.metadata.create_all(bind=engine)

# Schema for storing pipeline
class PipelineSchema(BaseModel):
    name: str
    nodes: list
    edges: list

@app.post("/save-pipeline/")
async def save_pipeline(pipeline: PipelineSchema):
    db = SessionLocal()
    db_pipeline = Pipeline(
        name=pipeline.name,
        nodes=json.dumps(pipeline.nodes),
        edges=json.dumps(pipeline.edges),
    )
    db.add(db_pipeline)
    db.commit()
    return {"message": "Pipeline saved successfully"}

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
    return {"message": "Pipeline not found"}
