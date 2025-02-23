from fastapi import APIRouter
from services import TopicModelingService
from models import TopicModel
from typing import List

router = APIRouter()
service = TopicModelingService()

@router.post("/train/")
def train_model():
    topics = service.train_model()
    return {"message": "Model trained successfully", "topics": topics}

@router.get("/topics/", response_model=list[TopicModel])
def get_topics():
    return service.get_topics()
