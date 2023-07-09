from pydantic import BaseModel
from typing import List


class Trainer(BaseModel):
    process_training: bool


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class ChildQuickTest(BaseModel):
    text: str
    bounding_boxes: BoundingBox
    confidence: float
    class_name: str


class QuickTest(BaseModel):
    quick_test: List[ChildQuickTest]


class Transform(BaseModel):
    transform_list: str


class Data(BaseModel):
    images: str
    labels: str


class DocumentTemplateTrainingRequestModel(BaseModel):
    epochs: int
    modelType: str
    data: List[Data]
    listLabel: list

