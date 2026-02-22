"""Pydantic schemas for API request/response contracts."""

from pydantic import BaseModel


class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class DetectionResponse(BaseModel):
    count: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str = "ok"
