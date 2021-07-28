from pydantic import BaseModel


class Crime(BaseModel):
    year: int
    month: int
    area1: float
    area2: float
    crimeType: int


class Crime_Wo_Districts(BaseModel):
    year: int
    month: int
    crimeType: int
