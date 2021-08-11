from pydantic import BaseModel
from typing import List

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

class Crime_Data(BaseModel):
    year: str
    month: str
    lat: str
    log: str
    crimeType: str
    reported: str

class Csv_Data(BaseModel):
    data: List[List[str]]

