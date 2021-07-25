from pydantic import BaseModel


class Crime(BaseModel):
    year: int
    month: int
    area1: float
    area2: float
    crimeType: int
