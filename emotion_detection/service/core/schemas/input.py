from pydantic import BaseModel


class OutputSchema(BaseModel):
    emotion: str
    probability: float