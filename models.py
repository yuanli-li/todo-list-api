from typing import Optional
from pydantic import BaseModel


class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    priority: int = 1
    due_date: Optional[str] = None  # "YYYY-MM-DD"


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    completed: bool | None = None
    priority: Optional[int] = None
    due_date: Optional[str] = None
