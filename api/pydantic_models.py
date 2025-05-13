from pydantic import BaseModel, Field
"""Библиотека для создания моделей данных. 
Модель данных - это способо описать какие данные должны присутствовать + функции для их проверки.  
Если данные не подходящего формата будет ошибка. 
"""
from enum import Enum #Библиотека для перечислений
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    qwen3 = "qwen3:1.7b"
    deepseek = "deepseek-r1:1.5b"
    gemma3 = "gemma3:1b"

class LLMSettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 2000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = "You are a helpful AI script assistant..."

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.gemma3)
    temperature: float = 0.7
    max_tokens: int = 2000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = "You are a helpful AI script assistant..."

# ... остальные классы ...

class QueryResponse(BaseModel): #Данные которые обязаны вернуть пользователю
    answer: str
    session_id: str
    model: ModelName
    context: str


class DocumentInfo(BaseModel):  #Данные которые нужны чтобы вывести файл
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel): #Данные которые нужны для удаления файла
    file_id: int