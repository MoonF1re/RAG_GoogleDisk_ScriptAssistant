from pydantic import BaseModel, Field
"""Библиотека для создания моделей данных. 
Модель данных - это способо описать какие данные должны присутствовать + функции для их проверки.  
Если данные не подходящего формата будет ошибка. 
"""
from enum import Enum #Библиотека для перечислений
from datetime import datetime
class ModelName(str, Enum): #Все модели которые может выбрать пользователь.
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    LLAMA3_2 = "llama3.2"
class QueryInput(BaseModel): #Данные которые пользователь обязан дать
    question: str
    session_id: str = Field(default=None) #не обязательный параметр. Пользователь его не передаёт, мы их генерируем в main.
    model: ModelName = Field(default=ModelName.LLAMA3_2)

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