from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid #отвечает за генерацию уникального айди сессии
import logging
#==Импорт из других модулей системы==
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest #Импорт моделей данных для защиты от дурака.
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
#==Библиотеки для работы с файлами===
import os
import shutil

logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({ #Отправляем словарь в нашу цепочку, на выходе получаем ответ.
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)): #Загрузка файла через HTTP
    allowed_extensions = ['.pdf', '.docx', '.html'] #Доступные форматы файлов
    file_extension = os.path.splitext(file.filename)[1].lower() #Выбираем формат файла

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"  #Временно сохраняем файл на Диск

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename) #Сохраняем файл в БД и возращаем файл айди
        success = index_document_to_chroma(temp_file_path, file_id) #Разбиваем файл на части и сохраняем их в векторное хранилище.

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally: #Всегда удаляем временный файл
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id) #Удаляем все части файла с этим айди

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id) #Удаляем запись из БД (таблицы)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {
                "error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}