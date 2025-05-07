import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, UploadGDriveRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from gdrive_extractor import GoogleDriveExtractor
import shutil
import tempfile

logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()

# Изменение: Добавлена поддержка CORS для кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Изменение: Эндпоинт /chat переименован в /query и стал асинхронным
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    try:
        chat_history = get_chat_history(session_id)
        rag_chain, hybrid_retriever_obj = get_rag_chain(query_input.model.value)
        answer = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })['answer']

        context_used = hybrid_retriever_obj.last_context

        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
        logging.info(f"Session ID: {session_id}, AI Response: {answer}")
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model, context=context_used)
    except Exception as e:
        logging.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Изменение: Эндпоинт /upload-doc стал асинхронным
@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Изменение: Добавлен новый эндпоинт для загрузки документов с Google Drive
@app.post("/upload-gdrive-doc")
async def upload_gdrive_document(request: UploadGDriveRequest = Body(...)):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        credentials_path = os.path.join(base_dir, "credentials.json")
        token_path = os.path.join(base_dir, "token.json")
        logging.info(f"Looking for credentials at: {credentials_path}")
        logging.info(f"Looking for token at: {token_path}")

        extractor = GoogleDriveExtractor(credentials_path=credentials_path, token_path=token_path)
        documents = extractor.process_folder(request.folder_url)

        if not documents:
            raise ValueError("No documents extracted from folder")

        files_documents = {}
        for doc in documents:
            file_name = doc.metadata.get('source', 'unknown')
            if file_name not in files_documents:
                files_documents[file_name] = []
            files_documents[file_name].append(doc)

        file_ids = []
        for file_name, docs in files_documents.items():
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension not in ['.pdf', '.docx', '.html']:
                logging.info(f"Skipping unsupported file: {file_name}")
                continue
            file_id_db = insert_document_record(file_name)
            success = index_document_to_chroma(docs, file_id_db)
            if success:
                logging.info(f"File {file_name} has been successfully uploaded and indexed.")
                file_ids.append(file_id_db)
            else:
                delete_document_record(file_id_db)
                logging.error(f"Failed to index {file_name}")

        if not file_ids:
            raise HTTPException(status_code=500, detail="No files were successfully indexed")

        return {"message": f"Successfully indexed {len(file_ids)} files from folder", "file_ids": file_ids}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Изменение: Эндпоинт /list-docs стал асинхронным
@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    try:
        return get_all_documents()
    except Exception as e:
        logging.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Изменение: Эндпоинт /delete-doc стал асинхронным
@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    try:
        chroma_delete_success = delete_doc_from_chroma(request.file_id)

        if chroma_delete_success:
            db_delete_success = delete_document_record(request.file_id)
            if db_delete_success:
                logging.info(f"Successfully deleted document with file_id {request.file_id} from the system.")
                return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
            else:
                logging.error(
                    f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database.")
                return {
                    "error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
        else:
            logging.error(f"Failed to delete document with file_id {request.file_id} from Chroma.")
            return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
    except Exception as e:
        logging.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))