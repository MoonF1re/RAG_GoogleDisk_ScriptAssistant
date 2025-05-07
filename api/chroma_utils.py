import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Изменение: Удалены импорты устаревших модулей (langchain_text_splitters, langchain_huggingface, SentenceTransformerEmbeddings)
# Изменение: Используется langchain.text_splitter вместо langchain_text_splitters
# Изменение: Используется langchain_community.embeddings вместо langchain_huggingface

# Инициализация векторного хранилища
# Изменение: Переименовано embedding_function в embedding_model для ясности
# Изменение: Добавлено collection_name="rag_collection" для явного указания имени коллекции
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_model, persist_directory="./chroma_db")

# Изменение: Функция load_and_split_document удалена, её логика перенесена в index_document_to_chroma
# Изменение: index_document_to_chroma теперь принимает input_data (путь или список документов) вместо file_path
def index_document_to_chroma(input_data, file_id):
    try:
        # Проверяем, является ли input_data строкой (путь к файлу) или списком документов
        if isinstance(input_data, str):
            file_path = input_data
            file_extension = os.path.splitext(file_path)[1].lower()

            # Загрузка документа в зависимости от типа файла
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.html':
                loader = UnstructuredHTMLLoader(file_path)
            else:
                return False  # Изменение: Упрощена обработка ошибок (без ValueError)

            documents = loader.load()
        elif isinstance(input_data, list):
            # Предполагаем, что input_data — это список объектов Document
            documents = input_data
        else:
            raise ValueError("Input must be a file path (str) or a list of Documents")

        # Разбиение документов на чанки
        # Изменение: chunk_size уменьшен с 1500 до 1000 для более мелких чанков
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Добавление метаданных с file_id
        for chunk in chunks:
            chunk.metadata["file_id"] = file_id

        # Индексация в Chroma
        vectorstore.add_documents(chunks)
        return True
    except Exception as e:
        print(f"Error indexing document: {str(e)}")
        return False

# Изменение: Упрощена функция delete_doc_from_chroma, удалён промежуточный get и логи
def delete_doc_from_chroma(file_id):
    try:
        # Удаление документов с указанным file_id
        vectorstore.delete(where={"file_id": file_id})
        return True
    except Exception as e:
        print(f"Error deleting document from Chroma: {str(e)}")
        return False