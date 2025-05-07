import streamlit as st
from api_utils import upload_document, list_documents, delete_document
import requests

# Изменение: Добавлен импорт requests для отправки запросов к Google Drive API
def display_sidebar():
    # Выпадющий список с моделями
    model_options = ["llama3.2", "gpt-4o", "gpt-4o-mini"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Загрузка локальных документов
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(
                        f"File '{uploaded_file.name}' uploaded successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents()

    # Изменение: Добавлен интерфейс для загрузки папок с Google Drive
    st.sidebar.header("Upload Google Drive Folder")
    gdrive_folder_url = st.sidebar.text_input("Google Drive Folder URL", placeholder="https://drive.google.com/drive/folders/...")
    if st.sidebar.button("Upload from Google Drive"):
        if not gdrive_folder_url:
            st.sidebar.error("Please enter a Google Drive Folder URL.")
        else:
            with st.spinner("Uploading from Google Drive..."):
                try:
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(
                        "http://localhost:8000/upload-gdrive-doc",
                        json={"folder_url": gdrive_folder_url},
                        headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.sidebar.success(f"{data['message']}: {data['file_ids']}")
                    st.session_state.documents = list_documents()
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Failed to upload folder: {str(e)}")

    # Просмотр списка документов
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    # Инициализируем список документов, если он отсутствует
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")

        # Удаление документа
        selected_file_id = st.sidebar.selectbox("Select a document to delete", options=[doc['id'] for doc in documents],
                                               format_func=lambda x: next(
                                                   doc['filename'] for doc in documents if doc['id'] == x))
        if st.sidebar.button("Delete Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                    st.session_state.documents = list_documents()
                else:
                    st.sidebar.error(f"Failed to delete document with ID {selected_file_id}.")