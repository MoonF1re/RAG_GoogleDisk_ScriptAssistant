import requests #Библиотека для отправки HTTP-запросов. Через него мы связываемся с нашим api
import streamlit as st #Через него мы создаём веб приложение. Конкретно тут для отображения ошибок

"""Тут описаны как мы отправляем запросы на наш Api и как мы получаем ответы"""
import requests
import streamlit as st

def get_api_response(question, session_id, model):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    data = {
        "question": question,
        "model": model,
        "temperature": st.session_state.llm_settings['temperature'],
        "max_tokens": st.session_state.llm_settings['max_tokens'],
        "frequency_penalty": st.session_state.llm_settings['frequency_penalty'],
        "presence_penalty": st.session_state.llm_settings['presence_penalty'],
        "system_prompt": st.session_state.llm_settings['system_prompt']
    }
    
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(
            "http://localhost:8000/chat",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def upload_document(file):
    print("Uploading file...")
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents():
    try:
        response = requests.get("http://localhost:8000/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"file_id": file_id}

    try:
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None