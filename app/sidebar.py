import streamlit as st
from api_utils import upload_document, list_documents, delete_document


def display_sidebar():
    # Инициализация состояния для LLM настроек
    if 'llm_settings' not in st.session_state:
        st.session_state.llm_settings = {
            'temperature': 0.7,
            'max_tokens': 2000,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'system_prompt': "Ты нейросетевой помощник по написанию сценариев."
        }

    # Выпадющий список с моделями
    model_options = ["gemma3:1b", "qwen3:1.7b", "deepseek-r1:1.5b"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Кнопка для загрузки документов
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
        # Список загруженных документов
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")

        # Удаление документа
        selected_file_id = st.sidebar.selectbox(
            "Select a document to delete",
            options=[doc['id'] for doc in documents],
            format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x)
        )
        if st.sidebar.button("Delete Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                    st.session_state.documents = list_documents()
                else:
                    st.sidebar.error(f"Failed to delete document with ID {selected_file_id}.")

    # Секция настроек LLM
    st.sidebar.header("LLM Settings")
    
    # Ползунок температуры
    st.session_state.llm_settings['temperature'] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.llm_settings['temperature'],
        step=0.1,
        key="temperature_slider",
        help="Контролирует случайность ответов. 0 = детерминировано, 1 = креативно"
    )
    
    # Максимальное количество токенов
    st.session_state.llm_settings['max_tokens'] = st.sidebar.number_input(
        "Max Tokens",
        min_value=100,
        max_value=8000,
        value=st.session_state.llm_settings['max_tokens'],
        step=100,
        key="max_tokens_input",
        help="Максимальная длина ответа в токенах"
    )
    
    # Штрафы
    st.session_state.llm_settings['frequency_penalty'] = st.sidebar.slider(
        "Frequency Penalty",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.llm_settings['frequency_penalty'],
        step=0.1,
        key="frequency_penalty_slider",
        help="Штраф за повторяющиеся фразы"
    )
    
    st.session_state.llm_settings['presence_penalty'] = st.sidebar.slider(
        "Presence Penalty",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.llm_settings['presence_penalty'],
        step=0.1,
        key="presence_penalty_slider",
        help="Штраф за повторение тем"
    )
    
    # Системный промт
    st.session_state.llm_settings['system_prompt'] = st.sidebar.text_area(
        "System Prompt",
        value=st.session_state.llm_settings['system_prompt'],
        key="system_prompt_area",
        help="Инструкции для модели, определяющие её поведение"
    )

    # Подсказки
    with st.sidebar.expander("❓ Помощь по настройкам"):
        st.markdown("""
        - **Temperature**: Чем выше, тем креативнее ответы
        - **Max Tokens**: Ограничивает длину ответа
        - **Frequency Penalty**: Снижает повторения
        - **Presence Penalty**: Поощряет новые темы
        - **System Prompt**: Определяет роль ИИ
        """)

    # Кнопка сброса с callback
    def reset_llm_settings():
        st.session_state.llm_settings = {
            'temperature': 0.7,
            'max_tokens': 2000,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'system_prompt': "You are a helpful AI script assistant. Use the following context to answer the user's question."
        }

    st.sidebar.button(
        "Сбросить настройки",
        key="reset_llm_settings",
        on_click=reset_llm_settings
    )