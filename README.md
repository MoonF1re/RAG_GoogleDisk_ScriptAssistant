# RAG GoogleDisk Script Assistant

Это веб-приложение, которое использует подход Retrieval-Augmented Generation (RAG) для создания чат-бота с использованием LangChain и FastAPI. Приложение также позволяет загружать документы для индексирования и семантического поиска, чтобы улучшить ответы ИИ.

## Функциональность

- **Чат с ИИ:**  
  Ведите диалог с помощником, который учитывает историю переписки и может использовать внешний контекст из документов.

- **Загрузка документов:**  
  Загружайте документы в форматах PDF, DOCX и HTML для индексирования и дальнейшего использования при генерации ответов.

- **Управление документами:**  
  Просматривайте список загруженных документов и удаляйте ненужные.

## Стек технологий

- **FastAPI:** Сервер для обработки API запросов.
- **LangChain:** Обработка цепочек (chains) для генерации ответов с учетом контекста и истории.
- **Chroma:** Векторное хранилище для семантического поиска по документам.
- **SQLite:** Локальная база данных для хранения истории чата и информации о документах.
- **Streamlit:** Веб-интерфейс для взаимодействия с пользователем.
- **Git:** Контроль версий.

## Структура проекта

- **api_utils.py:** Функции для отправки HTTP-запросов к серверу (чат, загрузка, получение и удаление документов).
- **chat_interface.py:** Реализация чат-интерфейса на Streamlit.
- **sidebar.py:** Боковая панель для выбора модели, загрузки и управления документами.
- **streamlit_app.py:** Главный файл для запуска приложения Streamlit.
- **db_utils.py:** Работа с базой данных (история чата и документы).
- **chroma_utils.py:** Индексация документов в векторное хранилище Chroma.
- **langchain_utils.py:** Создание цепочек обработки запросов с использованием LangChain и ChatOllama.

## Как запустить

1. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt

2. **Запустите сервер API:**

   ```bash
   cd api
   uvicorn app.main:app --reload

3. **Запустите приложение Streamlit:**

   ```bash
   cd app
   streamlit run streamlit_app.py

4. **Откройте веб-браузер и перейдите по адресу:**

   ```bash
   http://localhost:8501
