import sqlite3
from datetime import datetime

"""Тут мы сохраняем нашу историю чата. Чтобы позже её анализировать и использовать для контекста.
   Так же тут мы сохраняем таблицу со всеми документами."""

DB_NAME = "rag_app.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME) #Подключаемся к файлу с БД.
    conn.row_factory = sqlite3.Row #Настройка, которая позволяет обращаться к столбцам по именам, а не по индексам.
    return conn


def create_application_logs():
    #Тут мы создаём таблицу где будет хранится история чата.
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT,
                      user_query TEXT,
                      gpt_response TEXT,
                      model TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit() #Сохраняем изменения в БД.
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at',
                   (session_id,)) # Выбираем записи где session_id равен нашему session_id.
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ]) #Создаём два словаря и записываем их в список.
    conn.close()
    return messages

def create_document_store():
    # Тут мы создаём таблицу где будут хранится информация про файлы.
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT,
                      upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id #Возращаем айди что позже использовать его для связи с индексированными частями


def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True


def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


create_application_logs() #Автоматически создаём таблицу для истории чата
create_document_store() # Автоматически создаём таблицу для документов