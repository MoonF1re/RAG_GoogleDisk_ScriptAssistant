import streamlit as st #Через него мы создаём веб приложение.
from sidebar import display_sidebar #Функция для отображения боковой панели
from chat_interface import display_chat_interface

st.title("Сценарный Ассистент") #Навзвание приложения

#Если это новая сессия создаём списки для сообщений и айди
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

display_sidebar()#Включаем боковую панель

display_chat_interface()#Включаем чат