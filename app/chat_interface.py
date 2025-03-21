import streamlit as st #Через него мы создаём веб приложение. Конкретно тут чат
from api_utils import get_api_response


def display_chat_interface():
    # Выводим всю историю чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
#Эта структура^^^ работает так: Создать чат-сообщение для указанной роли и вывести содержимое сообщения в виде отформатированного текста.

    #Пользователь вводит запрос и мы добавляем его в историю чата
    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."): #Временное сообщение пока ии думает
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                with st.chat_message("assistant"): #Добавляем ответ ИИ
                    st.markdown(response['answer'])

                    with st.expander("Details"): #Доп. Информация для откладки
                        st.subheader("Generated Answer")
                        st.code(response['answer'])
                        st.subheader("Model Used")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])
            else:
                st.error("Failed to get a response from the API. Please try again.")