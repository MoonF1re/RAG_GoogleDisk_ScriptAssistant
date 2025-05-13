from api_utils import get_api_response
import streamlit as st

def display_chat_interface():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            try:
                response = get_api_response(
                    prompt,
                    st.session_state.session_id,
                    st.session_state.model
                )
                
                if response and isinstance(response, dict):
                    answer = response.get('answer', 'No answer provided')
                    st.session_state.session_id = response.get('session_id', st.session_state.session_id)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        with st.expander("Details"):
                            st.subheader("Generated Answer")
                            st.code(answer)
                            st.subheader("Model Used")
                            st.code(response.get('model', 'Unknown'))
                            st.subheader("Session ID")
                            st.code(response.get('session_id', 'None'))
                            st.subheader("Context Used")
                            st.code(response.get('context', 'No context'))
                else:
                    st.error("Invalid response format from API")
                    
            except Exception as e:
                st.error(f"Failed to get response: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I couldn't process your request."
                })