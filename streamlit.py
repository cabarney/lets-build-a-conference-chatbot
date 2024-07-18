from chatbot import ChatBot
import streamlit as st

bot = ChatBot("huggingface")
    
st.set_page_config(page_title="Nebraska.Code() Assistant")
with st.sidebar:
    st.title('Nebraska.Code() Assistant')

def generate_response(input):
    result = bot.chain.invoke(input)
    return result

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Nebraska.Code(). Can I answer any questions about the schedule?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("..."):
            response = generate_response(input)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)