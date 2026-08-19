import streamlit as st
from chatbot import ask_bot

st.set_page_config(
    page_title="College AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College AI Chatbot")
st.write("Ask any question about the college.")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What courses are offered?"
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = ask_bot(question)

        st.markdown("### Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")