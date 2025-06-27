import streamlit as st
from rag_bot import ask_question

st.set_page_config(page_title="Shopping Assistant Bot", page_icon="🛍️")
st.title("🛍️ Smart Shopping Assistant")
st.markdown("Ask me anything about available products!")

# Input from user
user_input = st.text_input("🧾 Type your query here:")

if st.button("Ask"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching products and generating answer..."):
            response = ask_question(user_input)
        st.success("Here's what I found:")
        st.write(response)

# Footer
st.markdown("---")
st.caption("Powered by RAG & OpenAI | Built by Areeba 💡")
