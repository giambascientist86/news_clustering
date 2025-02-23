import streamlit as st

def show_topic(topic_id, words):
    st.subheader(f"Topic {topic_id}")
    st.write(", ".join(words))
