import streamlit as st
from api_client import get_topics, train_model

st.title("🔍 Topic Modeling with BERTopic")

# Stato di sessione per evitare chiamate premature alla GET
if "trained" not in st.session_state:
    st.session_state["trained"] = False

if st.button("📈 Train Model"):
    with st.spinner("⏳ TRAINING..."):
        response = train_model()
        if "message" in response:
            st.success(response["message"])
            st.session_state["trained"] = True  # Segna il modello come addestrato

st.header("📌 Topics Extracted:")

if st.session_state["trained"]:  # Solo se il modello è addestrato
    topics = get_topics()
    if topics:
        for topic in topics:
            # Estrai e visualizza i dettagli del topic
            topic_id = topic.get('Topic', 'N/A')
            topic_name = topic.get('Name', 'N/A')
            representation = topic.get('Representation', [])
            representative_docs = topic.get('Representative_Docs', [])

            # Visualizza le informazioni del topic
            st.subheader(f"**Topic {topic_id}: {topic_name}**")
            st.write(f"**Rappresentazione:** {', '.join(representation)}")
            st.write(f"**Documenti Rappresentativi:**")
            for doc in representative_docs:
                st.write(f"- {doc}")

    else:
        st.warning("⚠️ Nessun topic trovato. Forse il training non è ancora completo?")
else:
    st.info("🔵 Addestra prima il modello per vedere i topics.")
