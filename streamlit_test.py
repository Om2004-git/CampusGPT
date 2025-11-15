import streamlit as st
import requests

st.set_page_config(page_title="CampusGPT", layout="wide")

st.title("🎓 CampusGPT – Your College Chat Assistant")
st.write("Ask questions and get answers directly from your embedded college documents.")

# ----------------------------
# Backend API URL
# ----------------------------
st.sidebar.header("⚙️ Backend Settings")
api_url = st.sidebar.text_input("FastAPI Backend URL", "http://127.0.0.1:8000")

if not api_url.startswith("http"):
    st.sidebar.error("API URL must start with http:// or https://")


# ----------------------------
# Ask Query Section
# ----------------------------
st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question:")

if st.button("🔍 Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer from RAG system..."):
            try:
                response = requests.post(
                    f"{api_url}/gpt/ask",
                    json={"query": query}      # IMPORTANT: JSON BODY
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display Answer
                    st.markdown("### 🧠 Answer")
                    st.write(result.get("answer", ""))

                    # Display Sources
                    sources = result.get("sources", [])
                    if sources:
                        st.markdown("### 📚 Sources Used")
                        for src in sources:
                            st.write(f"**File:** {src['filename']}")
                            st.write(f"**Snippet:** {src['snippet']}")
                            st.markdown("---")
                    else:
                        st.info("No sources found.")

                else:
                    st.error(f"Backend Error: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error("Failed to connect to backend.")
                st.write(str(e))
