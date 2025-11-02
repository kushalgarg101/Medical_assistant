import streamlit as st
import requests

# ---------------- Config ----------------
BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Patient Assistant", page_icon="🩺", layout="centered")
st.title("🩺 Medical Assistant")

# ---------------- Session Init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}

# ---------------- Chat Display ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat Input ----------------
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend /chat
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            params={"question": user_input},
            timeout=300
        )
    except requests.RequestException as e:
        st.error(f"⚠️ Could not reach server: {e}")
    else:
        if response.status_code == 200:
            try:
                data = response.json()
                ai_messages = data.get("responses", [])
            except Exception:
                ai_messages = [response.text]

            # Display
            for ai_msg in ai_messages:
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                with st.chat_message("assistant"):
                    st.markdown(ai_msg)
        else:
            st.error(f"❌ Server returned {response.status_code}: {response.text}")

# Footer
st.write("---")
st.caption("💬 Ask health-related medical questions. Responses are for informational purposes only.")
