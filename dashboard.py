import dashboard as st

st.title("My Streamlit App")
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")