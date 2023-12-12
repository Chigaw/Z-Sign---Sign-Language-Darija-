import streamlit as st

# Set the background color using HTML and CSS
background_color = """
    <style>
        body {
            background-color: #730068; /* Replace with your desired background color code */
        }
    </style>
"""

# Display the background color
st.markdown(background_color, unsafe_allow_html=True)

# Your Streamlit app content goes here
st.title("My Streamlit App")
st.write("This is an example Streamlit app with a custom background color.")
