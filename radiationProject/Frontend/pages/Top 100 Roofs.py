import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heatmap",
    page_icon="👋",
    layout="wide"
)

def load_css(name):
    file_path = os.path.join(os.path.dirname(__file__), name)
    with open(file_path, "r") as f:
        css = f.read()
    return css

css_file = "styles.css"
css_content = load_css("../styles.css")
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

with st.container():
    st.title("Top 100 Roofs")

    file_path = os.path.join(os.path.dirname(__file__), "../citiesRadiation_NoDuplicates_Merged.csv")
    df = pd.read_csv(file_path)

    table_width = 1000  # Width in pixels
    table_height = 600  # Height in pixels

    # Display the DataFrame with the specified width and height
    st.dataframe(df, width=table_width, height=table_height)