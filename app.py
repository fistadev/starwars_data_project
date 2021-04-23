import streamlit as st
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
# import plotly.express as px
# import plotly.figure_factory as ff


header = st.beta_container()
team = st.beta_container()
dataset = st.beta_container()
footer = st.beta_container()


@st.cache(persist=True)
def load_data(data):
    data = pd.read_csv('./data/galaxies.csv')
    return data


with header:
    st.title('Searching for Baby Yoda')  # site title h1
    st.text(' ')
    st.header('May the force be with you')
    st.text('Clustering Project')
    st.text(' ')
    st.text(' ')
    image = Image.open('data/baby-yoda.jpg')
    st.image(image, caption="")
    st.text(' ')
    with team:
        # meet the team button
        st.header('Team')
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            # image = Image.open('imgs/fabio.jpeg')
            # st.image(image, caption="")
            st.markdown(
                '[Fabio Fistarol](https://github.com/fistadev)')
        with col2:
            # image = Image.open('imgs/hedaya.jpeg')
            # st.image(image, caption="")
            st.markdown(
                '[Hedaya Ali](https://github.com/)')
        with col3:
            # image = Image.open('imgs/thomas.jpg')
            # st.image(image, caption="")
            st.markdown(
                '[Thomas Johnson-Ellis](https://github.com/Tomjohnsonellis)')

        st.text(' ')
        st.text(' ')


with footer:
    st.header("")

    # Footer
    st.markdown(
        "Thanks!")
    st.text(' ')
