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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

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
    st.markdown("""---""")
    st.header('May the force be with you')
    st.text('A Clustering Project it is.')
    st.text(' ')
    st.text(' ')
    image = Image.open('data/baby-yoda.jpg')
    st.image(image, caption="This is the way")
    st.text(' ')
    with team:
        # meet the team button
        st.header('Team Yoda')
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
                '[Hedaya Ali](https://github.com/HedayaAli)')
        with col3:
            # image = Image.open('imgs/thomas.jpg')
            # st.image(image, caption="")
            st.markdown(
                '[Thomas Johnson-Ellis](https://github.com/Tomjohnsonellis)')

        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.markdown("""---""")
        st.text(' ')
        st.text(' ')
        image = Image.open('data/long_time_ago.jpg')
        st.image(image, caption="")

        # Add audio
        audio_file = open('data/star_wars_theme_song.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')

with dataset:
    st.header("")
    # st.subheader("Galaxies")
    st.markdown("")
    st.markdown("")

########## k-means ###########
    st.markdown("")
    st.markdown("")
    st.subheader("K-means")
    st.text("Compare the clusters and keep track of which was highest, that's the galaxy we care about")
    st.markdown("")

    fig = plt.figure(figsize=(15, 10))

    df = pd.read_csv("data/galaxies.csv")
    plt.scatter(df.X, df.Y)

    coords = df.values
    model = KMeans(n_clusters=3)
    model.fit(coords)
    labels = model.predict(coords)

    def separate_labels(labels, points):
        data_0 = []
        data_1 = []
        data_2 = []
        for i in range(len(labels)):
            if labels[i] == 0:
                data_0.append(points[i])
            if labels[i] == 1:
                data_1.append(points[i])
            if labels[i] == 2:
                data_2.append(points[i])

        data_0 = np.array(data_0)
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)

        return data_0, data_1, data_2

    # print(points)

    data_0, data_1, data_2 = separate_labels(labels, coords)
    plt.scatter(data_0[:, 0], data_0[:, 1],
                color="rebeccapurple", label="Galaxy Cluster 0")
    plt.scatter(data_1[:, 0], data_1[:, 1],
                color="darkcyan", label="Galaxy Cluster 1")
    plt.scatter(data_2[:, 0], data_2[:, 1],
                color="steelblue", label="Galaxy Cluster 2")

    centroids = model.cluster_centers_
    # centroids
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]
    plt.scatter(centroids_x, centroids_y, marker="x", s=200,
                color="black", linewidths=5, label="Centre of each Galaxy")
    glx = max(centroids_y)

    # Compare them and keep track of which was highest, that's the galaxy we care about
    def find_uppermost_galaxy(data_0, data_1, data_2):
        name = ""
        if (data_1[:, 1].mean()) > (data_0[:, 1].mean()):
            uppermost_galaxy = data_1
            name = "Galaxy 1"
        else:
            uppermost_galaxy = data_0
            name = "Galaxy 0"

        if (data_2[:, 1].mean()) > (uppermost_galaxy[:, 1].mean()):
            uppermost_galaxy = data_2
            name = "Galaxy 2"

        # print("Highest Galaxy: {}".format(name))
        return uppermost_galaxy

    # Find the values of the planet with highest X value of that galaxy

    def find_rightmost_coords(galaxy):
        rightmost_point = max(galaxy[:, 0])
        coords = []
        for planet in galaxy:
            if planet[0] == rightmost_point:
                print("Coords: ({},{})".format(planet[0], planet[1]))
                coords = [planet[0], planet[1]]

        return coords

    baby_planet = find_rightmost_coords(
        find_uppermost_galaxy(data_0, data_1, data_2))

    # agree1 = st.checkbox("Highest Galaxy:")
    # if agree1:
    #     st.write("Highest Galaxy: {}".format(name))
    #     st.write(("Coords: ({},{})".format(planet[0], planet[1])))

    # plt.scatter(baby_planet[0], baby_planet[1], marker="*", s=200)

    plt.scatter(baby_planet[0], baby_planet[1], marker="*", s=250,
                alpha=0.75, label="Baby Yoda's Planet", color="red")
    plt.title("Map of Galaxies", fontsize=20)
    plt.legend()
    st.write(fig)

#################################
    st.markdown("")
    st.markdown("")
    # image = Image.open('data/k-means.png')
    # st.image(image, caption="")

########## 3D plot ###########
    st.markdown("")
    st.markdown("")
    st.subheader("Map of Baby Yoda's planet")
    st.markdown("")
    st.markdown("")

    planet_df = pd.read_csv("data/planet.csv")

    agree = st.checkbox("Show dataframe:")
    if agree:
        planet_df
        st.write("Planet.csv columns: {}".format(planet_df.columns))

        planet_points = planet_df[["X", "Y", "Z"]]
        planet_points

    fig = plt.figure(figsize=(15, 10))
    # syntax for 3-D projection
    #ax = plt.axes(projection ='3d')

    # defining all 3 axes
    z = planet_df["Z"]
    x = planet_df["X"]
    y = planet_df["Y"]

    # plotting
    #fig, axs = plt.subplots(2)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x, y, z, c=planet_df["Temp"])
    plt.title("Temperature", fontsize=14)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x, y, z, c=planet_df["climate"])
    plt.title("Climate", fontsize=14)
    #ax.set_title('3D line plot')
    plt.show()
    st.write(fig)

#################################
    st.markdown("")
    st.markdown("")
    # image = Image.open('data/3d_plot.png')
    # st.image(image, caption="")

########## PCA ###########
    st.markdown("")
    st.markdown("")
    st.subheader("Baby Yoda's Location")
    st.markdown("")
    st.markdown("")

    fig = plt.figure(figsize=(15, 10))
    scaler = StandardScaler()
    pca = PCA(n_components=5)

    # planet_df = pd.read_csv("data/planet.csv")
    # planet_df.columns
    points = planet_df.values
    scaled = scaler.fit_transform(points)
    pca.fit_transform(points)
    new_data = pca.fit_transform(points)

    # print("Principal Components:")
    # print(pca.components_)
    # How much info are in our PCs
    # print(np.cumsum(pca.explained_variance_ratio_))

    # print("We only need the first 2")
    # I think these are the "Force concentrations?"
    #plt.scatter(pca.components_[0],pca.components_[1], s=250)
    xs = new_data[:, 0]
    ys = new_data[:, 1]
    locations = np.array([xs, ys])
    plt.scatter(xs, ys, alpha=0.75, color="plum")

    force_spots = new_data
    # print(force_spots)
    force_x_centre = force_spots[:, 0].mean()
    force_y_centre = force_spots[:, 1].mean()
    yoda_core = [force_x_centre, force_y_centre]
    plt.scatter(force_x_centre, force_y_centre, marker="*",
                color="green", s=100, label="Centre")

    st.write("Centre of Force Gravity: {}".format(yoda_core))

    locations = locations.transpose()
    guess = locations[0, 0]
    best_guess = guess
    smallest_distance = np.linalg.norm(yoda_core - guess)
    # print(smallest_distance)
    for place in locations:
        guess = place
        guess_distance = np.linalg.norm(yoda_core - guess)

        if guess_distance < smallest_distance:
            smallest_distance = guess_distance
            best_guess = guess
            #print("That's a better guess")
        else:
            pass

    # print("Finally:")
    # print(smallest_distance)
    # print(best_guess)

    agree3 = st.checkbox("We believe Baby Yoda to be at:")
    if agree3:
        st.write("We believe Baby Yoda to be at: {}".format(best_guess))

    plt.scatter(best_guess[0], best_guess[1], marker="x",
                s=100, color="red", label="Baby Yoda"),
    plt.title("Baby Yoda's Location", fontsize=20)
    plt.legend()
    st.write(fig)

#################################
    st.markdown("")
    st.markdown("")
    # image = Image.open('data/planets.png')
    # st.image(image, caption="")


##################################################################################

with footer:
    st.markdown("""---""")
    st.subheader("The alliance manage to find Grogu and the galaxy is saved")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    image = Image.open('data/ship.jpg')
    st.image(image, caption="")
    st.text(' ')

    # Footer
    st.markdown("")
    st.markdown("")
    st.markdown(
        "If you have any questions, checkout our [documentation](https://github.com/fistadev/starwars_data_project) ")
    st.text(' ')
