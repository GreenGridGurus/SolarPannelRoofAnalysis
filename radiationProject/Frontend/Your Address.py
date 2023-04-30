import googlemaps
import streamlit as st
import os
import gmaps
import config
import numpy as np
import time


st.set_page_config(
    page_title="Heatmap",
    page_icon="👋",
    layout="wide"
)

def load_assets():
    # Load CSS
    css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_file_path, "r") as f:
        css = f.read()

    # Load JavaScript
    js_file_path = os.path.join(os.path.dirname(__file__), "fixes.js")
    with open(js_file_path, "r") as f:
        js = f.read()
    return css, js


def getImagesBackend(latitude, longitude):
    label_classes_super = ['pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow',
                           'tree', 'unknown', "nothing"]  #
    n_classes_superstructures = len(label_classes_super)
    label_classes_segment = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat', "background"]
    n_classes_segment = len(label_classes_segment)
    activation = 'softmax'
    BACKBONE = "resnet34"
    segment_model = sm.Unet(BACKBONE, classes=n_classes_segment, activation=activation)
    superstructures_model = sm.Unet(BACKBONE, classes=n_classes_superstructures, activation=activation)
    print('model_type not defined, choose between UNet, FPT or PSPNet')

    preprocess_input = sm.get_preprocessing(BACKBONE)

    weights_segment = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-instance-go/code/Users/ruslan.mammadov/RID/results/UNet_2_initial_segments.h5"
    segment_model.load_weights(weights_segment)

    weights_super = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-instance-go/code/Users/ruslan.mammadov/RID/results/UNet_2_initial.h5"
    superstructures_model.load_weights(weights_super)

    image, pixel_size = get_aerial_image_from_lat_lon_as_numpy(latitude, longitude)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    # gt = cv2.imread(gt_path, 0)

    segment_mask = segment_model.predict(image)
    segment_vector = np.argmax(segment_mask.squeeze(), axis=2)

    super_mask = superstructures_model.predict(image)
    super_vector = np.argmax(super_mask.squeeze(), axis=2)

    super_vector_updated = super_vector + n_classes_segment
    background_class = n_classes_segment + n_classes_superstructures - 1
    merged_vector = super_vector_updated

    no_superstructures = super_vector_updated == background_class
    merged_vector[no_superstructures] = segment_vector[no_superstructures]
    legend_labels = label_classes_segment + label_classes_super[:-1]

    from matplotlib import pyplot as plt
    plt.imshow(image[0])
    plt.show()

    # Create a colormap
    cmap = plt.cm.get_cmap('tab20', len(legend_labels))

    # Plot the segmentation image with a colorbar
    plt.imshow(merged_vector, cmap=cmap, vmin=0, vmax=len(legend_labels) - 1)
    plt.colorbar()

    # Create a legend
    # legend_patches = [plt.Rectangle((1, 1), 40, 11, color=cmap(i)) for i in range(len(legend_labels))]
    plt.legend(legend_labels, loc='lower right')
    # plt.legend({i: legend_labels[i] for i in range(len(legend_labels))})
    # Show the plot
    plt.show()

    {i: legend_labels[i] for i in range(len(legend_labels))}

    cmap = plt.cm.get_cmap('tab20', len(label_classes_segment))
    plt.imshow(segment_vector, cmap=cmap, vmin=0, vmax=len(label_classes_segment) - 1)
    plt.colorbar()
    plt.legend(label_classes_segment, loc='lower right')

    plt.show()

    {i: label_classes_segment[i] for i in range(len(label_classes_segment))}

    cmap = plt.cm.get_cmap('tab20', len(label_classes_super))
    plt.imshow(super_vector, cmap=cmap, vmin=0, vmax=len(label_classes_super) - 1)
    plt.colorbar()
    plt.legend(label_classes_super, loc='lower right')

    plt.show()

    {i: label_classes_super[i] for i in range(len(label_classes_super))}

    number_not_flat = (merged_vector < 16).sum(axis=None)
    number_flat = (merged_vector == 16).sum(axis=None)
    pv_modules = (merged_vector == 18).sum(axis=None)
    number_not_flat * pixel_size, number_flat * pixel_size, pv_modules * pixel_size

    # print(f"Flat surface:\t\t{number_flat * pixel_size} m^2")
    # print(f"Not-flat surface:\t{number_not_flat * pixel_size} m^2")
    # print(f"PV modules:\t\t{pv_modules * pixel_size} m^2")

    return number_flat*pixel_size, number_not_flat*pixel_size, pv_modules*pixel_size


css_content, js_content = load_assets()
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)

with st.container():
    st.title("Your data")
    # Create a 2-column layout
    col1, spacing, col2 = st.columns([1, 0.1, 1])  # Adjust the middle column's width for desired spacing

    # Left column content
    with col1:
        # Set the coordinates for the map
        latitude = 37.57749
        longitude = -122.4194

        # Initialize the Google Maps client
        gmaps = googlemaps.Client(key=config.API_KEY)

        # Use streamlit.text_input() to get the user's address input
        address = st.text_input("Enter your address:")
        if address:
            geocode_result = gmaps.geocode(address)

            if geocode_result:
                latitude = geocode_result[0]["geometry"]["location"]["lat"]
                longitude = geocode_result[0]["geometry"]["location"]["lng"]

                with st.spinner("Loading images..."):
                    # Introduce a 3-second delay before loading the images
                    time.sleep(3)


        # Define the HTML string that will contain the map
        html_string = f"""
        <!DOCTYPE html>
        <div class="map-container">
            <iframe class="map-frame" width="100%" height="500px"  src="https://maps.google.com/maps?q={latitude},{longitude}&t=k&output=embed"></iframe>
        </div>
        """
        # Display the HTML iframe using the st.components.v1.html function
        # st.components.v1.html(html_string)
        st.components.v1.html(html_string, height=500)


    # Right column content
    with col2:
        st.subheader("Information about your location")
        location = "Sample Location"
        roof_size = 100
        radiation = 200
        max_pv_output = 300
        max_panels = 10
        max_energy = 400
        euro_generated = 500
        efficiency = 600
        azimuth = 700

        # Create two columns inside col2
        info_col, value_col = st.columns([0.5, 1])

        with info_col:
            st.markdown("Location:")
            st.markdown("Size of roof:")
            st.markdown("Radiation:")
            st.markdown("Max PV output:")
            st.markdown("Max panels on roof:")
            st.markdown("Max energy:")
            st.markdown("Efficiency:")
            st.markdown("Azimuth:")
            st.markdown("€ generated:")

        with value_col:
            st.markdown(f"**{location}**")
            st.markdown(f"**{roof_size}**")
            st.markdown(f"**{radiation}**")
            st.markdown(f"**{max_pv_output}**")
            st.markdown(f"**{max_panels}**")
            st.markdown(f"**{max_energy}**")
            st.markdown(f"**{efficiency}**")
            st.markdown(f"**{azimuth}**")
            st.markdown(f"**{euro_generated}**")

    col1, col2, col3 = st.columns(3)

    file_path = os.path.join(os.path.dirname(__file__), "image.png")
    col1.image(file_path, caption="Image 1", use_column_width=True)
    col2.image(file_path, caption="Image 2", use_column_width=True)
    col3.image(file_path, caption="Image 3", use_column_width=True)