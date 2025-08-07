import streamlit as st
# import matplotlib
from io import BytesIO
from PIL import Image
import requests
from model import style_transfer_model, show_image_color_corrected

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: Navy Blue;'>Image Style Transfer</h1>", unsafe_allow_html=True)

# Main container for the images
image_container = st.empty() 

mode = st.sidebar.radio("Choose Input Mode", ["Upload", "Use Example Images"])

if mode == "Upload":
    content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png"])
else:
    content_examples = ['Img3.jpg', 'Img1.jpg', 'Img2.jpg']
    style_examples = ['Img6.png', 'Img4.jpg', 'Img5.jpg']
    content_path = f"Image/{st.sidebar.selectbox('Select Content Image', content_examples)}"
    style_path = f"Image/{st.sidebar.selectbox('Select Style Image', style_examples)}"

    # We need to handle the case where the app first loads and the files are not yet "open"
    # This ensures a clean state before the user makes a selection
    try:
        content_file = open(content_path, 'rb')
        style_file = open(style_path, 'rb')
    except FileNotFoundError:
        content_file = None
        style_file = None


st.sidebar.markdown("## Model Parameters")
steps = st.sidebar.slider("Training Steps", 1, 500, 50, 2)
content_weight = st.sidebar.number_input("Content Weight", min_value=10, value=100)
style_weight = st.sidebar.number_input("Style Weight", min_value=100, value=3000)
st.sidebar.markdown("---")

# Use a single conditional block to manage what's displayed in the main area
if content_file and style_file:
    # If files are selected, display them side-by-side
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    with image_container.container():
        img_cols = st.columns(2)
        with img_cols[0]:
            st.subheader("Content Image")
            st.image(content_image, use_column_width=True)
        with img_cols[1]:
            st.subheader("Style Image")
            st.image(style_image, use_column_width=True)
else:
    # If no files are selected, display the initial placeholder image
    with image_container.container():
        img = Image.open('Image/stylized_output.png').convert("RGB")
        st.image(img, width=800)


if content_file and style_file:
    if st.button("Apply Style Transfer"):
        with st.spinner("Transforming..."):
            output_tensor = style_transfer_model(content_file, style_file, steps, content_weight, style_weight)
            stylized_img = show_image_color_corrected(output_tensor)

        st.subheader("Stylized Output")
        st.image(stylized_img, width=600)

        buf = BytesIO()
        stylized_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Stylized Image", data=buf, file_name="stylized_output.png", mime="image/png")
