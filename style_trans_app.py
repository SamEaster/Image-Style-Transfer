import streamlit as st
# import matplotlib
from io import BytesIO
from PIL import Image
import requests
from model import style_transfer_model, show_image_color_corrected

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: Navy Blue;'>Image Style Transfer</h1>", unsafe_allow_html=True)

content_examples = ['Img3.jpg', 'Img1.jpg', 'Img2.jpg']
style_examples = ['Img6.png', 'Img4.jpg', 'Img5.jpg']

mode = st.sidebar.radio("Choose Input Mode", ["Upload", "Use Example Images"])

if mode == "Upload":
    content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png"])
else:
    content_path = f"Image/{st.sidebar.selectbox('Select Content Image', content_examples)}"
    style_path = f"Image/{st.sidebar.selectbox('Select Style Image', style_examples)}"

    content_file = open(content_path, 'rb')
    style_file = open(style_path, 'rb')


st.sidebar.markdown("## Model Parameters")
steps = st.sidebar.slider("Training Steps", 1, 500, 50, 2)
content_weight = st.sidebar.number_input("Content Weight", min_value=10, value=100)
style_weight = st.sidebar.number_input("Style Weight", min_value=100, value=3000)
st.sidebar.markdown("---")


if content_file and style_file:
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    # with center_col:
    img_cols = st.columns(2)
    with img_cols[0]:
        st.subheader("Content Image")
        st.image(content_image, use_column_width=True)
    with img_cols[1]:
        st.subheader("Style Image")
        st.image(style_image, use_column_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Transforming..."):
            print("Applying transfer...")
            output_tensor = style_transfer_model(content_file, style_file, steps, content_weight, style_weight)
            print("Transformation complete.")
            stylized_img = show_image_color_corrected(output_tensor)

        st.subheader("Stylized Output")
        st.image(stylized_img, width = 600)

        buf = BytesIO()
        stylized_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Stylized Image", data=buf, file_name="stylized_output.png", mime="image/png")