import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras import load_model


import io

# Load pre-trained SRGAN model
@st.cache_resource
def load_gan_model():
    model = tf.keras.models.load_model('srgan_model.h5', compile=False)
    return model

# Preprocess the input image
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    if len(image_array.shape) == 2:  # Grayscale to RGB
        image_array = np.stack((image_array,)*3, axis=-1)
    return np.expand_dims(image_array, axis=0)

# Post-process the output image
def postprocess_image(output_array: np.ndarray):
    output_array = (output_array[0] * 255.0).astype(np.uint8)  # Denormalize
    return Image.fromarray(output_array)

# Streamlit App
def main():
    st.title("Foggy Image Cleaner with GAN")

    st.write("Upload a foggy or blurred UAV image to enhance it.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        input_image = Image.open(uploaded_image)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        st.write("Enhancing the image using GAN...")
        gan_model = load_gan_model()
        
        # Preprocess
        processed_image = preprocess_image(input_image)
        
        # Enhance
        with st.spinner("Processing..."):
            enhanced_image_array = gan_model.predict(processed_image)
        
        # Post-process
        enhanced_image = postprocess_image(enhanced_image_array)
        
        # Display enhanced image
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
        
        # Download option
        buf = io.BytesIO()
        enhanced_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
