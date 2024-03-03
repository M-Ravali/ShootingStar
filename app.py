from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
# streamlit_app.py
import streamlit as st

def main():

    styling = '''
    <style>
    [data-testid="stApp"]{
        background-image: url("https://cdn-useast1.kapwing.com/teams/65e419417547a6d2eca88a42/jobs/65e41959e28bc77ba6861efa/final_65e4194263612ea158b3d93b_953719.gif?GoogleAccessId=prod-sa-videoprocessing%40kapwing-prod.iam.gserviceaccount.com&Expires=1710052427&Signature=MDuN5UhLUa1KXdWDPAoryoeuWaSdLWDWHPcKtXE4FsCbU%2F1wDn0Eee1Z2VMS%2BG1Y%2BUd3liOoBwYlItMfU8%2Bi71TjvuGi44bdx4X85cB5x1omO81jL9HNmlKCwB7GLi47dXZ5MdYtQHCyifScxbWpC%2FE%2BTyFXpId%2B24PCyLCJ2a5wl%2FqerV%2BooTnLwvW3UpXbLnMNY7LTk9eRKBfO%2FRnqeMiYnLIRkL8zCBp1Ek6bohPHrgfYNoPCu0jU7omz62Zk0otp0tNHPIXAEY%2ByU%2BjQmvalQV3f6ZYp9Kxp6YvFs3j50MxHzMpT4oWOsYZbGEFvi8v5sjBFy2lN%2BXuIhvHWJA%3D%3D");
        background-size: cover;
        color: white
        
        }
    [data-testid="stHeader"]{
        background: none;
    }
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="baseButton-secondary"]{
        background: rgb(134 130 130 / 50%);
        color: white;
    }
    [data-testid="baseButton-header"]{
        background: white;
        color: black;
        
    }
    [data-testid="stFileUploadDropzone"]{
        background-color: rgb(186 188 190);
        color: black;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="StyledLinkIconContainer"],
    [data-testid="stImageCaption"]{
        color: white
    }
    </style>
    '''
    st.markdown(styling, unsafe_allow_html=True)

with st.container(height=800):
    st.title("Shooting Star")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        uploaded_file = "image.png"
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
    else:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

    # Load the saved model
    saved_model_path = "C:/Users/madde/Downloads/team-ignite-main[1]/team-ignite-main/web app/alexnet_model_iter_10.h5"  
    # Update with the actual path
    alexnet_model = load_model(saved_model_path)

    # Assuming you have a dictionary mapping class labels to constellation names
    label_mapping = {'Cassiopeia': 0, 'Gemini': 1, 'Orion': 2, 'Perseus': 3, 'SummerTriangle': 4, 'Ursa Major': 5}

 

    # Load the image
    input_image = Image.open(uploaded_file)

    # Preprocess the image (resize, convert to grayscale, etc.)
    # Example preprocessing steps:
    input_image = input_image.resize((256, 256))  # Resize the image to match the input size of the model
    input_image = input_image.convert('L')  # Convert the image to grayscale

    # Convert the image to a numpy array
    input_array = np.array(input_image) / 255.0  # Normalize pixel values (assuming the model was trained with normalized inputs)

    # Add a batch dimension and a channel dimension to match the model input shape
    input_array = np.expand_dims(input_array, axis=(0, -1))

    # Make a prediction
    predicted_prob = alexnet_model.predict(input_array)
    predicted_label = np.argmax(predicted_prob)

    # Get the predicted constellation name
    predicted_constellation = next((name for name, label in label_mapping.items() if label == predicted_label), "Unknown")


        # For now, let's assume the model predicts a class named "Unknown"
       
    st.write(f"Prediction: {predicted_constellation}")

if __name__ == "__main__":
    main()