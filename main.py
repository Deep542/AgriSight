import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pygments.styles.dracula import background
from streamlit import container
from streamlit_option_menu import option_menu
# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#NavBar

app_mode = option_menu(
    menu_title= None,
    options= ["Home", "About", "Disease Recognition"],
    icons= ["house-door-fill","person-circle","search"],
    default_index= 0,
    orientation= "horizontal",

    styles={
        "container":{"padding":"10px", "background-color":"#D4F6FF"},
        "icons": {"color": "C70039", "font-size":"15px"},
        "nav-link":{"font-size":"10px",
                    "text-align":"left",
                    "font-family":"Georgia",
                    "margin":"0px",
                    "--hover-color": "#FFE6E6"
                    },
        "nav-link-selected":{"background-color":"#86D293"},


},
)
# Home Page
if (app_mode == "Home"):
    st.header("_AGRISIGHT_")
    image_path = "AgriSight logo.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to AgriSight the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the Navigation bar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

# About Page
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test 
    ### Team Members
    1. Deep Ranjan Das    
    2. Gautam Kumar Gupta
""")

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")


    test_image = st.file_uploader("Choose an Image:")
    if (st.button("Show Image")):
        st.image(test_image, use_container_width=True)
      #Predict Button
    if (st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Define Class
            class_name = ['Apple___Apple_scab , Recommendation: use fungicides, practice sanitation, choose resistant varieties',
                          'Apple___Black_rot , Recommendation: Prune trees when temperatures are below 32°F, Cut 2 to 4 inches below the growth, removing all infected wood, Use a fungicide like Trianum Shield. ',
                          'Apple___Cedar_apple_rust , Recommendation: Remove Nearby Cedar Hosts, Use Resistant Apple Varieties,Apply Fungicides at the Right Time',
                          'Apple___healthy',
                          'Blueberry___healthy',
                          'Cherry_(including_sour)___Powdery_mildew , Recommendation: Prune for Air Circulation, Apply Fungicides, Remove Infected Leaves',
                          'Cherry_(including_sour)___healthy',
                          'Corn_(maize)___Cercospora_leaf_spot Grey_leaf_spot , Recommendation: Plant Resistant Varieties, Rotate Crops, Apply Fungicides',
                          'Corn_(maize)___Common_rust_ , Recommendation: Plant Resistant Varieties , Monitor and Apply Fungicides, Improve Field Conditions',
                          'Corn_(maize)___Northern_Leaf_Blight , Recommendation:Plant Resistant Hybrids, Crop Rotation and Residue Management, Use Fungicides if Necessary',
                          'Corn_(maize)___healthy',
                          'Grape___Black_rot , Recommendation:Prune and Remove Infected Material, Apply Fungicides, Improve Airflow and Drainage',
                          'Grape___Esca_(Black_Measles) , Recommendation:Prune and Remove Infected Wood, Avoid Excessive Wounding, Maintain Vineyard Health',
                          'Grape___Leaf_blight_(Isariopsis_Leaf_Spot) , Recommendation:Prune and Remove Infected Leaves, Improve Air Circulation, Apply Fungicides',
                          'Grape___healthy',
                          'Orange___Haunglongbing_(Citrus_greening) , Recommendation:Control Asian Citrus Psyllid,Remove Infected Trees,Plant Disease-Resistant Varieties',
                          'Peach___Bacterial_spot , Recommendation:Plant Resistant Varieties,Apply Copper-Based Sprays,Improve Orchard Sanitation',
                          'Peach___healthy',
                          'Pepper__bell___Bacterial_spot , Recommendation:Use Disease-Resistant Varieties,Apply Copper-Based Sprays,Practice Crop Rotation and Sanitation',
                          'Pepper__bell___healthy',
                          'Potato___Early_blight , Recommendation:Use Resistant Varieties,Practice Crop Rotation,Apply Fungicides',
                          'Potato___Late_blight , Recommendation:Plant Resistant Varieties, Apply Fungicides,Improve Field Drainage and Ventilation',
                          'Potato___healthy',
                          'Raspberry___healthy',
                          'Soybean___healthy',
                          'Squash___Powdery_mildew , Recommendation:Remove Infected Leaves,Use Organic Treatments,Ensure Proper Sunlight',
                          'Strawberry___Leaf_scorch , Recommendation:Improve Air Circulation,Apply Fungicides,Remove Infected Leaves',
                          'Strawberry___healthy',
                          'Tomato___Bacterial_spot , Recommendation:Use Disease-Free Seeds and Transplants,Apply Copper-Based Sprays,Avoid Overhead Watering',
                          'Tomato___Early_blight , Recommendation:Rotate Crops Annually,Mulch Around Plants,Use Fungicides When Needed',
                          'Tomato___Late_blight , Recommendation:Remove and Destroy Infected Plants,Apply Preventative Fungicides,Avoid Prolonged Leaf Wetness',
                          'Tomato___Leaf_Mold , Recommendation:Improve Air Circulation,Avoid Overhead Watering,Apply Fungicides if Necessary',
                          'Tomato___Septoria_leaf_spot , Recommendation:Remove Infected Leaves,Apply Copper-Based Fungicides,Practice Crop Rotation',
                          'Tomato___Spider_mites Two-spotted_spider_mite , Recommendation:Spray Water to Dislodge Mites,Introduce Natural Predators (e.g., Ladybugs),Apply Neem Oil or Insecticidal Soap',
                          'Tomato___Target_Spot , Recommendation:Remove and Destroy Infected Leaves,Apply Fungicides Like Chlorothalonil,Avoid Overhead Watering',
                          'Tomato___Tomato_Yellow_Leaf_Curl_Virus , Recommendation:Control Whiteflies with Insecticides,Plant Resistant Tomato Varieties,Use Physical Barriers Like Row Covers',
                          'Tomato___Tomato_mosaic_virus , Recommendation:Use Disease-Free Seeds and Sanitize Tools,Remove and Destroy Infected Plants Immediately,Avoid Handling Plants After Touching Infected Materia',
                          'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))