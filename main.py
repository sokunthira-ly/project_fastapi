from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained model (update the path if needed)
model = load_model(r'D:\Project_Machine_learning\Image_classify.keras')

# List of category names
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Define image size expected by the model
img_height = 180
img_width = 180

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>Image Classification</title>
        </head>
        <body>
            <h2>Upload an image for classification</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file contents
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Convert image to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image to the target size
    image = image.resize((img_width, img_height))
    
    # Convert image to numpy array and expand dimensions to match model input
    img_array = np.array(image)
    img_bat = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = data_cat[np.argmax(score)]
    accuracy = float(np.max(score) * 100)
    
    return {
        "predicted_class": predicted_class,
        "accuracy": accuracy
    }
