from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import ImageUploadForm
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from django.conf import settings

# Load your model
model_path = os.path.join(settings.BASE_DIR, 'sequential.keras')
model = load_model(model_path)

# Define your class names (replace with your actual class list)
card_name = [f'Card {i}' for i in range(53)]

def predict_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    return card_name[predicted_index], float(np.max(predictions))

def index(request):
    prediction = None
    confidence = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = Image.open(request.FILES['image'])
            prediction, confidence = predict_image(img)
    else:
        form = ImageUploadForm()
    return render(request, 'predictor/index.html', {'form': form, 'prediction': prediction, 'confidence': confidence})
