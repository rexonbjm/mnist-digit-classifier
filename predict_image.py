from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model("mnist_model.keras")

# Ask user for an image file
path = input("Enter the path of your image: ")

# Open the image
img = Image.open(path).convert("L")  # convert to grayscale

# Resize to 28x28
img = img.resize((28, 28))

# Convert to numpy array
img_arr = np.array(img)

# Invert colors if needed (MNIST digits are black background and white digit)
if img_arr.mean() > 127:
    img_arr = 255 - img_arr

# Normalize (0–1)
img_arr = img_arr / 255.0

# Reshape for model: (1, 28, 28)
img_arr = img_arr.reshape(1, 28, 28)

# Predict
pred = model.predict(img_arr)
digit = np.argmax(pred)

print("Predicted digit:", digit)
