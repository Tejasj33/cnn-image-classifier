from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. Load the saved model
model = load_model("cnn_image_classifier.h5")

# 2. Load the CIFAR-10 dataset (correct unpacking)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 3. Normalize the pixel values
x_test = x_test / 255.0

# 4. Loop to predict and display multiple images from the test set
for i in range(5):  # Change range value to predict multiple images
    img = x_test[i]  # Get the i-th image from the test set

    # 5. Make a prediction
    img_input = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_input)

    # 6. Print the predicted class
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    print(f"Predicted class for image {i}: {predicted_class}")

    # Optionally, display the image
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class}")
    plt.show()
