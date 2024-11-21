import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt

def load_data(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            file_extension = os.path.splitext(img_name)[1].lower()
            if file_extension not in ['.jpg', '.jpeg', '.png']:
                print(f"Skipping non-image file: {img_path}")
                continue
            
            try:
                img = Image.open(img_path)
                img = img.resize(img_size)
                img = np.array(img)

                if img.shape != (img_size[0], img_size[1], 3):
                    print(f"Skipping invalid image (incorrect shape): {img_path}")
                    continue

                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue

    return np.array(images), np.array(labels), class_names

train_data_dir = 'd:/ML/thaifoodclassify/data/train'
val_data_dir = 'd:/ML/thaifoodclassify/data/val'

X_train, y_train, class_names = load_data(train_data_dir)
X_val, y_val, _ = load_data(val_data_dir)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=len(class_names))
y_val = keras.utils.to_categorical(y_val, num_classes=len(class_names))

model = Sequential([
    layers.InputLayer(input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

test_loss, test_acc = model.evaluate(X_val, y_val)

print(f"Test accuracy: {test_acc * 100:.2f}%")

print(f"Training completed in {elapsed_time_minutes:.2f} minutes.")

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

print("Train accuracy per epoch:")
for epoch, accuracy in enumerate(train_accuracy, 1):
    print(f"Epoch {epoch}: {accuracy * 100:.2f}%")

print("Validation accuracy per epoch:")
for epoch, accuracy in enumerate(val_accuracy, 1):
    print(f"Epoch {epoch}: {accuracy * 100:.2f}%")

avg_train_accuracy = np.mean(train_accuracy) * 100
avg_val_accuracy = np.mean(val_accuracy) * 100

print(f"\nAverage Training Accuracy: {avg_train_accuracy:.2f}%")
print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")

plt.plot(train_accuracy, label='Train accuracy')
plt.plot(val_accuracy, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()