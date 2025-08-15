import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

IMG_DIR = "data/ecg_images"
CSV_PATH = "data/demographics.csv"

df = pd.read_csv(CSV_PATH)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

images = []
for img_name in df['image']:
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    images.append(img)

images = np.array(images) / 255.0
images = images.reshape(-1, 128, 128, 1)

demographics = df[['age', 'height', 'weight']].values
scaler = StandardScaler()
demographics = scaler.fit_transform(demographics)

labels = to_categorical(df['label'], num_classes=3)

X_img_train, X_img_test, X_demo_train, X_demo_test, y_train, y_test = train_test_split(
    images, demographics, labels, test_size=0.2, random_state=42
)

img_input = Input(shape=(128, 128, 1))
x = Conv2D(32, (3,3), activation='relu')(img_input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

demo_input = Input(shape=(3,))
y = Dense(16, activation='relu')(demo_input)

combined = Concatenate()([x, y])
z = Dense(32, activation='relu')(combined)
output = Dense(3, activation='softmax')(z)

model = Model(inputs=[img_input, demo_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_img_train, X_demo_train], y_train,
    validation_data=([X_img_test, X_demo_test], y_test),
    epochs=10, batch_size=32
)

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

model.save("ecg_health_model.h5")
