import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

csv_files = [
    "../dataset/squat_landmarks.csv",
    "../dataset/pushup_landmarks.csv",
    "../dataset/stretching_landmarks.csv",
    #"../dataset/lifting_landmarks.csv"
]

dataframes = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dataframes, ignore_index=True)

X = df.drop('label', axis=1).values  
y = df['label'].values            

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

model.save("../models/pose_classifier_model.h5")

import pickle
with open("../models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Training complete. Model and label encoder saved.")
