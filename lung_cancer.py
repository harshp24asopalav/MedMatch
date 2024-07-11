import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

class LungCancer:
    def __init__(self, use_pretrained=True):
        if use_pretrained:
            self.model = tf.keras.models.load_model("./trained-model/lc_version_1.h5")
        else:
            self.load_data()
            self.build_model()
            self.train_model()
            self.evaluate_model()
            self.save_model()

    def load_data(self):
        self.train_path = './data-collection/Lung_cancer/train'
        self.val_path = './data-collection/Lung_cancer/valid'
        self.test_path = './data-collection/Lung_cancer/test'
        self.input_shape = (224, 224, 3)
        self.num_classes = 4

        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            dtype='float32'
        )

        val_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            dtype='float32'
        )

        test_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            dtype='float32'
        )

        self.train_data = train_gen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )

        self.val_data = val_gen.flow_from_directory(
            self.val_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )

        self.test_data = test_gen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False
        )

    def build_model(self):
        VGG16_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape
        )
        for layer in VGG16_model.layers:
            layer.trainable = False

        self.model = Sequential()
        self.model.add(VGG16_model)
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.summary()

        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
        )

    def train_model(self):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('./trained-model/lc_version_1.h5', save_best_only=True),
        ]

        self.history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=100,
            verbose=1,
            callbacks=callbacks
        )

    def evaluate_model(self):
        loss, acc = self.model.evaluate(self.test_data, verbose=1)
        print(f"Test Loss: {loss}, Test Accuracy: {acc}")

    def save_model(self):
        self.model.save('./trained-model/lc_version_1.h5')

    def predict(self, image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)

        return predicted_class


#lung_cancer_model = LungCancer(use_pretrained=False)
