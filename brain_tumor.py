import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout
from tensorflow import keras
from dotenv import load_dotenv

load_dotenv()

class BrainTumor:
    def __init__(self, use_pretrained = True) -> None:

        # TODO
        self.bt_model_path = os.getenv('BT_MODEL', './trained-model/tumor_version_1.h5')
        self.bt_epochs = int(os.getenv('BT_EPOCHS', '50'))
        self.bt_test_size = float(os.getenv('BT_TEST_SIZE', '0.2'))
        self.bt_random_state = int(os.getenv('BT_RANDOM_STATE', '101'))



        if use_pretrained:
            self.model = keras.models.load_model(self.bt_model_path)
        else:
            self.load_data()
            self.process_data()
            self.build_cnn_model()
            self.train_model()
            self.save_model()


    def load_data(self):
        normal_cells=os.listdir('./data-collection/Brain_tumor_data/no')
        tumor_cells=os.listdir('./data-collection/Brain_tumor_data/yes')
        normal_label=[0]*98
        tumor_label=[1]*155

        labels=normal_label+tumor_label
        self.normal_cells = normal_cells
        self.tumor_cells = tumor_cells
        self.labels = labels

    def process_data(self):
        normal_path=('./data-collection/Brain_tumor_data/no/')
        data=[]

        for img_file in self.normal_cells:
            image=Image.open(normal_path + img_file)
            image=image.resize((128,128))
            image=image.convert('RGB')
            image=np.array(image)
            data.append(image)
            
        tumor_path=('./data-collection/Brain_tumor_data/yes/')

        for img_file in self.tumor_cells:
            image=Image.open(tumor_path + img_file)
            image=image.resize((128,128))
            image=image.convert('RGB')
            image=np.array(image)
            data.append(image)

        X=np.array(data)
        Y=np.array(self.labels)

        # Splitting data
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=self.bt_test_size, random_state=self.bt_random_state)

        # Scaling data
        self.X_train=X_train/255
        self.X_test=X_test/255
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def build_cnn_model(self):
        num_of_classes=2
        model=Sequential()
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_of_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    

    def train_model(self):
        self.model.fit(self.X_train,self.Y_train, epochs=self.bt_epochs, validation_split=0.1, verbose=1)
    
    def save_model(self):
        self.model.save(f'./trained-model/tumor_version_1.h5')
    
    def evaluate(self):
        self.model.evaluate(self.X_test,self.Y_test)
  
    def predict(self, image_bytes):
        # Read the image bytes and preprocess the image
        input_image = Image.open(io.BytesIO(image_bytes))
        input_image = input_image.resize((128, 128))
        input_image = input_image.convert('RGB')
        input_image = np.array(input_image)
        image_normalized = input_image / 255.0
        img_reshape = np.reshape(image_normalized, (1, 128, 128, 3))

        # Predict the class
        input_prediction = self.model.predict(img_reshape)
        print('Prediction Probabilities are: ', input_prediction)  # Debug statement
        input_pred_label = np.argmax(input_prediction)

        return bool(input_pred_label)

        