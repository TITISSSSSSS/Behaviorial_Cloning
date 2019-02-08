import cv2
import os
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, Dense, Activation, MaxPooling2D

def read_image(dataset, data_dir, img_index, img_position = 'center'):
        
    imgs = dataset.iloc[img_index]
    center_steering = imgs['steering']
    img_path = os.path.join(data_dir, imgs[img_position].strip())
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, center_steering 

def change_brightness(image):
    br_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .2 + np.random.uniform(0.2, 0.6)
    br_img[:,:,2] = br_img[:,:,2] * random_bright
    result = cv2.cvtColor(br_img,cv2.COLOR_HSV2RGB)
    return result

def resize_image(image, target_size = (64,64)):
     return cv2.resize(image, target_size) 
    
def crop_image(image):
    return image[55:135,:,:]

def flip_image(image):
    return cv2.flip(image, 1)

def preprocess_image(image):
    img = crop_image(image)
    img = resize_image(img, target_size = (64,64))
    img = img.astype(np.float32)
    img = (img/127.5) - 1.0
    return img

def generate_augmented_image(dataset, img_index):
    # randomly choose camera position
    img_position = np.random.choice(['center', 'left', 'right']);
    
    #read image 
    image, center_steering = read_image(dataset, data_dir, img_index, img_position)
    
    # adjust the steering angle based on cameras position 
    if img_position == 'left':
        steering = center_steering + 0.25
    elif img_position == 'right':
        steering = center_steering - 0.25
    elif img_position == 'center':
        steering = center_steering
    
    #randomly change image brightness 
    image = change_brightness(image)
    
    #randomly flip image 
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        image = flip_image(image)
        steering = steering * (-1)
    
    #fianlly, crop, resize and normalize image 
    image = preprocess_image(image)
    return image, steering

def batch_generator(dataset, batch_size =32):
    N = dataset.shape[0]
    num_batches = N // batch_size
    while True:
        for i in range(num_batches):
            start = i*batch_size
            end = start-1 + batch_size
            X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
            y_batch = np.zeros((batch_size,), dtype=np.float32)

            for batch_index, img_index in zip(range(batch_size),range(start,end)):
                X_batch[batch_index], y_batch[batch_index] = generate_augmented_image(dataset, img_index)

            yield X_batch, y_batch


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 3
    
    data_dir = './data'
    cvs_file = os.path.join(data_dir, 'driving_log.csv')
    dataset=pd.read_csv(cvs_file)

    # shuffle the data
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    training_data = int(dataset.shape[0] * 0.8)
    training_dataset = dataset.loc[0:training_data-1]
    validation_dataset = dataset.loc[training_data:]

    # release the main data_frame from memory
    dataset = None

    trainingSet_generator = batch_generator(training_dataset, batch_size=BATCH_SIZE)
    validationSet_generator = batch_generator(validation_dataset, batch_size=BATCH_SIZE)

    model = Sequential()
    model.add(Convolution2D(24,5,5, input_shape=(64, 64, 3), subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(38,5,5, subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(64,3,3, activation= "relu"))
    model.add(Convolution2D(64,3,3, activation= "relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    # train model
    print('Training ...')
    nb_samples = (20000//BATCH_SIZE) * BATCH_SIZE # samples per epoch
    nb_valid = 5000
    model.compile(optimizer="adam", loss="mse")
    model.fit_generator(trainingSet_generator, validation_data=validationSet_generator,
                        samples_per_epoch=nb_samples, nb_epoch=EPOCHS, nb_val_samples=nb_valid)

    ## save model
    model.save('model.h5')
    print("Saved model to disk")
