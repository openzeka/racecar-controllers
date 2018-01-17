
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pylab as plt
import random
# In[ ]:


df = pd.read_csv('/home/nvidia/racecar-ws/src/racecar-controllers/deep_learning/data/034/seyir.csv')


df.Angle.describe()

images = list(df.FileName[30:])
labels = list(df.Angle[30:])

# add more data which steering angle bigger then 0.5 or smaller then -0.25 for three times
nitem = len(images)
for i in range(nitem):
    if labels[i] > 0.15 or labels[i] < -0.00:
        for j in range(5):
            images.append(images[i])
            labels.append(labels[i])     


# batch size 32
# split data %80 for training, %20 for validation
bsize = 8
dlen = len(labels)
splitpoint = int(0.8*dlen)
reindex = list(range(len(labels)))
random.seed(1234)
random.shuffle(reindex)


# Augmentation function (taken from github)
def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# I will use augment_brightness with random (nearly half of the data will use augment_brightness function)
def get_matrix(fname):
    img = cv2.imread(fname)
    #print(img.shape)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if random.randint(0,1) == 1 :
        img = augment_brightness(img)        
    return img/home/nvidia/racecar-ws/src/racecar-controllers/deep_learning/scripts/


# Generate data for training
def generate_data():
    i = 0
    while True:
        x = []
        y = []
        for j in range(i,i+bsize):  
            ix = reindex[j]
            img = get_matrix(images[ix])
            lbl = np.array([labels[ix]])
            flip = random.randint(0,1)
            if flip == 1:
                img = cv2.flip(img,1)
                lbl = lbl*-1.0
            x.append(img)
            y.append(lbl)
        x = np.array(x)
        y = np.array(y)       
        yield (x,y)    
        i +=bsize
        if i+bsize > splitpoint:
            i = 0
            
# Generate data for validation                  
def generate_data_val():
    i = splitpoint
    while True:
        x = []
        y = []
        for j in range(i,i+bsize): 
            ix = reindex[j]
            x.append(get_matrix(images[ix]))
            y.append(np.array([labels[ix]]))
        x = np.array(x)
        y = np.array(y)       
        yield (x,y)    
        i +=bsize
        if i+bsize > dlen:
            i = splitpoint


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD


# Model based on NVIDIA's End to End Learning for Self-Driving Cars model
model = Sequential()
# Cropping
model.add(Cropping2D(cropping=((124,126),(0,0)), input_shape=(376,1344,3)))
# Normalization
model.add(Lambda(lambda x: (2*x / 255.0) - 1.0))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))/home/nvidia/racecar/home/nvidia/racecar-ws/src/racecar-controllers/deep_learning/scripts/-ws/src/racecar-controllers/deep_learning/scripts/
model.compile(loss='mse', optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


labels = 3*np.array(labels)


# In[ ]:


model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)


# In[ ]:

del df

hs = model.fit_generator(generate_data(),steps_per_epoch=int(splitpoint/ bsize),
                    validation_data=generate_data_val(), 
                    validation_steps=(dlen-splitpoint)/bsize, epochs=1,callbacks=[model_checkpoint])


import json
# Save model weights and json.
mname = 'model_new'
model.save_weights(mname+'.h5')
model_json  = model.to_json()
with open(mname+'.json', 'w') as outfile:
    json.dump(model_json, outfile)


# Train and validation loss chart
print(hs.history.keys())

plt.plot(hs.history['loss'])
plt.plot(hs.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# Compare actual and predicted steering
for i in range(10):
    ix = random.randint(0,len(df)-1)
    out = model.predict(get_matrix(df.image_name[ix]).reshape(1,376,1344,3))
    print(df.angle[ix], ' - > ', out[0][0])

