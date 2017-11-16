import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def get_images(df):
    '''Create 3-channel 'images'. Return rescale-normalised images.'''
    images = []
    for i, col in df.iterrows():
        # Formulate the bands as 75x75 arrays
        band_1 = np.array(col['band_1']).reshape(75, 75)
        band_2 = np.array(col['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        # Rescale
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        images.append(rgb)
    return np.array(images)


train_data = pd.read_json('../data/train.json')
X = get_images(train_data)
y = np.array(train_data["is_iceberg"])

model = Sequential()
#model.add(Dropout(0.2, input_shape=(75, 75, 3)))
model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Dropout(0.2))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(2, (3, 3), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])
model.summary()


model.fit(X, y, validation_split=0.2, epochs=60, batch_size=32)

test_data = pd.read_json('../data/test.json')
X_test = get_images(test_data)

# Make predictions
prediction = model.predict(X_test, verbose=1)

submit_df = pd.DataFrame({'id': test_data["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv("./mysub.csv", index=False)
