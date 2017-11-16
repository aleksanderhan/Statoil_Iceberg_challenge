import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../data"]).decode("utf8"))

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/test.json")


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
x_band3 = x_band1 / x_band2

r = (x_band1 - x_band1.min()) / (x_band1.max() - x_band1.min())
g = (x_band2 - x_band2.min()) / (x_band2.max() - x_band2.min())
b = (x_band3 - x_band3.min()) / (x_band3.max() - x_band3.min())

X_train = np.concatenate([r[:, :, :, np.newaxis], g[:, :, :, np.newaxis], b[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])
print("Xtrain:", X_train.shape)


# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
x_band3 = x_band1 / x_band2

r = (x_band1 - x_band1.min()) / (x_band1.max() - x_band1.min())
g = (x_band2 - x_band2.min()) / (x_band2.max() - x_band2.min())
b = (x_band3 - x_band3.min()) / (x_band3.max() - x_band3.min())

X_test = np.concatenate([r[:, :, :, np.newaxis], g[:, :, :, np.newaxis], b[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)


# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout

model = Sequential()
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 3)))
model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 3)))
model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.28))
model.add(Dense(1, activation="sigmoid"))
model.compile("Adam", "mean_squared_error", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, validation_split=0.2, epochs=15)

# Make predictions
prediction = model.predict(X_test, verbose=1)
print "prediction len: " + str(prediction.shape)

submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv("./naive_submission.csv", index=False)
