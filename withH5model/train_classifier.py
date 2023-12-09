import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data from the pickle file
data_dict = pickle.load(open('/Users/pc/withH5model/data.pickle', 'rb'))

# Convert labels to numerical format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data_dict['labels'])

# Convert data to NumPy arrays
x_train = np.array(data_dict['data'])
y_train = labels_encoded

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(data_dict['data'][0]),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
accuracy = model.evaluate(x_test, y_test)[1]
print(f'{accuracy * 100}% of samples were classified correctly!')

# Save the Keras model to an h5 file
model.save('/Users/pc/withH5model/model.h5')

# Save label mapping to a text file
# Save label mapping to a text file
with open('/Users/pc/withH5model/labels.txt', 'w') as f:
    for class_label, class_name in enumerate(label_encoder.classes_):
        f.write(f"{class_label}:{chr(ord('A') + int(class_name))}\n")

