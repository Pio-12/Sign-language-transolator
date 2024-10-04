import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the collected data
def load_data():
    data = []
    labels = []
    
    for sign in ['A', 'B', 'C', 'D', 'E']:  # Add more signs as needed
        csv_data = pd.read_csv(f'asl_data/{sign}_data.csv', header=None)
        data.append(csv_data.iloc[:, :-1].values)  # All but last column (features)
        labels.append(csv_data.iloc[:, -1].values)  # Last column (labels)
    
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    return data, labels

# Load data and split into train and test sets
data, labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')  # Adjust output neurons to number of signs
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('asl_recognition_model.h5')
