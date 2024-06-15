from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data  # Assuming you have this function defined

def train_bot_model(train_x, train_y):
    # Define the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save the model
    model.save('chatbot_model.h5')

    print("Model File Created & Saved")

# Example function call assuming preprocess_train_data prepares train_x and train_y correctly
train_x, train_y = preprocess_train_data()

# Train the model
train_bot_model(train_x, train_y)
