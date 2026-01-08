# Import necessary libraries

# Data manipulation and analysis
import pandas as pd  # For handling and analyzing data in tabular format
import numpy as np  # For numerical computations and array manipulations

# Visualization
import matplotlib.pyplot as plt  # For plotting and visualizing data and results

# Handling imbalanced datasets
from imblearn.over_sampling import RandomOverSampler  # For oversampling minority classes in imbalanced datasets

# Cross-validation
from sklearn.model_selection import KFold  # For k-fold cross-validation

# TensorFlow Keras Sequential Model
from keras.models import Sequential  # For creating a linear stack of layers

# Neural network layers
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization  # Common layers for CNNs

# Dataset splitting
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Training callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  # For adaptive learning rate and early stopping during training

df = pd.read_csv('/Users/leonardoangellotti/Desktop/universita/Machine Learning Operation/CancerSpot/archive/hmnist_28_28_RGB.csv')

# Splits the DataFrame into features (X) and labels (Y).
# X contains all columns except the last one, and Y contains only the last column.
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Initialize RandomOverSampler with a fixed random state for reproducibility
oversample = RandomOverSampler(random_state=42)

# Apply oversampling to balance the dataset
X, Y = oversample.fit_resample(X, Y)

# Convert resampled data to numpy arrays for compatibility with machine learning models
X = X.to_numpy()
Y = Y.to_numpy()

# Reshapes the features to have a 4D shape
# compatible with convolutional layers (height, width, channels)
# and normalizes the pixel values to be between 0 and 1.

# Reshape label data to match the expected model input format
Y = np.reshape(Y, (46935, 1))  # Reshape Y into a column vector with 46935 rows

# Reshape feature data to fit CNN input format (28x28 images with 3 channels)
X = np.reshape(X, (46935, 28, 28, 3))

# Store original reshaped data for reference or debugging
Actual_X = X  # Backup original X data
Actual_Y = Y  # Backup original Y data

# Normalize pixel values to range [0, 1] for better training stability
X = X / 256

# Callback to reducing the learning rate during training if
# monitored metric (val_accuracy in this case) does not improve.

# Reduce learning rate when a metric has stopped improving
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',  # Monitor validation accuracy for changes
    patience=2,  # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,  # Print messages when reducing learning rate
    factor=0.5,  # Reduce learning rate by a factor of 0.5
    min_lr=0.00001  # Minimum learning rate limit
)

# Stop training early if validation accuracy doesn't improve
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the best epoch
)

# Model architecture
def model_function():

    model = Sequential([
        # First convolutional block
        Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'),  # First Conv layer with 32 filters
        MaxPooling2D((2, 2)),  # Downsample feature maps
        BatchNormalization(),  # Normalize activations
        
        # Second convolutional block
        Convolution2D(64, (3, 3), activation='relu', padding='same'),  # Second Conv layer with 64 filters
        Convolution2D(64, (3, 3), activation='relu', padding='same'),  # Another Conv layer with 64 filters
        MaxPooling2D((2, 2)),  # Downsample feature maps
        BatchNormalization(),  # Normalize activations
        
        # Third convolutional block
        Convolution2D(128, (3, 3), activation='relu', padding='same'),  # Conv layer with 128 filters
        Convolution2D(128, (3, 3), activation='relu', padding='same'),  # Another Conv layer with 128 filters
        MaxPooling2D((2, 2)),  # Downsample feature maps
        BatchNormalization(),  # Normalize activations
        
        # Fourth convolutional block
        Convolution2D(256, (3, 3), activation='relu', padding='same'),  # Conv layer with 256 filters
        Convolution2D(256, (3, 3), activation='relu', padding='same'),  # Another Conv layer with 256 filters
        MaxPooling2D((2, 2)),  # Downsample feature maps
        
        # Flatten and fully connected layers
        Flatten(),  # Flatten feature maps into a vector
        Dropout(rate=0.2),  # Prevent overfitting by randomly dropping neurons
        Dense(128, activation='relu'),  # Dense layer with 128 neurons
        BatchNormalization(),  # Normalize activations
        Dense(64, activation='relu'),  # Dense layer with 64 neurons
        BatchNormalization(),  # Normalize activations
        Dense(32, activation='relu'),  # Dense layer with 32 neurons
        BatchNormalization(),  # Normalize activations
        
        # Output layer
        Dense(7, activation='softmax')  # Output layer with 7 classes using softmax activation
    ])

    # Compile the model with loss function, optimizer, and evaluation metrics
    model.compile(
        loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
        optimizer='adam',  # Adam optimizer for adaptive learning
        metrics=['accuracy']  # Evaluate model based on accuracy
    )

    return model

# Initialize the model by calling the model_function
model = model_function()

# model.summary()

# Initialize KFold cross-validation with 5 splits and shuffling of the data
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a variable for the fold counter
i = 0

# Loop over each split (train and test indices) generated by KFold
for train_index, test_index in kfold.split(X):
    
    # Create the training and test data subsets using the indices provided by KFold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Train the model on the training data with validation on the test data
    # Use a learning rate reduction callback, and train for 25 epochs with a batch size of 128
    fitting_stats = model.fit(X_train, y_train, epochs=25, batch_size=128, 
                              validation_data=(X_test, y_test),
                              callbacks=[learning_rate_reduction])

    # Evaluate the model on the test data and output the test accuracy
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # print(f"Test Accuracy: {accuracy}")

    # Increment the fold counter for the next iteration
    i = i + 1

# Save the entire model
model.save('cnn_model.h5')

