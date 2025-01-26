import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Define constants
FILE_PATH = 'AndMal2020-dynamic-BeforeAndAfterReboot/Adware_after_reboot_Cat.csv'
TARGET_COLUMN = 'Category'

# Load dataset
print("Loading dataset...")
data = pd.read_csv(FILE_PATH)

# Check if the target column exists
if TARGET_COLUMN not in data.columns:
    raise KeyError(f"'{TARGET_COLUMN}' not found in dataset columns. Available columns: {list(data.columns)}")

# Separate numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# Check for invalid values in numeric columns
print("Checking for invalid values in numeric columns...")
if np.any(np.isinf(data[numeric_columns].values)):
    print("Found 'inf' values in the numeric columns. Replacing with NaN...")
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)

if np.any(np.isnan(data[numeric_columns].values)):
    print("Found 'NaN' values in the numeric columns. Filling with column means...")
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Extract features and target
X = data[numeric_columns]
y = data[TARGET_COLUMN]

# Encode the target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define FFNN model
def build_ffnn(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train FFNN
print("Training FFNN model...")
ffnn = build_ffnn((X_train.shape[1],), y_train.shape[1])
history_ffnn = ffnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Train CNN
print("Training CNN model...")
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]
cnn = build_cnn((X_train.shape[1], 1), y_train.shape[1])
history_cnn = cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate models
print("Evaluating models...")
ffnn_score = ffnn.evaluate(X_test, y_test, verbose=0)
cnn_score = cnn.evaluate(X_test_cnn, y_test, verbose=0)

# Save performance metrics
with open('model_performance.txt', 'w') as f:
    f.write(f"FFNN Test Accuracy: {ffnn_score[1]:.4f}\n")
    f.write(f"CNN Test Accuracy: {cnn_score[1]:.4f}\n")
    f.write("\nClassification Report for FFNN:\n")
    y_pred_ffnn = np.argmax(ffnn.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    f.write(classification_report(y_test_classes, y_pred_ffnn, target_names=encoder.classes_))
    f.write("\nClassification Report for CNN:\n")
    y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)
    f.write(classification_report(y_test_classes, y_pred_cnn, target_names=encoder.classes_))

# Generate confusion matrix and classification report
y_pred_ffnn = np.argmax(ffnn.predict(X_test), axis=1)
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)

conf_matrix_ffnn = confusion_matrix(y_test_classes, y_pred_ffnn)
conf_matrix_cnn = confusion_matrix(y_test_classes, y_pred_cnn)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ffnn, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("FFNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("ffnn_confusion_matrix.png")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("cnn_confusion_matrix.png")

# Plot model performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_ffnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_ffnn.history['val_accuracy'], label='Validation Accuracy')
plt.title("FFNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_ffnn.history['loss'], label='Train Loss')
plt.plot(history_ffnn.history['val_loss'], label='Validation Loss')
plt.title("FFNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("ffnn_performance.png")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("cnn_performance.png")

print("Model training and evaluation complete.")