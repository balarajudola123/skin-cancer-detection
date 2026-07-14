import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Convert dataset images to .npz
def images_to_npz(data_dir, img_size=(64, 64), save_path='real_fake_data.npz'):
    images = []
    labels = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for f in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, f)
            try:
                with Image.open(fpath) as img:
                    img = img.convert('RGB')
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(cls)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    np.savez_compressed(save_path, images=images, labels=labels_encoded, classes=le.classes_)
    print(f"Saved {len(images)} images and labels to {save_path}")

# Load .npz dataset
def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['images'], data['labels'], data['classes']

# For traditional ML models
def prepare_ml_data(X):
    return X.reshape(X.shape[0], -1)

# ANN model
def create_ann(input_shape=(64, 64, 3), num_classes=2):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# CNN model
def create_cnn(input_shape=(64, 64, 3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training all models
def train_and_evaluate(X_train, X_test, y_train, y_test, classes):
    results = {}

    # SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf')
    svm.fit(prepare_ml_data(X_train), y_train)
    svm_preds = svm.predict(prepare_ml_data(X_test))
    results['SVM'] = accuracy_score(y_test, svm_preds)
    print(f"SVM accuracy: {results['SVM']:.4f}")

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(prepare_ml_data(X_train), y_train)
    rf_preds = rf.predict(prepare_ml_data(X_test))
    results['Random Forest'] = accuracy_score(y_test, rf_preds)
    print(f"Random Forest accuracy: {results['Random Forest']:.4f}")

    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb.fit(prepare_ml_data(X_train), y_train)
    xgb_preds = xgb.predict(prepare_ml_data(X_test))
    results['XGBoost'] = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost accuracy: {results['XGBoost']:.4f}")

    # ANN
    print("Training ANN...")
    ann = create_ann(input_shape=(64, 64, 3), num_classes=len(classes))
    ann.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    _, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
    results['ANN'] = ann_acc
    print(f"ANN accuracy: {ann_acc:.4f}")

    # CNN
    print("Training CNN...")
    cnn = create_cnn(input_shape=(64, 64, 3), num_classes=len(classes))
    cnn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    _, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
    results['CNN'] = cnn_acc
    print(f"CNN accuracy: {cnn_acc:.4f}")

    return results, cnn

# Plot and save results
def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison (Real vs Fake Images)')
    plt.ylabel('Accuracy')
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.savefig("model_comparison.png")  # ✅ Save the plot
    print("Saved accuracy plot as 'model_comparison.png'")
    plt.show()  # You can keep or remove this depending on IDLE behavior

# Image classification using CNN
def classify_image(img_path, model, classes, img_size=(64, 64)):
    img = Image.open(img_path).convert('RGB').resize(img_size)
    img_arr = np.array(img) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(input_arr)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    print(f"Predicted class: {classes[class_idx]}, Confidence: {confidence:.2f}")

# Main Execution
if __name__ == '__main__':
    dataset_path = r'D:\SAIRIGAPU GANESH\dataset\training'  # update this
    npz_file = 'real_fake_data.npz'
    image_size = (64, 64)

    print("Converting images to .npz format...")
    images_to_npz(dataset_path, img_size=image_size, save_path=npz_file)

    print("Loading dataset...")
    X, y, classes = load_data(npz_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training and evaluating models...")
    results, cnn_model = train_and_evaluate(X_train, X_test, y_train, y_test, classes)

    print("Saving CNN model and class labels...")
    cnn_model.save('real_fake_cnn_model.h5')
    np.save('real_fake_classes.npy', classes)
    print("Model and labels saved successfully.")

    print("Plotting comparison chart...")
    plot_results(results)

    # Classify a test image
    test_img_path = r'C:\Users\Bala raju\Downloads\SAIRIGAPU GANESH\dataset\test\real_01062.jpg'  # Update this
    if os.path.exists(test_img_path):
        print("Classifying single image...")
        classify_image(test_img_path, cnn_model, classes, image_size)
    else:
        print(f"Test image not found: {test_img_path}")
