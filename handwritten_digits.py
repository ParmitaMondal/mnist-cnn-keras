# --- Imports & seeds ---
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

# --- Data ---
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Show sample grid
import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
for k in range(9):
    plt.subplot(3,3,k+1)
    plt.tight_layout()
    plt.imshow(X_train[k], cmap='gray')
    plt.title(f"Digit: {y_train[k]}")
    plt.axis('off')
plt.show()

# Reshape & scale
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)  # (N,28,28,1)
X_test  = np.expand_dims(X_test,  -1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# --- Model ---
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Helpful callbacks
cb = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, monitor='val_loss')
]

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,  # smaller val split
    shuffle=True,
    callbacks=cb,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss:     {test_loss:.4f}")

# Plots
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.show()

# Prediction demo
probs = model.predict(X_test, verbose=0)
idx = 565
plt.figure()
plt.imshow(X_test[idx].squeeze(), cmap='gray'); plt.axis('off')
plt.title(f"Predicted: {np.argmax(probs[idx])}")
plt.show()
