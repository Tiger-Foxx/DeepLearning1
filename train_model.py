import tensorflow as tf
import keras
import numpy as np

# 1. Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalisation des données (valeurs entre 0 et 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Redimensionnement des images (28x28 → 784)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 4. Construction du modèle fully connected
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# 5. Compilation du modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Entraînement du modèle
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 7. Évaluation sur le jeu de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test : {test_acc:.4f}")

# 8. Sauvegarde du modèle
model.save("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")
