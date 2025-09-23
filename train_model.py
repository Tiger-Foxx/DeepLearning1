import mlflow
import mlflow.tensorflow  # ou mlflow.keras (les deux fonctionnent)
import tensorflow as tf
from mlflow.models.signature import infer_signature

# Variables pour les paramètres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Chargement des données MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Construction du modèle (utilisation d'Input pour éviter le warning)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),     # plus propre que passer input_shape à Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Lancement de la session de suivi MLflow
with mlflow.start_run():
    # Enregistrement des paramètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    # Entraînement
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        verbose=2)

    # Évaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Enregistrement des métriques
    mlflow.log_metric("test_accuracy", float(test_acc))
    mlflow.log_metric("test_loss", float(test_loss))

    # Éviter le warning sur la signature : inférer et logger la signature
    sample_input = x_test[:10]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    # Enregistrement du modèle (avec signature)
    mlflow.keras.log_model(model, "mnist-model", signature=signature)

    print(f"Test accuracy: {test_acc:.4f}")
