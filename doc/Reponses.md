
# Responses.MD

**Practical Work:** From Design to Deployment of Deep Learning Mode3. **Model-specific metrics:** Prediction distribution drift, input data drift, outlier detection, and periodic accuracy checks if labels are available.


**End of Responses.MD**



# TP2 Responses

## Part 1: Theory and Key Concepts

### Data Splitting
- **Training set:** Used to update model parameters.
- **Validation (dev) set:** Used to tune hyperparameters and diagnose bias/variance.
- **Test set:** Used only for final evaluation to avoid overfitting.

### Results Analysis
- High training error and high validation error: high bias (underfitting).
- Low training error but high validation error: high variance (overfitting).

### L2 Regularization
Penalizes large weights by adding λ||w||² to the loss, where λ is the regularization strength, encouraging smaller weights and reducing overfitting.

### Dropout
Randomly sets a fraction of neurons to zero during training, preventing co-adaptation and acting as regularization.

### Batch Normalization
Normalizes the inputs to each layer, stabilizing training by reducing internal covariate shift and allowing higher learning rates.

### Optimization Algorithms
- **Momentum:** Accelerates SGD by adding a fraction of the previous update to the current one.
- **RMSprop:** Adapts learning rates by dividing by the root mean square of recent gradients.
- **Adam:** Combines momentum and RMSprop, providing adaptive learning rates; preferred as default due to robustness.

## Part 2: Practical Exercises

### Exercise 1: Bias/Variance Analysis
After training, training accuracy was ~0.98, validation ~0.97, indicating mild overfitting (high variance). The model performs well but could generalize better.

### Exercise 2: Applying Regularization
With L2 (0.001) and Dropout (0.2), validation loss decreased, improving generalization. Regularization reduced overfitting.

### Exercise 3: Comparing Optimizers
- Adam: Best performance, accuracy ~0.97, fast convergence.
- RMSprop: Good, slightly lower accuracy.
- SGD with momentum: Stable but slower, lower accuracy.

### Exercise 4: Batch Normalization
Adding BatchNormalization improved convergence speed and final accuracy by stabilizing activations.*M2-GI — Donfack Pascal Arthur**

---

## Part 1 — Foundations of Deep Learning

**Q1 (Part 1.1)** — Difference between batch gradient descent and stochastic gradient descent (SGD), and why SGD is preferred in deep learning:

* **Batch gradient descent** computes gradients using the entire dataset. Stable but slow and memory intensive.
* **Stochastic gradient descent (SGD)** computes gradients per sample (or mini-batch). Faster updates, can escape shallow local minima, and scales to large datasets.
* **Preference in deep learning:** Datasets and models are large, making full-batch impractical; SGD enables faster convergence and better generalization.

**Q2 (Part 1.1)** — Roles of layers and backpropagation:

* **Input layer:** receives raw data, shapes it for the network.
* **Hidden layers:** extract and transform features through weighted combinations and nonlinearities.
* **Output layer:** produces predictions in the desired format (e.g., probabilities).
* **Backpropagation:** computes gradients of the loss w\.r.t each parameter, propagating the error backward to update weights and improve predictions.

---

## Exercise 1 — Keras MNIST

**Q1** — Why Dense and Dropout layers are used, and why softmax for output:

* **Dense layers:** fully-connected, learn weighted combinations of features.
* **Dropout:** regularization to prevent overfitting by randomly disabling neurons during training.
* **Softmax:** converts logits to a probability distribution over classes; suitable for multi-class classification like MNIST.

**Q2** — Why Adam optimizer is an improvement over SGD:

* Adam adapts learning rates per parameter using first and second moments of gradients.
* Benefits: faster convergence, less sensitive to learning rate, handles sparse gradients, often better out-of-the-box than vanilla SGD.

**Q3** — Vectorization and batching in the code:

* **Vectorization:** operations are applied on whole arrays instead of element-wise loops, enabling GPU acceleration and faster computation.
* **Batching:** processes multiple samples per update (`batch_size=128`) for efficiency and stability of gradient estimates.

---

## Part 2 — Engineering Deep Learning

### Exercise 2 — Git / GitHub

No theoretical questions; practical commands are in the README.

---

### Exercise 3 — MLflow

**Q** — Purpose of MLflow tracking:

* **Parameters:** record hyperparameters for reproducibility.
* **Metrics:** monitor performance (loss, accuracy) across runs.
* **Artifacts:** store models and outputs for deployment or inspection.
* **Benefits:** enables comparison of experiments, model selection, reproducibility, and deployment readiness.

---

### Exercise 4 — Containerization & API

No explicit questions in this exercise; it focuses on practical Docker/Flask setup.

---

### Exercise 5 — CI/CD and Monitoring

**Q1** — How a CI/CD pipeline could automate Docker build and deployment:

* Trigger pipeline on code push.
* Build Docker image and run automated tests.
* Push image to registry (DockerHub, GCR, ECR).
* Deploy image to cloud service (Cloud Run, ECS) automatically.
* Optional: run smoke tests and roll back on failure.

**Q2** — Key monitoring indicators in production (minimum three types):

1. **Performance metrics:** request latency (p50/p95/p99), throughput, CPU/RAM/GPU usage.
2. **Reliability metrics:** error rate (4xx/5xx), crash/restart counts, timeouts.
3. **Model-specific metrics:** prediction distribution drift, input data drift, outlier detection, and periodic accuracy checks if labels are available.


**End of Responses.MD**


