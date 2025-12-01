import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=100, noise_factor=10):
    np.random.seed(42)
    X = 2 * np.random.rand(num_points, 1)
    y = 4 + 3 * X + np.random.randn(num_points, 1) * (noise_factor / 10.0)
    return X, y

def main():
    # 1. Data Preparation
    X, y = generate_data()
    
    # 2. Model Definition
    # In Keras, a single Dense layer with 1 unit and linear activation is equivalent to Linear Regression.
    # y = w * x + b
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    
    # 3. Compilation
    # We use SGD (Stochastic Gradient Descent) and MSE (Mean Squared Error)
    # to match our scratch implementation.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  loss='mean_squared_error')
    
    print("Starting TensorFlow Training...")
    
    # 4. Training
    history = model.fit(X, y, epochs=100, verbose=0)
    
    # 5. Results
    w, b = model.layers[0].get_weights()
    print(f"Final Loss: {history.history['loss'][-1]:.4f}")
    print(f"Learned Parameters: Intercept={b[0]:.2f}, Slope={w[0][0]:.2f}")
    
    # 6. Visualization (Optional comparison)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='orange', linewidth=2, label='TF Prediction')
    plt.title('TensorFlow Implementation')
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, 'tf_fit.png'))
    print("Saved tf_fit.png")

if __name__ == "__main__":
    main()
