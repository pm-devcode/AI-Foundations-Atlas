import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=100):
    np.random.seed(42)
    x0 = np.random.randn(num_points // 2, 2) + 2
    y0 = np.zeros((num_points // 2, 1))
    x1 = np.random.randn(num_points // 2, 2) + 6
    y1 = np.ones((num_points // 2, 1))
    X = np.vstack((x0, x1))
    y = np.vstack((y0, y1))
    return X, y

def main():
    X, y = generate_data()
    
    # Model: Dense(1) with sigmoid activation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Starting TensorFlow Training...")
    history = model.fit(X, y, epochs=100, verbose=0)
    
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Final Loss: {loss:.4f}, Accuracy: {acc*100:.1f}%")
    
    # Viz
    w, b = model.layers[0].get_weights()
    # w is shape (2, 1), b is shape (1,)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y.flatten()==0][:, 0], X[y.flatten()==0][:, 1], color='blue')
    plt.scatter(X[y.flatten()==1][:, 0], X[y.flatten()==1][:, 1], color='red')
    
    x1_vals = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    # w1*x1 + w2*x2 + b = 0  =>  x2 = -(b + w1*x1) / w2
    x2_vals = -(b[0] + w[0][0] * x1_vals) / w[1][0]
    
    plt.plot(x1_vals, x2_vals, "k--", linewidth=3, label='TF Boundary')
    plt.legend()
    plt.title('TensorFlow Logistic Regression')
    plt.savefig(os.path.join(ASSETS_DIR, 'tf_boundary.png'))
    print("Saved tf_boundary.png")

if __name__ == "__main__":
    main()
