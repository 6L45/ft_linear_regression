import numpy as np
import sys
import matplotlib.pyplot as plt
import shared


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2


def linear_regression(data):
    x = data[:, 0]
    y = data[:, 1]

    # Normalisation
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Init params
    a = np.random.randn()
    b = np.random.randn()
    lr = 0.01
    epoch = 300

    #history = []

    # Préparation figure double (2D + 3D)
    plt.ion()
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1 : Régression linéaire (2D)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x, y, label='data')
    line, = ax1.plot(x, y, color='red', label='linear regression')
    ax1.set_xlabel('Km')
    ax1.set_ylabel('Price')
    ax1.set_title('Linear Regression')
    ax1.legend()

    # Subplot 2 : Surface 3D du loss + point de descente
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('Gradient descent on loss surface')
    ax2.set_xlabel('theta0 (b)')
    ax2.set_ylabel('theta1 (a)')
    ax2.set_zlabel('Loss')

    # Pré-calcul surface de perte
    theta0_vals = np.linspace(-3, 3, 100)
    theta1_vals = np.linspace(-3, 3, 100)
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
    Loss = np.zeros_like(T0)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            pred = T1[i, j] * x_norm + T0[i, j]
            Loss[i, j] = ((pred - y_norm) ** 2).mean()
    ax2.plot_surface(T0, T1, Loss, cmap='viridis', alpha=0.6)

    point = ax2.plot([], [], [], 'ro')[0]

    for i in range(epoch):
        # Prédiction et loss
        y_pred = a * x_norm + b
        error = y_pred - y_norm
        loss = (error ** 2).mean()
        print(f"Epoch {i+1} - Loss: {loss:.6f}")

        # Gradients
        da = 2 * (error * x_norm).mean()
        db = 2 * error.mean()

        # Update
        a -= lr * da
        b -= lr * db

        # Sauvegarde pour tracé
        #history.append((b, a, loss))

        # Conversion pour le tracé réel
        a_real = a * y_std / x_std
        b_real = y_mean + y_std * b - a_real * x_mean
        y_pred_real = a_real * x + b_real

        # Update du subplot 1 (droite)
        line.set_ydata(y_pred_real)

        # Update du subplot 2 (point 3D sur surface)
        point.set_data([b], [a])
        point.set_3d_properties([loss])

        plt.pause(0.01)

    plt.ioff()
    print("\nq to quit graphs")
    plt.show()

    return (a_real, b_real)


def main(file = None):
    shared.data = np.loadtxt(file, delimiter=',', skiprows=1)
    (shared.theta0, shared.theta1) = linear_regression(shared.data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main("data.csv")
    else:
        main(sys.argv[1])
