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

    # Mode interactif
    plt.ion()
    fig = plt.figure(figsize=(18, 6))

    # === SUBPLOT 1 : RÉGRESSION LINÉAIRE ================================
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(x, y, label='data')
    ax1.set_xlabel('Km')
    ax1.set_ylabel('Price')
    ax1.set_title('Linear Regression')
    ax1.legend()
    line, = ax1.plot(x, y, color='red', label='linear regression')
    # === SUBPLOT 2 : SURFACE 3D DU LOSS =================================
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title('Gradient descent on loss surface')
    ax2.set_xlabel('theta0 (b)')
    ax2.set_ylabel('theta1 (a)')
    ax2.set_zlabel('Loss')
    # === SUBPLOT 3 : LOSS VS EPOCH =====================================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Loss Function")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_xlim(0, epoch)
    loss_history = []
    loss_line, = ax3.plot([], [], color='purple')

    # 2. Calculer l'erreur et la perte initiale (MSE) pour Y scale du graph
    y_pred_init = a * x_norm + b
    loss_init = ((y_pred_init - y_norm) ** 2).mean()
    y_max = loss_init * 1.10
    ax3.set_ylim(0, y_max)

    # Pré-calcul surface de perte (subplot 2)
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

    # === TRAINING LOOP ==================================================
    for i in range(epoch):
        # Prédiction + Loss
        y_pred = a * x_norm + b
        error = y_pred - y_norm
        loss = (error ** 2).mean()
        loss_history.append(loss)

        print(f"Epoch {i+1}/{epoch} - Loss: {loss:.6f}")

        # Gradients
        da = 2 * (error * x_norm).mean()
        db = 2 * error.mean()

        # Update params
        a -= lr * da
        b -= lr * db

        # Convert back to real scale
        a_real = a * y_std / x_std
        b_real = y_mean + y_std * b - a_real * x_mean
        y_pred_real = a_real * x + b_real


    # === Plots update ==================================================
        # Update subplot 1
        line.set_ydata(y_pred_real)

        # Update subplot 2 (point)
        point.set_data([b], [a])
        point.set_3d_properties([loss])

        # Update subplot 3 (loss curve)
        loss_line.set_data(range(len(loss_history)), loss_history)
        ax3.relim()
        ax3.autoscale_view()

        plt.pause(0.01)

    plt.ioff()
    print("\nq to quit graphs")
    plt.show()

    return (a_real, b_real)


def main(file=None):
    shared.data = np.loadtxt(file, delimiter=',', skiprows=1)
    (shared.theta0, shared.theta1) = linear_regression(shared.data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main("data.csv")
    else:
        main(sys.argv[1])

