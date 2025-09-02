import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour sauvegarde
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from filterpy.kalman import KalmanFilter

# Paramètres de la simulation
np.random.seed(42)
n_samples = 500
initial_prob = 0.01
peak_threshold = 0.8
alpha_ema = 0.2  # Facteur de lissage pour EMA
k_window = 10  # Taille de la fenêtre pour SMA

# Fonction pour détecter le premier pic
def find_first_peak(data, threshold=0.5):
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] > data[i+1] and data[i] > threshold:
            return i, data[i]  # Index et valeur du premier pic
    return None, None  # Aucun pic trouvé

# Fonction pour générer les probabilités
def generate_probabilities(n_samples, initial_prob, peak_threshold):
    probs = np.zeros(n_samples)
    probs[0] = initial_prob
    for i in range(1, n_samples):
        if probs[i-1] < 0.3:
            probs[i] = probs[i-1] + np.random.uniform(0, 0.02)
        elif probs[i-1] >= peak_threshold and np.random.random() < 0.3:
            probs[i] = probs[i-1] - np.random.uniform(0.2, 0.5)
        else:
            probs[i] = probs[i-1] + np.random.uniform(-0.05, 0.1)
        probs[i] = np.clip(probs[i], 0, 1)
    return probs

# Filtre de Kalman
def apply_kalman_filter(data, index):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data[0]], [0]])
    kf.F = np.array([[1, 1], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 1000
    kf.R = 0.01
    kf.Q = np.array([[0.001, 0], [0, 0.001]])
    smoothed = np.zeros(index + 1)
    for i in range(index + 1):
        kf.predict()
        kf.update(np.array([[data[i]]]))
        smoothed[i] = kf.x[0, 0]
    return smoothed

# EMA avec réinitialisation au premier pic
def apply_ema(data, alpha, index, peak_index=None, peak_value=None):
    ema = np.zeros(index + 1)
    ema[0] = data[0]
    for i in range(1, index + 1):
        if peak_index is not None and i == peak_index:
            ema[i] = peak_value  # Réinitialiser l'EMA au pic
        else:
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

# SMA avec fenêtre de k points
def apply_sma(data, index, k, peak_index=None, peak_value=None):
    sma = np.zeros(index + 1)
    sma[0] = data[0]
    for i in range(1, index + 1):
        if peak_index is not None and i == peak_index:
            sma[i] = peak_value  # Réinitialiser au pic
        else:
            start = max(0, i - k + 1)
            sma[i] = np.mean(data[start:i+1])
    return sma

# Générer les données
probs = generate_probabilities(n_samples, initial_prob, peak_threshold)

# Trouver le premier pic
peak_index, peak_value = find_first_peak(probs)
if peak_index is not None:
    print(f"Premier pic détecté à l'index {peak_index} avec la valeur {peak_value:.3f}")
else:
    print("Aucun pic détecté")

# Configuration de l'animation
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, n_samples)
ax.set_ylim(0, 1)
ax.set_xlabel('Temps')
ax.set_ylabel('Probabilité')
ax.set_title('Lissage des probabilités avec Kalman, EMA et SMA (Animation)')
ax.grid(True)

# Marquer le premier pic
if peak_index is not None:
    ax.plot(peak_index, peak_value, 'ro', label='Premier pic')

# Initialisation des lignes
line_raw, = ax.plot([], [], label='Probabilités brutes', alpha=0.5)
line_kalman, = ax.plot([], [], label='Filtre de Kalman', linewidth=2)
line_ema, = ax.plot([], [], label=f'EMA (alpha={alpha_ema})', linewidth=2)
line_sma, = ax.plot([], [], label=f'SMA (k={k_window})', linewidth=2)
ax.legend()

# Fonction d'initialisation pour l'animation
def init():
    line_raw.set_data([], [])
    line_kalman.set_data([], [])
    line_ema.set_data([], [])
    line_sma.set_data([], [])
    return line_raw, line_kalman, line_ema, line_sma

# Fonction de mise à jour pour l'animation
def update(frame):
    x = np.arange(frame + 1)
    line_raw.set_data(x, probs[:frame + 1])
    line_kalman.set_data(x, apply_kalman_filter(probs, frame))
    line_ema.set_data(x, apply_ema(probs, alpha_ema, frame, peak_index, peak_value))
    line_sma.set_data(x, apply_sma(probs, frame, k_window, peak_index, peak_value))
    return line_raw, line_kalman, line_ema, line_sma

# Créer l'animation
ani = FuncAnimation(fig, update, frames=n_samples, init_func=init, blit=True, interval=50)

# Sauvegarder l'animation comme vidéo
ani.save('probability_animation.mp4', writer='ffmpeg', fps=20)
print("Animation sauvegardée dans probability_animation.mp4")

# Sauvegarder une image statique finale
plt.figure(figsize=(12, 6))
plt.plot(probs, label='Probabilités brutes', alpha=0.5)
#plt.plot(apply_kalman_filter(probs, n_samples-1), label='Filtre de Kalman', linewidth=2)
plt.plot(apply_ema(probs, alpha_ema, n_samples-1, peak_index, peak_value), label=f'EMA (alpha={alpha_ema})', linewidth=2)
plt.plot(apply_sma(probs, n_samples-1, k_window, peak_index, peak_value), label=f'SMA (k={k_window})', linewidth=2)
if peak_index is not None:
    plt.plot(peak_index, peak_value, 'ro', label='Premier pic')
plt.xlabel('Temps')
plt.ylabel('Probabilité')
plt.title('Lissage des probabilités avec EMA et SMA')
plt.legend()
plt.grid(True)
plt.savefig('final_plot.png')
print("Image statique sauvegardée dans final_plot.png")