import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'docs/lectures/1_vvedenie/1_1_neurons/img'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 13})

CAPTION_KW = dict(ha='center', fontsize=12, color='#455a64')

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)

# Рисунок 1. Нейрон (подпись снизу — как было)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10.6); ax.set_ylim(0.2, 6.4); ax.axis('off')
items = [('x1', 5.6), ('x2', 4.5), ('x3', 3.4), ('xn', 1.0)]
for label, y in items:
    ax.text(1, y, label, ha='center', va='center', fontsize=15,
            bbox=dict(boxstyle='round,pad=0.35', fc='#e3f2fd', ec='#1565c0', lw=1.5))
ax.text(1, 2.3, '...', ha='center', fontsize=18)
ax.text(1, 6.1, 'входы', ha='center', fontsize=11, color='#455a64')
sx, sy = 5.2, 3.3
ax.add_patch(plt.Circle((sx, sy), 0.45, fc='#fff3e0', ec='#e65100', lw=2))
ax.text(sx, sy, 'Σ', ha='center', va='center', fontsize=18, color='#e65100')
for (label, y), wl in zip(items, ['w1', 'w2', 'w3', 'wn']):
    ax.annotate('', xy=(sx - 0.45, sy), xytext=(1.45, y),
                arrowprops=dict(arrowstyle='->', lw=1.6, color='#37474f'))
    ax.text(3.0, (sy + y) / 2 + 0.2, wl, fontsize=14)
ax.annotate('', xy=(sx, sy - 0.45), xytext=(sx, 1.2),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#37474f'))
ax.text(sx + 0.18, 1.75, 'b', fontsize=15, style='italic')
ax.annotate('', xy=(6.7, sy), xytext=(sx + 0.45, sy),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#37474f'))
ax.text(6.05, sy + 0.28, 'z', fontsize=14, style='italic')
ax.text(7.55, sy, 'φ(z)', ha='center', va='center', fontsize=15,
        bbox=dict(boxstyle='round,pad=0.4', fc='#e8f5e9', ec='#2e7d32', lw=1.5))
ax.annotate('', xy=(9.8, sy), xytext=(8.6, sy),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#37474f'))
ax.text(9.95, sy, 'y', fontsize=16, va='center', style='italic')
ax.text(5.3, 0.35, 'Рисунок 1. Математическая модель искусственного нейрона', **CAPTION_KW)
plt.tight_layout()
plt.savefig(f'{OUT}/neuron.png', dpi=150, bbox_inches='tight')
plt.close()

# Рисунок 2. MLP — подпись перенесена вниз
fig, ax = plt.subplots(figsize=(10, 6.5)); ax.axis('off')
ax.set_xlim(-0.8, 12.3); ax.set_ylim(-2.6, 5.6)
layers = [(0.4, 5, 'Входной слой', '784 нейрона'),
          (4.2, 6, 'Скрытый слой', '128 нейронов'),
          (8.0, 6, 'Скрытый слой', '64 нейрона'),
          (11.6, 3, 'Выходной слой', '10 нейронов (softmax)')]
pos = []
for x, n, _, _ in layers:
    ys = np.linspace(3 - (n - 1) * 0.6, 3 + (n - 1) * 0.6, n) if n > 1 else [3]
    pos.append([(x, y) for y in ys])
for a, b in zip(pos[:-1], pos[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b:
            ax.plot([x1, x2], [y1, y2], color='#90a4ae', lw=0.6, alpha=0.35, zorder=1)
for (x, n, t1, t2), pts in zip(layers, pos):
    for (px, py) in pts:
        ax.add_patch(plt.Circle((px, py), 0.27, fc='#ffffff', ec='#1565c0', lw=1.8, zorder=2))
    ax.text(x, -1.05, t1, ha='center', fontsize=12, weight='bold')
    ax.text(x, -1.75, t2, ha='center', fontsize=11, color='#455a64')
ax.text(5.75, -2.45, 'Рисунок 2. Полносвязная сеть (MLP)', **CAPTION_KW)
plt.tight_layout()
plt.savefig(f'{OUT}/mlp.png', dpi=150, bbox_inches='tight')
plt.close()

# Рисунок 3. Активации — подпись перенесена вниз
x = np.linspace(-4, 4, 500)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
panels = [('Sigmoid', sigmoid(x)), ('Tanh', np.tanh(x)),
          ('ReLU', relu(x)), ('Leaky ReLU', np.where(x > 0, x, 0.1 * x))]
for (title, yv), axi in zip(panels, axes.ravel()):
    axi.plot(x, yv, lw=2.5, color='#1565c0')
    axi.axhline(0, color='gray', lw=0.8); axi.axvline(0, color='gray', lw=0.8)
    axi.set_title(title, fontsize=14); axi.grid(alpha=0.3)
fig.text(0.5, 0.005, 'Рисунок 3. Основные функции активации', **CAPTION_KW)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(f'{OUT}/activations.png', dpi=150, bbox_inches='tight')
plt.close()

# Рисунок 4. Градиенты — подпись перенесена вниз
xg = np.linspace(-6, 6, 500)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(xg, sigmoid(xg) * (1 - sigmoid(xg)), lw=2.5, color='#6a1b9a')
axes[0].set_title('Производная сигмоиды', fontsize=13)
axes[0].grid(alpha=0.3); axes[0].set_ylim(-0.02, 0.3)
axes[1].plot(xg, (xg > 0).astype(float), lw=2.5, color='#2e7d32', label='ReLU')
axes[1].plot(xg, np.where(xg > 0, 1.0, 0.1), lw=2.0, ls='--', color='#ef6c00', label='Leaky ReLU')
axes[1].set_title('Производные ReLU / Leaky ReLU', fontsize=13)
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=11); axes[1].set_ylim(-0.1, 1.3)
fig.text(0.5, 0.005, 'Рисунок 4. Сравнение поведения градиента', **CAPTION_KW)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(f'{OUT}/gradients.png', dpi=150, bbox_inches='tight')
plt.close()

print('Готово:', sorted(os.listdir(OUT)))
