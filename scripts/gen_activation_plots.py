"""Generate activation function plots for ai-kb/fundamentals/activation_functions.md"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import os

OUTPUT_DIR = os.path.expanduser("~/Documents/ai-kb/assets/activation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global style
BG_COLOR = "#1a1a2e"
GRID_COLOR = "#333355"
TEXT_COLOR = "#e0e0e0"
ZERO_LINE_COLOR = "#555577"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "font.size": 11,
    "font.family": ["Heiti TC", "Hei", "DejaVu Sans"],
})

COLORS = plt.cm.Set1.colors


# ── Activation functions ──

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_fn(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, lam=1.0507, alpha=1.6733):
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))

def gelu_deriv(x):
    phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return phi + x * pdf

def silu(x):
    return x * sigmoid(x)

def silu_deriv(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def mish(x):
    sp = np.log1p(np.exp(x))
    return x * np.tanh(sp)

def mish_deriv(x):
    sp = np.log1p(np.exp(x))
    tsp = np.tanh(sp)
    sig = sigmoid(x)
    return tsp + x * sig * (1 - tsp**2)


def add_zero_lines(ax):
    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=0.8)
    ax.axvline(x=0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=0.8)


# ── Plot 1: Classic Trio (Sigmoid, Tanh, ReLU) with derivatives ──

def plot_classic_trio():
    x = np.linspace(-6, 6, 500)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("经典三件套：Sigmoid / Tanh / ReLU", fontsize=15, fontweight="bold")

    funcs = [
        ("Sigmoid", sigmoid, sigmoid_deriv),
        ("Tanh", tanh_fn, tanh_deriv),
        ("ReLU", relu, relu_deriv),
    ]

    for col, (name, fn, fn_d) in enumerate(funcs):
        # Function
        ax = axes[0, col]
        ax.plot(x, fn(x), color=COLORS[col], linewidth=2.5)
        ax.set_title(f"{name}(x)", fontsize=13)
        ax.grid(True)
        add_zero_lines(ax)
        ax.set_ylim(-1.5, 2.0 if name == "ReLU" else 1.5)

        # Derivative
        ax = axes[1, col]
        ax.plot(x, fn_d(x), color=COLORS[col], linewidth=2.5, linestyle="-")
        ax.set_title(f"{name}'(x)", fontsize=13)
        ax.grid(True)
        add_zero_lines(ax)
        ax.set_ylim(-0.5, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "classic_trio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 2: ReLU Family ──

def plot_relu_family():
    x = np.linspace(-5, 5, 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("ReLU 家族", fontsize=15, fontweight="bold")

    members = [
        ("ReLU", relu(x)),
        ("LeakyReLU (α=0.1)", leaky_relu(x, 0.1)),
        ("PReLU (α=0.25)", prelu(x, 0.25)),
        ("ELU (α=1)", elu(x)),
        ("SELU", selu(x)),
    ]
    for i, (name, y) in enumerate(members):
        ax.plot(x, y, color=COLORS[i], linewidth=2.2, label=name)

    add_zero_lines(ax)
    ax.grid(True)
    ax.legend(fontsize=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "relu_family.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 3: Modern Smooth Activations (GELU, SiLU, Mish) with derivatives ──

def plot_modern_smooth():
    x = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("现代平滑激活：GELU / SiLU(Swish) / Mish", fontsize=15, fontweight="bold")

    funcs = [
        ("GELU", gelu(x), gelu_deriv(x)),
        ("SiLU/Swish", silu(x), silu_deriv(x)),
        ("Mish", mish(x), mish_deriv(x)),
    ]

    # Values
    ax = axes[0]
    for i, (name, y, _) in enumerate(funcs):
        ax.plot(x, y, color=COLORS[i], linewidth=2.5, label=name)
    ax.set_title("函数值 f(x)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True)
    add_zero_lines(ax)
    ax.set_ylim(-1, 5)

    # Derivatives
    ax = axes[1]
    for i, (name, _, yd) in enumerate(funcs):
        ax.plot(x, yd, color=COLORS[i], linewidth=2.5, label=f"{name}'")
    ax.set_title("导数 f'(x)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True)
    add_zero_lines(ax)
    ax.set_ylim(-0.5, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "modern_smooth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 4: GLU Family gate mechanism ──

def plot_glu_family():
    x = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("GLU 家族：门控机制对比（固定线性路 = x）", fontsize=15, fontweight="bold")

    # Simulate: linear path = x, gate path uses different activations
    linear = x  # xW component (identity for illustration)

    gates = [
        ("GLU: σ(x)", sigmoid(x)),
        ("SwiGLU: SiLU(x)", silu(x)),
        ("GeGLU: GELU(x)", gelu(x)),
    ]

    for i, (name, gate) in enumerate(gates):
        ax = axes[i]
        output = linear * gate
        ax.plot(x, linear, color="#666688", linewidth=1.5, linestyle="--", label="线性路 xW", alpha=0.6)
        ax.plot(x, gate, color=COLORS[i + 3], linewidth=2.0, linestyle=":", label="门控 g(xV)")
        ax.plot(x, output, color=COLORS[i], linewidth=2.5, label="输出 xW⊗g(xV)")
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True)
        add_zero_lines(ax)
        ax.set_ylim(-3, 6)
        ax.set_xlabel("x")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, "glu_family.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 5: All Comparison ──

def plot_all_comparison():
    x = np.linspace(-5, 5, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("激活函数全家福对比", fontsize=15, fontweight="bold")

    all_funcs = [
        ("Sigmoid", sigmoid(x)),
        ("Tanh", tanh_fn(x)),
        ("ReLU", relu(x)),
        ("LeakyReLU", leaky_relu(x, 0.1)),
        ("ELU", elu(x)),
        ("SELU", selu(x)),
        ("GELU", gelu(x)),
        ("SiLU/Swish", silu(x)),
        ("Mish", mish(x)),
    ]

    colors_extended = list(COLORS) + ["#00ff88", "#ff6699"]
    for i, (name, y) in enumerate(all_funcs):
        ax.plot(x, y, color=colors_extended[i % len(colors_extended)], linewidth=2.0, label=name)

    add_zero_lines(ax)
    ax.grid(True)
    ax.legend(fontsize=9, ncol=3, loc="upper left")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "all_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


if __name__ == "__main__":
    print("Generating activation function plots...")
    plot_classic_trio()
    plot_relu_family()
    plot_modern_smooth()
    plot_glu_family()
    plot_all_comparison()
    print(f"\nDone! All plots saved to {OUTPUT_DIR}/")
