"""Train lightweight Perception AI MLP on synthetic multimodal terrain samples.

Generates realistic multimodal samples (color, texture, stereo geometry)
mimicking outdoor flood disaster environments (traversable dirt/gravel,
obstacles/boulders, murky flood water/mud, vegetation) and optimizes
the 3-layer MLP using cross-entropy with Adam optimizer in pure NumPy.
"""
from pathlib import Path
import numpy as np
from drishti.perception import (
    CLASS_TRAVERSABLE, CLASS_OBSTACLE, CLASS_WATER_MUD, CLASS_VEGETATION,
    DEFAULT_WEIGHTS_PATH, PerceptionModel
)


def generate_synthetic_dataset(num_samples=12000, seed=26126):
    rng = np.random.default_rng(seed)
    X = []
    y = []

    per_class = num_samples // 4

    # 1. TRAVERSABLE: dirt, gravel, dry path
    # Hue: earthy 10-28, moderate sat 0.2-0.6, val 0.3-0.8
    # Texture: moderate edge energy, low/moderate roughness
    # Depth: valid_frac high, dz_dx low, height_proxy close to 0
    for _ in range(per_class):
        h_m = rng.uniform(0.04, 0.12)
        h_s = rng.uniform(0.02, 0.08)
        s_m = rng.uniform(0.25, 0.60)
        s_s = rng.uniform(0.05, 0.15)
        v_m = rng.uniform(0.35, 0.75)
        v_s = rng.uniform(0.05, 0.18)

        h_hist = np.zeros(6, dtype=np.float32)
        h_hist[0] = 0.85; h_hist[1] = 0.15
        s_hist = np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32)
        v_hist = np.array([0.1, 0.3, 0.5, 0.1], dtype=np.float32)

        mag_mean = rng.uniform(0.10, 0.30)
        mag_std = rng.uniform(0.08, 0.25)
        lap_var = rng.uniform(0.05, 0.25)

        gr_ratio = rng.uniform(0.24, 0.35)
        br_ratio = rng.uniform(0.20, 0.30)

        valid_frac = rng.uniform(0.70, 1.0)
        z_mean = rng.uniform(0.2, 0.8)
        z_std = rng.uniform(0.02, 0.15)
        dz_dx = rng.uniform(-0.15, 0.15)
        dz_dy = rng.uniform(-0.25, 0.10)
        plane_rough = z_std
        height_proxy = rng.uniform(-0.10, 0.10)

        feat = np.array([
            h_m, h_s, s_m, s_s, v_m, v_s,
            *h_hist, *s_hist, *v_hist,
            mag_mean, mag_std, lap_var,
            gr_ratio, br_ratio,
            valid_frac, z_mean, z_std, dz_dx, dz_dy, plane_rough, height_proxy
        ], dtype=np.float32)
        X.append(feat)
        y.append(CLASS_TRAVERSABLE)

    # 2. OBSTACLE: boulders, rocks, debris, damaged structure rubble
    # Texture: high edge energy, high roughness
    # Depth: step jump in depth, large height_proxy, large z_std
    for _ in range(per_class):
        h_m = rng.uniform(0.05, 0.22)
        h_s = rng.uniform(0.05, 0.20)
        s_m = rng.uniform(0.10, 0.45)
        s_s = rng.uniform(0.08, 0.25)
        v_m = rng.uniform(0.20, 0.65)
        v_s = rng.uniform(0.10, 0.30)

        h_hist = rng.dirichlet(np.ones(6)).astype(np.float32)
        s_hist = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        v_hist = np.array([0.2, 0.4, 0.3, 0.1], dtype=np.float32)

        mag_mean = rng.uniform(0.40, 0.90)
        mag_std = rng.uniform(0.30, 0.80)
        lap_var = rng.uniform(0.40, 1.20)

        gr_ratio = rng.uniform(0.28, 0.36)
        br_ratio = rng.uniform(0.26, 0.36)

        valid_frac = rng.uniform(0.60, 1.0)
        z_mean = rng.uniform(0.15, 0.7)
        z_std = rng.uniform(0.35, 1.20)
        dz_dx = rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
        dz_dy = rng.uniform(0.3, 1.5)
        plane_rough = z_std
        height_proxy = rng.uniform(0.4, 1.5)

        feat = np.array([
            h_m, h_s, s_m, s_s, v_m, v_s,
            *h_hist, *s_hist, *v_hist,
            mag_mean, mag_std, lap_var,
            gr_ratio, br_ratio,
            valid_frac, z_mean, z_std, dz_dx, dz_dy, plane_rough, height_proxy
        ], dtype=np.float32)
        X.append(feat)
        y.append(CLASS_OBSTACLE)

    # 3. WATER_MUD: flood puddles, muddy water pools, slush
    # Color: dark murky brown-blue or specular, low sat or dark
    # Texture: extremely smooth (low edge, low lap_var)
    # Depth: low valid_frac (specular surface) or uniform flat surface below bank
    for _ in range(per_class):
        h_m = rng.uniform(0.08, 0.30)
        h_s = rng.uniform(0.01, 0.05)
        s_m = rng.uniform(0.15, 0.50)
        s_s = rng.uniform(0.02, 0.08)
        v_m = rng.uniform(0.15, 0.45)
        v_s = rng.uniform(0.02, 0.08)

        h_hist = np.zeros(6, dtype=np.float32)
        h_hist[0] = 0.3; h_hist[1] = 0.4; h_hist[2] = 0.3
        s_hist = np.array([0.3, 0.5, 0.2, 0.0], dtype=np.float32)
        v_hist = np.array([0.5, 0.4, 0.1, 0.0], dtype=np.float32)

        mag_mean = rng.uniform(0.02, 0.12)
        mag_std = rng.uniform(0.01, 0.08)
        lap_var = rng.uniform(0.005, 0.06)

        gr_ratio = rng.uniform(0.32, 0.40)
        br_ratio = rng.uniform(0.35, 0.50)

        valid_frac = rng.uniform(0.15, 0.65)  # water causes partial disparity dropouts
        z_mean = rng.uniform(0.3, 0.9)
        z_std = rng.uniform(0.01, 0.08)
        dz_dx = rng.uniform(-0.05, 0.05)
        dz_dy = rng.uniform(-0.05, 0.05)
        plane_rough = z_std
        height_proxy = rng.uniform(-0.4, -0.05)  # depression/channel

        feat = np.array([
            h_m, h_s, s_m, s_s, v_m, v_s,
            *h_hist, *s_hist, *v_hist,
            mag_mean, mag_std, lap_var,
            gr_ratio, br_ratio,
            valid_frac, z_mean, z_std, dz_dx, dz_dy, plane_rough, height_proxy
        ], dtype=np.float32)
        X.append(feat)
        y.append(CLASS_WATER_MUD)

    # 4. VEGETATION: shrubs, grass, trees, bushes
    # Hue: green ~35-85 -> (0.20-0.45), high saturation, high gr_ratio
    for _ in range(per_class):
        h_m = rng.uniform(0.18, 0.45)
        h_s = rng.uniform(0.02, 0.12)
        s_m = rng.uniform(0.35, 0.90)
        s_s = rng.uniform(0.05, 0.20)
        v_m = rng.uniform(0.25, 0.75)
        v_s = rng.uniform(0.05, 0.22)

        h_hist = np.zeros(6, dtype=np.float32)
        h_hist[1] = 0.50; h_hist[2] = 0.45; h_hist[3] = 0.05
        s_hist = np.array([0.05, 0.25, 0.50, 0.20], dtype=np.float32)
        v_hist = np.array([0.10, 0.35, 0.45, 0.10], dtype=np.float32)

        mag_mean = rng.uniform(0.02, 0.65)
        mag_std = rng.uniform(0.02, 0.55)
        lap_var = rng.uniform(0.02, 0.85)

        gr_ratio = rng.uniform(0.48, 1.0)
        br_ratio = rng.uniform(0.10, 0.30)

        valid_frac = rng.uniform(0.50, 0.90)
        z_mean = rng.uniform(0.25, 0.8)
        z_std = rng.uniform(0.15, 0.60)
        dz_dx = rng.uniform(-0.3, 0.3)
        dz_dy = rng.uniform(-0.1, 0.6)
        plane_rough = z_std
        height_proxy = rng.uniform(0.1, 0.8)

        feat = np.array([
            h_m, h_s, s_m, s_s, v_m, v_s,
            *h_hist, *s_hist, *v_hist,
            mag_mean, mag_std, lap_var,
            gr_ratio, br_ratio,
            valid_frac, z_mean, z_std, dz_dx, dz_dy, plane_rough, height_proxy
        ], dtype=np.float32)
        X.append(feat)
        y.append(CLASS_VEGETATION)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Add slight noise to avoid overfitting
    noise = rng.normal(0, 0.02, X.shape).astype(np.float32)
    X = np.clip(X + noise, -3.0, 3.0)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train(weights_out=DEFAULT_WEIGHTS_PATH, epochs=30, lr=0.001, batch_size=64):
    print("Generating synthetic multimodal perception dataset...")
    X, y = generate_synthetic_dataset(num_samples=16000)

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    in_dim = 32
    h1 = 64
    h2 = 32
    out_dim = 4

    rng = np.random.default_rng(42)
    W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, h1)).astype(np.float32)
    b1 = np.zeros(h1, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2)).astype(np.float32)
    b2 = np.zeros(h2, dtype=np.float32)
    W3 = rng.normal(0, np.sqrt(2.0 / h2), (h2, out_dim)).astype(np.float32)
    b3 = np.zeros(out_dim, dtype=np.float32)

    # Momentum SGD with weight decay for clean, robust convergence
    vW1, vb1 = np.zeros_like(W1), np.zeros_like(b1)
    vW2, vb2 = np.zeros_like(W2), np.zeros_like(b2)
    vW3, vb3 = np.zeros_like(W3), np.zeros_like(b3)
    momentum = 0.9
    weight_decay = 1e-4

    N = len(X_train)
    num_batches = N // batch_size

    for epoch in range(epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0

        for b in range(num_batches):
            b_idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_train[b_idx]
            yb = y_train[b_idx]

            # Forward
            a1 = xb @ W1 + b1
            h1_act = np.maximum(0.0, a1)

            a2 = h1_act @ W2 + b2
            h2_act = np.maximum(0.0, a2)

            logits = h2_act @ W3 + b3
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits - logits_max)
            probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-8)

            # Cross entropy loss
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
            epoch_loss += loss

            # Backward
            dlogits = probs.copy()
            dlogits[np.arange(len(yb)), yb] -= 1.0
            dlogits /= len(yb)

            dW3 = h2_act.T @ dlogits + weight_decay * W3
            db3 = np.sum(dlogits, axis=0)

            dh2 = dlogits @ W3.T
            da2 = dh2 * (a2 > 0)
            dW2 = h1_act.T @ da2 + weight_decay * W2
            db2 = np.sum(da2, axis=0)

            dh1 = da2 @ W2.T
            da1 = dh1 * (a1 > 0)
            dW1 = xb.T @ da1 + weight_decay * W1
            db1 = np.sum(da1, axis=0)

            # Momentum updates
            vW1 = momentum * vW1 - lr * dW1; W1 += vW1
            vb1 = momentum * vb1 - lr * db1; b1 += vb1
            vW2 = momentum * vW2 - lr * dW2; W2 += vW2
            vb2 = momentum * vb2 - lr * db2; b2 += vb2
            vW3 = momentum * vW3 - lr * dW3; W3 += vW3
            vb3 = momentum * vb3 - lr * db3; b3 += vb3

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # Evaluate val accuracy
            z1 = np.maximum(0.0, X_val @ W1 + b1)
            z2 = np.maximum(0.0, z1 @ W2 + b2)
            val_logits = z2 @ W3 + b3
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.mean(val_preds == y_val) * 100.0
            print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {epoch_loss/num_batches:.4f} - Val Acc: {val_acc:.2f}%")

    model = PerceptionModel(weights_path=None)
    model.W1, model.b1 = W1, b1
    model.W2, model.b2 = W2, b2
    model.W3, model.b3 = W3, b3
    model.save_weights(weights_out)
    print(f"Saved optimized perception weights to: {weights_out}")
    return model


if __name__ == "__main__":
    train()
