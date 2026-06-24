"""
Surrogate trainer — Paradigm A (ODE-based) and Paradigm B (input-output).

Paradigm A trainers (GP, NN)
  fit(model_class, *state_arrays, u, target_highest_deriv, ...)
  Predictor interface:  predict(*states, u) → acceleration scalar/array
  Used by ODE integration surrogate.

Paradigm B trainers (NARX, RNN, Transformer)
  fit_narx / fit_rnn / fit_transformer
  Predictor interface:  predict_sequence(y_init, u_seq) → y trajectory
  Used by rolling-prediction validation simulator.

All predictors are picklable (weights stored as numpy arrays where torch is used).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.surrogate.model_class_selector import ModelClass

MAX_GP_POINTS = 400   # subsample to this when GP is selected and N > MAX_GP_POINTS


# ══════════════════════════════════════════════════════════════════════════════
# Paradigm A — ODE-based predictor (generalised for any system order)
# ══════════════════════════════════════════════════════════════════════════════

class SurrogatePredictor:
    """
    Picklable callable that wraps either a _NumpyGP or a numpy-weights MLP.

    Interface (variable-arg, works for any system order):
        predict(*states, u)                  → float | np.ndarray
        predict_with_std(*states, u)         → (mean, std)

    For a 2nd-order system: predict(theta, theta_dot, u)  — backward-compatible.
    For a 3rd-order system: predict(q, q_dot, q_ddot, u).
    The last positional argument is always the input u.
    """

    def __init__(
        self,
        model_class: ModelClass,
        gp:          Optional["_NumpyGP"] = None,
        mlp_state:   Optional[Dict[str, np.ndarray]] = None,
        X_mean:      Optional[np.ndarray] = None,
        X_std:       Optional[np.ndarray] = None,
        y_mean:      float = 0.0,
        y_std:       float = 1.0,
    ):
        self._cls    = model_class
        self._gp     = gp
        self._mlp    = mlp_state
        self._X_mean = X_mean
        self._X_std  = X_std
        self._y_mean = float(y_mean)
        self._y_std  = float(y_std)

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, *args: "float | np.ndarray") -> "float | np.ndarray":
        """
        Predict the highest-order derivative from (state0, state1, ..., u).

        Works for any number of state variables — the last arg is always u.
        Scalars return a float; arrays return an ndarray.
        """
        scalar = all(np.isscalar(a) for a in args)
        X      = self._stack(*args)
        Xm, Xs = self._normalise_params()
        X_n    = (X - Xm) / (Xs + 1e-8)
        y_n    = self._predict_normalised(X_n)
        y      = y_n * self._y_std + self._y_mean
        return float(y[0]) if scalar else y

    def predict_with_std(
        self, *args: "float | np.ndarray"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays; std is GP posterior or heuristic for NN."""
        X = self._stack(*args)
        Xm, Xs = self._normalise_params()
        X_n = (X - Xm) / (Xs + 1e-8)

        if self._cls == ModelClass.GP:
            assert self._gp is not None
            mu_n, std_n = self._gp.predict(X_n, return_std=True)
        else:
            mu_n  = self._predict_normalised(X_n)
            std_n = np.full_like(mu_n, 0.1)   # heuristic for NN

        return mu_n * self._y_std + self._y_mean, std_n * self._y_std

    # ── Private ───────────────────────────────────────────────────────────────

    def _normalise_params(self) -> Tuple[np.ndarray, np.ndarray]:
        Xm = self._X_mean if self._X_mean is not None else np.zeros(3)
        Xs = self._X_std  if self._X_std  is not None else np.ones(3)
        return Xm, Xs

    @staticmethod
    def _stack(*args) -> np.ndarray:
        return np.column_stack([
            np.atleast_1d(np.asarray(a, dtype=float)) for a in args
        ])

    def _predict_normalised(self, X_n: np.ndarray) -> np.ndarray:
        if self._cls == ModelClass.GP:
            assert self._gp is not None
            return self._gp.predict(X_n)
        else:
            return self._mlp_forward(X_n)

    def _mlp_forward(self, X_n: np.ndarray) -> np.ndarray:
        s = self._mlp
        x = X_n
        x = np.tanh(x @ s["w1"].T + s["b1"])
        x = np.tanh(x @ s["w2"].T + s["b2"])
        x = x @ s["w3"].T + s["b3"]
        return x.ravel()


# ══════════════════════════════════════════════════════════════════════════════
# Paradigm B — input-output sequence predictors
# ══════════════════════════════════════════════════════════════════════════════

class NARXPredictor:
    """
    NARX (Nonlinear AutoRegressive with eXogenous inputs) predictor.

    y(k) = f(y(k-1), ..., y(k-lag_y), u(k), ..., u(k-lag_u+1))

    Fitted with a GP or numpy MLP on the lag-feature matrix.
    Interface: predict_sequence(y_init, u_seq) → trajectory
    """

    def __init__(
        self,
        lag_y:   int,
        lag_u:   int,
        model:   object,               # _NumpyGP or mlp_state dict
        mc:      ModelClass,
        X_mean:  np.ndarray,
        X_std:   np.ndarray,
        y_mean:  float,
        y_std:   float,
    ):
        self.lag_y  = lag_y
        self.lag_u  = lag_u
        self._mc    = mc
        self._model = model
        self._X_mean = X_mean
        self._X_std  = X_std
        self._y_mean = float(y_mean)
        self._y_std  = float(y_std)

    def predict_one(self, y_buf: np.ndarray, u_buf: np.ndarray) -> float:
        """Predict next output from lag buffers."""
        X = np.concatenate([y_buf, u_buf]).reshape(1, -1)
        X_n = (X - self._X_mean) / (self._X_std + 1e-8)
        y_n = self._forward(X_n)
        return float(y_n[0] * self._y_std + self._y_mean)

    def predict_sequence(self, y_init: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """
        Roll out NARX prediction from an initial output history.

        y_init : (lag_y,) — initial output values (most recent last)
        u_seq  : (T,)     — future input sequence
        Returns : (T,) — predicted output trajectory
        """
        y_buf = list(np.asarray(y_init, dtype=float)[-self.lag_y:])
        if len(y_buf) < self.lag_y:
            y_buf = [0.0] * (self.lag_y - len(y_buf)) + y_buf
        u_buf = [0.0] * self.lag_u

        preds = []
        for u_k in np.asarray(u_seq, dtype=float):
            u_buf = u_buf[1:] + [float(u_k)]
            y_next = self.predict_one(np.array(y_buf), np.array(u_buf))
            preds.append(y_next)
            y_buf = y_buf[1:] + [y_next]

        return np.array(preds)

    def _forward(self, X_n: np.ndarray) -> np.ndarray:
        if self._mc == ModelClass.GP:
            return self._model.predict(X_n)
        else:
            s = self._model
            x = X_n
            x = np.tanh(x @ s["w1"].T + s["b1"])
            x = np.tanh(x @ s["w2"].T + s["b2"])
            x = x @ s["w3"].T + s["b3"]
            return x.ravel()


class RNNPredictor:
    """
    LSTM-based recurrent predictor.  Weights stored as numpy for pickle safety.

    Interface: predict_sequence(y_init, u_seq) → trajectory
    """

    def __init__(
        self,
        weights: Dict[str, np.ndarray],
        y_mean:  float,
        y_std:   float,
        u_mean:  float,
        u_std:   float,
    ):
        self._w      = weights
        self._y_mean = float(y_mean)
        self._y_std  = float(y_std)
        self._u_mean = float(u_mean)
        self._u_std  = float(u_std)

    def predict_sequence(self, y_init: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        w = self._w
        hidden_size = w["W_o"].shape[0]

        h = np.zeros(hidden_size)
        c = np.zeros(hidden_size)

        y_buf = float(np.asarray(y_init, dtype=float).ravel()[-1])
        preds = []

        for u_k in np.asarray(u_seq, dtype=float):
            y_n = (y_buf - self._y_mean) / (self._y_std + 1e-8)
            u_n = (float(u_k) - self._u_mean) / (self._u_std + 1e-8)
            x_t = np.array([y_n, u_n])

            h, c = self._lstm_step(x_t, h, c, w)
            y_next_n = float(np.dot(h, w["W_out"]) + np.asarray(w["b_out"]).ravel()[0])
            y_next   = float(y_next_n * self._y_std + self._y_mean)
            preds.append(y_next)
            y_buf = y_next

        return np.array(preds)

    @staticmethod
    def _lstm_step(x, h, c, w):
        xh = np.concatenate([x, h])
        i = _sigmoid(xh @ w["W_i"].T + w["b_i"])
        f = _sigmoid(xh @ w["W_f"].T + w["b_f"])
        g = np.tanh(xh @ w["W_g"].T + w["b_g"])
        o = _sigmoid(xh @ w["W_o"].T + w["b_o"])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new


class TransformerPredictor:
    """
    Causal (decoder-only) Transformer predictor.  Weights stored as numpy.

    Interface: predict_sequence(y_init, u_seq) → trajectory
    The context window is reset each call; past tokens are prepended from y_init/u_init.
    """

    def __init__(
        self,
        weights:      Dict[str, np.ndarray],
        context_len:  int,
        y_mean:       float,
        y_std:        float,
        u_mean:       float,
        u_std:        float,
    ):
        self._w          = weights
        self._ctx        = context_len
        self._y_mean     = float(y_mean)
        self._y_std      = float(y_std)
        self._u_mean     = float(u_mean)
        self._u_std      = float(u_std)

    def predict_sequence(self, y_init: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        w = self._w
        d_model = w["tok_emb"].shape[1]

        y_hist = list(np.asarray(y_init, dtype=float).ravel())
        u_hist = [0.0] * len(y_hist)

        preds = []
        for u_k in np.asarray(u_seq, dtype=float):
            # Build context tokens: (ctx, 2) with (y_n, u_n) pairs
            ctx_len = min(self._ctx, len(y_hist))
            y_ctx   = np.array(y_hist[-ctx_len:])
            u_ctx   = np.array(u_hist[-ctx_len:])

            y_n = (y_ctx - self._y_mean) / (self._y_std + 1e-8)
            u_n = (u_ctx - self._u_mean) / (self._u_std + 1e-8)

            tokens = np.column_stack([y_n, u_n])   # (ctx_len, 2)
            y_pred_n = self._forward(tokens, w)
            y_next   = float(y_pred_n * self._y_std + self._y_mean)

            preds.append(y_next)
            y_hist.append(y_next)
            u_hist.append(float(u_k))

        return np.array(preds)

    @staticmethod
    def _forward(tokens: np.ndarray, w: Dict[str, np.ndarray]) -> float:
        """
        Single-layer causal Transformer forward pass in numpy.
        tokens : (T, 2)   input token matrix
        Returns the next-step prediction (scalar) from the last position.
        """
        T, _ = tokens.shape
        d_model = w["tok_emb"].shape[1]

        # Token embedding
        x = tokens @ w["tok_emb"] + w["tok_emb_bias"]   # (T, d_model)

        # Sinusoidal positional encoding (precomputed or on-the-fly)
        pos = np.arange(T)[:, None]
        i   = np.arange(0, d_model, 2)
        pe  = np.zeros((T, d_model))
        pe[:, 0::2] = np.sin(pos / 10000 ** (i / d_model))
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(pos / 10000 ** (i / d_model))
        else:
            pe[:, 1::2] = np.cos(pos / 10000 ** (i[:-1] / d_model))
        x = x + pe

        # Single-head causal self-attention
        Q = x @ w["W_q"]   # (T, d_model)
        K = x @ w["W_k"]
        V = x @ w["W_v"]

        scale = d_model ** 0.5
        scores = Q @ K.T / scale   # (T, T)

        # Causal mask
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + mask
        attn = _softmax(scores, axis=-1)

        x = attn @ V                             # (T, d_model)
        x = x + tokens @ w["tok_emb"] + w["tok_emb_bias"]  # residual (approx)

        # Feed-forward
        x2 = np.tanh(x @ w["ff_w1"] + w["ff_b1"])
        x  = x + x2 @ w["ff_w2"] + w["ff_b2"]   # (T, d_model)

        # Output projection from last position
        return float(x[-1] @ w["out_w"] + w["out_b"])


# ══════════════════════════════════════════════════════════════════════════════
# Numpy RBF-GP (shared between Paradigm A and NARX GP)
# ══════════════════════════════════════════════════════════════════════════════

class _NumpyGP:
    """
    Exact numpy RBF-GP.  O(N³) — only for N ≤ MAX_GP_POINTS.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        sigma_f:      float = 1.0,
        sigma_n:      float = 0.05,
    ):
        self.ls      = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X_train: Optional[np.ndarray] = None
        self._alpha:  Optional[np.ndarray] = None
        self._K_inv:  Optional[np.ndarray] = None

    def _rbf(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return self.sigma_f ** 2 * np.exp(-0.5 * d2 / (self.ls ** 2 + 1e-12))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X.copy()
        K = self._rbf(X, X) + self.sigma_n ** 2 * np.eye(len(X))
        self._alpha = np.linalg.solve(K, y)
        self._K_inv = np.linalg.inv(K)

    def predict(
        self,
        X_new:      np.ndarray,
        return_std: bool = False,
    ) -> "np.ndarray | Tuple[np.ndarray, np.ndarray]":
        K_star = self._rbf(X_new, self.X_train)
        mu     = K_star @ self._alpha
        if not return_std:
            return mu
        K_ss = np.diag(self._rbf(X_new, X_new))
        var  = K_ss - np.sum((K_star @ self._K_inv) * K_star, axis=1)
        std  = np.sqrt(np.clip(var, 0.0, None))
        return mu, std


# ══════════════════════════════════════════════════════════════════════════════
# Training result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    predictor:    object          # SurrogatePredictor | NARXPredictor | RNNPredictor | TransformerPredictor
    model_class:  ModelClass
    n_train:      int
    train_rmse:   float
    length_scale: Optional[float] = None   # GP only; None for NN/RNN/Transformer
    extra_meta:   Optional[dict]  = None   # paradigm-B specific metadata (lags, context_len, …)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

class SurrogateTrainer:
    """Fits any supported model class from training data."""

    # ── Paradigm A: ODE-based ─────────────────────────────────────────────────

    def fit(
        self,
        model_class:  ModelClass,
        theta:        np.ndarray,    # output (state 0)
        theta_dot:    np.ndarray,    # first derivative (state 1)
        u:            np.ndarray,    # input
        theta_ddot:   np.ndarray,    # target: highest-order derivative
        n_epochs:     int = 200,
        seed:         int = 0,
    ) -> TrainingResult:
        """
        Fit a Paradigm A surrogate from pointwise (state, u) → acceleration data.

        Backward-compatible signature: takes theta, theta_dot, u, theta_ddot.
        Internally generalises to (N, n_features) where n_features = 3 here.
        """
        valid = (
            np.isfinite(theta)
            & np.isfinite(theta_dot)
            & np.isfinite(u)
            & np.isfinite(theta_ddot)
            & (np.abs(theta_ddot) < 1e4)
        )
        X = np.column_stack([theta[valid], theta_dot[valid], u[valid]])
        y = theta_ddot[valid]

        if len(X) < 4:
            raise ValueError(f"Too few valid samples for surrogate fitting: {len(X)}")

        return self._fit_ode_from_XY(X, y, model_class, n_epochs, seed)

    def fit_ode_from_states(
        self,
        states:      np.ndarray,    # (system_order, N) all state estimates
        u:           np.ndarray,    # (N,) input
        target:      np.ndarray,    # (N,) highest-order derivative
        model_class: ModelClass,
        n_epochs:    int = 200,
        seed:        int = 0,
    ) -> TrainingResult:
        """Generalised Paradigm A fit for any system order."""
        state_rows = [states[i] for i in range(states.shape[0])]
        valid = (
            all(np.isfinite(s).all() for s in state_rows)
            & np.isfinite(u)
            & np.isfinite(target)
            & (np.abs(target) < 1e4)
        )
        valid = np.isfinite(u) & np.isfinite(target) & (np.abs(target) < 1e4)
        for s in state_rows:
            valid &= np.isfinite(s)

        cols = [s[valid] for s in state_rows] + [u[valid]]
        X    = np.column_stack(cols)
        y    = target[valid]

        if len(X) < 4:
            raise ValueError(f"Too few valid samples: {len(X)}")

        return self._fit_ode_from_XY(X, y, model_class, n_epochs, seed)

    # ── Paradigm B: NARX ──────────────────────────────────────────────────────

    def fit_narx(
        self,
        y:           np.ndarray,    # (N,) observed output sequence
        u:           np.ndarray,    # (N,) input sequence
        lag_y:       int = 5,
        lag_u:       int = 3,
        model_class: ModelClass = ModelClass.NARX,
        seed:        int = 0,
    ) -> TrainingResult:
        """
        Fit a NARX model from an observed I/O sequence.

        Builds a lag-feature matrix and fits a GP (for small data) or MLP
        (for larger data).
        """
        N = len(y)
        min_lag = max(lag_y, lag_u)

        if N < min_lag + 10:
            raise ValueError(f"Not enough data for NARX fitting: N={N}, min_lag={min_lag}")

        rows, targets = [], []
        for k in range(min_lag, N):
            y_feats = y[k - lag_y:k]
            u_feats = u[k - lag_u:k + 1] if k >= lag_u else np.zeros(lag_u)
            rows.append(np.concatenate([y_feats, u_feats[-lag_u:]]))
            targets.append(y[k])

        X = np.array(rows, dtype=float)
        t = np.array(targets, dtype=float)

        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0) + 1e-8
        y_mean = float(t.mean())
        y_std  = float(t.std() + 1e-8)
        X_n    = (X - X_mean) / X_std
        t_n    = (t - y_mean) / y_std

        # Pick regressor: GP for small data, MLP for larger
        if len(X) <= MAX_GP_POINTS:
            regressor_mc = ModelClass.GP
            rng = np.random.default_rng(seed)
            stride = max(1, len(X_n) // 50)
            Xs = X_n[::stride]
            d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
            dists = np.sqrt(d2[d2 > 0])
            ls = float(np.median(dists)) if len(dists) > 0 else 1.0

            gp = _NumpyGP(length_scale=ls, sigma_f=1.0, sigma_n=0.05)
            gp.fit(X_n, t_n)
            model_obj = gp

            t_pred_n = gp.predict(X_n)
        else:
            regressor_mc = ModelClass.NN
            model_obj, t_pred_n = self._fit_mlp_numpy(X_n, t_n, n_epochs=200, seed=seed)

        t_pred   = t_pred_n * y_std + y_mean
        train_rmse = float(np.sqrt(np.mean((t_pred - t) ** 2)))

        predictor = NARXPredictor(
            lag_y=lag_y, lag_u=lag_u,
            model=model_obj, mc=regressor_mc,
            X_mean=X_mean, X_std=X_std,
            y_mean=y_mean, y_std=y_std,
        )
        return TrainingResult(
            predictor=predictor,
            model_class=model_class,
            n_train=len(X),
            train_rmse=train_rmse,
            extra_meta={"lag_y": lag_y, "lag_u": lag_u},
        )

    # ── Paradigm B: RNN (LSTM) ────────────────────────────────────────────────

    def fit_rnn(
        self,
        y:           np.ndarray,
        u:           np.ndarray,
        seq_len:     int = 50,
        hidden_size: int = 64,
        n_epochs:    int = 150,
        seed:        int = 0,
    ) -> TrainingResult:
        """Fit an LSTM on (y, u) sequences; store weights as numpy for pickling."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError("torch is required for the RNN surrogate path.") from exc

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        y_arr = np.asarray(y, dtype=float)
        u_arr = np.asarray(u, dtype=float)

        y_mean = float(y_arr.mean()); y_std = float(y_arr.std() + 1e-8)
        u_mean = float(u_arr.mean()); u_std = float(u_arr.std() + 1e-8)

        y_n = (y_arr - y_mean) / y_std
        u_n = (u_arr - u_mean) / u_std

        N = len(y_n)
        if N < seq_len + 2:
            raise ValueError(f"Too few samples for RNN: N={N}, seq_len={seq_len}")

        # Build sequences: input=(y[t-1], u[t-1])→target=y[t]
        seqs_X, seqs_y = [], []
        for start in range(0, N - seq_len - 1, seq_len // 2):
            end = start + seq_len
            if end + 1 > N:
                break
            x_seq = np.column_stack([y_n[start:end], u_n[start:end]])
            y_seq = y_n[start + 1:end + 1]
            seqs_X.append(x_seq)
            seqs_y.append(y_seq)

        if not seqs_X:
            raise ValueError("Could not build any RNN training sequences.")

        X_t = torch.tensor(np.array(seqs_X), dtype=torch.float32)  # (B, T, 2)
        y_t = torch.tensor(np.array(seqs_y), dtype=torch.float32)  # (B, T)

        lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        out_proj = nn.Linear(hidden_size, 1)
        params = list(lstm.parameters()) + list(out_proj.parameters())
        opt = optim.Adam(params, lr=1e-3)

        B = X_t.shape[0]
        lstm.train(); out_proj.train()
        for _ in range(n_epochs):
            idx    = rng.integers(0, B, min(32, B))
            X_b    = X_t[idx]
            y_b    = y_t[idx]
            h_out, _ = lstm(X_b)                        # (batch, T, hidden)
            preds  = out_proj(h_out).squeeze(-1)         # (batch, T)
            loss   = nn.functional.mse_loss(preds, y_b)
            opt.zero_grad(); loss.backward(); opt.step()

        lstm.eval(); out_proj.eval()
        with torch.no_grad():
            h_out, _ = lstm(X_t)
            preds_n  = out_proj(h_out).squeeze(-1).numpy()
        preds_all  = preds_n.ravel() * y_std + y_mean
        target_all = np.array(seqs_y).ravel() * y_std + y_mean
        train_rmse = float(np.sqrt(np.mean((preds_all - target_all) ** 2)))

        # Extract weights as numpy for pickle-safe inference
        sd = lstm.state_dict()
        # PyTorch LSTM packs all gates: (4*hidden, input_size) for ii gates
        # We store the full weight matrices (IFGO ordering)
        W_ih = sd["weight_ih_l0"].numpy().copy()  # (4*H, 2)
        W_hh = sd["weight_hh_l0"].numpy().copy()  # (4*H, H)
        b_ih = sd["bias_ih_l0"].numpy().copy()
        b_hh = sd["bias_hh_l0"].numpy().copy()
        W_out = out_proj.weight.detach().numpy().copy().T    # (H, 1) → use (H,)
        b_out = out_proj.bias.detach().numpy().copy()

        H = hidden_size
        def _slice(M, r): return M[r * H:(r + 1) * H]

        weights = {
            # IFGO ordering: 0=input, 1=forget, 2=cell, 3=output
            "W_i": np.concatenate([_slice(W_ih, 0), _slice(W_hh, 0)], axis=1),
            "b_i": _slice(b_ih, 0) + _slice(b_hh, 0),
            "W_f": np.concatenate([_slice(W_ih, 1), _slice(W_hh, 1)], axis=1),
            "b_f": _slice(b_ih, 1) + _slice(b_hh, 1),
            "W_g": np.concatenate([_slice(W_ih, 2), _slice(W_hh, 2)], axis=1),
            "b_g": _slice(b_ih, 2) + _slice(b_hh, 2),
            "W_o": np.concatenate([_slice(W_ih, 3), _slice(W_hh, 3)], axis=1),
            "b_o": _slice(b_ih, 3) + _slice(b_hh, 3),
            "W_out": W_out.ravel(),
            "b_out": float(b_out.ravel()[0]),   # scalar — avoids numpy 2.x float() on (1,) array
        }

        predictor = RNNPredictor(
            weights=weights,
            y_mean=y_mean, y_std=y_std,
            u_mean=u_mean, u_std=u_std,
        )
        return TrainingResult(
            predictor=predictor,
            model_class=ModelClass.RNN,
            n_train=len(y),
            train_rmse=train_rmse,
            extra_meta={"hidden_size": hidden_size, "seq_len": seq_len},
        )

    # ── Paradigm B: Transformer ───────────────────────────────────────────────

    def fit_transformer(
        self,
        y:           np.ndarray,
        u:           np.ndarray,
        context_len: int = 50,
        d_model:     int = 32,
        n_heads:     int = 4,
        n_epochs:    int = 100,
        seed:        int = 0,
    ) -> TrainingResult:
        """Fit a causal decoder Transformer; store weights as numpy."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError("torch is required for the Transformer surrogate path.") from exc

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        y_arr = np.asarray(y, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        y_mean = float(y_arr.mean()); y_std = float(y_arr.std() + 1e-8)
        u_mean = float(u_arr.mean()); u_std = float(u_arr.std() + 1e-8)
        y_n = (y_arr - y_mean) / y_std
        u_n = (u_arr - u_mean) / u_std

        N = len(y_n)
        if N < context_len + 2:
            raise ValueError(f"Too few samples for Transformer: N={N}")

        # Build context windows: tokens = (y[t-ctx:t], u[t-ctx:t]), target = y[t]
        seqs_X, seqs_y = [], []
        for t in range(context_len, N):
            tok = np.column_stack([y_n[t - context_len:t], u_n[t - context_len:t]])
            seqs_X.append(tok)
            seqs_y.append(y_n[t])

        X_t = torch.tensor(np.array(seqs_X), dtype=torch.float32)  # (B, ctx, 2)
        y_t = torch.tensor(np.array(seqs_y), dtype=torch.float32)  # (B,)

        # Simple single-layer causal Transformer
        # Use PyTorch MultiheadAttention for training
        d_ff = d_model * 2
        tok_emb_w = nn.Linear(2, d_model)
        attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        ff_w1 = nn.Linear(d_model, d_ff)
        ff_w2 = nn.Linear(d_ff, d_model)
        out_w = nn.Linear(d_model, 1)

        params = (list(tok_emb_w.parameters()) + list(attn.parameters()) +
                  list(ff_w1.parameters()) + list(ff_w2.parameters()) +
                  list(out_w.parameters()))
        opt = optim.Adam(params, lr=1e-3)

        B = X_t.shape[0]
        for epoch in range(n_epochs):
            idx   = rng.integers(0, B, min(32, B))
            x_b   = X_t[idx]                     # (batch, ctx, 2)
            y_b   = y_t[idx]

            # Token embedding
            x_emb = tok_emb_w(x_b)               # (batch, ctx, d_model)

            # Causal mask
            ctx  = x_b.shape[1]
            mask = torch.triu(torch.full((ctx, ctx), float('-inf')), diagonal=1)

            x_att, _ = attn(x_emb, x_emb, x_emb, attn_mask=mask)
            x_att = x_emb + x_att                 # residual

            x_ff  = ff_w2(torch.tanh(ff_w1(x_att)))
            x_out = x_att + x_ff                  # residual

            pred = out_w(x_out[:, -1, :]).squeeze(-1)  # last token → scalar
            loss = nn.functional.mse_loss(pred, y_b)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            x_emb = tok_emb_w(X_t)
            ctx   = X_t.shape[1]
            mask  = torch.triu(torch.full((ctx, ctx), float('-inf')), diagonal=1)
            x_att, _ = attn(x_emb, x_emb, x_emb, attn_mask=mask)
            x_att = x_emb + x_att
            x_ff  = ff_w2(torch.tanh(ff_w1(x_att)))
            x_out = x_att + x_ff
            preds_n = out_w(x_out[:, -1, :]).squeeze(-1).numpy()

        preds_all  = preds_n * y_std + y_mean
        target_all = np.array(seqs_y) * y_std + y_mean
        train_rmse = float(np.sqrt(np.mean((preds_all - target_all) ** 2)))

        # Extract numpy weights for pure-numpy inference
        weights = {
            "tok_emb":       tok_emb_w.weight.detach().numpy().T.copy(),  # (2, d_model)
            "tok_emb_bias":  tok_emb_w.bias.detach().numpy().copy(),
            # Attention weights (in_proj = W_q+W_k+W_v concatenated for first head)
            # Use the full in_proj and out_proj for single-head approximation
            "W_q": attn.in_proj_weight[:d_model].detach().numpy().copy(),
            "W_k": attn.in_proj_weight[d_model:2 * d_model].detach().numpy().copy(),
            "W_v": attn.in_proj_weight[2 * d_model:].detach().numpy().copy(),
            "ff_w1": ff_w1.weight.detach().numpy().T.copy(),              # (d_model, d_ff)
            "ff_b1": ff_w1.bias.detach().numpy().copy(),
            "ff_w2": ff_w2.weight.detach().numpy().T.copy(),              # (d_ff, d_model)
            "ff_b2": ff_w2.bias.detach().numpy().copy(),
            "out_w": out_w.weight.detach().numpy().ravel().copy(),        # (d_model,)
            "out_b": out_w.bias.item(),
        }

        predictor = TransformerPredictor(
            weights=weights, context_len=context_len,
            y_mean=y_mean, y_std=y_std,
            u_mean=u_mean, u_std=u_std,
        )
        return TrainingResult(
            predictor=predictor,
            model_class=ModelClass.TRANSFORMER,
            n_train=len(y),
            train_rmse=train_rmse,
            extra_meta={
                "context_len": context_len,
                "d_model":     d_model,
                "n_heads":     n_heads,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fit_ode_from_XY(
        X: np.ndarray, y: np.ndarray,
        model_class: ModelClass,
        n_epochs: int,
        seed: int,
    ) -> TrainingResult:
        """Common ODE-paradigm training logic for any feature matrix X."""
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0) + 1e-8
        y_mean = float(y.mean())
        y_std  = float(y.std() + 1e-8)
        X_n    = (X - X_mean) / X_std
        y_n    = (y - y_mean) / y_std

        if model_class == ModelClass.GP:
            return SurrogateTrainer._fit_gp(
                X_n, y_n, X, y, X_mean, X_std, y_mean, y_std, seed
            )
        else:
            return SurrogateTrainer._fit_nn(
                X_n, y_n, X, y, X_mean, X_std, y_mean, y_std, n_epochs, seed
            )

    @staticmethod
    def _fit_gp(
        X_n, y_n, X, y, X_mean, X_std, y_mean, y_std, seed
    ) -> TrainingResult:
        rng = np.random.default_rng(seed)
        N   = len(X_n)
        if N > MAX_GP_POINTS:
            idx = rng.choice(N, MAX_GP_POINTS, replace=False)
            X_n = X_n[idx]
            y_n = y_n[idx]

        stride = max(1, len(X_n) // 50)
        Xs = X_n[::stride]
        d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
        dists = np.sqrt(d2[d2 > 0])
        ls = float(np.median(dists)) if len(dists) > 0 else 1.0

        gp = _NumpyGP(length_scale=ls, sigma_f=1.0, sigma_n=0.05)
        gp.fit(X_n, y_n)

        y_pred_n   = gp.predict(X_n)
        y_pred     = y_pred_n * y_std + y_mean
        train_rmse = float(np.sqrt(np.mean((y_pred - y[:len(y_pred)]) ** 2)))

        predictor = SurrogatePredictor(
            model_class=ModelClass.GP,
            gp=gp,
            X_mean=X_mean, X_std=X_std,
            y_mean=y_mean, y_std=y_std,
        )
        return TrainingResult(
            predictor=predictor,
            model_class=ModelClass.GP,
            n_train=len(X_n),
            train_rmse=train_rmse,
            length_scale=ls,
        )

    @staticmethod
    def _fit_nn(
        X_n, y_n, X, y, X_mean, X_std, y_mean, y_std, n_epochs, seed
    ) -> TrainingResult:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError(
                "torch is required for the NN surrogate path. "
                "Install it via: pip install torch"
            ) from exc

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        N   = len(X_n)
        n_features = X.shape[1]   # generalised: works for any number of states + u

        Xt = torch.tensor(X_n, dtype=torch.float32)
        yt = torch.tensor(y_n, dtype=torch.float32)

        hidden = 64
        model  = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        batch     = min(128, N)

        model.train()
        for _ in range(n_epochs):
            idx  = rng.integers(0, N, batch)
            X_b  = Xt[idx]
            y_b  = yt[idx]
            pred = model(X_b).squeeze(-1)
            loss = nn.functional.mse_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_n = model(Xt).squeeze(-1).numpy()
        y_pred     = y_pred_n * y_std + y_mean
        train_rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

        sd = model.state_dict()
        mlp_state = {
            "w1": sd["0.weight"].numpy().copy(),
            "b1": sd["0.bias"].numpy().copy(),
            "w2": sd["2.weight"].numpy().copy(),
            "b2": sd["2.bias"].numpy().copy(),
            "w3": sd["4.weight"].numpy().copy(),
            "b3": sd["4.bias"].numpy().copy(),
        }

        predictor = SurrogatePredictor(
            model_class=ModelClass.NN,
            mlp_state=mlp_state,
            X_mean=X_mean, X_std=X_std,
            y_mean=y_mean, y_std=y_std,
        )
        return TrainingResult(
            predictor=predictor,
            model_class=ModelClass.NN,
            n_train=N,
            train_rmse=train_rmse,
        )

    @staticmethod
    def _fit_mlp_numpy(
        X_n: np.ndarray, y_n: np.ndarray, n_epochs: int, seed: int
    ) -> "Tuple[dict, np.ndarray]":
        """Fit a small MLP using numpy gradients (no torch required)."""
        rng = np.random.default_rng(seed)
        N, n_feat = X_n.shape
        H = 32
        lr = 5e-3

        # Xavier initialization
        W1 = rng.standard_normal((H, n_feat)) * np.sqrt(2 / n_feat)
        b1 = np.zeros(H)
        W2 = rng.standard_normal((H, H)) * np.sqrt(2 / H)
        b2 = np.zeros(H)
        W3 = rng.standard_normal((1, H)) * np.sqrt(2 / H)
        b3 = np.zeros(1)

        def _forward(X):
            h1 = np.tanh(X @ W1.T + b1)
            h2 = np.tanh(h1 @ W2.T + b2)
            return (h2 @ W3.T + b3).ravel()

        batch = min(64, N)
        for _ in range(n_epochs):
            idx = rng.integers(0, N, batch)
            Xb, yb = X_n[idx], y_n[idx]

            h1_pre = Xb @ W1.T + b1
            h1 = np.tanh(h1_pre)
            h2_pre = h1 @ W2.T + b2
            h2 = np.tanh(h2_pre)
            yp = (h2 @ W3.T + b3).ravel()

            d_out = (yp - yb) / batch
            dW3 = d_out[None, :] @ h2          # (1, batch) @ (batch, H) = (1, H)
            db3 = d_out.sum(keepdims=True)      # (1,)

            d_h2 = d_out[:, None] * W3 * (1 - h2 ** 2)
            dW2 = d_h2.T @ h1
            db2 = d_h2.sum(axis=0)

            d_h1 = d_h2 @ W2 * (1 - h1 ** 2)
            dW1 = d_h1.T @ Xb
            db1 = d_h1.sum(axis=0)

            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
            W3 -= lr * dW3; b3 -= lr * db3

        y_pred_n = _forward(X_n)
        mlp_state = {
            "w1": W1.copy(), "b1": b1.copy(),
            "w2": W2.copy(), "b2": b2.copy(),
            "w3": W3.copy(), "b3": b3.copy(),
        }
        return mlp_state, y_pred_n


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)
