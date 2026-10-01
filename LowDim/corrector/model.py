from __future__ import annotations

import tensorflow as tf


class ResidualTCNCorrector(tf.keras.Model):
    """
    Gated residual temporal convolutional network for invariant-measure correction.

    Architecture:
      Input (B, T, 3)
        ↓
      Dense(hidden_dim, tanh)
        ↓
      Conv1D(hidden_dim, kernel=5, dilation=1, tanh)
      Conv1D(hidden_dim, kernel=5, dilation=2, tanh)
      Conv1D(hidden_dim, kernel=5, dilation=4, tanh)
        ↓
      Dense(3, linear) -> delta
      Dense(1, sigmoid) -> gate approximately [0, 1] initialized near 0
        ->
      corrected = input + gate * delta

    Loss (hybrid):
      L_path = weighted_mse(pred, target)     # Pointwise supervised
      L_delta = mean(||delta||^2)              # Regularize residual size
      L_moment = ||mean(pred)-mean(target)||^2 + ||std(pred)-std(target)||^2
      L_tail = sum_k (exceed_pred_k - exceed_target_k)^2
      L_total = L_path + lambda_delta*L_delta + lambda_moment*L_moment + lambda_tail*L_tail
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        context_dim: int = 0,
        film_hidden_dim: int = 32,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Input projection
        self.dense_in = tf.keras.layers.Dense(hidden_dim, activation="tanh")

        # Dilated convolutions
        self.conv1 = tf.keras.layers.Conv1D(hidden_dim, kernel_size=5, padding="same",
                                            activation="tanh", dilation_rate=1)
        self.conv2 = tf.keras.layers.Conv1D(hidden_dim, kernel_size=5, padding="same",
                                            activation="tanh", dilation_rate=2)
        self.conv3 = tf.keras.layers.Conv1D(hidden_dim, kernel_size=5, padding="same",
                                            activation="tanh", dilation_rate=4)

        if context_dim > 0:
            self.film_h = tf.keras.layers.Dense(film_hidden_dim, activation="tanh")
            self.film_gamma = tf.keras.layers.Dense(
                hidden_dim,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
            self.film_beta = tf.keras.layers.Dense(
                hidden_dim,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        else:
            self.film_h = None
            self.film_gamma = None
            self.film_beta = None

        # Output: delta and gate
        self.delta_out = tf.keras.layers.Dense(state_dim, activation=None)
        # Gate with negative bias initialization (start near zero)
        self.gate_out = tf.keras.layers.Dense(
            1,
            activation=None,
            bias_initializer=tf.keras.initializers.Constant(gate_init_bias)
        )

    def _apply_film(self, h: tf.Tensor, c_seq: tf.Tensor | None) -> tf.Tensor:
        if self.context_dim == 0 or c_seq is None:
            return h
        u = self.film_h(c_seq)
        gamma = self.film_gamma(u)
        beta = self.film_beta(u)
        return (1.0 + gamma) * h + beta

    def call(self, z_seq: tf.Tensor, c_seq: tf.Tensor | None = None) -> tf.Tensor:
        """
        Apply residual correction with gated output.

        Args:
            z_seq: Input sequence (B, T, state_dim)

        Returns:
            corrected: Corrected sequence (B, T, state_dim)
        """
        h = self.dense_in(z_seq)       # (B, T, hidden_dim)
        h = self.conv1(h)              # (B, T, hidden_dim)
        h = self.conv2(h)              # (B, T, hidden_dim)
        h = self.conv3(h)              # (B, T, hidden_dim)
        h = self._apply_film(h, c_seq)

        delta = self.delta_out(h)      # (B, T, state_dim)
        gate_logits = self.gate_out(h) # (B, T, 1)
        gate = tf.sigmoid(gate_logits) # (B, T, 1) ∈ [0, 1]

        corrected = z_seq + gate * delta  # Gated residual
        return corrected

    def sequence_loss(
        self,
        obs_seq: tf.Tensor,
        target_seq: tf.Tensor,
        c_seq: tf.Tensor | None = None,
        z3_tail_threshold: float = 0.0,
        lambda_delta: float = 1e-3,
        lambda_moment: float = 1e-2,
        lambda_tail: float = 1e-2,
        tail_weight_factor: float = 2.0,
        state_loss_weights: tuple[float, ...] | None = None,
        tail_exceedance_percentiles: list[float] | None = None,
        tail_temperature: float = 0.05,
    ) -> tuple[tf.Tensor, dict]:
        """
        Compute hybrid loss: pathwise + delta regularization + moment matching + tail exceedance.

        Args:
            obs_seq: Input (B, T, state_dim), e.g., coarse or nudged
            target_seq: Target (B, T, state_dim), normalized coordinates
            z3_tail_threshold: Fixed threshold in normalized coordinates
            lambda_delta: Weight on ||delta||^2 regularization
            lambda_moment: Weight on mean/std matching
            lambda_tail: Weight on tail exceedance loss
            tail_exceedance_percentiles: Percentile thresholds (list of floats in [0, 100])
            tail_temperature: Temperature for smooth indicator function

        Returns:
            total_loss: Scalar loss
            loss_dict: Dict with component losses (for logging)
        """
        # Forward pass
        pred = self.call(obs_seq, c_seq)  # (B, T, state_dim)
        delta = pred - obs_seq     # (B, T, state_dim)

        # 1. Pathwise MSE
        if state_loss_weights is None:
            state_loss_weights = (1.0,) * self.state_dim
        dim_weights = tf.constant(state_loss_weights, dtype=pred.dtype)
        sq_err = tf.square(pred - target_seq) * dim_weights
        mse_per_step = tf.reduce_mean(sq_err, axis=-1)  # (B, T)

        # Optional tail weighting: only weight if threshold is provided
        if z3_tail_threshold > 0.0:
            tail_mask_dim = tf.abs(target_seq) > z3_tail_threshold
            tail_mask = tf.cast(tf.reduce_any(tail_mask_dim, axis=-1), tf.float32)
            weights = 1.0 + tail_weight_factor * tail_mask
            L_path = tf.reduce_mean(weights * mse_per_step)
        else:
            L_path = tf.reduce_mean(mse_per_step)

        # 2. Residual regularization
        L_delta = tf.reduce_mean(tf.square(delta))

        # 3. Moment matching (mean and std per coordinate)
        mean_pred = tf.reduce_mean(pred, axis=1)  # (B, state_dim)
        mean_target = tf.reduce_mean(target_seq, axis=1)  # (B, state_dim)
        std_pred = tf.math.reduce_std(pred, axis=1)  # (B, state_dim)
        std_target = tf.math.reduce_std(target_seq, axis=1)  # (B, state_dim)

        L_moment = tf.reduce_mean(tf.square(mean_pred - mean_target)) + \
                   tf.reduce_mean(tf.square(std_pred - std_target))

        # 4. Tail exceedance loss (using smooth indicator)
        L_tail = tf.constant(0.0)
        if tail_exceedance_percentiles is not None and len(tail_exceedance_percentiles) > 0:
            tail_loss_components = []
            for dim in range(self.state_dim):
                target_flat = tf.reshape(tf.abs(target_seq[..., dim]), [-1])
                pred_flat = tf.reshape(tf.abs(pred[..., dim]), [-1])

                for pctl in tail_exceedance_percentiles:
                    n_total = tf.cast(tf.shape(target_flat)[0], tf.float32)
                    k = tf.maximum(1, tf.cast(n_total * (100.0 - pctl) / 100.0, tf.int32))
                    k = tf.minimum(k, tf.shape(target_flat)[0] - 1)
                    top_vals = tf.nn.top_k(target_flat, k=k, sorted=False).values
                    threshold_approx = top_vals[-1]

                    smooth_ind_pred = tf.nn.sigmoid((pred_flat - threshold_approx) / (tail_temperature + 1e-8))
                    smooth_ind_target = tf.nn.sigmoid((target_flat - threshold_approx) / (tail_temperature + 1e-8))
                    exceed_pred = tf.reduce_mean(smooth_ind_pred)
                    exceed_target = tf.reduce_mean(smooth_ind_target)
                    tail_loss_components.append(tf.square(exceed_pred - exceed_target))

            L_tail = tf.reduce_mean(tf.stack(tail_loss_components)) if tail_loss_components else tf.constant(0.0)

        # Total loss
        total_loss = L_path + lambda_delta * L_delta + lambda_moment * L_moment + lambda_tail * L_tail

        loss_dict = {
            "L_path": float(L_path.numpy()),
            "L_delta": float(L_delta.numpy()),
            "L_moment": float(L_moment.numpy()),
            "L_tail": float(L_tail.numpy()),
        }

        return total_loss, loss_dict


class ResidualAttentionCorrector(ResidualTCNCorrector):
    """Residual Transformer encoder corrector with the same FiLM/loss interface."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        context_dim: int = 0,
        film_hidden_dim: int = 32,
        gate_init_bias: float = -2.0,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_multiplier: int = 2,
        dropout_rate: float = 0.05,
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            film_hidden_dim=film_hidden_dim,
            gate_init_bias=gate_init_bias,
        )
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout_rate)
        key_dim = max(1, hidden_dim // max(1, int(num_heads)))

        self.input_projection = tf.keras.layers.Dense(hidden_dim)
        self.attn_layers = []
        self.ffn_layers = []
        self.norm_attn = []
        self.norm_ffn = []
        self.dropouts = []

        for _ in range(self.num_layers):
            self.attn_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=dropout_rate,
                )
            )
            self.ffn_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(ffn_multiplier * hidden_dim, activation="gelu"),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.Dense(hidden_dim),
                    ]
                )
            )
            self.norm_attn.append(tf.keras.layers.LayerNormalization(epsilon=1e-5))
            self.norm_ffn.append(tf.keras.layers.LayerNormalization(epsilon=1e-5))
            self.dropouts.append(tf.keras.layers.Dropout(dropout_rate))

    def _positional_encoding(self, t_len: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        positions = tf.cast(tf.range(t_len)[:, None], dtype)
        dims = tf.cast(tf.range(self.hidden_dim)[None, :], dtype)
        angle_rates = tf.pow(tf.cast(10000.0, dtype), -2.0 * tf.floor(dims / 2.0) / tf.cast(self.hidden_dim, dtype))
        angles = positions * angle_rates
        even = tf.sin(angles[:, 0::2])
        odd = tf.cos(angles[:, 1::2])
        stacked = tf.reshape(tf.stack([even, odd], axis=-1), (t_len, -1))
        return stacked[:, : self.hidden_dim][None, :, :]

    def call(self, z_seq: tf.Tensor, c_seq: tf.Tensor | None = None, training: bool = False) -> tf.Tensor:
        t_len = tf.shape(z_seq)[1]
        h = self.input_projection(z_seq)
        h = h + self._positional_encoding(t_len, h.dtype)

        for attn, ffn, norm_a, norm_f, dropout in zip(
            self.attn_layers, self.ffn_layers, self.norm_attn, self.norm_ffn, self.dropouts
        ):
            a = norm_a(h)
            a = attn(a, a, training=training)
            h = h + dropout(a, training=training)
            f = norm_f(h)
            h = h + dropout(ffn(f, training=training), training=training)

        h = self._apply_film(h, c_seq)
        delta = self.delta_out(h)
        gate = tf.sigmoid(self.gate_out(h))
        return z_seq + gate * delta
