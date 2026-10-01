# Author: Isabella Thiel <thieli@mit.edu>
"""Seeded numerical fixtures captured before the repository reorganization."""
import numpy as np
import tensorflow as tf

from test_lowdim import ResidualAttentionCorrector, ResidualTCNCorrector, data
from test_qg import baseline, film, loss_params_2D

def reference_outputs():
    """Evaluate the unchanged seeded pre-cleanup numerical cases."""
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 8, 3)).astype(np.float32)
    context = rng.normal(size=(2, 8, 3)).astype(np.float32)
    actual = {}
    for cls in (ResidualTCNCorrector, ResidualAttentionCorrector):
        for context_dim in (0, 3):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(17)
            kwargs = dict(state_dim=3, hidden_dim=8, context_dim=context_dim, film_hidden_dim=4)
            if cls is ResidualAttentionCorrector:
                kwargs.update(num_heads=2, num_layers=1, dropout_rate=0)
            model = cls(**kwargs)
            actual[cls.__name__ + str(context_dim)] = model(x, context if context_dim else None).numpy()
    actual['lowdim_context'] = data.context_from_coarse_trajectory(x[0], True, 1, 5)
    alpha = loss_params_2D(1, 1, 2, 2, 2, 0.1)
    x_qg = rng.normal(size=(2, 8, 8)).astype(np.float32)
    context_qg = rng.normal(size=(2, 8, 4)).astype(np.float32)
    inputs = {'field_in': x_qg, 'cond_in': context_qg}
    for name, module in [('s0', baseline), ('s4', film)]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)
        model = module.build_film_lstm(8, 8, 4, alpha)
        actual[name] = model(inputs)['field'].numpy()
        model.train_on_batch(inputs, {'field': x_qg})
        actual[name + '_trained'] = model(inputs)['field'].numpy()
    actual['qg_context'] = baseline._context_from_coarse_causal(rng.normal(size=(2, 2, 8, 2)), 1, 4)
    return actual
