import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateActionValue(nn.Module):
    base_cls: nn.Module
    kernel_init: object = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=self.kernel_init or default_init())(outputs)

        return jnp.squeeze(value, -1)
