import jax
import jax.numpy as jnp

def simnorm(x: jnp.ndarray, simnorm_dim=8):

    """ Group feature dim into k groups, do softmax locally 
    then reshape back into original shape
    """

    shp = x.shape
    x = x.reshape(*shp[:-1], -1, simnorm_dim)
    x = jax.nn.softmax(x, axis=-1)
    x = x.reshape(*shp)
    
    return x