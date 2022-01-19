"""Chain binomial process rippler algorithm"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from gemlib.util import compute_state
from gemlib.distributions import UniformInteger

tfd = tfp.distributions

__all__ = ["chain_binomial_rippler"]


def _log_factorial(x):
    """Computes x!"""
    return tf.math.lgamma(x + 1.0)


def _log_choose(n, k):
    """Computes nCk"""
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def hypergeom_prob(N, K, n, k):
    """Returns the PMF of a Hypergeometric distribution
    $$f(k | N, K, n) = \frac{ {K \choose k} {N-K \choose n-k} }
                            {N \choose n}$$

    :param N: size of population
    :param K: number of units of interest in population
    :param n: size of sample drawn from population
    :param k: number of units of interest in the sample
    :returns: the probability mass function value.
    """

    N = tf.convert_to_tensor(N, dtype_hint=tf.float32)
    K = tf.convert_to_tensor(K, dtype_hint=tf.float32)
    n = tf.convert_to_tensor(n, dtype_hint=tf.float32)
    k = tf.convert_to_tensor(k, dtype_hint=tf.float32)

    numerator = _log_choose(K, k) + _log_choose(N - K, n - k)
    denominator = _log_choose(N, n)
    return tf.math.exp(numerator - denominator)


def random_hypergeom(n_samples, N, K, n, seed=None, validate_args=False):
    """Returns draws from a Hypergeometric distribution with PMF
    $$f(k | N, K, n) = \frac{ {K \choose k} {N-K \choose n-k} }
                            {N \choose n}$$

    :param n_samples: number of required samples
    :param N: size of population
    :param K: number of units of interest in population
    :param n: size of sample drawn from population
    :param seed: an integer random number seed
    :param validate_args: if true ensure arguments are valid
    :returns: k, the number of units of interest in the sample
    """

    N = tf.cast(N, dtype=tf.int64)
    K = tf.convert_to_tensor(K)
    n = tf.cast(n, dtype=tf.int32)

    if validate_args is True:
        tf.debugging.assert_non_negative(N)
        tf.debugging.assert_non_negative(K)
        tf.debugging.assert_non_negative(n)

    # Create a `[N]` tensor of random uniforms,
    # the top `n` of which mark out our units of
    # interest.
    u = tfd.Uniform(low=tf.zeros(N), high=1.0, validate_args=validate_args,).sample(
        n_samples, seed=seed
    )

    # How to vectorize this wrt n?
    _, indices = tf.math.top_k(u, k=tf.squeeze(n))

    # Since order does not matter, `k` is just the sum of the indices < K
    return tf.reduce_sum(
        tf.cast(indices < tf.cast(K, indices.dtype), K.dtype), axis=-1,
    )


# Basic rippler sampling functions
def _psltp(z, p, ps, seed=None, validate_args=False):
    """ If $x^\star = x, p^\star \leq p$ then
         $$z^\star \sim \mbox{Binomial}(z, \frac{p_\star}{p})$$

    :param z: current $z$
    :param x: current $x$
    :param p: $p$ current probability
    :param xs: $x^\star$ new state
    :param ps: $p_star$ new probability
    :return: `z` the new number of events
    """
    return tfd.Binomial(
        total_count=tf.cast(z, p.dtype), probs=ps / p, validate_args=validate_args
    ).sample(seed=seed)


def _psgtp(z, x, p, ps, seed=None, validate_args=False):
    """ If $x^\star = x, p^\star > p$ then
         $$z^\star \sim z + \mbox{Binomial}(x-z, \frac{p_\star - p}{1 - p})$$
    
    :param z: current $z$
    :param x: current $x$
    :param p: $p$ current probability
    :param xs: $x^\star$ new state
    :param ps: $p_star$ new probability
    :return: the new number of events
    """
    prob = 1 - (1 - ps) / (1 - p)
    return z + tfd.Binomial(
        total_count=tf.cast(x - z, prob.dtype), probs=prob, validate_args=validate_args,
    ).sample(seed=seed)


def _xsgtx(z, x, xs, ps, seed=None, validate_args=False):
    """ If $x^\star > x, p^\star = p$ then
         $$z^\star \sim z + \mbox{Binomial}(xs-x, p_\star)$$

    :param z: current $z$
    :param x: current $x$
    :param p: $p$ current probability
    :param xs: $x^\star$ new state
    :param ps: $p_star$ new probability

    :return: the new number of events
    """
    return z + tfd.Binomial(
        total_count=tf.cast(xs - x, ps.dtype), probs=ps, validate_args=validate_args
    ).sample(seed=seed)


def _xsltx(z, x, xs, seed=None, validate_args=False):
    """ If $x^\star < x, p^\star = p$ then
         $$z^\star \sim \mbox{Hypergeometric}(x, z, xs)$$

    :param z: current $z$
    :param x: current $x$
    :param p: $p$ current probability
    :param xs: $x^\star$ new state
    :param ps: $p_star$ new probability

    :return: the new number of events
    """
    res = tf.cast(
        random_hypergeom(1, N=x, K=z, n=xs, seed=seed, validate_args=validate_args),
        z.dtype,
    )
    return res


def _dispatch_update(z, x, p, xs, ps, seed=None, validate_args=False):
    """Dispatches update function based on values of 
       parameters.

    **This is purely a scalar function due to random_hypergeom**

    :param z: current $z$
    :param x: current $x$
    :param p: $p$ current probability
    :param xs: $x^\star$ new state
    :param ps: $p_star$ new probability

    :returns: an updated number of events
    """
    p = tf.convert_to_tensor(p)
    ps = tf.convert_to_tensor(ps)
    z = tf.cast(z, p.dtype)
    x = tf.cast(x, p.dtype)
    xs = tf.cast(xs, p.dtype)

    seeds = samplers.split_seed(seed, salt="_dispatch_update")

    # Update for p->ps first
    idx = tf.cast(tf.math.sign(ps - p), tf.int32) + 1
    z_prime = tf.switch_case(
        tf.squeeze(idx),
        [
            lambda: _psltp(z, p, ps, seeds[0], validate_args),
            lambda: z,
            lambda: _psgtp(z, x, p, ps, seeds[0], validate_args),
        ],
    )

    idx = tf.cast(tf.math.sign(xs - x), tf.int32) + 1
    z_new = tf.switch_case(
        tf.squeeze(idx),
        [
            lambda: _xsltx(z_prime, x, xs, seeds[1], validate_args),
            lambda: z_prime,
            lambda: _xsgtx(z_prime, x, xs, ps, seeds[1], validate_args),
        ],
    )

    return z_new


# Tests
def test_dispatch():
    _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.1)
    _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.05)
    _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.2)
    _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.1)
    _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.05)
    _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.2)
    _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.1)
    _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.05)
    _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.2)


# @tf.function(jit_compile=False)
def chain_binomial_rippler(model, current_events, seed=None):

    seed = samplers.sanitize_seed(seed, salt="chain_binomial_rippler")
    seeds = samplers.split_seed(seed, n=3)

    num_steps = model.num_steps

    # Calculate current state
    current_state = compute_state(
        initial_state=model.initial_state,
        events=current_events,
        stoichiometry=model.stoichiometry,
    )

    # Choose timepoint, t
    proposed_time_idx = UniformInteger(low=0, high=num_steps).sample(seed=seeds[0])
    current_state_t = tf.gather(current_state, proposed_time_idx, axis=-2)

    # Choose new infection events at time t
    proposed_transition_rates = tf.stack(
        model.transition_rates(proposed_time_idx, current_state_t), axis=-1
    )
    prob_t = 1.0 - tf.math.exp(
        -proposed_transition_rates[..., 0]
    )  # First event to perturb.
    new_si_events_t = tfd.Binomial(
        total_count=current_state_t[:, 0], probs=prob_t,  # Perturb SI events here
    ).sample(seed=seeds[1])
    new_events_t = tf.tensor_scatter_nd_update(
        current_events[:, proposed_time_idx, :], [[0, 0]], new_si_events_t
    )

    # Propagate from t+1 up to end of the timeseries
    def draw_events(time, new_state_t):

        current_state_t = tf.gather(
            current_state, time, axis=-2
        )  # current_state.shape == [M, T, S]
        current_events_t = tf.gather(
            current_events, time, axis=-2
        )  # current_events.shape == [M, T, R]

        proposed_infec_rate = tf.stack(
            model.transition_rates(time, new_state_t), axis=0
        )
        current_infec_rate = tf.stack(
            model.transition_rates(time, current_state_t), axis=0
        )

        ps = 1 - tf.math.exp(-proposed_infec_rate)
        p = 1 - tf.math.exp(-current_infec_rate)

        # Iterate over transitions
        def dispatch_on_one_tx(i):
            return _dispatch_update(
                z=current_events_t[..., i],
                x=current_state_t[..., i],
                p=p[i],
                xs=new_state_t[..., i],
                ps=ps[i],
                seed=seed,
                validate_args=False,
            )

        update = tf.map_fn(
            dispatch_on_one_tx,
            elems=tf.range(current_events.shape[-1]),
            fn_output_signature=current_events.dtype,
        )
        update = tf.reshape(
            update, [current_events.shape[-3], current_events.shape[-1]],
        )
        tf.debugging.assert_non_negative(update)
        return update

    def propagate(accum, t):

        # Previous state
        events_t, state_t = accum
        tf.debugging.assert_non_negative(events_t, summarize=100)
        tf.debugging.assert_non_negative(state_t, summarize=100)
        tf.debugging.assert_rank(events_t, 2)
        tf.debugging.assert_rank(state_t, 2)

        # Propagate previous state to new state
        state_t1 = state_t + tf.einsum("...i,ij->...j", events_t, model.stoichiometry)
        tf.debugging.assert_non_negative(state_t1, summarize=100)

        # Infection rates and draw new events
        events_t1 = draw_events(t + 1, state_t1)

        return events_t1, state_t1

    timepoints = tf.range(proposed_time_idx, num_steps - 1)

    new_events, new_state = tf.scan(
        propagate,
        elems=timepoints,
        initializer=(new_events_t, current_state_t),  # MxR  # MxR
    )
    new_events = tf.transpose(new_events, perm=(1, 0, 2))

    concat_events = tf.concat(
        [
            current_events[:, :proposed_time_idx],
            tf.expand_dims(new_events_t, axis=-2),
            new_events,
        ],
        axis=-2,
    )

    concat_events = tf.ensure_shape(concat_events, current_events.shape)

    return (
        concat_events,
        {
            "delta": new_events_t - current_events[:, proposed_time_idx, :],
            "timepoint": proposed_time_idx,
            "new_si_events_t": new_si_events_t,
            "current_state_t": current_state_t,
            "prob_t": prob_t,
        },
    )
