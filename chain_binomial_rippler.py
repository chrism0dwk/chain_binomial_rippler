"""Chain binomial process rippler algorithm"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import prefer_static as ps

from gemlib.util import compute_state
from gemlib.distributions import UniformInteger

tfd = tfp.distributions

__all__ = ["chain_binomial_rippler"]


def _get_src_states(stoichiometry):
    """Iterate over rows in `stoichiometry` and return
       the index of the column containing a `-1`"""
    src_states = tf.where(tf.math.equal(stoichiometry, -1.0))
    return src_states[:, 1]


def _compute_state(initial_state, events, stoichiometry, closed=False):
    """Computes a state tensor from initial state and event tensor

    :param initial_state: a tensor of shape [S, M]
    :param events: a tensor of shape [T, R, M]
    :param stoichiometry: a stoichiometry matrix of shape [R, S] describing
                          how transitions update the state.
    :param closed: if `True`, return state in close interval [0, T], otherwise [0, T)
    :return: a tensor of shape [T, S, M] if `closed=False` or [T+1, S, M] if `closed=True`
             describing the state of the
             system for each batch M at time T.
    """
    if isinstance(stoichiometry, tf.Tensor):
        stoichiometry = ps.cast(stoichiometry, dtype=events.dtype)
    else:
        stoichiometry = tf.convert_to_tensor(stoichiometry, dtype=events.dtype)

    increments = tf.einsum("...trm,rs->...tsm", events, stoichiometry)

    if closed is False:
        cum_increments = tf.cumsum(increments, axis=-3, exclusive=True)
    else:
        cum_increments = tf.cumsum(increments, axis=-3, exclusive=False)
        cum_increments = tf.concat(
            [tf.zeros_like(cum_increments[..., 0:1, :, :]), cum_increments],
            axis=-2,
        )
    state = cum_increments + tf.expand_dims(initial_state, axis=-3)
    return state


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
    with tf.name_scope("random_hypergeom"):
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
        u = tfd.Uniform(
            low=tf.zeros(N), high=1.0, validate_args=validate_args,
        ).sample(n_samples, seed=seed)

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
    with tf.name_scope("psltp"):
        return tfd.Binomial(
            total_count=tf.cast(z, p.dtype),
            probs=ps / p,
            validate_args=validate_args,
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
    with tf.name_scope("psgtp"):
        prob = 1 - (1 - ps) / (1 - p)
        return z + tfd.Binomial(
            total_count=tf.cast(x - z, prob.dtype),
            probs=prob,
            validate_args=validate_args,
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
    with tf.name_scope("xsgtx"):
        return z + tfd.Binomial(
            total_count=tf.cast(xs - x, ps.dtype),
            probs=ps,
            validate_args=validate_args,
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
    with tf.name_scope("xsltx"):
        res = tf.cast(
            random_hypergeom(
                1, N=x, K=z, n=xs, seed=seed, validate_args=validate_args
            ),
            z.dtype,
        )
        return tf.squeeze(res)


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
    with tf.name_scope("dispatch_update"):
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
            name="switch_step_0",
        )

        idx = tf.cast(tf.math.sign(xs - x), tf.int32) + 1
        z_new = tf.switch_case(
            tf.squeeze(idx),
            [
                lambda: _xsltx(z_prime, x, xs, seeds[1], validate_args),
                lambda: z_prime,
                lambda: _xsgtx(z_prime, x, xs, ps, seeds[1], validate_args),
            ],
            name="switch_step_1",
        )

        return z_new


# Tests
def test_dispatch():
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.1)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.05)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=100, ps=0.2)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.1)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.05)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=50, ps=0.2)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.1)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.05)
    )
    tf.debugging.assert_scalar(
        _dispatch_update(z=10, x=100, p=0.1, xs=200, ps=0.2)
    )


def _initial_ripple(model, current_events, current_state, seed):

    init_time_seed, init_events_seed = samplers.split_seed(
        seed, salt="_initial_ripple"
    )

    # Choose timepoint, t
    proposed_time_idx = UniformInteger(low=0, high=model.num_steps).sample(
        seed=init_time_seed
    )
    current_state_t = tf.gather(current_state, proposed_time_idx, axis=-3)

    # Choose new infection events at time t
    proposed_transition_rates = tf.stack(
        model.transition_rates(
            proposed_time_idx, tf.transpose(current_state_t)
        ),
        axis=0,
    )
    prob_t = 1.0 - tf.math.exp(
        -proposed_transition_rates[0]
    )  # First event to perturb.
    new_si_events_t = tfd.Binomial(
        total_count=current_state_t[0], probs=prob_t,  # Perturb SI events here
    ).sample(seed=init_events_seed)
    new_events_t = tf.tensor_scatter_nd_update(
        current_events[proposed_time_idx], [[0, 0]], new_si_events_t
    )

    return proposed_time_idx, new_events_t, current_state_t


def chain_binomial_rippler(model, current_events, seed=None):

    init_seed, ripple_seed = samplers.split_seed(
        seed, salt="chain_binomial_rippler"
    )
    src_states = _get_src_states(
        model.stoichiometry
    )  # source state of each transition

    # Transpose to [T, S/R, M]
    current_events = tf.transpose(current_events, perm=(1, 2, 0))

    # Calculate current state
    current_state = _compute_state(
        initial_state=tf.transpose(model.initial_state),
        events=current_events,
        stoichiometry=model.stoichiometry,
    )

    # Begin the ripple by sampling a time point, and perturbing the events at that timepoint
    proposed_time_idx, new_events_t, current_state_t = _initial_ripple(
        model, current_events, current_state, init_seed
    )
    new_events = tf.tensor_scatter_nd_update(
        current_events, indices=[[proposed_time_idx]], updates=[new_events_t]
    )

    # Propagate from t+1 up to end of the timeseries
    def draw_events(
        time, new_state_t, current_events_t, current_state_t, seed
    ):
        with tf.name_scope("draw_events"):

            # Calculate transition rates for current and new states
            def transition_probs(state):
                rates = tf.stack(
                    model.transition_rates(time, tf.transpose(state)), axis=-2
                )
                return 1.0 - tf.math.exp(-rates * model.time_delta)

            current_p = transition_probs(current_state_t)
            new_p = transition_probs(new_state_t)

            # Each iteration requires a transition and its
            #  associated source state.
            elems = [
                tf.reshape(x, [-1])
                for x in (
                    current_events_t,
                    tf.gather(current_state_t, indices=src_states),
                    current_p,
                    tf.gather(new_state_t, indices=src_states),
                    new_p,
                )
            ] + [
                samplers.split_seed(
                    seed,
                    n=tf.reduce_prod(current_events_t.shape),
                    salt="draw_events",
                )
            ]
            update = tf.vectorized_map(
                lambda args: _dispatch_update(*args, validate_args=False),
                elems=elems,
            )
            update = tf.reshape(update, current_events_t.shape)
            tf.debugging.assert_non_negative(update)
            return update

    def time_loop_body(t, new_events_t, new_state_t, new_events_buffer, seed):

        sample_seed, next_seed = samplers.split_seed(
            seed, salt="time_loop_body"
        )

        # Propagate new_state[t] to new_state[t+1]
        new_state_t1 = new_state_t + tf.einsum(
            "...ik,ij->...jk", new_events_t, model.stoichiometry
        )
        tf.debugging.assert_non_negative(new_state_t1, summarize=100)

        # Gather current states and events, and draw new events
        new_events_t1 = draw_events(
            t + 1,
            new_state_t1,
            current_events[t + 1],
            current_state[t + 1],
            sample_seed,
        )

        # Update new_events_buffer
        new_events_buffer = tf.tensor_scatter_nd_update(
            new_events_buffer, indices=[[t + 1]], updates=[new_events_t1]
        )

        return t + 1, new_events_t1, new_state_t1, new_events_buffer, next_seed

    def time_loop_cond(t, _1, _2, new_events_buffer, _3):
        t_stop = t < (model.num_steps - 1)
        delta_stop = tf.reduce_any(new_events_buffer != current_events)
        return t_stop & delta_stop

    _, _, _, new_events, _ = tf.while_loop(
        time_loop_cond,
        time_loop_body,
        loop_vars=(
            proposed_time_idx,
            new_events_t,
            current_state_t,
            new_events,
            ripple_seed,
        ),
    )  # new_events.shape = [T, R, M]

    new_events = tf.transpose(new_events, perm=(2, 0, 1))

    return (
        new_events,
        {
            "delta": tf.transpose(
                new_events_t - current_events[proposed_time_idx]
            ),
            "timepoint": proposed_time_idx,
            "initial_ripple": new_events_t,
            "current_state_t": tf.transpose(current_state_t),
        },
    )
