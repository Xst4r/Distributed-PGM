import numpy as np


def squared_l2_regularization(state_p):
    state = state_p.contents
    lam = 0.1
    np.copyto(state.gradient, state.gradient + 2.0 * lam * state.weights)


def prox_l1(state_p):
    state = state_p.contents
    l = state.lam * state.stepsize

    x = state.weights_extrapolation - state.stepsize * state.gradient

    np.copyto(state.weights, 0, where=np.absolute(x) < l)
    np.copyto(state.weights, x - l, where=x > l)
    np.copyto(state.weights, x + l, where=-x > l)


def default(state_p):
    return
