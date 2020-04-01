import importlib
import numpy as np

if importlib.util.find_spec('sklearn'):
    sklearn_available = True
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


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


def _accuracy(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Predictions and Labels have to be of samel length")
    return np.where(y_true == y_pred)[0]/y_true.shape[0]


def sk_accuracy(y_true, y_pred):
    if sklearn_available:
        return accuracy_score(y_true, y_pred)
    else:
        return _accuracy(y_true, y_pred)


def sk_f1_score(y_true, y_pred):
    if sklearn_available:
        return f1_score(y_true, y_pred)


def sk_confusion_matrix(y_true, y_pred):
    if sklearn_available:
        return confusion_matrix(y_true, y_pred)


def fisher_information(model):
    mu, A = model.infer()
    states = np.copy(model.states)
    edgelist = np.copy(model.graph.edgelist)

    x = np.ascontiguousarray(np.zeros(edgelist.shape[0] + 1 , dtype=np.uint16) - 1)
    fisher_matrix = None
    offset = 0
    for u, v in edgelist:
        for i in range(states[u]):
            for j in range(states[v]):
                x[u] = i
                x[v] = j
                mu_i, _ = model.infer(observed=x)
                phi_x = model.phi(x)
                assert phi_x[offset + i*states[u] + j] == 1
                fisher_matrix = mu_i * mu[offset + i*states[u] + j] if fisher_matrix is None else np.hstack((fisher_matrix, mu_i * mu[offset + i*j]))
        offset += states[u] * states[v]
        x[u] = -1
        x[v] = -1
    return fisher_matrix
