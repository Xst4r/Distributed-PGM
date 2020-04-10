import importlib.util
import numpy as np

if importlib.util.find_spec('sklearn'):
    sklearn_available = True
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_squared_error


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
        if np.unique(y_true).shape[0] > 2 or np.unique(y_pred).shape[0] > 2:
            return f1_score(y_true, y_pred, average='macro')
        else:
            return f1_score(y_true, y_pred)


def sk_confusion_matrix(y_true, y_pred):
    if sklearn_available:
        return confusion_matrix(y_true, y_pred)


def sk_mse(theta_true, theta_pred):
    return mean_squared_error(theta_true, theta_pred)


def fisher_information(model):
    mu, A = model.infer()
    mu = mu[:model.weights.shape[0]]
    states = np.copy(model.states)
    edgelist = np.copy(model.graph.edgelist)

    x = np.ascontiguousarray(np.zeros(edgelist.shape[0] + 1 , dtype=np.uint16) - 1)
    test_x = np.ascontiguousarray(np.zeros(edgelist.shape[0] + 1 , dtype=np.uint16))
    fisher_matrix = None
    offset = np.insert(np.cumsum([states[u] * states[v] for u,v in edgelist]), 0, 0)
    node_offset = np.insert(np.cumsum(states), 0, 0)
    for idx, (u, v) in enumerate(edgelist):
        for i in range(states[u]):
            for j in range(states[v]):
                x[u] = i
                x[v] = j
                mu_, _ = model.infer(observed=x)
                mu_ij = mu_[:model.weights.shape[0]]
                prob_uv = model.prob(u, i, v, j)
                try:
                    if not (prob_uv == model.prob(u, i, v, j)):
                        print("Error: " + str(prob_uv) +  str(model.prob(u, i, v, j)))
                    model.prob(-1, i, 100, j)
                    model.prob(-1, 500, 100, 300)
                except (IndexError, ValueError):
                    print("Error: " + "e= (" + str(u) + ',' + str(v) + ")" + " state: " + str(i) + " " + str(j))
                fisher_matrix = mu_ij * prob_uv if fisher_matrix is None else np.vstack((fisher_matrix, mu_ij * prob_uv))
        x[u] = -1
        x[v] = -1
    return fisher_matrix
