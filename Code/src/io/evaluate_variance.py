import numpy as np
import pxpy as px
import os
import pandas as pd
from scipy.stats import random_correlation


def record_obj(baseline, model):
    weights = model
    mu, A = baseline.infer()
    if weights is not None:
        weights = np.ascontiguousarray(weights)
        np.copyto(baseline.weights, weights)
        mu, A = baseline.infer()
        ll = A - np.inner(baseline.statistics, baseline.weights)
    else:
        ll = A - np.inner(baseline.statistics, baseline.weights)
    return ll


def get_bounded_distance(d, n, delta=0.8):
    c = - (np.log(1 - np.sqrt(delta)) - np.log(2)) / (np.log(d))
    return 2 * np.sqrt(((1 + c) * np.log(d)) / (2 * n))


def gen_unif_cov(n_dim, eps=1e-1):
    return np.diag(np.ones(n_dim)) * eps


def gen_random_cov(n_dim):
    try:
        eigs = np.random.rand(n_dim)
        eigs = eigs / np.sum(eigs) * eigs.shape[0]
        cov = random_correlation.rvs(eigs)
        cov = np.multiply(cov, np.sqrt(np.outer(eigs, eigs)))
        return cov
    except Exception as e:
        cov = np.random.randn(n_dim, n_dim)
        return np.dot(cov, cov.T) / n_dim


def sample_parameters(model, covtype):
    r = model[0].weights.shape[0] + 2
    n_samples = int(r ** 1)
    samples_per_model = 25
    theta_old = []
    theta_samples = []
    eps = (get_bounded_distance(model[0].weights.shape[0], model[0].num_instances) / 2) ** 2

    for i, px_model in enumerate(model):
        if covtype == "unif":
            cov = gen_unif_cov(px_model.weights.shape[0], eps=eps)
        elif covtype == "random":
            cov = gen_random_cov(px_model.weights.shape[0])
        elif covtype == "fish":
            cov = gen_semi_random_cov(px_model, px_model.num_instances, eps)
        else:
            cov = gen_unif_cov(px_model.weights.shape[0], eps=eps)

        theta_old.append(px_model.weights)

        theta_samples.append(
            np.random.multivariate_normal(px_model.weights, cov, samples_per_model).T)

    return theta_samples, theta_old


def gen_semi_random_cov(model, n_local_data, eps=0):
    a = np.insert(np.cumsum([model.states[u] * model.states[v] for u, v in model.graph.edgelist]), 0, 0)
    marginals, A = model.infer()
    marginals = marginals[:a.max()]
    cov = np.zeros((model.weights.shape[0], model.weights.shape[0]))
    rhs = np.outer(marginals, marginals)
    diag = np.diag(marginals[:model.weights.shape[0]] - marginals[:model.weights.shape[0]] ** 2)
    for x in range(a.shape[0] - 1):
        cov[a[x]:a[x + 1], a[x]:a[x + 1]] = - rhs[a[x]:a[x + 1], a[x]:a[x + 1]]
    cov -= np.diag(np.diag(cov))
    cov += np.diag(marginals - np.diag(rhs))
    cov += diag
    cov *= n_local_data

    try:
        inv = np.linalg.inv(cov)
        return inv + np.diag(np.full(model.weights.shape[0], eps))
    except np.linalg.LinAlgError:
        return np.linalg.inv(np.diag(np.diag(cov))) + np.diag(np.full(model.weights.shape[0], eps))


def main():
    dataset = ["dota2", "covertype", "susy"]
    var_dfs_low = pd.DataFrame()
    var_dfs_high = pd.DataFrame()
    varvar_dfs_low = pd.DataFrame()
    varvar_dfs_high = pd.DataFrame()
    var_aggs_low = pd.DataFrame()
    var_aggs_high = pd.DataFrame()
    save_path = os.path.join("..", "Thesis", "kapitel", "figures")

    aggregates = {data:{"none":{"avg":[[] for i in range(15)], "kl":[[] for i in range(15)], "wa":[[] for i in range(15)], "radon":[[] for i in range(15)], "var":[[] for i in range(15)]},
                  "fish":{"avg":[[] for i in range(15)], "kl":[[] for i in range(15)], "wa":[[] for i in range(15)], "radon":[[] for i in range(15)], "var":[[] for i in range(15)]},
                  "random":{"avg":[[] for i in range(15)], "kl":[[] for i in range(15)], "wa":[[] for i in range(15)], "radon":[[] for i in range(15)], "var":[[] for i in range(15)]},
                  "unif":{"avg":[[] for i in range(15)], "kl":[[] for i in range(15)], "wa":[[] for i in range(15)], "radon":[[] for i in range(15)], "var":[[] for i in range(15)]}} for data in dataset}
    for data in dataset:
        root_path = os.path.join("..", "..", "Cluster", "dota2")

        col_names = {"none": "NoCov",
                     "l2":"l2",
                     "None": "NoReg",
                     "unif":"UnifCov",
                     "fish" :"FishCov",
                     "random":"Rand",
                     "avg":"Average",
                     "radon":"Radon",
                     "wa":"LLWeighted",
                     "kl":"Bootstrap",
                     "var":"AccWeighted"}

        for experiment in os.listdir(root_path):
            if experiment == "old" or experiment == "plots":
                continue
            index = []
            path = os.path.join(root_path, experiment)
            covtype = experiment.split("_")[2]
            reg =  experiment.split("_")[-1]
            eps = float(pd.read_csv(os.path.join(path, "readme.md")).loc[14][0].split(":")[1])
            cv = []
            baselines = []
            aggs = ['avg', 'kl', 'var', 'wa', 'radon']
            agg_dict = {a:[[] for i in range(15)] for a in aggs}
            for i in range(10):
                if i == 0:
                    index.append(px.load_model(os.path.join(path, 'baseline', "px_model" + str(i))).num_instances)
                baselines.append(px.load_model(os.path.join(path, 'baseline', "px_model" + str(i))).weights)
            for h in range(10):
                cvp = os.path.join(path, str(h))
                batches = []
                for i in range(15):
                    thetas = []
                    batch = os.path.join(cvp, "batch_n" + str(i))
                    agg_dict['avg'][i].append(np.load(os.path.join(batch, "weights_mean.npy")))
                    agg_dict['kl'][i].append(np.load(os.path.join(batch, "weights_kl.npy")))
                    agg_dict['var'][i].append(np.load(os.path.join(batch, "weights_var.npy")))
                    agg_dict['wa'][i].append(np.load(os.path.join(batch, "weights_wa.npy")))
                    if covtype.lower() != "none":
                        agg_dict['radon'][i].append(np.load(os.path.join(batch, "weights_radon.npy")))
                    for j in range(10):
                        if h == 0 and j == 0:
                            index.append(px.load_model(os.path.join(batch, "dist_pxmodel " + str(j) + ".px")).num_instances)
                        thetas.append(px.load_model(os.path.join(batch, "dist_pxmodel " + str(j) + ".px")).weights)
                    batches.append(thetas)
                cv.append(batches)
            if eps == 0.05:
                df = var_dfs_high
                var_df = varvar_dfs_high
                agg_df = var_aggs_high
            else:
                df = var_dfs_low
                var_df = varvar_dfs_low
                agg_df = var_aggs_low
            average_variance = []
            variance_variance = []
            max_variance = []
            avg_acc = {a: [] for a in aggs}
            for agg in aggs:
                if agg == 'radon' and covtype == 'none':
                    continue
                if agg == 'avg' and covtype =='fish' and reg.lower()=='none':
                    print("stop")
                tmp_avg_agg = []
                for j in range(15):
                    tmp_avg_agg.append(np.mean(np.var(agg_dict[agg][j], axis=0)))
                avg_acc[agg].append(tmp_avg_agg)

            for i in range(15):
                total_var = []
                tmp_var = [a[i] for a in cv]
                for split in tmp_var:
                    total_var.append(np.var(split, axis=0))
                average_variance.append(np.mean(total_var))
                variance_variance.append(np.var(total_var))
                max_variance.append(np.max(total_var))
            for agg in aggs:
                if agg=='radon' and covtype=='none':
                    continue
                agg_df[data.capitalize() + " " + col_names[covtype] + " " + col_names[reg] + " " + col_names[agg]] = avg_acc[agg][0]
            if data.capitalize() + " " + col_names[reg] not in df.columns:
                df[data.capitalize() + " " + col_names[reg]] = np.append(np.mean(np.var(baselines, axis=0)), average_variance)
                var_df[data.capitalize() + " " + col_names[reg]] = np.append(np.mean(np.var(baselines, axis=0)),
                                                                         variance_variance)
            else:
                df[data.capitalize() + " " + col_names[reg]] += np.append(np.mean(np.var(baselines, axis=0)), average_variance)
                var_df[data.capitalize() + " " + col_names[reg]] += np.append(np.var(np.var(baselines, axis=0)),
                                                                          variance_variance)
            df.index = index
            var_df.index = index
            agg_df.index = index[1:]

        var_dfs_high.index.name = "Num Samples"
        var_dfs_low.index.name = "Num Samples"
    var_dfs_low = var_dfs_low /4
    var_dfs_high = var_dfs_high /4
    varvar_dfs_high = varvar_dfs_high /4
    varvar_dfs_low = varvar_dfs_low /4
    with open(os.path.join(save_path, "average_std_0.05.tex"), "w+") as texfile:
        texfile.write(np.sqrt(var_dfs_high).to_latex(float_format="%.2f", label="", caption="", na_rep="---"))
    with open(os.path.join(save_path, "average_std_0.1.tex"), "w+") as texfile:
        texfile.write(np.sqrt(var_dfs_low).to_latex(float_format="%.2f", label="", caption="", na_rep="---"))
    with open(os.path.join(save_path, "std_std_0.05.tex"), "w+") as texfile:
        texfile.write(np.sqrt(varvar_dfs_high).to_latex(float_format="%.2f", label="", caption="", na_rep="---"))
    with open(os.path.join(save_path, "std_std_0.1.tex"), "w+") as texfile:
        texfile.write(np.sqrt(varvar_dfs_low).to_latex(float_format="%.2f", label="", caption="", na_rep="---"))
    return var_dfs_high, var_dfs_low

if __name__ == '__main__':
    main()