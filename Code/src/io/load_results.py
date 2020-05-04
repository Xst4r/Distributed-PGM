#   Experiment
#   ---- CV Split
#   ----    Batch
#   ----    obj.csv
#   ----    accuracy.csv
#   ----    f1.csv
#   ----    ...
#   ---- Baseline
#   ----    accuracy.csv
#   ----    baseline_metrics.csv

import os
import pandas as pd
import numpy as np
from src.conf.settings import CONFIG
import matplotlib.pyplot as plt
import itertools

EXPERIMENT_DIR = os.path.join(CONFIG.ROOT_DIR, "..", "..", "Cluster")

def plot_covs(groups, objs):
    sample_parameters = ["fish",'none','random', 'unif']
    agg_methods = ['KL', 'Mean', 'RadonMachine', 'WeightedAverage', 'Variance']
    markers=['x', 'o', '1', 'p']
    opac = [0.3, 0.5, 0.7, 0.9]
    width = [5,12,19,26]
    color = ['black', 'blue', 'orange','red']
    linestyle=['dashed', 'dotted', 'dashed', 'dotted']

    for m in agg_methods:
        fig, axs = plt.subplots(2, 2, dpi=400, figsize=(12, 12))
        for i, group in enumerate(groups):
            for j, idx in enumerate(group):
                curr_obj = objs[idx]
                data = curr_obj[curr_obj.name.str.strip() == m]
                #axs[np.unravel_index(i, axs.shape)].bar('n_data', 'normalized_obj', data=data,  alpha=opac[j], width=width[j], color=color[j])
                axs[np.unravel_index(i, axs.shape)].plot('n_data', 'normalized_obj', data=data, label=sample_parameters[j], marker='x', alpha=opac[j], linestyle=linestyle[j], color=color[j])
                axs[np.unravel_index(i, axs.shape)].legend()
        plt.savefig(m + ".png")


def plot_obj(obj):
    x = np.unique(obj['n_data'])
    fig, axs = plt.subplots(1, 5, dpi=250, figsize=(25, 4), sharey=True)
    i = 0
    local_avg = None
    for name, elem in obj.groupby('name'):
        if 'local' not in name:
            axs[i].plot('n_data', 'normalized_obj', data=elem, label=name, marker="x")
            axs[i].legend()
            i+=1
        else:
            local_avg = elem['normalized_obj'].to_numpy() if local_avg is None else local_avg +  elem['normalized_obj'].to_numpy()
    local_avg = local_avg / 10
    for ax in axs:
        ax.plot(x, local_avg, label="Local Average", linestyle="--", marker="x")
        ax.legend()
    axs[0].set_ylabel('- LogLikelihood')
    plt.show()


def load_results(experiment):
    baseline_stats = {}
    agg_stats = []
    objs = None
    for cv_split in os.listdir(experiment):
        if cv_split == 'baseline':
            b_path = os.path.join(experiment, cv_split)
            baseline_stats['baseline_metrics'] =  pd.read_csv(os.path.join(b_path, "baseline_metrics.csv"))
        elif os.path.isdir(os.path.join(experiment, cv_split)):
            curr_results = {}
            a_path = os.path.join(experiment, cv_split)
            curr_results['acc'] = pd.read_csv(os.path.join(a_path, "accuracy.csv"))
            curr_results['f1'] = pd.read_csv(os.path.join(a_path, "f1.csv"), header=None, names=curr_results['acc'].columns)
            if cv_split == '0':
                curr_results['obj'] = pd.read_csv(os.path.join(a_path, "obj.csv"))
                objs = curr_results['obj'][0:10]
                objs.columns = objs.columns.str.strip()
                curr_results['obj'] = curr_results['obj'][10:]
            else:
                curr_results['obj'] = pd.read_csv(os.path.join(a_path, "obj.csv"), header= None)
                curr_results['obj'].columns = agg_stats[0]['obj'].columns
            curr_results['obj'].columns = curr_results['obj'].columns.str.strip()
            agg_stats.append(curr_results)
        else:
            pass
    return baseline_stats, agg_stats, objs


def main(path):
    results = []
    sample_parameters = ["fish",'random','unif', 'none']
    reg = ['None', 'l2']
    eps = [1e-1, 5e-2]
    configurations = [element for element in itertools.product(*[reg, eps])]
    for experiment in os.listdir(path):
        stats = pd.read_csv(os.path.join(path, experiment, "readme.md"), header=None, sep=":")
        stats.index = stats[0].str.strip()
        results.append((stats, load_results(os.path.join(path, experiment))))
    m_accs = []
    m_f1 = []
    m_objs = []
    exp_obj = []
    for stats, result in results:
        baseline, local, objs = result
        accs = []
        f1 = []
        norm_objs = []
        for i, data in enumerate(local):
            baseline_obj = objs['obj'].iloc[i]
            accs.append(data['acc'])
            f1.append(data['f1'])
            data['obj']['normalized_obj'] = data['obj']['obj'] - baseline_obj
            norm_objs.append(data['obj'])
        m_accs.append(pd.concat(accs).groupby(level=0).mean())
        m_f1.append(pd.concat(f1).groupby(level=0).mean())
        avg_obj = None
        avg_norm_obj = None
        for loc_obj in norm_objs:
            avg_obj = loc_obj['obj'].to_numpy() if avg_obj is None else avg_obj + loc_obj['obj'].to_numpy()
            avg_norm_obj = loc_obj['normalized_obj'].to_numpy() if avg_norm_obj is None else avg_norm_obj + loc_obj['normalized_obj'].to_numpy()
        tmp = pd.concat([loc_obj['n_data'], loc_obj['name'], pd.Series(avg_obj/len(norm_objs)), pd.Series(avg_norm_obj/len(norm_objs))], axis=1, keys=[n for n in loc_obj.columns])
        tmp.columns = loc_obj.columns
        m_objs.append(tmp)
        exp_obj.append(norm_objs)
    grouped_by_cov = []
    for reg, eps in configurations:
        idx = []
        for i,  res in enumerate(results):
            if res[0].loc['reg'][1].strip() == reg.strip() and float(res[0].loc['hoefd_eps'][1]) == eps:
                idx.append(i)
        grouped_by_cov.append(idx)
    plot_covs(grouped_by_cov, m_objs)
    plot_obj(m_objs[0])
    return results


if __name__ == '__main__':
    for folder in os.listdir(EXPERIMENT_DIR):
        main(os.path.join(EXPERIMENT_DIR, folder))