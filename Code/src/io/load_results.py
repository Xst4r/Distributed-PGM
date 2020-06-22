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
import pxpy as px
from src.conf.settings import CONFIG
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from network2tikz import plot

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
        plt.close(fig)

def plot_obj(objs, accs, f1s):
    x = np.unique(objs[0]['n_data'])
    obj_names = [("Mean", "Average"),
             ("WeightedAverage", "LL-Weighted"),
             ("KL", "Bootstrap"),
                 ("RadonMachine", "RadonMachine"),
             ("Variance", "Acc. Weighted"),
            ]
    acc_names = [("mean_acc", "Average"),
             ("wa_acc", "LL-Weighted"),
             ("kl_acc", "Bootstrap"),
            ("radon_acc", "RadonMachine"),
             ("var_acc", "Acc. Weighted")
            ]
    markers = {"Average": "o",
     "LL-Weighted": "o",
     "Bootstrap": "+",
     "RadonMachine": "o",
     "Acc. Weighted": "+"
    }
    names = [obj_names, acc_names, acc_names]
    fig, axs = plt.subplots(nrows=3, ncols=len(objs), dpi=250, figsize=(16, 12), sharex=True, sharey="row")
    i = 0
    ordered_objs = []
    for obj in objs:
        new_df = pd.DataFrame()
        for name, elem in obj.groupby('name'):
            new_df[name] = elem['normalized_obj'].to_numpy()
        new_df.index = x
        if "RadonMachine" not in new_df.columns.str.strip():
            new_df.insert(loc=2, value=np.full(new_df.shape[0], fill_value=np.nan), column="RadonMachine")
        ordered_objs.append(new_df)
    if np.max([np.nanmax(ords) for ords in ordered_objs]) / np.min([np.nanmax(ords) for ords in ordered_objs]) > 100:
        axs[0,0].set_yscale('log')
        tmp_a = np.min([np.nanmin(ords) for ords in ordered_objs if np.nanmin(ords) > 0])
        axs[0,0].set_ylim(bottom=1*10**(np.floor(np.log10(np.abs(tmp_a)))), top=1e4)
    else:
        axs[0, 0].set_yscale('log')
    for acc, f1 in zip(accs,f1s):
        if "radon_acc" not in acc.columns.str.strip():
            acc.insert(loc=2, value=np.full(acc.shape[0], fill_value=np.nan), column="radon_acc")
            f1.insert(loc=2, value=np.full(f1.shape[0], fill_value=np.nan), column="radon_acc")

    for j, row in enumerate([ordered_objs, accs, f1s]):
        for i, data in enumerate(row):
            ax = axs[j, i]
            data_names = names[j]
            data.columns = data.columns.str.strip()
            data.index = x
            for old, new in data_names:
                data = rename_and_drop(data, old, new)
            tmp_average = None
            for name in [a[1] for a in data_names]:
                ax.plot(name, data=data, label=name, marker=markers[name], linestyle="--", alpha=0.7)
                ax.legend()

    local_avg = []
    local_var = []
    for j, row in enumerate([ordered_objs, accs, f1s]):
        for i, data in enumerate(row):
            local_avg.append(data[[col for col in data.columns if col.startswith("local")]].mean(axis=1))
            local_var.append(data[[col for col in data.columns if col.startswith("local")]].var(axis=1))

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.plot(x, local_avg[axs.shape[1] * i + j], label="Local Average", marker="x", c="black")
            ax.grid(True, "major", "y")
            ax.legend()
    return fig, axs


def rename_and_drop(df, col, new_col):
    if col == new_col:
        return df
    df[new_col] = df[col]
    df = df.drop([col], axis=1)
    return df


def reorder_df(df, order):
    return df[[df.columns[i] for i in order]]


def write_tables(tables, stats, path, dataset):
    order = [1, 2, 0, 3, 4, 5]
    names = [("Local_Average", "Local Average"),
             ("Mean", "Average"),
             ("WeightedAverage", "LL-Weighted"),
             ("KL", "Bootstrap"),
             ("Variance", "Acc. Weighted"),
            ]

    for i, table in enumerate(tables):
        label = "tab:" + str(i)
        caption = "Relative Likelihood when compared to the " \
                  "baseline likelihood on the test split. " \
                 "This table shows the results for {} using {} " \
                  "sampling with epsilon {} and {} regularization.".format(stats[i][1]['data'],
                                                                           stats[i][1]['covtype'],
                                                                           stats[i][1]['hoefd_eps'],
                                                                           stats[i][1]['Regularization'])
        if "RadonMachine" not in table.columns:
            table.insert(loc=2, value=np.full(table.shape[0], fill_value=np.nan), column="RadonMachine")
        tmp = table
        for old, new in names:
            tmp = rename_and_drop(tmp, old, new)
        tmp = reorder_df(tmp, order)


        with open(os.path.join(path, dataset + "_" + "table_" + str(i) + ".tex"), "w+") as texfile:
            texfile.write(tmp.to_latex(float_format="%.2f", label=label, caption=caption, na_rep="---"))


def load_results(experiment):
    baseline_stats = {}
    agg_stats = []
    graph = None
    objs = None
    test_objs = None
    for cv_split in os.listdir(experiment):

        # -------------------------------
        # Load Baseline Results
        # -------------------------------
        if cv_split == 'baseline':
            b_path = os.path.join(experiment, cv_split)
            baseline_stats['baseline_metrics'] =  pd.read_csv(os.path.join(b_path, "baseline_metrics.csv"))
            graph_model = px.load_model(os.path.join(b_path, "px_model0"))
            graph= (graph_model.graph.edgelist, graph_model.states)

        # -------------------------------
        # Load CV Results
        # -------------------------------
        elif os.path.isdir(os.path.join(experiment, cv_split)):
            curr_results = {}
            a_path = os.path.join(experiment, cv_split)

            # Load Accuracy
            curr_results['acc'] = pd.read_csv(os.path.join(a_path, "accuracy.csv"))

            # Load F1 Score
            curr_results['f1'] = pd.read_csv(os.path.join(a_path, "f1.csv"), header=None, names=curr_results['acc'].columns)

            # -------------------------------
            # Likelihood evaluated on Baseline
            #
            # File is obj.csv
            # -------------------------------
            if cv_split == '0':
                curr_results['obj'] = pd.read_csv(os.path.join(a_path, "obj.csv"))
                objs = curr_results['obj'][0:10]
                objs.columns = objs.columns.str.strip()
            else:
                curr_results['obj'] = pd.read_csv(os.path.join(a_path, "obj.csv"))
                curr_results['obj'].columns = agg_stats[0]['obj'].columns

            curr_results['obj'].columns = curr_results['obj'].columns.str.strip()
            curr_results['obj']['obj'] = curr_results['obj']['obj'].apply(float)
            curr_results['obj'] = curr_results['obj'][10:]

            # -------------------------------
            # Likelihood evaluated on Test Split
            #
            # File is test_likelihood.csv
            # -------------------------------
            if cv_split == '0':
                curr_results['test_ll'] = pd.read_csv(os.path.join(a_path, "test_likelihood.csv"))
                test_objs = curr_results['test_ll'][0:10]
                test_objs.columns = test_objs.columns.str.strip()
            else:
                curr_results['test_ll'] = pd.read_csv(os.path.join(a_path, "test_likelihood.csv"))
                curr_results['test_ll'].columns = agg_stats[0]['test_ll'].columns

            curr_results['test_ll'].columns = curr_results['test_ll'].columns.str.strip()
            curr_results['test_ll']['test_ll'] = curr_results['test_ll']['test_ll'].apply(float)
            curr_results['test_ll'] = curr_results['test_ll'][10:]

            agg_stats.append(curr_results)
        else:
            pass
    return baseline_stats, agg_stats, objs, test_objs, graph


def process_obj(norm_objs, col_name):
    avg_obj = None
    avg_norm_obj = None
    var_obj = None
    vars_norm = None
    for df in norm_objs:
        if vars_norm is None:
            var_obj = pd.DataFrame(df[col_name])
            var_obj.index = df.index
            vars_norm = pd.DataFrame(df['normalized_obj'])
            vars_norm.index = df.index
        else:
            var_obj = pd.concat([var_obj, df[col_name]], axis=1)
            vars_norm = pd.concat([vars_norm, df['normalized_obj']], axis=1)
        avg_obj = df[col_name].to_numpy() if avg_obj is None else avg_obj + df[col_name].to_numpy()
        avg_norm_obj = df['normalized_obj'].to_numpy() if avg_norm_obj is None else avg_norm_obj + df[
            'normalized_obj'].to_numpy()
    tmp = pd.concat([df['n_data'].reset_index(drop=True), df['name'].reset_index(drop=True),
                     var_obj.mean(axis=1).reset_index(drop=True),
                     vars_norm.mean(axis=1).reset_index(drop=True)],
                    axis=1, keys=[n for n in df.columns])
    tmp.columns = df.columns
    tmp_std = pd.concat([df['n_data'].reset_index(drop=True), df['name'].reset_index(drop=True),
                     var_obj.std(axis=1).reset_index(drop=True),
                     vars_norm.std(axis=1).reset_index(drop=True)],
                    axis=1, keys=[n for n in df.columns])
    tmp_std.column = ['n_data', 'name', 'obj_std', 'obj_norm_std']
    return tmp, tmp_std


def normalize_obj(results):
    average_accuracy = []
    var_accuracy = []
    average_f1 = []
    var_f1 = []
    average_objs = []
    var_objs = []
    average_test_objs = []
    var_test_objs = []
    experiment_obj = []
    experiment_test_obj = []
    for stats, result in results:
        baseline, local, objs, test_ll, _ = result
        accs = []
        f1 = []
        norm_objs = []
        normalized_test_objs = []

        for i, data in enumerate(local):
            accs.append(pd.concat([data['acc']['n_local_data'],
                                   data['acc'].iloc[:,1:].astype(np.float64) -  result[0]['baseline_metrics']['acc'][i]], axis=1))
            f1.append(pd.concat([data['f1']['n_local_data'],
                                data['f1'].iloc[:,1:].astype(np.float64) - result[0]['baseline_metrics']['f1'][i]], axis=1))
            #accs.append(result[0]['baseline_metrics']['acc'][i] - data['acc'])
            #f1.append(result[0]['baseline_metrics']['f1'][i] - data['f1'])

            baseline_obj = objs['obj'].iloc[i]
            test_obj = test_ll['test_ll'].iloc[i]

            data['obj']['normalized_obj'] = data['obj']['obj'] - np.float64(baseline_obj)
            data['test_ll']['normalized_obj'] = data['test_ll']['test_ll'] - np.float64(test_obj)
            data['obj']['normalized_obj'][data['obj']['normalized_obj'] < 0] = np.nan
            data['test_ll']['normalized_obj'][data['test_ll']['normalized_obj'] < 0] = np.nan
            norm_objs.append(data['obj'])
            normalized_test_objs.append(data['test_ll'])

        average_accuracy.append(pd.concat(accs).groupby(level=0).mean())
        var_accuracy.append(pd.concat(accs).groupby(level=0).var())
        average_f1.append(pd.concat(f1).groupby(level=0).mean())
        var_f1.append(pd.concat(f1).groupby(level=0).var())

        tmp, std = process_obj(norm_objs, col_name='obj')
        average_objs.append(tmp)
        tmp, std = process_obj(normalized_test_objs, col_name='test_ll')
        average_test_objs.append(tmp)
        experiment_obj.append(norm_objs)
        experiment_test_obj.append(normalized_test_objs)

    return {'avg_acc': average_accuracy,
            'avg_f1': average_f1,
            'avg_obj': average_objs,
            'avg_test_obj': average_test_objs,
            'baseline_ll': experiment_obj,
            'test_set_ll':experiment_test_obj}

def color_tree(g, root, col1="red", col2="blue", labelcol="red"):
    """
    Easy two coloring of trees from chosen root.
    Parameters
    ----------
    g :
    root :
    col1 :
    col2 :
    labelcol :

    Returns
    -------

    """
    from heapq import heappush, heappop
    gc = {True: col1, False:col2}
    rev_gc = {col1: True, col2:False}
    cols = {i: None for i in range(len(g))}
    current_color = True
    q = []
    heappush(q, root)
    while q:
        node = heappop(q)
        for v in g.neighbors(node):
            if cols[v] is None:
                heappush(q, v)
            else:
                current_color = not rev_gc[cols[v]]
        cols[node] = gc[current_color]
    if labelcol:
        cols[root] = labelcol
    return cols


def draw_graph(i, g, k, fname, node_size=500):
    pos = nx.spring_layout(g, k=k * 1 / np.sqrt(len(g.nodes())), iterations=100)
    res = color_tree(g, 0, "tugreen", "tuorange")
    plt.figure(3, figsize=(17, 13))
    nx.draw_networkx(g, node_size=node_size, pos=pos)
    plt.savefig(fname + "_" + str(i))
    plt.close()
    visual_style = {}
    visual_style['vertex_size'] = .5
    visual_style['vertex_opacity'] = .7
    visual_style['layout'] = pos
    visual_style['canvas'] = (17, 15)
    visual_style['margin'] = 1
    visual_style['vertex_color'] = res
    visual_style['edge_width'] = 0.1
    visual_style['edge_color'] = 'black'
    plot(g, fname + "_" + str(i) + '.tex', **visual_style)


def main(path):
    results = []
    sample_parameters = ["fish",'random','unif', 'none']
    reg = ['None', 'l2']
    eps = [1e-1, 5e-2]
    raw_stats = []
    configurations = [element for element in itertools.product(*[reg, eps])]
    for experiment in os.listdir(path):
        if experiment not in ["old", "plots"] and os.path.isdir(os.path.join(path,experiment)):
            stats = pd.read_csv(os.path.join(path, experiment, "readme.md"), header=None, sep=":")
            stats.index = stats[0].str.strip()
            results.append((stats, load_results(os.path.join(path, experiment))))
            raw_stats.append(stats)

    cv_scores = normalize_obj(results)
    grouped_by_cov = []
    for reg, eps in configurations:
        idx = []
        for i,  res in enumerate(results):
            if res[0].loc['reg'][1].strip() == reg.strip() and float(res[0].loc['hoefd_eps'][1]) == eps:
                idx.append(i)
        grouped_by_cov.append(idx)

    tables = []
    for experiment in cv_scores['avg_obj']:
        table = None
        for name, df in experiment.groupby('name'):
            df.index = df['n_data']
            df = df.drop(['n_data', 'name'], axis=1)
            if table is None:
                df[name] = df['normalized_obj']
                df = df.drop(['normalized_obj', 'obj'], axis=1)
                table = df
            else:
                table[name] = df['normalized_obj']
        table['Local_Average'] = table.iloc[:, 5:].mean(axis=1)
        table.columns = table.columns.str.strip()
        table = table.drop(['local_0','local_1','local_2','local_3','local_4','local_5','local_6','local_7','local_8','local_9'], axis=1)
        tables.append(table)
    figure_path = os.path.join(path, "..", ".." ,"Distributed-PGM", "Thesis", "kapitel" , "figures")
    for reg, eps in configurations:
        prepare_metrics(raw_stats, cv_scores, figure_path, path.split("\\")[-1] + "_" + reg + "_" + str(eps), reg, str(eps))
    plot_covs(grouped_by_cov, cv_scores['avg_obj'])
    for i, result in enumerate(results):
        g = nx.from_edgelist(result[1][-1][0])
        draw_graph(i, g, k=3, fname=os.path.join(os.path.join(figure_path, "graphs"),  result[0][1]['data'].strip() + "_" + 'graph'))
    write_tables(tables, raw_stats, os.path.join(figure_path, "tables"), dataset=path.split("\\")[-1])

    return results


def prepare_metrics(stats, cv_scores, path, fname, reg="None", eps="0.05"):
    x_titles = {"fish":"Diagonal Fisher Information", "none":"No Covariance", "unif":"Uniform", "random":"Davies&Higham Rnd. Corr"}
    y_titles = ["Rel. Average LL", "Rel. Accuracy", "Rel. F1-Score"]
    order = ["none", "unif", "random", "fish"]
    a = [stat.drop(0, axis=1) for stat in stats]
    b = [c.iloc[:, 0].str.strip() for c in a]
    obj_plot_idx = np.argwhere([c['reg'] == reg and c['hoefd_eps'] == eps for c in b]).flatten()
    tmp_a = [stats[i].loc['covtype', 1].strip() for i in obj_plot_idx]
    obj_plot_idx = [{a:b for a,b in zip(tmp_a, obj_plot_idx)}[i] for i in order]
    x_title = [x_titles[stats[i].loc['covtype', 1].strip()] for i in obj_plot_idx]
    tmp_list = []
    for i in obj_plot_idx:
        tmp_list.append((cv_scores['avg_obj'][i], cv_scores['avg_acc'][i], cv_scores['avg_f1'][i]))
    fig, axs = plot_obj([a[0] for a in tmp_list], [a[1] for a in tmp_list], [a[2] for a in tmp_list])
    for i in range(4):
        axs[0,i].set_title(str(x_title[i]))
        axs[2,i].set_xlabel("Number of Samples per Learner")
    for j in range(3):
        axs[j,0].set_ylabel(y_titles[j])
    #if reg.capitalize() == "None":
    #    fig.suptitle("Results for the {} data set without regularization and {}  = {}".format(path.split("\\")[-1].capitalize(), r'$\epsilon$', str(eps)))
    #else:
    #    fig.suptitle("Results for the {} data set with {} regularization and {} = {}".format(path.split("\\")[-1].capitalize(), reg, r'$\epsilon$' ,str(eps)))
    fig.tight_layout()
    fig.savefig(os.path.join(path, fname + "_neg_relative.pdf"), bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    folders = ['dota2', "covertype", 'susy']
    for folder in os.listdir(EXPERIMENT_DIR):
        if folder in folders:
            main(os.path.join(EXPERIMENT_DIR, folder))