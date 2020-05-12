from src.model.aggregation import RadonMachine, Mean, WeightedAverage, KL, Variance
from src.model.model import Susy as SusyModel
from src.preprocessing.sampling import Random
from src.data.metrics import sk_f1_score
from src.conf.settings import CONFIG, get_parser, CovType
from src.data.dataset import CoverType, Susy, Dota2
from multiprocessing import Process, Queue
from time import time
from guppy import hpy
import io, shlex, os
from shutil import copyfileobj

import sys
import numpy as np
import pandas as pd
import gc
from scipy.stats import random_correlation
import pxpy as px
import itertools

logger = CONFIG.get_logger()


class Coordinator(object):

    def __init__(self, data_set_name, Data, n, k, iters, h, epochs, n_models, n_test):
        self.name = data_set_name
        self.data_obj = Data
        self.num_local_samples = n
        self.k_fold = k
        self.iters = iters
        self.h = h
        self.r = 0
        self.rounds = epochs
        self.n_models = n_models
        self.save_path = os.path.join("experiments", self.name)
        self.n_test = n_test
        self.curr_split = 0
        self.mask = None
        self.curr_model = None
        self.curr_model = None
        self.sampler = None
        self.aggregates = {}

        covtype = "COV_" + CONFIG.ARGS.covtype
        reg = "REG_" + CONFIG.ARGS.reg
        curr_time = str(int(time()))
        self.experiment_path, self.seed = self.save_seed(
            os.path.join(self.save_path, "_".join([curr_time, covtype, reg])))

        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
        self.random_state = np.random.RandomState(seed=self.seed)
        self.csv_writer = io.StringIO()
        self.obj_writer = io.StringIO()
        self.ll_writer = io.StringIO()
        self.f1_writer = io.StringIO()
        print("n_data, name, obj", file=self.obj_writer)
        print("n_data, name, test_ll", file=self.ll_writer)
        self.baseline_models = []
        self.baseline_px_models = []
        self.local_models = []
        self.res = None
        self.baseline_metrics = {}
        CONFIG.write_readme(self.experiment_path)

    def baseline(self):

        models = []
        accs = []
        f1 = []
        for i in range(CONFIG.CV):
            self.data.load_cv_split(i)
            test_size = np.min([self.n_test, self.data.test.shape[0] - 1])
            model = SusyModel(self.data, path=self.name)
            self.curr_model = model
            model.train(split=None, epochs=self.rounds, iters=self.iters)

            predictions, test_ll = model.predict(n_test=test_size)
            y_pred = predictions[0][:, self.data.label_column]
            y_true = self.data.test_labels[:test_size]
            accuracy = np.where(np.equal(y_pred, y_true))[0].shape[0] / test_size
            accs.append(accuracy)
            f1.append(sk_f1_score(y_true, y_pred))
            print("GLOBAL Model " + str(i) + " : " + str(accuracy))

            if not os.path.isdir(os.path.join(self.experiment_path, 'baseline')):
                os.makedirs(os.path.join(self.experiment_path, 'baseline'))
            model.px_model[0].save(os.path.join(self.experiment_path, 'baseline', 'px_model' + str(i)))
            if CONFIG.MODELTYPE == px.ModelType.integer:
                self.baseline_px_models.append(model.px_model_scaled[0])
            else:
                self.baseline_px_models.append(model.px_model[0])
            self.record_obj(i, None, 'baseline_' + str(i))
            [self.record_test_ll(ll, 'baseline_' + str(i)) for ll in test_ll]
            baseline_path = os.path.join(self.experiment_path, 'baseline')
            np.save(os.path.join(baseline_path, 'y_true_' + str(i)), y_true)
            np.save(os.path.join(baseline_path, 'y_pred_' + str(i)), y_pred)
            np.save(os.path.join(baseline_path, 'mask_' + str(i)), self.data.mask)
            model.write_progress_hook(baseline_path, 'stats ' + str(i) + ".csv")
            self.baseline_models.append(model)
        self.baseline_metrics['acc'] = accs
        self.baseline_metrics['f1'] = f1
        for cv_idx, split in enumerate(self.data.split):
            np.save(os.path.join(self.experiment_path, 'baseline', 'split_' + str(cv_idx)), split)
        np.save(os.path.join(self.experiment_path, 'baseline', 'accuracy'), np.array(accs))
        pd.DataFrame(self.baseline_metrics).to_csv(os.path.join(baseline_path, "baseline_metrics.csv"))
        self.data.reset_cv()

    def load_model(self, path):
        pass

    def save_seed(self, path):
        os.makedirs(path)
        seed = np.random.randint(0, np.iinfo(np.int32).max, 1)
        np.save(os.path.join(path, "seed"), seed)
        return path, seed

    def load_experiment(self, path):
        mask = None
        models = [np.load(os.path.join(path, "weights", file)) for file in os.listdir(os.path.join(path, "weights"))]
        try:
            mask = np.load(os.path.join(path, "mask.npy"))
        except FileNotFoundError:
            pass
        return models, mask

    def load_seed(self, path):
        seed = np.load(os.path.join(path, "seed.npy"))
        return seed

    def aggregation_helper(self, aggregator, weights=None):
        states = self.curr_model.state_space
        test_size = np.min([self.n_test, self.data.test.shape[0] - 1])
        graph = px.create_graph(self.curr_model.edgelist)
        try:
            logger.debug("===AGGREGATION HELPER=== AGGREGATE ===")
            aggregator.aggregate(None)
            aggregate = aggregator.aggregate_models
            if aggregator.success:
                if np.sum([(states[row[0]] + 1) * (states[row[1]] + 1) for row in graph.edgelist]) != \
                        aggregate[0].shape[0]:
                    print("that's not right")
                logger.debug("===AGGREGATION HELPER=== NEW PX MODEL===")
                weights = np.ascontiguousarray(np.copy(aggregate[0]))
                logger.debug("===AGGREGATION HELPER=== PREDICT AGGREGATE===")
                predictions, test_ll = self.curr_model.predict(weights=weights, n_test=test_size)
                logger.debug("===AGGREGATION HELPER=== GET ACC AND F1===")
                y_pred = np.copy(predictions[:, self.curr_model.data_set.label_column])
                y_true = np.copy(self.data.test_labels[:test_size])
                accuracy = np.where(np.equal(y_pred, y_true))[0].shape[0] / y_true.shape[0]
                f1 = sk_f1_score(y_true, y_pred)
                logger.debug("===AGGREGATION HELPER=== RECORD AGGREGATE LL===")
                curr_ll = self.record_obj(self.curr_split, weights, aggregator.__class__.__name__)
                [self.record_test_ll(ll, aggregator.__class__.__name__) for ll in test_ll]
                logger.info(aggregator.__class__.__name__  + aggregator.hint + ": " + str(accuracy))
                logger.info(aggregator.__class__.__name__ + aggregator.hint + ": " + str(f1))
                logger.info(aggregator.__class__.__name__ + aggregator.hint + ": " + str(np.bincount(y_pred)))
                logger.debug("===AGGREGATION HELPER=== RETURN DICT===")
                return {'px_model': weights, 'y_pred': y_pred, 'y_true': y_true, 'acc': accuracy, 'f1': f1}, curr_ll
            return {}, np.infty
        except ValueError or TypeError as e:
            print(e)
            return {}, np.infty

    def aggregate(self, weights=None, idx=None, bootsmap=None):

        test_size = np.min([self.n_test, self.data.test.shape[0] - 1])
        num_models = int(self.r ** self.h)
        methods = ['mean', 'radon', 'wa', 'kl', 'var', 'bootsmap']
        aggregates = {k: [] for k in methods}
        sample_size = np.ceil(
            (self.curr_model.sample_func(self.curr_model.epoch) * self.curr_model.data_delta) / 2)
        bootsmap_size = int(np.ceil(
            (self.curr_model.sample_func(self.curr_model.epoch) * self.curr_model.data_delta) * self.n_models))
        logger.debug("===AGGREGATE=== CREATE AGGREGATORS===")
        if weights is not None:
            aggr = [Mean(weights[:num_models]),
                    RadonMachine(weights[:num_models], self.r, self.h),
                    WeightedAverage(weights[:num_models]),
                    KL(self.curr_model.px_model, graph=self.curr_model.graph,
                       states=self.curr_model.state_space, samples=None, n=int(sample_size)),
                    Variance(self.curr_model.px_model, graph=self.curr_model.graph,
                             states=self.curr_model.state_space, samples=idx, label=-1),
                    KL(self.curr_model.px_model, graph=self.curr_model.graph,
                       states=self.curr_model.state_space, samples=bootsmap, n=bootsmap_size)]
        else:
            aggr = [Mean(self.curr_model),
                    RadonMachine(self.curr_model, self.r, self.h),
                    WeightedAverage(self.curr_model),
                    KL(self.curr_model.px_model, graph=self.curr_model.graph,
                       states=self.curr_model.state_space, samples=None, n=int(sample_size)),
                    Variance(self.curr_model.px_model, graph=self.curr_model.graph,
                             states=self.curr_model.state_space, samples=idx, label=-1),
                    KL(self.curr_model.px_model, graph=self.curr_model.graph,
                       states=self.curr_model.state_space, samples=bootsmap, n=int(sample_size))
                    ]
        weights = None
        if isinstance(self.curr_model, np.ndarray):
            weights = self.curr_model
        logger.debug("===AGGREGATE=== CALL AGGREGATE===")
        best_ll = np.infty
        best_agg = None
        for name, aggregator in zip(methods, aggr):
            aggregate, ll = self.aggregation_helper(aggregator=aggregator, weights=weights)
            aggregates[name].append(aggregate)
            if ll < best_ll:
                best_ll = ll
                best_agg = aggregate['px_model']
                logger.info("NEW BEST AGGREGATE: " + name + " with Likelihood " + str(best_ll))

        return aggregates, best_agg

    def test_local_acc(self):
        local_predictions = None
        logger.debug("===RUN=== PREDICT LOCAL ACC===")
        local_predictions, test_ll = self.curr_model.predict(n_test=self.n_test)
        local_acc = []
        local_f1 = []
        local_y_pred = []
        logger.debug("===RUN=== PRINT AND RECORD LOCAL ACC===")
        test_size = np.min([self.n_test, self.data.test.shape[0] - 1])
        if not local_predictions is None:
            for local_indexer, local_pred in enumerate(local_predictions):
                y_pred = local_pred[:, self.curr_model.data_set.label_column]
                y_true = self.data.test_labels[:test_size]
                acc = np.where(np.equal(y_pred, y_true))[
                          0].shape[0] / self.data.test_labels[:self.n_test].shape[
                          0]
                local_f1.append(sk_f1_score(y_true, y_pred))
                local_acc.append(acc)
                local_y_pred.append(y_pred)
                logger.info(str(acc))
                self.record_obj(self.curr_split, np.copy(self.curr_model.px_model[local_indexer].weights),
                                "local_" + str(local_indexer))
                self.record_test_ll(test_ll[local_indexer], "local_" + str(local_indexer))
        return local_acc, local_f1, local_y_pred

    def generate_models(self):
        theta_samples = None
        theta_arr = None
        if CONFIG.COVTYPE != CovType.none:
            theta_samples, _ = self.sample_parameters(self.curr_model)
            theta_arr = np.concatenate(theta_samples, axis=1)
            del theta_samples
        test_arr = None
        if theta_arr is not None:
            for theta in theta_arr.T:
                px_map = px.Model(weights=np.ascontiguousarray(theta),
                                  states=np.copy(self.curr_model.state_space + 1),
                                  graph=px.create_graph(self.curr_model.edgelist))
                test_arr = px_map.MAP() if test_arr is None else np.vstack((test_arr, px_map.MAP()))
                px_map.delete()
                del px_map
        return theta_arr, test_arr

    def aggr_wrapper(self):
        logger.debug("Aggregating Model No. " + str(self.curr_model))
        radons = []
        vars = []
        local_acc, local_f1, local_y_pred = self.test_local_acc(self.curr_model)
        theta_arr, test_arr = self.generate_models()
        kl_samples = [np.ascontiguousarray(
            self.data.train.iloc[idx][:self.curr_model.data_delta * self.curr_model.epoch].values,
            dtype=np.uint16) for idx in self.sampler.split_idx]

        logger.debug("===RUN=== AGGREGATE LOCAL MODELS===")
        aggregation, best_agg = self.aggregate(theta_arr, kl_samples, test_arr)
        self.curr_model.best_aggregate = best_agg
        logger.debug("===RUN=== RECORD SCORES===")
        self.record_progress(aggregation, self.curr_split, local_acc,
                             self.csv_writer, 'acc')
        self.record_progress(aggregation, self.curr_split, local_f1,
                             self.f1_writer, 'f1')
        self.aggregates[self.curr_model.n_local_data] = aggregation
        radons.append(self.aggregates[self.curr_model.n_local_data]['radon'][0]['px_model'])
        self.check_convergence(radons)

    def check_convergence(self, radons):
        if len(radons) > 1:
            d = self.curr_model.get_num_of_states()
            c = - (np.log(1 - np.sqrt(0.5)) - np.log(2)) / (np.log(d))
            tmp_eps_small = 2 * np.sqrt(((1 + c) * np.log(d)) / (2 * 2000)) ** 2 / 4
            np.var(radons, axis=0)
            vars.append(np.var(radons, axis=0))
            if np.all(np.var(radons, axis=0) < tmp_eps_small):
                print("STOP")

    def train(self):
        self.data.load_cv_split(self.curr_split)
        self.curr_model = SusyModel(self.data,
                                    path=self.data.__class__.__name__, epochs=self.rounds)
        self.local_models.append(self.curr_model)
        while self.curr_model.n_local_data < self.curr_model.suff_data:
            # Training
            h = hpy()
            logger.info(h.heap())
            self.sampler = Random(self.data, n_splits=self.n_models, k=self.k_fold, seed=self.seed)
            self.sampler.create_split(self.data.train.shape, self.data.train)
            self.curr_model.train(split=self.sampler.split_idx,
                                  epochs=1,
                                  n_models=self.n_models,
                                  iters=self.iters)

            d, r, h, n = self.data.radon_number(r=self.curr_model.get_num_of_states() + 2,
                                                h=1,
                                                d=self.data.train.shape[0])
            self.r = r
            # Aggregation
            self.aggr_wrapper()
            gc.collect()
            if self.curr_model.n_local_data > self.sampler.split_idx[0].shape[0]:
                break

    def run(self):
        models = []
        k_aggregates = []
        self.sampler = None

        # Outer Cross-Validation Loop.
        for i in range(self.k_fold):
            self.curr_split = i
            aggregates = self.train()
            self.write_progress(os.path.join(self.experiment_path, str(i)), "accuracy.csv", "obj.csv", "f1.csv", "test_likelihood.csv")
            models.append(self.curr_model)
            self.curr_model.write_progress_hook(path=os.path.join(self.experiment_path, str(i)),
                                                fname="obj_progress" + ".csv")
            k_aggregates.append(aggregates)
            self.res = k_aggregates
            try:
                self.finalize(i, aggregates, self.sampler.split_idx)
                for mod in self.curr_model.px_model:
                    mod.delete()
                for _, item in self.curr_model.px_batch.items():
                    for mod in item:
                        mod.delete()
                del self.curr_model
                self.curr_model = None
                gc.collect()
            except Exception as e:
                print(e)
        return models, k_aggregates, self.sampler

    def finalize(self, i, aggregates, splits):
        mask = self.data.mask
        cv_path = os.path.join(self.experiment_path, str(i))
        for cnt, split in enumerate(splits):
            np.save(os.path.join(cv_path, "local_split_" + str(cnt)), split)
        for j, (n_data, model) in enumerate(aggregates.items()):
            sub_model_path = os.path.join(cv_path, "batch_n" + str(j))
            if not os.path.isdir(sub_model_path):
                os.makedirs(sub_model_path)
            if j == 0:
                np.save(os.path.join(cv_path, "mask"), mask)
                np.save(os.path.join(cv_path, 'edgelist'), self.curr_model.edgelist)
                np.save(os.path.join(cv_path, 'statespace'), self.curr_model.state_space)
            for k, px_model in enumerate(self.curr_model.px_batch[j + 1]):
                np.save(os.path.join(sub_model_path, "dist_weights " + str(k)), np.copy(px_model.weights))
            for m, (name, aggregation) in enumerate(model.items()):
                if m == 0:
                    np.save(os.path.join(sub_model_path, "y_true"), aggregation[0]['y_true'])
                if aggregation[0]:
                    np.save(os.path.join(sub_model_path, "y_pred_" + name), aggregation[0]['y_pred'])
                    np.save(os.path.join(sub_model_path, "weights_" + name), aggregation[0]['px_model'])
            for k, px_model in enumerate(self.curr_model.px_batch_local[j + 1]):
                px_model.save(os.path.join(sub_model_path, "dist_pxmodel " + str(k) + ".px"))

    def sample_parameters(self, model, perturb=False):
        n_samples = np.max([self.r ** self.h, self.curr_model.n_local_data * self.n_models])

        samples_per_model = int(np.ceil(n_samples / len(model.px_model)))
        theta_old = []
        theta_samples = []
        eps = (model.get_bounded_distance(model.delta) / 2) ** 2
        for i, px_model in enumerate(model.px_model):
            if CONFIG.COVTYPE == CovType.unif:
                cov = self.gen_unif_cov(px_model.weights.shape[0], eps=eps)
            elif CONFIG.COVTYPE == CovType.random:
                cov = self.gen_random_cov(px_model.weights.shape[0])
            elif CONFIG.COVTYPE == CovType.fish:
                cov = self.gen_semi_random_cov(px_model, eps)
            else:
                cov = self.gen_unif_cov(px_model.weights.shape[0], eps=eps)

            theta_old.append(px_model.weights)
            if np.mod(i, 3) == 0 and perturb:
                theta_samples.append(
                    self.random_state.multivariate_normal(px_model.weights, cov, samples_per_model).T *
                    self.random_state.multivariate_normal(
                        np.zeros(px_model.weights.shape[0]), cov, 1).ravel()[:, None]
                )
            else:
                theta_samples.append(
                    self.random_state.multivariate_normal(px_model.weights, cov, samples_per_model).T)

        return theta_samples, theta_old

    def gen_unif_cov(self, n_dim, eps=1e-1):
        return np.diag(np.ones(n_dim)) * eps

    def gen_random_cov(self, n_dim):
        try:
            eigs = self.random_state.rand(n_dim)
            eigs = eigs / np.sum(eigs) * eigs.shape[0]
            return random_correlation.rvs(eigs, random_state=self.random_state)
        except Exception as e:
            cov = self.random_state.randn(n_dim, n_dim)
            return np.dot(cov, cov.T) / n_dim

    def gen_fisher_cov(self, phi, mu):
        return np.outer(mu - phi, (mu - phi).T)

    def gen_semi_random_cov(self, model, eps=0):
        a = np.insert(np.cumsum([model.states[u] * model.states[v] for u, v in model.graph.edgelist]), 0, 0)
        marginals, A = model.infer()
        cov = np.zeros((model.weights.shape[0], model.weights.shape[0]))
        rhs = np.outer(marginals, marginals)
        diag = np.diag(marginals[:model.weights.shape[0]] - marginals[:model.weights.shape[0]] ** 2)
        for x in range(a.shape[0] - 1):
            cov[a[x]:a[x + 1], a[x]:a[x + 1]] = - rhs[a[x]:a[x + 1], a[x]:a[x + 1]]
        cov -= np.diag(np.diag(cov))
        cov += diag + np.diag(np.full(model.weights.shape[0], eps))

        return cov

    def prepare_and_run(self):

        logger.debug("=== PREPARE === DATA ===")
        self.data = self.data_obj(path=os.path.join("data", self.name), mask=self.mask, seed=self.seed,
                                  cval=self.k_fold)
        self.data.create_cv_split()
        logger.debug("=== PREPARE === BASELINE===")
        self.baseline()
        logger.debug("=== PREPARE === LOCAL AGG===")
        models, aggregate, sampler = self.run()
        logger.debug("=== PREPARE === DONE===")
        return models, aggregate

    def record_progress(self, model_dict, k, local_info, dest, metric):
        """

        Parameters
        ----------
        model_dict : dict
        d : int

        Returns
        -------

        """
        d = self.curr_model.n_local_data
        experiment_path = self.experiment_path
        write_str = str(d) + ", "
        path = os.path.join(experiment_path, str(k))

        if not os.path.isdir(path):
            header = "n_local_data, "
            os.makedirs(path)
            header += ", ".join(["local_" + metric + "_" + str(k) for k in range(len(local_info))])
            header += ", "
            header += ", ".join([name + "_" + metric for name, _ in model_dict.items()])
            print(header, file=dest)

        write_str += ", ".join([str(inf) for inf in local_info])
        for method, results in model_dict.items():
            for stats in results:
                if stats:
                    write_str += ", " + str(stats[metric])
                else:
                    write_str += ", " + 'nan'
        print(write_str, file=dest)

    def record_obj(self, i, weights, name):
        n = self.curr_model.n_local_data
        logger.debug("===RECORD OBJ=== GET BASELINE PX===")
        px_model = self.baseline_px_models[i]
        logger.debug("===RECORD OBJ=== INFER BASELINE PX===")
        mu, A = px_model.infer()
        if weights is not None:
            weights = np.ascontiguousarray(weights)
            logger.debug("===RECORD OBJ=== GET LOCAL LL===")
            np.copyto(px_model.weights, weights)
            mu, A = px_model.infer()
            ll = A - np.inner(px_model.statistics, px_model.weights)
        else:
            logger.debug("===RECORD OBJ=== GET GLOBAL LL===")
            ll = A - np.inner(px_model.statistics, px_model.weights)
        logger.debug("===RECORD OBJ=== WRITE LL===")
        obj_str = str(n) + ", " + str(name) + ", " + str(ll)
        print(obj_str, file=self.obj_writer)
        return ll

    def record_test_ll(self, ll, name):
        n = self.curr_model.n_local_data
        logger.debug("===RECORD OBJ=== WRITE LL===")
        obj_str = str(n) + ", " + str(name) + ", " + str(ll)
        print(obj_str, file=self.ll_writer)

    def write_progress(self, path, fname, fname2, fname3, fname4):
        with open(os.path.join(path, fname), "w+", encoding='utf-8') as f:
            self.csv_writer.seek(0)
            copyfileobj(self.csv_writer, f)
            del self.csv_writer
            self.csv_writer = io.StringIO()

        with open(os.path.join(path, fname2), "w+", encoding='utf-8') as f:
            self.obj_writer.seek(0)
            copyfileobj(self.obj_writer, f)
            baselines = self.obj_writer.getvalue().split('\n')
            del self.obj_writer
            self.obj_writer = io.StringIO()
            print("n_data, name, obj", file=self.obj_writer)
            for i in range(self.k_fold):
                print(baselines[i+1], file=self.obj_writer)

        with open(os.path.join(path, fname3), "w+", encoding='utf-8') as f:
            self.f1_writer.seek(0)
            copyfileobj(self.f1_writer, f)
            del self.f1_writer
            self.f1_writer = io.StringIO()

        with open(os.path.join(path, fname4), "w+", encoding='utf-8') as f:
            self.ll_writer.seek(0)
            copyfileobj(self.ll_writer, f)
            baselines = self.ll_writer.getvalue().split('\n')
            del self.ll_writer
            self.ll_writer = io.StringIO()
            print("n_data, name, obj", file=self.ll_writer)
            for i in range(self.k_fold):
                print(baselines[i+1], file=self.ll_writer)


def main():
    """
        data = Data(params)
        split = Split(params)

        model = Model(data, split)
        model.train()
        agg = Aggregation()
        agg.aggregate()
    """
    # Create Data and Model
    pass


def get_data_class(type):
    type = str.lower(type)
    choices = {'covertype': CoverType,
               'susy': Susy,
               'dota2': Dota2}
    return choices[type]


def start(cmd_args):
    try:
        data_class = get_data_class(cmd_args.data)

        CONFIG.setup(cmd_args)
        number_of_samples_per_model = 100
        coordinator = Coordinator(data_set_name=cmd_args.data,
                                  Data=data_class,
                                  n=number_of_samples_per_model,
                                  k=cmd_args.cv,
                                  iters=cmd_args.maxiter,
                                  h=cmd_args.h,
                                  epochs=cmd_args.epoch,
                                  n_models=cmd_args.n_models,
                                  n_test=cmd_args.n_test)
        result, agg = coordinator.prepare_and_run()
        del coordinator
        return
    except Exception as e:
        with open("exceptions.txt", "a+") as file:
            import traceback
            logger.error(
                "Experiment Failed in " + str(cmd_args.data) + " " + str(cmd_args.reg) + " " + str(cmd_args.covtype) + "\n")
            file.write(
                "Experiment Failed in " + str(cmd_args.data) + " " + str(cmd_args.reg) + " " + str(cmd_args.covtype) + "\n")
            file.write(str(e) + "\n")
            traceback.print_exc()
        return


if __name__ == '__main__':


    keywords = ['--data', '--covtype', '--reg', '--hoefd_eps']
    datasets = ['dota2']
    sample_parameters = ["fish", "unif", "random", "none"]
    reg = ['None', 'l2']
    eps = [1e-1, 5e-2]
    configurations = [element for element in itertools.product(*[datasets, sample_parameters, reg, eps])]
    func = lambda x: zip(keywords, x)
    kwargs = [func(x) for x in configurations]
    strargs = []
    for args in kwargs:
        strargs.append(" ".join([str(name) + " " + str(val) for name, val in args]))

    cmd_arg_list = []
    for arg in strargs:
        parser = get_parser()
        cmd_arg_list.append(parser.parse_args(shlex.split(arg)))

    parser = get_parser()
    # cmd_args = parser.parse_args()

    for cmd_args in cmd_arg_list:
        try:
            logger.debug("=== MAIN === DONE===")
            p = Process(target=start, args=(cmd_args,))
            p.start()
            p.join()
            gc.collect()
        except Exception as e:
            with open("exceptions.txt", "a+") as file:
                import traceback
                logger.error("Experiment Failed in " + str(cmd_args.data)  + " "+ str(cmd_args.reg) + " " + str(cmd_args.covtype) + "\n")
                file.write("Experiment Failed in " + str(cmd_args.data)  + " "+ str(cmd_args.reg) + " " + str(cmd_args.covtype) + "\n")
                file.write(str(e) + "\n")
                traceback.print_exc()


