from src.model.aggregation import RadonMachine, Mean, WeightedAverage, KL, Variance
from src.model.model import Susy as SusyModel
from src.preprocessing.sampling import Random
from src.conf.settings import CONFIG, get_parser, CovType
from src.data.dataset import CoverType, Susy, Dota2
from time import time
import io, shlex, os
from shutil import copyfileobj

import numpy as np
from scipy.stats import random_correlation
import pxpy as px
import itertools
logger = CONFIG.get_logger()


class Coordinator(object):

    def __init__(self, data_set_name, Data, exp_loader, n, k, iters, h, epochs, n_models, n_test):
        self.name = data_set_name
        self.data_obj = Data
        self.exp_loader = exp_loader
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
        self.model_loader = None
        self.mask = None
        if self.exp_loader is not None:
            self.experiment_path = os.path.join(self.save_path, "_".join(
                [str(self.num_local_samples), str(self.iters), str(self.rounds)]) + "_" + str(self.exp_loader))
            self.seed = self.load_seed(self.experiment_path)
            self.model_loader, self.mask = self.load_experiment(self.experiment_path)
        else:
            covtype = "COV_" + CONFIG.ARGS.covtype
            reg = "REG_" + CONFIG.ARGS.reg
            curr_time = str(int(time()))
            self.experiment_path, self.seed = self.save_seed(os.path.join(self.save_path, "_".join([curr_time, covtype, reg])))

        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
        self.random_state = np.random.RandomState(seed=self.seed)
        self.csv_writer = io.StringIO()
        self.obj_writer = io.StringIO()
        print("n_data, name, obj", file=self.obj_writer)
        self.baseline_models = []
        self.res = None
        CONFIG.write_readme(self.experiment_path)

    def baseline(self, data):

        models = []
        accs = []
        for i in range(CONFIG.CV):
            data.load_cv_split(i)
            test_size = np.min([self.n_test, data.test.shape[0] - 1])
            model = SusyModel(data, path=self.name)
            model.train(split=None, epochs=self.rounds, iters=self.iters)

            predictions = model.predict(n_test=test_size)
            y_pred = predictions[0][:, data.label_column]
            y_true = data.test_labels[:test_size]
            accuracy = np.where(np.equal(y_pred, y_true))[0].shape[0] / data.test_labels[:test_size].shape[0]
            accs.append(accuracy)
            print("GLOBAL Model " + str(i) + " : " + str(accuracy))

            if not os.path.isdir(os.path.join(self.experiment_path, 'baseline')):
                os.makedirs(os.path.join(self.experiment_path, 'baseline'))
            model.px_model[0].save(os.path.join(self.experiment_path, 'baseline', 'px_model' + str(i)))
            if CONFIG.MODELTYPE == px.ModelType.integer:
                self.baseline_models.append(model.px_model_scaled[0])
            else:
                self.baseline_models.append(model.px_model[0])
            self.record_obj(i, None, model.px_model[0].num_instances, 'baseline_' + str(i))
            np.save(os.path.join(self.experiment_path, 'baseline', 'y_true_' + str(i)), y_true)
            np.save(os.path.join(self.experiment_path, 'baseline', 'y_pred_' + str(i)), y_pred)
            np.save(os.path.join(self.experiment_path, 'baseline', 'mask_' + str(i)), data.mask)
            model.write_progress_hook(os.path.join(self.experiment_path, 'baseline'), 'stats ' + str(i) + ".csv")
            models.append(model)
        np.save(os.path.join(self.experiment_path, 'baseline', 'split'), np.stack(data.split))
        np.save(os.path.join(self.experiment_path, 'baseline', 'accuracy'), np.array(accs))
        data.reset_cv()
        return models

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

    def aggregation_helper(self, model, aggregator, data, graph, states, weights=None):
        test_size = np.min([self.n_test, data.test.shape[0] - 1])
        try:
            aggregator.aggregate(None)
            aggregate = aggregator.aggregate_models
            if aggregator.success:
                if np.sum([(states[row[0]] + 1) * (states[row[1]] + 1) for row in graph.edgelist]) != \
                        aggregate[0].shape[0]:
                    print("that's not right")
                aggregate_model = px.Model(weights=np.ascontiguousarray(aggregate[0]), graph=graph,
                                           states=np.ascontiguousarray(states + 1))
                predictions = model.predict(aggregate_model, test_size)
                y_pred = np.copy(predictions[:, model.data_set.label_column])
                y_true = np.copy(data.test_labels[:test_size])
                accuracy = np.where(np.equal(y_pred, y_true))[0].shape[0] / y_true.shape[0]
                self.record_obj(self.curr_split, aggregate_model.weights, model.n_local_data, aggregator.__class__.__name__)
                if accuracy < 0.4:
                    print("test")
                logger.info(aggregator.__class__.__name__ + ": " + str(accuracy))
                return {'px_model': aggregate_model, 'y_pred': y_pred, 'y_true': y_true, 'acc': accuracy}
            return {}
        except ValueError or TypeError as e:
            print(e)
            return {}

    def train(self, data, i, sampler, experiment_path, model=None):
        model.train(split=sampler.split_idx,
                    epochs=1,
                    n_models=self.n_models,
                    iters=self.iters)

        return model

    def aggregate(self, distributed_models, data, graph, states, weights=None, idx=None):
        test_size = np.min([self.n_test, data.test.shape[0] - 1])
        methods = ['mean', 'radon', 'wa', 'kl', 'var']
        aggregates = {k: [] for k in methods}
        sample_size = np.ceil(
            (distributed_models.sample_func(distributed_models.epoch) * distributed_models.data_delta) / 2)
        if weights is not None:
            aggr = [Mean(weights),
                    RadonMachine(weights, self.r, self.h),
                    WeightedAverage(weights),
                    KL(distributed_models.px_model, graph=distributed_models.graph,
                       states=distributed_models.state_space, samples=None, n=int(sample_size)),
                    Variance(distributed_models.px_model, graph=distributed_models.graph,
                             states=distributed_models.state_space, samples=idx, label=-1)]
        else:
            aggr = [Mean(distributed_models),
                    RadonMachine(distributed_models, self.r, self.h),
                    WeightedAverage(distributed_models),
                    KL(distributed_models.px_model, graph=distributed_models.graph,
                       states=distributed_models.state_space, samples=None, n=int(sample_size)),
                    Variance(distributed_models.px_model, graph=distributed_models.graph,
                             states=distributed_models.state_space, samples=idx, label=-1)]
        weights = None
        if isinstance(distributed_models, np.ndarray):
            weights = distributed_models
        for name, aggregator in zip(methods, aggr):
            aggregates[name].append(
                self.aggregation_helper(model=distributed_models,
                                        aggregator=aggregator,
                                        data=data,
                                        weights=weights,
                                        graph=graph,
                                        states=states))

        return aggregates

    def run(self, data, loaded_model):
        models = []
        k_aggregates = []
        sampler = None

        if loaded_model is not None:
            models = loaded_model
            dummy_model = SusyModel(data, path="SUSY")
            r = loaded_model[0].shape[0]
            d, r, h, n = data.radon_number(r=r + 2, h=self.h, d=data.train.shape[0])
        else:
            # Outer Cross-Validation Loop.
            for i in range(self.k_fold):
                self.curr_split = i
                aggregates = {}
                data.load_cv_split(i)
                model = SusyModel(data,
                                  path=data.__class__.__name__, epochs=self.rounds)
                theta_samples = None
                while model.n_local_data < model.suff_data:
                    # Training
                    sampler = Random(data, n_splits=self.n_models, k=self.k_fold, seed=self.seed)
                    sampler.create_split(data.train.shape, data.train)
                    trained_model = self.train(data=data, i=i, sampler=sampler, experiment_path=self.experiment_path,
                                               model=model)

                    d, r, h, n = data.radon_number(r=trained_model.get_num_of_states() + 2,
                                                   h=1,
                                                   d=data.train.shape[0])
                    self.r = r

                    theta_arr = None
                    if CONFIG.COVTYPE != CovType.none:
                        theta_samples_unif, theta_samples_fisher, theta_old = self.sample_parameters(trained_model)
                        theta_arr = np.concatenate(theta_samples_fisher, axis=1)

                    # Aggregation
                    logger.info("Aggregating Model No. " + str(i))
                    kl_samples = [np.ascontiguousarray(
                        data.train.iloc[idx][:trained_model.data_delta * trained_model.epoch].values,
                        dtype=np.uint16) for idx in sampler.split_idx]
                    local_predictions = None
                    local_predictions = trained_model.predict(n_test=self.n_test)
                    local_acc = []
                    local_y_pred = []
                    if not local_predictions is None:
                        for local_indexer, local_pred in enumerate(local_predictions):
                            y_pred = local_pred[:, model.data_set.label_column]
                            y_true = data.test_labels[:self.n_test]
                            acc = np.where(np.equal(y_pred, y_true))[
                                      0].shape[0] / data.test_labels[:self.n_test].shape[
                                      0]
                            local_acc.append(acc)
                            local_y_pred.append(y_pred)
                            logger.info(str(acc))
                            self.record_obj(i, model.px_model[local_indexer].weights, model.n_local_data, "local_" + str(local_indexer) )
                    aggregation = self.aggregate(trained_model, data, trained_model.graph, trained_model.state_space,
                                                 theta_arr, kl_samples)
                    self.record_progress(aggregation, model.n_local_data, i, self.experiment_path, local_acc)
                    aggregates[model.n_local_data] = aggregation
                    if model.n_local_data > sampler.split_idx[0].shape[0]:
                        break
                self.write_progress(os.path.join(self.experiment_path, str(i)), "accuracy.csv", "obj.csv")
                models.append(trained_model)
                model.write_progress_hook(path=os.path.join(self.experiment_path, str(i)),
                                          fname="obj_progress" + ".csv")
                k_aggregates.append(aggregates)
                self.res = k_aggregates
                try:
                    self.finalize(i, trained_model, aggregates, data.mask, sampler.split_idx, local_y_pred)
                except Exception as e:
                    print(e)
        np.save(os.path.join(self.experiment_path, 'split'), np.stack(data.split))
        return models, k_aggregates, sampler

    def finalize(self, i, models, aggregates, mask, splits, local_y_pred):
        cv_path = os.path.join(self.experiment_path, str(i))
        for cnt, split in enumerate(splits):
            np.save(os.path.join(cv_path, "local_split_" + str(cnt)), split)
        for cnt, y_pred in enumerate(local_y_pred):
            np.save(os.path.join(cv_path, "local_pred_" + str(cnt)), y_pred)
        for j, (n_data, model) in enumerate(aggregates.items()):
            sub_model_path = os.path.join(cv_path, "batch_n" + str(j))
            if not os.path.isdir(sub_model_path):
                os.makedirs(sub_model_path)
            if j == 0:
                np.save(os.path.join(cv_path, "mask"), mask)
                np.save(os.path.join(cv_path, 'edgelist'), models.edgelist)
                np.save(os.path.join(cv_path, 'statespace'), models.state_space)
            for k, px_model in enumerate(models.px_batch[j + 1]):
                np.save(os.path.join(sub_model_path, "dist_weights " + str(k)), np.copy(px_model.weights))
            for m, (name, aggregation) in enumerate(model.items()):
                if m == 0:
                    np.save(os.path.join(sub_model_path, "y_true"), aggregation[0]['y_true'])
                if aggregation[0]:
                    np.save(os.path.join(sub_model_path, "y_pred_" + name), aggregation[0]['y_pred'])
                    np.save(os.path.join(sub_model_path, "weights_" + name), aggregation[0]['px_model'].weights)
            for k, px_model in enumerate(models.px_batch_local[j + 1]):
                px_model.save(os.path.join(sub_model_path, "dist_pxmodel " + str(k) + ".px"))

    def sample_parameters(self, model, perturb=False):
        n_samples = self.r ** self.h
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
                                                    np.zeros(px_model.weights.shape[0]), cov,1).ravel()[:, None]
                )
            else:
                theta_samples.append(
                    self.random_state.multivariate_normal(px_model.weights, cov, samples_per_model).T)

        return theta_samples, theta_old

    def gen_unif_cov(self, n_dim, eps=1e-1):
        return np.diag(np.ones(n_dim)) * eps

    def gen_random_cov(self, n_dim):
        eigs = self.random_state.rand(n_dim)
        eigs = eigs / np.sum(eigs) * eigs.shape[0]
        return random_correlation.rvs(eigs, random_state=self.random_state)

    def gen_fisher_cov(self, phi, mu):
        return np.outer(mu - phi, (mu - phi).T)

    def gen_semi_random_cov(self, model, eps = 0):
        a = np.insert(np.cumsum([model.states[u] * model.states[v] for u, v in model.graph.edgelist]), 0, 0)
        marginals, A = model.infer()
        eigs = self.random_state.rand(model.weights.shape[0])
        eigs = eigs / np.sum(eigs) * eigs.shape[0]
        # cov = random_correlation.rvs(eigs, random_state=self.random_state)
        cov = np.zeros((model.weights.shape[0],model.weights.shape[0]))
        rhs = np.outer(marginals, marginals)
        diag = np.diag(marginals[:model.weights.shape[0]] - marginals[:model.weights.shape[0]]**2)
        for x in range(a.shape[0] - 1):
            cov[a[x]:a[x + 1], a[x]:a[x + 1]] = - rhs[a[x]:a[x + 1], a[x]:a[x + 1]]
        cov -= np.diag(np.diag(cov))
        cov += diag + np.diag(np.full(model.weights.shape[0], eps))

        return cov

    def prepare_and_run(self):

        data = self.data_obj(path=os.path.join("data", self.name), mask=self.mask, seed=self.seed, cval=self.k_fold)
        data.create_cv_split()
        self.baseline(data)
        models, aggregate, sampler = self.run(data, self.model_loader)

        return models, aggregate

    def record_progress(self, model_dict, d, k, experiment_path, local_info):
        """

        Parameters
        ----------
        model_dict : dict
        d : int

        Returns
        -------

        """
        write_str = str(d) + ", "
        path = os.path.join(experiment_path, str(k))

        if not os.path.isdir(path):
            header = "n_local_data, "
            os.makedirs(path)
            header += ", ".join(["local_acc_" + str(k) for k in range(len(local_info))])
            header += ", "
            header += ", ".join([name + "_acc" for name, _ in model_dict.items()])
            print(header, file=self.csv_writer)

        write_str += ", ".join([str(inf) for inf in local_info]) + ", "
        for method, results in model_dict.items():
            for stats in results:
                if stats:
                    write_str += ", " + str(stats['acc'])
                else:
                    write_str += ", " + 'nan'
        print(write_str, file=self.csv_writer)

    def record_obj(self, i, weights, n, name):
        px_model = self.baseline_models[i]
        mu, A = px_model.infer()
        if weights is not None:
            np.copyto(px_model.weights, weights)
            mu, A = px_model.infer()
            ll = A - np.inner(px_model.statistics, px_model.weights)
        else:
            ll = A - np.inner(px_model.statistics, px_model.weights)
        if ll <= 0:
            print("P = NP !")
        obj_str = str(n) + ", " + str(name) + ", " + str(ll)
        print(obj_str , file=self.obj_writer)

    def write_progress(self, path, fname, fname2):
        with open(os.path.join(path, fname), "w+", encoding='utf-8') as f:
            self.csv_writer.seek(0)
            copyfileobj(self.csv_writer, f)
            del self.csv_writer
            self.csv_writer = io.StringIO()

        with open(os.path.join(path, fname2), "w+", encoding='utf-8') as f:
            self.obj_writer.seek(0)
            copyfileobj(self.obj_writer, f)
            del self.obj_writer
            self.obj_writer = io.StringIO()


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
               'dota2' : Dota2}
    return choices[type]


if __name__ == '__main__':

    keywords = ['--data', '--covtype', '--reg', '--graphtype']
    datasets = ['susy', 'dota2', 'covertype']
    sample_parameters = ['none', 'unif', 'random', 'fish']
    reg = ['None', 'l2']
    configurations = [element for element in itertools.product(*[datasets, sample_parameters, reg])]
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
    #cmd_args = parser.parse_args()

    for cmd_args in cmd_arg_list:
        data_class = get_data_class(cmd_args.data)

        CONFIG.setup(cmd_args)
        number_of_samples_per_model = 100
        coordinator = Coordinator(data_set_name=cmd_args.data,
                                  Data=data_class,
                                  exp_loader=cmd_args.load,
                                  n=number_of_samples_per_model,
                                  k=cmd_args.cv,
                                  iters=cmd_args.maxiter,
                                  h=cmd_args.h,
                                  epochs=cmd_args.epoch,
                                  n_models=cmd_args.n_models,
                                  n_test=cmd_args.n_test)

        result, agg = coordinator.prepare_and_run()
