import csv
import matplotlib.pyplot as plt
from hyperparameter_tuning import HyperparameterTuner
import lasagne.nonlinearities as NL
import copy

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import os.path
EXPDIR = '/home/wesley/install/rllab/data/local/experiment'

class ExperimentRunner:
    def __init__(self, hyperparam_dict, env_name):
        self.hyperparam_dict = hyperparam_dict
        self.env_name = env_name
        self.exp_fields = ['AverageDiscountedReturn', 'AverageReturn']
        self.nonlinearities = {'tanh': NL.tanh, 'rectify': NL.rectify, 'elu': NL.elu, 'softmax': NL.softmax}

    def run_experiments(self, plot=True):
        hyperparam_lists = HyperparameterTuner(self.hyperparam_dict, 'grid')\
            .generate_hyperparam_lists()

        exp_results = [self.run_single_experiment(hyperparams)
                       for hyperparams in hyperparam_lists]
        if plot:
            self.plot_results(exp_results)
        return exp_results

    def run_single_experiment(self, hyperparams):
        stub(globals())
        env = normalize(GymEnv(self.env_name))
        wrapped_hyperparams = copy.deepcopy(hyperparams)
        wrapped_hyperparams['hidden_nonlinearity'] = self.nonlinearities[hyperparams['hidden_nonlinearity']]
        policy = CategoricalMLPPolicy(env_spec=env.spec, **wrapped_hyperparams)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=50,
            discount=0.99,
            step_size=0.01,
            plot=True,
        )
        exp_name='environment:{}_hyperparmams:{}'.format(self.env_name, str(hyperparams))
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=2,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            plot=True,
            exp_name=exp_name
        )
        exp_filepath = os.path.join(EXPDIR, exp_name, 'progress.csv')
        result_dict = self.parse_experiment_results(exp_filepath)
        return hyperparams, result_dict

    def parse_experiment_results(self, exp_filepath):
        print(exp_filepath)
        with open(exp_filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [r for r in reader]
            filtered_rows = [{field: float(row[field]) for field in self.exp_fields}
                             for row in rows]
            print(filtered_rows)
            return filtered_rows

    def plot_results(self, exp_results):
        for field in self.exp_fields:
            # Plot distribution of field entry across all hyperparameters
            final_field_entries = [result[1][-1][field] for result in exp_results]
            plt.figure()
            plt.hist(final_field_entries)
            plt.title('Distribution of {}, Environment {}'.format(field, self.env_name))
            plt.savefig('hyperparameter_distribution_{}_env_{}'.format(field, self.env_name))

            # Plot field entry vs iteration for best result
            best_hyperparams, best_result = max(exp_results, key=lambda x: x[1][-1][field])
            field_vs_iter = [row[field] for row in best_result]
            plt.figure()
            plt.plot(field_vs_iter)
            plt.xlabel('Iteration')
            plt.ylabel('Value of Field: {}'.format(field))
            plt.title('Learning Curve for Field: {}, Environment: {}'.format(field, self.env_name))
            plt.savefig('learning_curve_{}_env_{}:hyperparams_{}'.format(field, self.env_name, best_hyperparams))
            print('Best Result, Hyperparameters for Field {}: {}, {}'.format(field, best_result, best_hyperparams))



