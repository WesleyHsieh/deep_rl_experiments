import csv
from hyperparameter_tuning import HyperparameterTuner

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

    def run_experiments(self):
        hyperparam_lists = HyperparameterTuner(self.hyperparam_dict, 'grid')\
            .generate_hyperparam_lists()

        exp_results = [self.run_single_experiment(exp_num, hyperparam_lists[exp_num])
                       for exp_num in range(len(hyperparam_lists))]
        return exp_results

    def run_single_experiment(self, exp_num, hyperparams):
        stub(globals())
        env = normalize(GymEnv(self.env_name))
        policy = CategoricalMLPPolicy(env_spec=env.spec, **hyperparams)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=env.horizon,
            n_itr=40,
            discount=0.99,
            step_size=0.01,
            plot=True,
        )
        exp_name='sample_experiment_{}'.format(exp_num)
        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            plot=True,
            exp_name=exp_name
        )
        exp_filepath = os.path.join(EXPDIR, exp_name, 'progress.csv')
        result_dict = self.parse_results(exp_filepath)
        return hyperparams, result_dict

    def parse_results(self, exp_filepath):
        fields = ['AverageDiscountedReturn', 'AverageReturn']
        with open(exp_filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            row = [r for r in reader][-1]
            result_dict = {field: row[field] for field in fields}
            return result_dict

