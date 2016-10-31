from experiment_runner import ExperimentRunner
import argparse

hyperparam_dict = {'hidden_sizes': [(32, 32), (32, 16), (32, 32, 32), (32, 32, 16), (32, 16, 16), (32, 32, 32, 32)],
                   'hidden_nonlinearity': ['tanh', 'rectify', 'elu', 'softmax']
                  }
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument('environment_name', type=str,
                        help='OpenAI Gym Environment Name')
    args = parser.parse_args()
    env_name = args.environment_name
    print('Running Experiment on Environment: {}'.format(env_name))
    runner = ExperimentRunner(hyperparam_dict, env_name)
    exp_results = runner.run_experiments()
    print('Experiment Results')
    print(exp_results)
