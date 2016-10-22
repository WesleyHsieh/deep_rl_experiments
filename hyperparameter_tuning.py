import itertools

class HyperparameterTuner:
    def __init__(self, hyperparam_dict, search_type):
        """
        :param hyperparam_dict: Dictionary mapping
         hyperparameter name to list of possible values.
        :param search_type: String ('random', 'grid')
        """
        self.hyperparam_dict = hyperparam_dict
        self.search_type = search_type
    def generate_hyperparam_lists(self):
        """
        Generates list of all possible combinations
        of hyperparameters. Combination represented as
        dictionary mapping hyperparameter name to value.
        :return: List of all possible hyperparameter dicts.
        """
        if self.search_type == 'random':
            pass
        elif self.search_type == 'grid':
           items = self.hyperparam_dict.items()
           keys, vals = [x[0] for x in items], [x[1] for x in items]
           product = itertools.product(*vals)
           kwdicts = [dict(zip(keys, param_vals)) for param_vals in product]
        return kwdicts
