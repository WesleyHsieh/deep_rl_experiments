class HyperparameterTuner:
    def __init__(self, hyperparam_dict, search_type):
        self.hyperparam_dict = hyperparam_dict
        self.search_type = search_type
    def generate_hyperparam_lists(self):
        if self.search_type == 'random':
            pass
        elif self.search_type == 'grid':
            pass