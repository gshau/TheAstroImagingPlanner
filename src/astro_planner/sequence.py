class Sequence:
    def __init__(self, target, filter_sequence, profile, n_rounds=1):
        self.target = target
        self.filter_sequence = filter_sequence
        self.n_rounds = n_rounds
        self.profile = profile
