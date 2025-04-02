
def linear_scheduler(pruning_ratio_dict, steps):
    return [((i) / float(steps)) * pruning_ratio_dict for i in range(steps+1)]

def exponential_scheduler(pruning_ratio_dict, steps):
    # interpolate i in range(steps + 1) to [0, 1] exponentially
    return [1 - (1 - pruning_ratio_dict) ** (i / steps) for i in range(steps+1)]
