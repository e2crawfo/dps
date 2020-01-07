from collections import defaultdict
import numpy as np
from tabulate import tabulate
import torch

from dps.utils.base import RenderHook as _RenderHook, Config, map_structure


def walk_variable_scopes(model, max_depth=None):
    def _fmt(i):
        return "{:,}".format(i)

    fixed_vars = set()

    n_fixed = defaultdict(int)
    n_trainable = defaultdict(int)
    shapes = {}

    for name, v in model.named_parameters():
        n_variables = int(np.prod(tuple(v.size())))

        if name in fixed_vars:
            n_fixed[""] += n_variables
            n_trainable[""] += 0
        else:
            n_fixed[""] += 0
            n_trainable[""] += n_variables

        shapes[name] = tuple(v.shape)

        name_so_far = ""

        for token in name.split("."):
            name_so_far += token
            if v in fixed_vars:
                n_fixed[name_so_far] += n_variables
                n_trainable[name_so_far] += 0
            else:
                n_fixed[name_so_far] += 0
                n_trainable[name_so_far] += n_variables
            name_so_far += "."

    table = ["scope shape n_trainable n_fixed total".split()]

    any_shapes = False
    for scope in sorted(n_fixed, reverse=True):
        depth = sum(c == "." for c in scope) + 1

        if max_depth is not None and depth > max_depth:
            continue

        if scope in shapes:
            shape_str = "{}".format(shapes[scope])
            any_shapes = True
        else:
            shape_str = ""

        table.append([
            scope,
            shape_str,
            _fmt(n_trainable[scope]),
            _fmt(n_fixed[scope]),
            _fmt(n_trainable[scope] + n_fixed[scope])])

    if not any_shapes:
        table = [row[:1] + row[2:] for row in table]

    print("PyTorch variable scopes (down to maximum depth of {}):".format(max_depth))
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


class RenderHook(_RenderHook):
    def get_tensors(self, data, updater):
        tensors, recorded_tensors, losses = updater.model(data, plot=True, is_training=False)
        tensors = Config(tensors)
        tensors = map_structure(
            lambda t: t.cpu().detach().numpy() if isinstance(t, torch.Tensor) else t,
            tensors, is_leaf=lambda rec: not isinstance(rec, dict))
        return tensors, recorded_tensors, losses
