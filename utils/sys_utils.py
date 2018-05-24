import os


def ensure_dir(fn):
    directory = os.path.dirname(fn)
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dirs(fns):
    if isinstance(fns, str):
        ensure_dir(fns)
    else:
        for fn in fns:
            ensure_dir(fn)


def listify_dict(d):
    return {k: list(v) for k, v in d.items()}


def setify_dict(d):
    return {k: set((i, j) for i, j in v) for k, v in d.items()}

