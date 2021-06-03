import argparse
import json
import numpy as np
import collections
import os
import sys
import subprocess
from contextlib import contextmanager


def flatten_dict(obj):
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            v = flatten_dict(v)
            if isinstance(v, dict):
                for inner_k, inner_v in v.items():
                    new_d['{}.{}'.format(k, inner_k)] = inner_v
            else:
                new_d[k] = v
        return new_d
    elif isinstance(obj, list):
        return flatten_dict({str(i): v for i, v in enumerate(obj)})
    else:
        return obj


def reduce_losses(losses, sep='/'):
    if len(losses) == 0:
        return {}

    keys = collections.OrderedDict()
    for elem in losses:
        for key in elem:
            keys[key] = True
    keys = list(keys.keys())

    new_losses = {}
    for key in keys:
        vals = []
        for x in losses:
            if key in x:
                vals.append(x[key])
        new_losses['{}{}mean'.format(key, sep)] = np.mean(vals)
        new_losses['{}{}std'.format(key, sep)] = np.std(vals)
    return new_losses


def reduce_infos(losses, sep='/'):
    if len(losses) == 0:
        return {}

    keys = collections.OrderedDict()
    for elem in losses:
        for key in elem:
            keys[key] = True
    keys = list(keys.keys())

    new_losses = {}
    for key in keys:
        vals = []
        for x in losses:
            if key in x:
                vals.append(x[key])
        new_losses['{}{}episode_mean'.format(key, sep)] = np.mean(vals)
        new_losses['{}{}episode_min'.format(key, sep)] = np.min(vals)
        new_losses['{}{}episode_max'.format(key, sep)] = np.max(vals)
        new_losses['{}{}episode_sum'.format(key, sep)] = np.sum(vals)
    return new_losses


def deep_update_list(to, fr):
    for i in range(len(to)):
        if type(to[i]) is dict and type(fr[i]) is dict:
            deep_update_dict(to[i], fr[i])
        elif type(to[i]) is list and type(fr[i]) is list and len(to[i]) == len(fr[i]):
            deep_update_list(to[i], fr[i])
        else:
            to[i] = fr[i]
    return to


def deep_update_dict(to, fr):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict and k in to and type(to[k]) is dict:
            deep_update_dict(to[k], v)
        elif type(v) is list and k in to and type(to[k]) is list and len(v) == len(to[k]):
            deep_update_list(to[k], v)
        else:
            to[k] = v
    return to


def deep_update_binding(to, binding, value):
    this_key = binding[0]
    next_binding = binding[1:]

    if type(to) is list:
        this_key = int(this_key)

    if len(next_binding) == 0:
        to[this_key] = value
    else:
        deep_update_binding(to[this_key], next_binding, value)

    return to


def prefix_dict(d, prefix, sep='/'):
    return {'{}{}{}'.format(prefix, sep, k): v for k, v in d.items()}


def get_config():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--gin_file', action='append', default=[])
    parser.add_argument('--gin_binding', action='append', default=[])
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    default_config_file = os.path.join(cur_dir, 'configs', 'default.json')

    config = dict()
    for file in [default_config_file] + list(args.gin_file):
        with open(file, 'r') as f:
            file = json.load(f)
        deep_update_dict(config, file)

    for binding in args.gin_binding:
        key, value = binding.split('=')
        value = eval(value)
        sub_keys = key.split('.')
        deep_update_binding(config, sub_keys, value)

    return config


def create_log_dir(base_log_dir):
    t = 0
    while True:
        try:
            log_dir = os.path.join(base_log_dir, str(t))
            os.makedirs(log_dir, exist_ok=False)
            return log_dir
        except Exception:
            t += 1
        if t == 100:
            raise ValueError


def print_bar():
    print('=' * 50)

# coding=utf-8

# Taken from https://raw.githubusercontent.com/IDSIA/sacred/master/sacred/stdout_capturing.py


def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, OSError):
        pass  # unsupported


# Duplicate stdout and stderr to a file. Inspired by:
# http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# http://stackoverflow.com/a/651718/1388435
# http://stackoverflow.com/a/22434262/1388435
@contextmanager
def capture_output(output_file, error_file):
    """Duplicate stdout and stderr to a file on the file descriptor level."""

    @contextmanager
    def inner(output_target, error_target):
        original_stdout_fd = 1
        original_stderr_fd = 2

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # start_new_session=True to move process to a new process group
        # this is done to avoid receiving KeyboardInterrupts (see #149)
        tee_stdout = subprocess.Popen(
            ["tee", "-a", output_target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=1,
        )
        tee_stderr = subprocess.Popen(
            ["tee", "-a", error_target.name],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=2,
        )

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            yield None  # let the caller do their printing
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            try:
                tee_stdout.wait(timeout=1)
                tee_stderr.wait(timeout=1)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
            except Exception:
                pass

    with open(output_file, "w+") as output_target:
        with open(error_file, "w+") as error_target:
            with inner(output_target, error_target):
                yield
