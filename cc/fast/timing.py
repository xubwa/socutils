'''Accumulating timers for per-term profiling: wall and CPU time.

``timer(name)`` is a context manager.  Timers nest: each one is booked
under the path of the timers that enclose it, so that the time of a stage
can be shown next to the times of its parts (tree) while report() keeps the
flat view by name.  The CPU time is that of the whole process, i.e. of all
threads, and of its child processes that have ended (cpu(): the processes
of a pass over the quadruples of CC4), so CPU / wall is the number of cores
kept busy.
'''

import os
import time
from collections import defaultdict
from contextlib import contextmanager

_wall = defaultdict(float)          # path (tuple of names) -> seconds
_cpu = defaultdict(float)
_calls = defaultdict(int)
_stack = []


def cpu():
    '''CPU seconds of this process and of its child processes that have ended.'''
    t = os.times()
    return time.process_time() + t[2] + t[3]


@contextmanager
def timer(name):
    _stack.append(name)
    path = tuple(_stack)
    w0, c0 = time.perf_counter(), cpu()
    try:
        yield
    finally:
        _wall[path] += time.perf_counter() - w0
        _cpu[path] += cpu() - c0
        _calls[path] += 1
        _stack.pop()


def add(name, wall, cpu=0.0):
    '''Book ``wall`` (and ``cpu``) seconds measured elsewhere, e.g. by a C
    kernel, under ``name`` below the timers that are running.'''
    path = tuple(_stack) + (name,)
    _wall[path] += wall
    _cpu[path] += cpu
    _calls[path] += 1


def reset():
    _wall.clear()
    _cpu.clear()
    _calls.clear()


def report(per=1):
    '''Return [(name, seconds / per, calls / per)] sorted by time: the wall
    time of each name summed over the places it was called from.'''
    wall, calls = defaultdict(float), defaultdict(int)
    for path, w in _wall.items():
        wall[path[-1]] += w
        calls[path[-1]] += _calls[path]
    return sorted(((k, v / per, calls[k] / per) for k, v in wall.items()),
                  key=lambda x: -x[1])


def snapshot():
    '''{path: (wall, cpu, calls)} of everything booked so far.'''
    return {p: (_wall[p], _cpu[p], _calls[p]) for p in _wall}


def since(before, after=None):
    '''{path: (wall, cpu, calls)} booked between two snapshots (``after``:
    now).'''
    after = snapshot() if after is None else after
    out = {}
    for p, (w, c, n) in after.items():
        w0, c0, n0 = before.get(p, (0.0, 0.0, 0))
        if n > n0:
            out[p] = (w - w0, c - c0, n - n0)
    return out


def children(times, parent=()):
    '''[(name, wall, cpu, calls)] of the timers directly below ``parent``
    (a path) in a {path: (wall, cpu, calls)} dict, the longest first.'''
    k = len(parent)
    rows = [(p[k], w, c, n) for p, (w, c, n) in times.items()
            if len(p) == k + 1 and p[:k] == tuple(parent)]
    return sorted(rows, key=lambda r: -r[1])
