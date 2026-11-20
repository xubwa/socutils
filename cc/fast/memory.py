'''Memory available to a calculation.

``available()`` is what the process can still allocate: MemAvailable of the
node, and not more than the memory cgroups of the process leave (a Slurm job
or step with a memory limit).  File pages count as free in both: the page
cache is given up before a limit is hit.

``budget(max_memory)`` is the number of bytes the arrays of a calculation may
take: ``max_memory`` in MB as in PySCF (the limit for the whole process, so
what is resident already is subtracted), or by default a fraction of what is
available.
'''

import os

UNLIMITED = 1 << 60
FRACTION = 0.8          # default share of the available memory


def _read_int(path):
    try:
        with open(path) as f:
            s = f.read().strip()
        return None if s == 'max' else int(s)
    except (OSError, ValueError):
        return None


def _stat(path, keys):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, val = line.partition(' ')
                if k in keys:
                    out[k] = int(val)
    except (OSError, ValueError):
        pass
    return out


def node_available():
    '''MemAvailable of /proc/meminfo in bytes, or None.'''
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return 1024 * int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def cgroup_available(cgroup_file='/proc/self/cgroup', root='/sys/fs/cgroup'):
    '''Bytes left under the tightest memory limit among the cgroups of this
    process and their parents (cgroup v1 and v2), file pages counted as free;
    None if there is no limit.'''
    try:
        with open(cgroup_file) as f:
            lines = f.read().split()
    except OSError:
        return None
    best = None
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        if 'memory' in parts[1].split(','):                 # v1 memory controller
            top = os.path.join(root, 'memory')
            names = ('memory.limit_in_bytes', 'memory.usage_in_bytes',
                     ('total_inactive_file', 'total_active_file'))
        elif parts[1] == '':                                # v2 unified hierarchy
            top = root
            names = ('memory.max', 'memory.current', ('inactive_file', 'active_file'))
        else:
            continue
        d = os.path.normpath(top + '/' + parts[2])
        while d.startswith(top):
            limit = _read_int(os.path.join(d, names[0]))
            usage = _read_int(os.path.join(d, names[1]))
            if limit is not None and usage is not None and limit < UNLIMITED:
                cache = sum(_stat(os.path.join(d, 'memory.stat'), names[2]).values())
                free = limit - usage + cache
                best = free if best is None else min(best, free)
            if d == top:
                break
            d = os.path.dirname(d)
    return best


def available():
    '''Bytes this process can still allocate (None if unknown).'''
    vals = [x for x in (node_available(), cgroup_available()) if x is not None]
    return max(0, min(vals)) if vals else None


def resident():
    '''Anonymous resident memory of this process in bytes (0 if unknown).'''
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('RssAnon'):
                    return 1024 * int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def ao2mo_memory(max_memory=None):
    '''MB for the buffers of a PySCF AO-to-MO transformation (its
    ``max_memory``): a quarter of budget(max_memory), at most PySCF's default
    of 4000 MB and at least 200 MB.  PySCF sizes its buffers by this number
    whatever the memory limit of the job, so every transformation is given it.'''
    b = budget(max_memory)
    return 4000.0 if b is None else float(min(4000.0, max(200.0, b / 4e6)))


def budget(max_memory=None):
    '''Bytes the arrays of a calculation may take from now on: ``max_memory``
    (MB, for the whole process) minus what is resident, else FRACTION of
    available(); None if neither is known.'''
    if max_memory is not None:
        return max(0, int(max_memory * 1e6) - resident())
    avail = available()
    return None if avail is None else int(FRACTION * avail)
