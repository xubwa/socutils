#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Helper class to record time consumption
'''

import time

process_clock = time.process_time
perf_counter = time.perf_counter

class Timer:
    def __init__(self):
        self.cpu_time = 0.0
        self.wall_time = 0.0
        self.t0 = process_clock()
        self.w0 = perf_counter()

    def start(self):
        t0, w0 = process_clock(), perf_counter()
        self.t0, self.w0 = t0, w0
        return t0, w0

    def accumulate(self, info='this step'):
        t0, w0 = process_clock(), perf_counter()
        self.cpu_time += t0 - self.t0
        self.wall_time += w0 - self.w0
        self.t0, self.w0 = t0, w0
        print(f'CPU time for {info}:{self.cpu_time:.2f}, wall time {self.wall_time:.2f}')
    
