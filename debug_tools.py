import time


def __benchmark_init():
    global BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = time.time()


def benchmark(name='', print_benchmark=True):
    global BENCHMARK_INIT_TIME
    now = time.time()
    diff = now - BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = now
    if print_benchmark: print('{} time: {:.4f} seconds'.format(name, diff))
    return diff