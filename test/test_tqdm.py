import time
import concurrent.futures
from tqdm import tqdm


def f(x):
    time.sleep(0.001)
    return x ** 2


def run(f, my_iter):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results


my_iter = range(100000)
run(f, my_iter)
