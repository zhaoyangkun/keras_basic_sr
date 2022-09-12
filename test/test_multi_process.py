from concurrent.futures import ThreadPoolExecutor


def worker(item, flag):
    return (item, flag)


processed = []
with ThreadPoolExecutor(max_workers=2) as pool:
    for (item, flag) in pool.map(worker, [1, 2, 3], [False] * 3):
        print(item, flag)

