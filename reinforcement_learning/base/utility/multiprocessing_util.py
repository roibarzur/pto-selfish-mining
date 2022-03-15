import multiprocessing as mp


def get_process_name() -> str:
    return mp.current_process().name


def get_process_index() -> int:
    return int(get_process_name().split(' ')[-1])
