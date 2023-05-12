import os


def memory_usage():
    p = os.popen(f"nvidia-smi | grep ' 2196 ' | awk '{{print $8}}'")
    usage = p.read().replace('MiB', '')[:-1].split('\n')
    usage = [int(s) for s in usage]
    p.close()
    return usage


def get_memory_usage():
    pid = os.getpid()
    p = os.popen(f"nvidia-smi | grep ' {pid} ' | awk '{{print $8}}'")
    usage = p.read().replace('\n', ' ')
    p.close()
    return usage
