from typing import List


def GetGpuMemory() -> List[int]:
    import os
    import random

    tmp_file_name = f'gpu_mem_list_{random.randint(0, 10000000)}.txt'
    while os.path.exists(tmp_file_name):
        tmp_file_name = f'gpu_mem_list_{random.randint(0, 10000000)}.txt'

    os.system(
        f'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > {tmp_file_name}')
    gpu_mem_list = [int(x.split()[2])
                    for x in open(tmp_file_name, 'r').readlines()]
    os.system(f'rm {tmp_file_name}')
    return gpu_mem_list


def IdxOfMaxElement(list: List[int]) -> int:
    return list.index(max(list))


# This is is the only API you should call from outside
# 'display' determines if the chosen result will be printed
def GpuWithMaxFreeMem(display: bool = True) -> int:
    result = IdxOfMaxElement(GetGpuMemory())
    if display:
        print(f'GPU {result} is chosen')
    return result
