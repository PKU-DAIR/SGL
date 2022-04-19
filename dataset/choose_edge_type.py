from typing import List, Tuple
import random
import math
import warnings

EDGE_TYPE_DELIMITER = '__to__'


# Without sorting. Keep the original order.
def EdgeTypeStr2Tuple(edge_type: str) -> Tuple[str]:
    edge_type_list = edge_type.split(EDGE_TYPE_DELIMITER)
    return tuple(edge_type_list)


# Due to the existence of directed edges in both directions,
# e.g., paper__to__author and author__to__paper may exist simultaneously.
def RemoveDuplicateEdgeType(edge_types: List) -> List[str]:
    unique_edge_types = []
    for et in edge_types:
        et_tuple = EdgeTypeStr2Tuple(et)
        reversed_et = et_tuple[1]+EDGE_TYPE_DELIMITER+et_tuple[0]
        if reversed_et not in unique_edge_types:
            unique_edge_types.append(et)
    return unique_edge_types


def ChooseEdgeType(edge_type_num: int, edge_types: List, predict_class: str) -> Tuple[str]:
    chosen_edge_types_list = []
    explored_node_type_set = {predict_class}
    unique_edge_type = RemoveDuplicateEdgeType(edge_types)
    remaining_edge_types = unique_edge_type.copy()

    # Estimate by "coupon collector"
    maximal_reasonable_steps = 10 * edge_type_num * \
        int(math.log2(edge_type_num)+1)
    step_cnt = 0

    for _ in range(edge_type_num):
        while True:
            # Avoid infinite loop.
            step_cnt += 1
            if step_cnt > maximal_reasonable_steps or len(remaining_edge_types) == 0:
                warnings.warn(
                    f"Can't find enough ({edge_type_num}) edge types!", UserWarning)
                break

            edge_type_idx = random.randint(0, len(remaining_edge_types)-1)
            new_edge_type = remaining_edge_types[edge_type_idx]
            new_edge_type_tuple = EdgeTypeStr2Tuple(new_edge_type)
            if len(set(new_edge_type_tuple) & explored_node_type_set) == 0:
                continue
            chosen_edge_types_list.append(new_edge_type)
            explored_node_type_set |= set(new_edge_type_tuple)
            remaining_edge_types.pop(edge_type_idx)
            break
    return tuple(sorted(chosen_edge_types_list))


# Ways of choosing k items from n items.
def Combination(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise ValueError('n < 0 or k < 0!')
    result = 1
    for i in range(k):
        result = result*(n-i)//(i+1)
    return result


def ChooseMultiSubgraphs(subgraph_num: int, edge_type_num: int,
                         edge_types: List, predict_class: str) -> List[Tuple[str]]:
    subgraph_edge_types_list = []
    unique_edge_type = RemoveDuplicateEdgeType(edge_types)
    if edge_type_num > len(unique_edge_type):
        return subgraph_edge_types_list

    # Estimate by "coupon collector"
    maximal_reasonable_steps = 10 * Combination(len(unique_edge_type), edge_type_num) *\
        (math.log2(Combination(len(unique_edge_type), edge_type_num))+1)
    step_cnt = 0

    for _ in range(subgraph_num):
        while True:
            # Avoid infinite loop.
            step_cnt += 1
            if step_cnt > maximal_reasonable_steps:
                warnings.warn(
                    f"Can't find enough ({subgraph_num}) subgraphs!", UserWarning)
                break

            new_subgraph_edge_types = ChooseEdgeType(
                edge_type_num, unique_edge_type, predict_class)
            if new_subgraph_edge_types in subgraph_edge_types_list:
                continue
            if len(new_subgraph_edge_types) > 0:
                subgraph_edge_types_list.append(new_subgraph_edge_types)
            break
    return subgraph_edge_types_list
