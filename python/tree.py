import random
import torch
import sys
import os

sys.setrecursionlimit(10000)

NODE_START_TIMES = []
NODE_END_TIMES = []
LINEAR_REMAINING = 0
RANDOM_REMAINING = 0

class Node:
    def __init__(self):
        self.children = []
        self.start_time = None
        self.end_time = None

def add_random_nodes(root):
    # print(linear_remaining, random_remaining)
    global LINEAR_REMAINING, RANDOM_REMAINING
    if LINEAR_REMAINING == 0 and RANDOM_REMAINING == 0:
        return
    if LINEAR_REMAINING > 0:
        new_node = Node()
        root.children.append(new_node)
        # print("Linear Node: ", LINEAR_REMAINING, RANDOM_REMAINING)
        LINEAR_REMAINING -= 1
        add_random_nodes(new_node)
    else:
        num_children = random.randint(1, min(RANDOM_REMAINING, 10))
        RANDOM_REMAINING -= num_children
        # print("Random Node: ", LINEAR_REMAINING, RANDOM_REMAINING, num_children)
        for _ in range(num_children):
            new_node = Node()
            root.children.append(new_node)
            add_random_nodes(new_node)

def generate_tree():
    global LINEAR_REMAINING
    root = Node()
    LINEAR_REMAINING -= 1
    add_random_nodes(root)
    return root

def dfs_(node, time):
    if node is not None:
        node.start_time = time
        time += 1
        for child in node.children:
            time = dfs_(child, time)
        node.end_time = time
        time += 1
    return time

def collect_node_times(node):
    global NODE_START_TIMES, NODE_END_TIMES
    if node is not None:
        NODE_START_TIMES.append(node.start_time)
        NODE_END_TIMES.append(node.end_time)
        for child in node.children:
            collect_node_times(child)

def create_causal_mask(start_times, end_times):
    start_I = start_times[:,:,None]
    start_J = start_times[:,None,:]
    end_I = end_times[:,:,None]
    end_J = end_times[:,None,:]
    return ((start_I >= start_J) & (end_I <= end_J)).contiguous()

def dfs(root):
    global NODE_START_TIMES, NODE_END_TIMES
    NODE_START_TIMES = []
    NODE_END_TIMES = []
    dfs_time = 0
    dfs_(root, dfs_time)
    collect_node_times(root)
    return NODE_START_TIMES, NODE_END_TIMES


def generate_random_trees(num_trees, num_nodes, root_chain):
    file_name = os.path.join("../data", f"tree_dfs_data_BS_{num_trees}_RN_{num_nodes}_LN_{root_chain}.pth")
    if os.path.exists(file_name):
        print("Using Cached Trees")
        loaded_tree_data = torch.load(file_name)
        start_times = loaded_tree_data['start_times'].cuda()
        end_times = loaded_tree_data['end_times'].cuda()
        causal_masks = create_causal_mask(start_times, end_times)
        return start_times, end_times, causal_masks

    print("Generating New Trees")
    global LINEAR_REMAINING, RANDOM_REMAINING
    tree_start_times = []
    tree_end_times = []
    for _ in range(num_trees):
        LINEAR_REMAINING = root_chain
        RANDOM_REMAINING = num_nodes
        root = generate_tree()
        _start_time, _end_times = dfs(root)
        _start_time = torch.tensor(_start_time, dtype=torch.float32).cuda()
        _end_times = torch.tensor(_end_times, dtype=torch.float32).cuda()
        tree_start_times.append(_start_time)
        tree_end_times.append(_end_times)

    start_times = torch.stack(tree_start_times)
    end_times = torch.stack(tree_end_times)
    causal_masks = create_causal_mask(start_times, end_times)

    if not os.path.exists("../data"):
        os.makedirs("../data")
    torch.save({
        'start_times': start_times.cpu(),
        'end_times': end_times.cpu(),
    }, file_name)

    return start_times.cuda(), end_times.cuda(), causal_masks.cuda()

if __name__ == "__main__":
    tree_start_times, tree_end_times, causal_mask = generate_random_trees(1, 10, 3)
    print(tree_start_times)
    print(tree_end_times)
    print(causal_mask)