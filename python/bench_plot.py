import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os
import matplotlib.pyplot as plt

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'  # For A6000
from tree import generate_random_trees

# Load the CUDA kernel as a python module
minimal_attn = load(
    name='minimal_attn',
    sources = list(map(lambda x: '../src/' + x, ['main.cpp', 'forward_1.cu', 'forward_2.cu', \
                                                 'forward_3.cu', 'forward_4.cu', 'forward_5.cu', \
                                                 'forward_6.cu', 'forward_7.cu', 'forward_8.cu',
                                                 'forward_base_4.cu', 'forward_base_5.cu',
                                                 'forward_base_6.cu', 'forward_base_8.cu'])),
    extra_cuda_cflags=['-O3', '--use_fast_math']
)




def plot_results(results, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # Define a nice color palette:
    colors = ["#2B2F42", "#8D99AE", "#EF233C", "#D90429", "#A90000", "#7B0000", "#4D0000", "#1F0000"]

    # Plot each of the main lines
    x = [result[0] for result in results]
    torch_times = [result[1] for result in results]
    forward_times = [result[2] for result in results]

    ax.plot(x, torch_times, label="PyTorch", color=colors[0], linewidth=2, marker='o')
    for i in range(len(forward_times[0])):
        forward_time = [result[2][i] for result in results]
        ax.plot(x, forward_time, label=f"Forward {i+1}", color=colors[i+1], linewidth=2, marker='o')

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x), max(x))

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_experiments_b(batch_sizes, num_tree_nodes, prompt_length, n_head, head_embd, IsTree):
    results = []
    for batch_size in batch_sizes:
        seq_len = num_tree_nodes + prompt_length
        q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
        k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
        v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

        start_times, end_times, causal_masks = generate_random_trees(batch_size, num_tree_nodes, prompt_length)

        def torch_attention(q, k, v, mask):
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return output

        with torch.no_grad():
            causal_mask = causal_masks.unsqueeze(1)  # .broadcast_to((batch_size, n_head, seq_len, seq_len)).contiguous()
            if not IsTree:
                causal_mask = None
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            manual_result_torch = torch_attention(q, k, v, causal_mask)
            end.record()
            torch.cuda.synchronize()
            torch_time = start.elapsed_time(end)

        forward_times = []
        with torch.no_grad():
            for i in range(1, 8 + 1):
                forward_func = getattr(minimal_attn, f'forward_{i}')
                (minimal_result, time_taken, max_shared_mem, used_shared_mem) = forward_func(q, k, v, start_times, end_times, IsTree)
                forward_times.append(time_taken.item())

        results.append((batch_size, torch_time, forward_times))

    return results


def run_experiments_n(batch_size, tree_sizes, prompt_length, n_head, head_embd, IsTree):
    results = []
    for num_tree_nodes in tree_sizes:
        seq_len = num_tree_nodes + prompt_length
        q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
        k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
        v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

        start_times, end_times, causal_masks = generate_random_trees(batch_size, num_tree_nodes, prompt_length)

        def torch_attention(q, k, v, mask):
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return output

        with torch.no_grad():
            causal_mask = causal_masks.unsqueeze(1)  # .broadcast_to((batch_size, n_head, seq_len, seq_len)).contiguous()
            if not IsTree:
                causal_mask = None
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            manual_result_torch = torch_attention(q, k, v, causal_mask)
            end.record()
            torch.cuda.synchronize()
            torch_time = start.elapsed_time(end)

        forward_times = []
        with torch.no_grad():
            for i in range(1, 8 + 1):
                forward_func = getattr(minimal_attn, f'forward_{i}')
                (minimal_result, time_taken, max_shared_mem, used_shared_mem) = forward_func(q, k, v, start_times, end_times, IsTree)
                forward_times.append(time_taken.item())

        results.append((num_tree_nodes, torch_time, forward_times))

    return results


# Experiment 1: Varying batch sizes 1, 100 with default tree size
batch_sizes_exp1 = [1, 100]
tree_sizes_exp1 = [2 ** 6 - 10, 2 ** 8 - 10, 2 ** 10 - 10, 2 ** 12 - 10, 2 ** 14 - 10]
prompt_length_exp1 = 10
n_head_exp1 = 10
head_embd_exp1 = 64
IsTree_exp1 = True

results_exp1 = run_experiments_n(batch_sizes_exp1[0], tree_sizes_exp1, prompt_length_exp1, n_head_exp1, head_embd_exp1, IsTree_exp1)
plot_results(results_exp1, "Execution Time vs. Batch Size (Default Tree Size)", "exp1.png")

results_exp2 = run_experiments_n(batch_sizes_exp1[1], tree_sizes_exp1, prompt_length_exp1, n_head_exp1, head_embd_exp1, IsTree_exp1)
plot_results(results_exp2, "Execution Time vs. Batch Size (Default Tree Size)", "exp2.png")

# Experiment 2: Varying batch sizes 1, 5, 25, 100, 1000 with tree sizes 64-10 and 1024-10
batch_sizes_exp2 = [1, 5, 25, 100, 1000]
tree_sizes_exp2 = [64 - 10, 1024 - 10]
prompt_length_exp2 = 10
n_head_exp2 = 10
head_embd_exp2 = 64
IsTree_exp2 = True

results_exp3 = run_experiments_b(batch_sizes_exp2, tree_sizes_exp2[0], prompt_length_exp2, n_head_exp2, head_embd_exp2, IsTree_exp2)
plot_results(results_exp3, "Execution Time vs. Batch Size (Tree Size 64-10)", "exp3.png")

results_exp4 = run_experiments_b(batch_sizes_exp2, tree_sizes_exp2[1], prompt_length_exp2, n_head_exp2, head_embd_exp2, IsTree_exp2)
plot_results(results_exp4, "Execution Time vs. Batch Size (Tree Size 1024-10)", "exp4.png")