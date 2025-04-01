import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import os, json

def plot(ax, means, stds, labels, ylabel, title):
    """
    Plot a bar chart with error bars on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        means (list): List of mean values.
        stds (list): List of standard deviations.
        labels (list): Labels for each bar.
        ylabel (str): Y-axis label.
        title (str): Title for the subplot.
    """
    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, align='center', alpha=0.6,
           ecolor='black', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    ax.set_ylim(bottom=0)

def load_stats(rank, epochs=10, workdir='./workdir'):
    """
    Load training times from result JSON files for a given rank.

    Args:
        rank (int): The rank ID (e.g., 0 or 1).
        epochs (int): Number of epochs.
        workdir (str): Path to workdir containing result JSON files.

    Returns:
        (mean_time, std_time): Tuple of mean and std of training time.
    """
    training_times = []
    for epoch in range(epochs):
        json_path = os.path.join(workdir, f'rank{rank}_results_epoch{epoch}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                result = json.load(f)
                training_times.append(result["training_time"])
        else:
            print(f"[Warning] Missing file: {json_path}")
    
    if training_times:
        return np.mean(training_times), np.std(training_times)
    else:
        return 0.0, 0.0
    
def load_tokens_per_second(rank, epochs=10, workdir='./workdir'):
    """
    Load tokens per second from result JSON files for a given rank.

    Args:
        rank (int): The rank ID (e.g., 0 or 1).
        epochs (int): Number of epochs.
        workdir (str): Path to workdir containing result JSON files.

    Returns:
        (mean_tokens_per_sec, std_tokens_per_sec): Tuple of mean and std of tokens per second.
    """
    tokens_per_sec = []
    for epoch in range(epochs):
        json_path = os.path.join(workdir, f'rank{rank}_results_epoch{epoch}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                result = json.load(f)
                tokens_per_sec.append(result["tokens_per_sec"])
        else:
            print(f"[Warning] Missing file: {json_path}")
    
    if tokens_per_sec:
        return np.mean(tokens_per_sec), np.std(tokens_per_sec)
    else:
        return 0.0, 0.0
    
def plot_1_3():
        # Load Single GPU stats
    single_mean_time, single_std_time = load_stats(rank=0, workdir="./workdir/singleGPU")
    single_mean_tokens, single_std_tokens = load_tokens_per_second(rank=0, workdir="./workdir/singleGPU")

    # Load Data Parallel stats from both GPUs
    device0_mean_time, device0_std_time = load_stats(rank=0, workdir="./workdir/parallelGPU")
    device1_mean_time, device1_std_time = load_stats(rank=1, workdir="./workdir/parallelGPU")
    device0_mean_tokens, device0_std_tokens = load_tokens_per_second(rank=0, workdir="./workdir/parallelGPU")
    device1_mean_tokens, device1_std_tokens = load_tokens_per_second(rank=1, workdir="./workdir/parallelGPU")

    # Combine Data Parallel stats
    data_parallel_mean_time = np.mean([device0_mean_time, device1_mean_time])
    data_parallel_std_time = np.sqrt(device0_std_time**2 + device1_std_time**2) / 2  # Assuming independent std deviations
    data_parallel_mean_tokens = np.mean([device0_mean_tokens, device1_mean_tokens])
    data_parallel_std_tokens = np.sqrt(device0_std_tokens**2 + device1_std_tokens**2) / 2  # Assuming independent std deviations

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Execution Time Comparison
    plot(axs[0],
         [data_parallel_mean_time, single_mean_time],
         [data_parallel_std_time, single_std_time],
         ['Data Parallel - 2 GPUs', 'Single GPU'],
         'Execution Time (Seconds)')
    axs[0].set_title('Execution Time Comparison')

    # Plot Tokens Per Second Comparison
    plot(axs[1],
         [data_parallel_mean_tokens, single_mean_tokens],
         [data_parallel_std_tokens, single_std_tokens],
         ['Data Parallel - 2 GPUs', 'Single GPU'],
         'Tokens Per Second')
    axs[1].set_title('Tokens Per Second Comparison')

    # Adjust layout and save the combined figure
    plt.tight_layout()
    plt.savefig('combined_comparison.png')
    plt.close(fig)

def plot_2_3():
    # === Manually defined stats ===
    labels = [ 'Model Parallel', 'Pipeline Parallel']

    # Replace these with your actual numbers:
    time_means = [52.86895143985748, 48.65150594711304]         # Example: average training times
    time_stds = [0.032663941383361816, 0.14301300048828125]

    tokens_means = [12105.40831368303, 13154.89660099964]    # Example: tokens per second
    tokens_stds = [7.479065440319573, 38.669331964094454]

    # === Plotting ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot(axs[0], time_means, time_stds, labels, 'Execution Time (s)', 'Training Time Comparison')
    plot(axs[1], tokens_means, tokens_stds, labels, 'Tokens per Second', 'Throughput Comparison')

    plt.tight_layout()
    plt.savefig('pp_vs_mp_comparison.png')
    plt.close(fig)
    print("Plot saved to 'pp_vs_mp_comparison.png'")

if __name__ == '__main__':
    plot_2_3()