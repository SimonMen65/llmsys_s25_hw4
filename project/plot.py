import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import os, json

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

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

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = load_stats(rank=0, workdir = "./workdir/singleGPU")
    device0_mean, device0_std =  load_stats(rank=0, workdir = "./workdir/parallelGPU")
    device1_mean, device1_std =  load_stats(rank=1, workdir = "./workdir/parallelGPU")
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png')

    # pp_mean, pp_std = None, None
    # mp_mean, mp_std = None, None
    # plot([pp_mean, mp_mean],
    #     [pp_std, mp_std],
    #     ['Pipeline Parallel', 'Model Parallel'],
    #     'pp_vs_mp.png')