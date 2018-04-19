import matplotlib.pyplot as plt


def plot_convergence(optimisers):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    lines = ['--bs', '--kd', '--ro']
    for i, optimiser in enumerate(optimisers):
        ax.plot(optimiser.checkpoints, optimiser.Y_best, lines[i], label=optimiser.name)

    plt.ylabel('Min Validation Error')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()
