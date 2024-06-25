import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from torchmetrics import ROC, AUROC


plt.rcParams.update(
    {
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    }
)

SHOW = True


def calculate_ece(probabilities, labels, n_bins=10, rms=False, quantile_based=True):
    assert len(probabilities) == len(labels)
    if quantile_based:
        # Bin based on balancing number of samples in each bin
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(probabilities, quantiles)
    else:
        # Equal size bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(probabilities, bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_pred = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_probs = [[] for _ in range(n_bins)]

    for i in range(len(probabilities)):
        bin_index = bin_indices[i]
        bin_counts[bin_index] += 1
        bin_pred[bin_index] += probabilities[i]
        bin_true[bin_index] += labels[i]
        bin_probs[bin_index].append(probabilities[i])

    assert np.sum(bin_counts) == len(probabilities)

    # Avoid division by zero
    valid_bins = bin_counts > 0
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_acc[valid_bins] = bin_true[valid_bins] / bin_counts[valid_bins]
    bin_conf[valid_bins] = bin_pred[valid_bins] / bin_counts[valid_bins]

    if quantile_based:
        bin_centers = np.array([np.mean(probs) if probs else 0 for probs in bin_probs])
    else:
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

    if rms:
        ece = np.sqrt(np.sum(((bin_conf - bin_acc) ** 2) * bin_counts / len(probabilities)))
    else:
        ece = np.sum(np.abs(bin_conf - bin_acc) * bin_counts / len(probabilities))

    return ece, bin_centers, bin_acc, bin_conf, bin_counts


def plot_calibration_curve(ece, bin_centers, bin_acc, bin_probs, bin_counts, labels, include_bars=False,
                           title='Calibration Curves', save_file=None, show=True, log=False):
    if isinstance(ece, float):
        ece = [ece]
        bin_centers = [bin_centers]
        bin_acc = [bin_acc]
        bin_probs = [bin_probs]
        bin_counts = [bin_counts]
        labels = [labels]

    def transform(vals):
        # Note - this will break if give 100% confidence, so only use for log probs and it should be fine
        return -np.log(1 - vals)

    plt.figure(figsize=(8, 8))

    max_x, max_y = 0.0, 0.0

    for i in range(len(ece)):
        label = f'{labels[i]} (ECE: {ece[i]:.3f})'
        sizes = bin_counts[i] + 1

        non_empty_bins = bin_counts[i].astype(bool)

        x_values = bin_centers[i][non_empty_bins]
        y_values = bin_acc[i][non_empty_bins]
        if log:
            x_values = transform(x_values)
            y_values = transform(y_values)

        max_x = max(np.max(x_values), max_x)
        max_y = max(np.max(y_values), max_y)

        plt.scatter(x_values, y_values, s=sizes[non_empty_bins])
        plt.plot(x_values, y_values, '-', label=label)

        if include_bars:
            # TODO - this only works if all bins have the same width so doesn't work with
            # the quantile approach.
            bin_width = bin_centers[i][1] - bin_centers[i][0] if len(bin_centers[i]) > 1 else 1.0
            plt.bar(
                bin_centers[i], bin_acc[i], width=bin_width, align='center', alpha=0.7,
                label=f'Accuracy in Bin ({labels[i]})'
            )
            plt.bar(
                bin_centers[i], bin_probs[i], width=bin_width, align='center', alpha=0.3,
                label=f'Confidence in Bin ({labels[i]})'
            )

    perfect_line = np.array([0, 1])
    if log:
        perfect_line = [0, max_x]
    plt.plot(perfect_line, perfect_line, linestyle='--', color='gray', label='Perfect Calibration')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    if not log:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.xlim(0, max_x * 1.1)
        plt.ylim(0, np.max(max_y * 1.5, max_x * 1.1))
    plt.legend()

    if save_file:
        plt.savefig(save_file)
    if show:
        plt.show()
    else:
        plt.close()


def plot_confidence_against_iterations(data, output_folder: str = None, show=SHOW):
    std_devs = [np.std([p for p in probs if p is not None]) for _, _, probs in data]

    min_std, max_std = min(std_devs), max(std_devs)
    normalized_stds = [(std - min_std) / (max_std - min_std) for std in std_devs]

    cmap = plt.cm.viridis
    colors = cmap(normalized_stds)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    name = ''
    status = set()
    for idx, item in enumerate(data):
        id, success, probs = item
        status.add(success)
        probs = [p for p in probs if p is not None]
        if not probs:
            continue
        plt.plot(probs, marker='.', color=colors[idx], label=id)
        final_color = 'green' if success else 'red'
        plt.plot(len(probs) - 1, probs[-1], marker='o' if success else 'x', color=final_color)
    plt.xlabel('Iteration')
    plt.ylabel('Confidence')
    plt.ylim(-4, 104)
    plt.minorticks_on()

    plt.grid(visible=True, axis='y', which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width * 0.9, ax_pos.height])

    cax = fig.add_axes([ax_pos.x0 + ax_pos.width * 0.92, ax_pos.y0, 0.02, ax_pos.height])

    norm = Normalize(vmin=min_std, vmax=max_std)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Standard Deviation of Confidences (%)')

    if output_folder is not None:
        if len(status) == 1:
            name = '_success' if status.pop() else '_failure'
        plt.savefig(output_folder + f'/iterations{name}.pdf')

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(preds_t, target_t, labels, path=None, show=SHOW):
    plt.figure(figsize=(8, 8))
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random guess')

    roc = ROC(num_classes=1, task='binary')
    auroc = AUROC(num_classes=1, task='binary')

    for i in range(preds_t.shape[0]):
        fpr, tpr, thresholds = roc(preds_t[i], target_t[i])
        roc_auc = auroc(preds_t[i], target_t[i])

        # Determine the best threshold
        distances = (fpr ** 2 + (1 - tpr) ** 2) ** 0.5
        min_distance_index = np.argmin(distances)
        best_threshold = thresholds[min_distance_index]

        plt.plot(fpr, tpr, label=f'{labels[i]} (AUROC: {roc_auc:.3f})')
        plt.annotate(
            f'{best_threshold:.3f}',
            (fpr[min_distance_index], tpr[min_distance_index]),
            textcoords="offset points",
            xytext=(-10, 10),
            ha='center',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none')
        )
        plt.plot(
            fpr[min_distance_index], tpr[min_distance_index], 'x', color='red',
        )

    plt.plot([], [], 'x', color='red', label='Best threshold')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey', alpha=0.5)

    if path:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def prob_to_logit(probabilities):
    """Convert probabilities to logits. Assumes probabilities are not exactly 0 or 1 for stability."""
    eps = 1e-8
    probabilities = np.clip(probabilities, eps, 1 - eps)
    logits = np.log(probabilities / (1 - probabilities))
    return logits


def scale_logits_with_temp(logits, temperature):
    """Apply temperature scaling to 1D array of logits and return the calibrated probabilities for the positive class."""
    scaled_logits = logits / temperature
    calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
    return calibrated_probs


def scale_probs_with_temp(preds_np, temperature):
    non_extremes = (preds_np != 0.0) & (preds_np != 1.0)
    scaled_percentages = scale_logits_with_temp(prob_to_logit(preds_np[non_extremes]), temperature)
    new_percentages = preds_np.copy()
    new_percentages[non_extremes] = scaled_percentages
    return new_percentages


def temperature_objective(temperature, logits, labels):
    """ Objective function for temperature scaling which minimizes negative log likelihood. """
    probabilities = scale_logits_with_temp(logits, temperature)
    return log_loss(labels, probabilities)


def calculate_temperature_scale(preds_np_train, targets_np_train):
    """
    Use the resulting temperature like this
        calibrated_probs_test = apply_temperature_scaling(logits_test, optimal_temperature)

    """
    non_extremes = (preds_np_train != 0.0) & (preds_np_train != 1.0)
    preds_np_train = preds_np_train[non_extremes]
    targets_np_train = targets_np_train[non_extremes]

    # Convert to logits
    logits_val = prob_to_logit(preds_np_train)

    result = minimize(
        temperature_objective,
        x0=np.array([1.0]),
        bounds=[(0.01, 100)],
        args=(logits_val, targets_np_train),
    )

    optimal_temperature = result.x[0]

    return optimal_temperature


def calculate_overall_benchmark_score(eces, aurocs):
    return np.mean((1-np.array(eces)) * np.array(aurocs))
