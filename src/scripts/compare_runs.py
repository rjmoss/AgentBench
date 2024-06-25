import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from configs import ConfigLoader
from scripts.parse_output import ResultsParser, TaskResult
from typings import AssignmentConfig


SHOW = False

plt.rcParams.update(
    {
        'font.size': 12,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 12,
        'figure.titlesize': 20
    }
)


def flatten_dict(d, parent_key='', sep='_', use_separators=True, check_clash=False, clash_dict=None):
    if clash_dict is None:
        clash_dict = {}

    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key and use_separators else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep, use_separators, check_clash, clash_dict).items())
        else:
            if check_clash and new_key in clash_dict:
                if clash_dict[new_key] != v:
                    new_key = f'{parent_key}{sep}{k}'
                    if new_key in clash_dict:
                        raise ValueError(
                            f'Key clash detected for key "{new_key}" with different values: '
                            f'{clash_dict[new_key]} and {v}'
                        )
            clash_dict[new_key] = v
            items.append((new_key, v))
    return dict(items)


def read_task_results(outputs_to_parse: dict[str, str], skip_dev=True):
    task_results = []
    for output, name in outputs_to_parse.items():
        path = "outputs/" + output
        loader = ConfigLoader()
        config_ = loader.load_from(path + "/config.yaml")
        value = AssignmentConfig.parse_obj(config_)
        value = AssignmentConfig.post_validate(value)

        parser = ResultsParser(value)
        parser.read_results(from_analysis=True)
        for tr in sorted(parser.task_results, key=lambda tr: tr.assignment.agent):
            if skip_dev:
                if tr.assignment.task == 'os-dev':
                    continue
            tr.overall["run_time"] = output
            task_results.append((tr, name))

    return task_results


def task_result_to_flat_dict(task_result: TaskResult, name: str):
    dict_ = dict(task_result.overall)
    dict_.update(task_result.assignment.dict())

    # For now using run id as the series label but later can use description or similar.
    dict_["run_full_id"] = (
        f'{task_result.overall["run_time"]}/{task_result.assignment.agent}/{task_result.assignment.task}'
    )
    dict_["suffix"] = name

    td = dict(task_result.task_def)
    td['parameters'].pop("data_config", None)
    td['parameters'].pop("docker_config", None)
    td['parameters'].pop("scripts", None)
    dict_.update(td)

    return flatten_dict(dict_, use_separators=False, check_clash=True)


def create_confidence_benchmark_table(run_results: list[tuple[TaskResult, str]], save_path: str):
    df = results_to_df(run_results)

    metrics = {
        'acc': 'ACC$^\\uparrow$',
        'confidence_ece': 'ECE$^\\downarrow$',
        'confidence_brier': 'BS$^\\downarrow$',
        'confidence_auroc': 'AUC$^\\uparrow$',
        'overall_benchmark_score': 'OBS$^\\uparrow$',
    }
    confidence_metrics = [m for m in metrics if m.startswith('confidence')]

    times = {
        'first': 'First Confidence',
        'last': 'Last Confidence'
    }
    table_data = []
    for i, label in enumerate(df['label']):
        row_data = [label, df.loc[i, 'suffix']]
        row_data.append(df.loc[i, 'acc'])
        for time in times:
            for metric in confidence_metrics:
                full_metric_name = f"{time}_{metric}"
                row_data.append(df.loc[i, full_metric_name])
        row_data.append(df.loc[i, 'overall_benchmark_score'])
        table_data.append(row_data)

    column_names = ['Model', 'Type', 'acc']
    for time in times:
        for metric in confidence_metrics:
            column_names.append(f"{time}_{metric}")
    column_names.append('overall_benchmark_score')

    table_df = pd.DataFrame(table_data, columns=column_names)

    # Format data to 2 decimal places
    table_df.iloc[:, 2:] = table_df.iloc[:, 2:].round(2)

    def format_values(df, col_name, metric_type):
        formatted_values = [f"{v}" for v in df[col_name]]

        sorted_values = df[col_name].sort_values(ascending=metric_type in ['ece', 'brier'])
        best_value = sorted_values.iloc[0]
        second_best_value = sorted_values.iloc[1] if len(sorted_values) > 1 else None

        # Find indices of all instances matching the best value
        best_indices = df[col_name][df[col_name] == best_value].index
        # Apply bold to all best values
        for idx in best_indices:
            formatted_values[idx] = f"\\textbf{{{formatted_values[idx]}}}"

        # Find indices of all instances matching the second best value (excluding best values)
        if second_best_value is not None and second_best_value != best_value:
            second_best_indices = df[col_name][df[col_name] == second_best_value].index
            # Apply italic to all second best values
            for idx in second_best_indices:
                formatted_values[idx] = f"\\textit{{{formatted_values[idx]}}}"

        return formatted_values

    # Apply the formatting function to each applicable column in the DataFrame
    for col in table_df.columns[2:]:
        metric_type = col.split('_')[-1]
        table_df[col] = format_values(table_df, col, metric_type)

    header = "\\begin{tabular}{l l c " + " ".join(["c"] * 2 * len(confidence_metrics)) + " c}\n"
    header += "\\toprule\n"

    primary_headers = " & ".join(["", "", ""] + [f"\\multicolumn{{3}}{{c}}{{{time}}}" for time in times.values()] + [""])
    header += f"{primary_headers} \\\\\n"

    # multicolumn midrules
    cmidrules = []
    cumulative_columns = 4  # starting point
    for time in times:
        span = len(confidence_metrics)
        cmidrules.append(f"\\cmidrule(lr){{{cumulative_columns}-{cumulative_columns + span - 1}}}")
        cumulative_columns += span

    header += " ".join(cmidrules) + "\n"

    subheaders = " & ".join(
        ['Model', 'Type'] +
        [metrics['acc']] +
        [f"{metrics[m]}" for time in times for m in confidence_metrics] +
        [metrics['overall_benchmark_score']]
    )
    header += f"{subheaders} \\\\\n"

    body = table_df.to_latex(index=False, escape=False, header=False).split('\n', 2)[-1]

    latex_table = header + body

    with open(save_path, 'w') as f:
        f.write(latex_table)


def create_confidence_details_table(run_results: list[tuple[TaskResult, str]], save_path: str):
    df = results_to_df(run_results)

    statuses = ['success', 'fail']
    times = ['first', 'avg', 'last']

    table_data = []
    for i, label in enumerate(df['label']):
        for status in statuses:
            row_data = [label, df.loc[i, 'suffix'], status]
            for time in times:
                mean_key = f'{time}_confidence_{status}_avg_conf'
                std_key = f'{time}_confidence_{status}_std_conf'

                mean_std = f"{df.loc[i, mean_key]:.3f} ± {df.loc[i, std_key]:.3f}"
                row_data.append(mean_std)
            table_data.append(row_data)

    column_names = ['model', 'type', 'outcome'] + [f'{time} (mean ± std)' for time in times]

    table_df = pd.DataFrame(table_data, columns=column_names)
    table_df.sort_values(by=['model', 'type'], inplace=True)

    latex_table = ""
    current_model = None
    current_type = None
    first_row = True

    for idx, row in table_df.iterrows():
        model, type_, outcome = row['model'], row['type'], row['outcome']
        row_data = row[3:]

        if model != current_model:
            if not first_row:
                latex_table += "\\cmidrule(lr){1-6}\n"
            current_model = model
            current_type = type_
            model_rowspan = (table_df['model'] == model).sum()
            type_rowspan = (table_df[(table_df['model'] == model) & (table_df['type'] == type_)].shape[0])
            latex_table += f"\\multirow{{{model_rowspan}}}{{*}}{{{model}}} & \\multirow{{{type_rowspan}}}{{*}}{{{type_}}} & {outcome} "
        elif type_ != current_type:
            current_type = type_
            type_rowspan = (table_df[(table_df['model'] == model) & (table_df['type'] == type_)].shape[0])
            latex_table += f"& \\multirow{{{type_rowspan}}}{{*}}{{{type_}}} & {outcome} "
        else:
            latex_table += f"& & {outcome} "

        latex_table += " & " + " & ".join(row_data) + " \\\\\n"
        first_row = False

    latex_table = (
            "\\begin{tabular}{lllccc}\n" +
            "\\toprule\n" +
            " & ".join(name.capitalize() for name in column_names) + "\\\\\n" +
            "\\midrule\n" +
            latex_table +
            "\\bottomrule\n" +
            "\\end{tabular}"
    )

    print(latex_table)

    with open(save_path, 'w') as f:
        f.write(latex_table)


def results_to_df(run_results):
    df = pd.DataFrame([task_result_to_flat_dict(item, name) for item, name in run_results])
    # df['label'] = df['agent'] + df['suffix']
    df['label'] = df['agent']
    df.insert(0, 'label', df.pop('label'))
    return df


def comparison_bar_chart(run_results: list[tuple[TaskResult, str]], save_path: str):
    df = results_to_df(run_results)

    metrics = {
        'confidence_ece': 'ECE$^\\downarrow$',
        'confidence_brier': 'Brier Score$^\\downarrow$',
        'confidence_auroc': 'AUROC$^\\uparrow$',
        'acc': 'Accuracy$^\\uparrow$',
    }
    df.to_csv(save_path + "/data.csv", index=False)
    num_colors = len(df)
    cmap = plt.get_cmap('tab10')
    base_colors = [cmap(i) for i in range(num_colors)]

    times = ['first', 'avg', 'last']
    intensity_factors = [0.4, 0.7, 1.0]

    def adjust_color_lightness(color, alpha_factor):
        color = np.array(color)
        color[3] = np.clip(color[3] * alpha_factor, 0, 1)
        return color

    bar_width = 0.8 / len(df)

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, label in enumerate(df['label']):
        for j, metric in enumerate(metrics.keys()):

            base_color = base_colors[i]
            if metric.startswith('confidence'):
                for k, time in enumerate(times):
                    full_metric_name = f"{time}_{metric}"
                    base_color = base_colors[i]
                    plt.bar(
                        1 + j + i * bar_width + k * (bar_width / len(times)),
                        df.loc[i, full_metric_name],
                        bar_width / len(times),
                        color=adjust_color_lightness(base_color, intensity_factors[k]),
                        label=f"{label} - {time.capitalize()}" if i == 0 else ""
                    )
            else:
                plt.bar(
                    1 + j + i * bar_width + bar_width / 2,
                    df.loc[i, metric],
                    bar_width,
                    color=base_color,
                    label=f"{label}" if i == 0 else ""
                )

    plt.ylim(0, 1)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Verbalised self-confidence comparison')

    xtick_labels = [metrics[metric] for metric in metrics]
    xtick_locations = 1 + np.arange(len(metrics)) + bar_width * len(df) / 2 - bar_width / (2 * len(times))
    xtick_locations[-1] = xtick_locations[-1] + bar_width / (2 * len(times))
    plt.xticks(xtick_locations, xtick_labels)

    legend_elements = []
    for i, label in enumerate(df['label']):
        for t, time in enumerate(times):
            legend_elements.append(
                Patch(facecolor=adjust_color_lightness(base_colors[i], intensity_factors[t]), label=f'{label} ({time})')
            )
    ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1), loc='upper left', frameon=True)

    plt.tight_layout()

    plt.savefig(save_path + '/metrics_grouped.pdf')
    if SHOW:
        plt.show()


def plot_not_grouped(df, save_path):
    metrics = {
        'ece': 'ECE',
        'auroc': 'AUROC',
        'acc': 'ACC',
    }

    bar_width = 0.8 / len(df)
    index = np.arange(len(metrics))
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(df['label']):
        plt.bar(index + i * bar_width, df.loc[i, metrics.keys()], bar_width, label=label)
    plt.ylim(0, 1)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Run metric comparison')
    plt.xticks(index + bar_width * (len(df['label']) - 1) / 2, list(metrics.values()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/metrics.pdf')
    if SHOW:
        plt.show()


def main(args):
    save_path = args.save
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    all_original_os = "2024-08-05-15-10-54"
    final_fixed = "2024-08-25-21-34-31"

    conf_final_fixed = "2024-08-27-11-41-48"
    conf_final_fixed_scaled = "2024-08-27-11-41-48_scaled"
    conf_final_logits = "2024-08-28-15-49-29"
    conf_final_logits_scaled = "2024-08-28-15-49-29_scaled"
    conf_final_fixed_gtp35_base = "2024-08-28-19-22-48"
    conf_final_fixed_gtp4_turbo_base = "2024-08-29-12-11-11"

    outputs_to_parse = {
        conf_final_fixed: 'verbalised',
        conf_final_fixed_scaled: 'verbalised-t',
        conf_final_logits: 'logits',
        conf_final_logits_scaled: 'logits-t',
        conf_final_fixed_gtp35_base: 'gpt-3.5 base',
        conf_final_fixed_gtp4_turbo_base: 'gpt-4-turbo base',
    }

    task_results_flat = read_task_results(outputs_to_parse)
    core = [(tr, name) for (tr, name) in task_results_flat if 'base' not in name and 'logits' not in name and '-t' not in name]
    based = [(tr, name) for (tr, name) in task_results_flat if 'base' in name]

    comparison_bar_chart(core, save_path)

    create_confidence_benchmark_table(
        [(tr, name) for (tr, name) in task_results_flat if name in ['verbalised', 'logits-t']],
        f'{save_path}/confidence_table.tex'
    )
    create_confidence_benchmark_table(
        [(tr, name) for (tr, name) in task_results_flat if 'base' not in name],
        f'{save_path}/confidence_table_non_based.tex'
    )
    create_confidence_benchmark_table(based, f'{save_path}/confidence_table_based.tex')

    create_confidence_details_table(
        [(tr, name) for (tr, name) in task_results_flat if 'base' not in name],
        f'{save_path}/confidence_table_details.tex'
    )


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--save", type=str, default="analysis")
    args = arg_parser.parse_args()
    main(args)
