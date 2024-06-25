from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


SHOW = False


plt.rcParams.update(
    {
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 14
    }
)


PROBLEM_MAPPING = {
    'Missing information': 'Instructions',
    'Asks agent to setup problem': 'Instructions',
    'Asking for wrong format': 'Instructions',
    'No question asked': 'Instructions',
    'Vague instructions': 'Instructions',
    'Incorrect assessment': 'Evaluation',
    'Incorrect environment setup': 'Evaluation',
    'Multiline execution failed': 'Conversation',
    'Incorrect init environment setup': 'Initialisation',
    'Incorrect eval environment setup': 'Evaluation',
    'Other': 'Other',
    'Ok': 'Ok'
}

COLOR_MAPPING = {
    'Missing information': '#FF6F61',  # Soft Red
    'Asks agent to setup problem': '#FFA07A',  # Light Salmon
    'Asking for wrong format': '#FFD700',  # Gold
    'No question asked': '#FF8C00',  # Dark Orange
    'Vague instructions': '#FF6347',  # Tomato
    'Incorrect assessment': '#4682B4',  # Steel Blue
    'Incorrect environment setup': '#6495ED',  # Cornflower Blue
    'Multiline execution failed': '#32CD32',  # Lime Green
    'Incorrect init environment setup': '#87CEFA',  # Light Sky Blue
    'Incorrect eval environment setup': '#00BFFF',  # Deep Sky Blue
    'Other': '#D3D3D3',  # Light Grey
    'Ok': '#228B22'  # Forest Green (Darker Green)
}

step_colours = {
    'Ok': '#228B22',
    'Initialisation': '#FF8C00',
    'Instructions': '#8B4513',
    'Conversation': '#B22222',
    'Evaluation': '#8B0000',
    'Other': '#000000',
}


def create_pie_chart(merged_df_sorted: pd.DataFrame, scenario):
    filtered_df = merged_df_sorted.copy(deep=True)
    filtered_df = filtered_df.loc[merged_df_sorted['Reasons'] != 'Total']

    grouped_df = filtered_df.groupby('Reasons').sum('Breaking Error')

    ordered_index = list(step_colours.keys())
    grouped_df = grouped_df.reindex(ordered_index).dropna()

    labels = grouped_df.index
    sizes = grouped_df['Breaking Error']

    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, colors=[step_colours[label] for label in labels], autopct='%1.1f%%', startangle=0,
        textprops={'fontsize': 16}
    )
    plt.axis('equal')

    for i, text in enumerate(autotexts):
        if labels[i] in ['Instructions', 'Conversation', 'Evaluation', 'Other']:
            text.set_color('#A9A9A9')

    plt.savefig(f'../../results/pie_{scenario}.pdf', dpi=300, bbox_inches='tight')
    if SHOW:
        plt.show()

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, sizes, color=[step_colours[label] for label in labels])

    plt.ylabel('Samples Count', fontsize=16)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Error stage', fontsize=16)
    plt.tight_layout()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'../../results/bar_{scenario}.pdf', dpi=300, bbox_inches='tight')
    if SHOW:
        plt.show()


def create_stacked_bar_chart(merged_dfs: list, scenarios: list):
    width = 0.8
    fig, ax = plt.subplots(figsize=(10, 7))

    fig.subplots_adjust(right=0.75)

    n_scenarios = len(scenarios)
    indices = np.arange(n_scenarios)

    for index, (df, scenario) in enumerate(zip(merged_dfs, scenarios)):
        filtered_df = df.copy(deep=True)
        filtered_df = filtered_df[filtered_df['Reasons'] != 'Total']

        grouped_df = filtered_df.groupby('Reasons').sum('Breaking Error')
        ordered_index = list(step_colours.keys())
        grouped_df = grouped_df.reindex(ordered_index).dropna()

        labels = grouped_df.index
        sizes = grouped_df['Breaking Error']
        percentages = sizes / sizes.sum() * 100

        # Create a stacked bar for each scenario
        bottom_heights = np.zeros(n_scenarios)
        for label, percentage in zip(labels, percentages):
            ax.bar(
                indices[index], percentage, width, bottom=bottom_heights[index],
                color=step_colours[label], label=label if index == 0 else ""
            )
            bottom_heights[index] += percentage

            color = '#A9A9A9' if label in ['Instructions', 'Conversation', 'Evaluation', 'Other'] else 'black'

            ax.text(
                indices[index], min(bottom_heights[index] - (percentage / 2), 98.2), f'{percentage:.1f}%', ha='center',
                va='center', color=color, fontsize=14
            )

    ax.set_xticks(indices)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Percentage (%)')

    legend_elements = [Patch(facecolor=color, label=label) for label, color in step_colours.items()]
    ax.legend(handles=legend_elements, title='Error Stage', loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f'../../results/stacked_bar_chart.pdf', dpi=300, bbox_inches='tight')
    if SHOW:
        plt.show()
    else:
        plt.close()


def create_summary_table(df_sorted):
    primary_count_dict = defaultdict(int)
    secondary_count_dict = defaultdict(int)
    both_count_dict = defaultdict(int)

    for index, row in df_sorted.iterrows():
        primary_problem = row['Problem 1']
        secondary_problem = row['Problem 2']

        if primary_problem in PROBLEM_MAPPING:
            primary_count_dict[(PROBLEM_MAPPING[primary_problem], primary_problem)] += 1
            both_count_dict[(PROBLEM_MAPPING[primary_problem], primary_problem)] += 1
        else:
            raise ValueError(f'Unrecognised problem {primary_problem}')

        if secondary_problem in PROBLEM_MAPPING:
            secondary_count_dict[(PROBLEM_MAPPING[secondary_problem], secondary_problem)] += 1
            both_count_dict[(PROBLEM_MAPPING[secondary_problem], secondary_problem)] += 1
        else:
            if not np.isnan(secondary_problem):
                raise ValueError(f'Unrecognised problem {secondary_problem}')

    primary_count_df = pd.DataFrame(
        [(reason, result, count) for (reason, result), count in primary_count_dict.items()],
        columns=['Reasons', 'Result', 'Breaking Error']
    )

    secondary_count_df = pd.DataFrame(
        [(reason, result, count) for (reason, result), count in secondary_count_dict.items()],
        columns=['Reasons', 'Result', 'Secondary Error']
    )

    both_count_df = pd.DataFrame(
        [(reason, result, count) for (reason, result), count in both_count_dict.items()],
        columns=['Reasons', 'Result', 'Either']
    )

    merged_df = pd.merge(both_count_df, primary_count_df, on=['Reasons', 'Result'], how='outer')
    merged_df = pd.merge(merged_df, secondary_count_df, on=['Reasons', 'Result'], how='outer')

    merged_df = merged_df.fillna(0).astype({'Breaking Error': 'int', 'Secondary Error': 'int', 'Either': 'int'})

    total_issues = merged_df[merged_df['Reasons'] != 'Ok'][['Breaking Error', 'Secondary Error', 'Either']].sum()
    total_issues_row = pd.DataFrame(
        [['Total', '', total_issues['Breaking Error'], total_issues['Secondary Error'], total_issues['Either']]],
        columns=['Reasons', 'Result', 'Breaking Error', 'Secondary Error', 'Either']
    )
    merged_df = pd.concat([merged_df, total_issues_row], ignore_index=True)

    # Define the custom order
    order = list(PROBLEM_MAPPING.values()) + ['Total', 'Ok']
    order_mapping = {key: idx for idx, key in enumerate(order)}

    merged_df['Order'] = merged_df['Reasons'].map(order_mapping)
    merged_df_sorted = merged_df.sort_values(by=['Order', 'Result']).drop(columns=['Order'])
    merged_df_sorted = merged_df_sorted[['Reasons', 'Result', 'Breaking Error', 'Secondary Error', 'Either']]

    return merged_df_sorted


def write_count_table(merged_df_sorted, scenario, secondary=False):
    if secondary:
        latex_table = "\\begin{tabular}{l l c c c}\n"
        latex_table += "\\toprule\n"
        latex_table += "Stage & Category & First Error & Second Error & Either \\\\\n"
        latex_table += "\\midrule\n"
    else:
        latex_table = "\\begin{tabular}{l l c}\n"
        latex_table += "\\toprule\n"
        latex_table += "Stage & Category & Error \\\\\n"
        latex_table += "\\midrule\n"

    current_reason = ""
    for _, row in merged_df_sorted.iterrows():
        if row['Reasons'] == 'Ok':
            continue

        if row['Reasons'] != current_reason:
            current_reason = row['Reasons']
            reason_str = f"\\multirow{{}}*{{{current_reason}}}"
        else:
            reason_str = ""

        if current_reason == 'Total':
            latex_table += "\\midrule\n"

        if secondary:
            latex_table += f"{reason_str} & {row['Result']} & {row['Breaking Error']} & {row['Secondary Error']} & {row['Either']} \\\\\n"
        else:
            latex_table += f"{reason_str} & {row['Result']} & {row['Breaking Error']} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    with open(f'../../results/count_table_{scenario}.tex', 'w') as f:
        f.write(latex_table)


def write_samples_table(df_sorted, scenario):
    df_selected = df_sorted[['Index', 'Problem 1', 'Problem 2', 'Status']].copy()
    df_selected.columns = ['Index', 'Primary Problem', 'Secondary Problem', 'Status']
    df_selected.loc[df_selected['Status'].isna() & (df_selected['Primary Problem'] == 'Ok'), 'Status'] = 'Ok'
    df_selected.loc[df_selected['Status'].isna() & (df_selected['Primary Problem'] != 'Ok'), 'Status'] = 'Fixed'
    df_selected = df_selected.replace(np.NaN, '-')

    df_selected['Primary Problem'] = df_selected['Primary Problem'].str.replace('environment', 'env')
    df_selected['Secondary Problem'] = df_selected['Secondary Problem'].str.replace('environment', 'env')

    long = len(df_selected) > 50
    latex_table = df_selected.to_latex(index=False, longtable=long)
    if long:
        latex_table = "{\\small\n" + latex_table + "\n}"

    with open(f'../../results/table_{scenario}.tex', 'w') as f:
        f.write(latex_table)


def identify_encoding_issues(file_path):
    problematic_lines = []
    with open(file_path, 'rb') as f:
        for idx, line in enumerate(f, start=1):
            try:
                line.decode('utf-8')
            except UnicodeDecodeError as e:
                problematic_lines.append((idx, line, str(e)))

    return problematic_lines


def read_samples_csv(file_path):
    problematic_lines = identify_encoding_issues(file_path)
    if problematic_lines:
        print("Problematic lines:")
        for idx, line, error in problematic_lines:
            print(f"Line {idx}: {line}")
            print(f"Error: {error}")
    else:
        print("No encoding issues found.")

    if not problematic_lines:
        df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', on_bad_lines='skip')
    else:
        raise ValueError(problematic_lines)

    try:
        df_sorted = df.sort_values(by='Key')
    except KeyError:
        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        df_sorted = df.sort_values(by='Key')

    return df_sorted


def create_stacked_column_results(df, scenario, original, fixed):
    df = df.rename(
        columns={
            original[0]: original[1],
            fixed[0]: fixed[1],
        }
    )
    df['Broken Originally'] = df['Problem 1'] != 'Ok'
    df['Broken Now'] = (df['Status'] != 'Fixed') & (df['Status'] != 'Ok') & (df['Status'] != '') & (
        ~pd.isna(df['Status']))

    colors = {
        'pass': 'green',
        'pass_broken': '#002B14',
        'fail': 'red',
        'fail_broken': '#600000'
    }

    # Define conditions for original and fixed
    conditions = [
        (df['Original'] == True) & (df['Broken Originally'] == False),
        (df['Original'] == True) & (df['Broken Originally'] == True),
        (df['Original'] == False) & (df['Broken Originally'] == False),
        (df['Original'] == False) & (df['Broken Originally'] == True)
    ]

    conditions_fixed = [
        (df['Fixed'] == True) & (df['Broken Now'] == False),
        (df['Fixed'] == True) & (df['Broken Now'] == True),
        (df['Fixed'] == False) & (df['Broken Now'] == False),
        (df['Fixed'] == False) & (df['Broken Now'] == True)
    ]

    num_fixed = df[(df['Broken Originally'] == True) & (df['Broken Now'] == False)].shape[0]
    num_fixed_changed_status = \
    df[(df['Broken Originally'] == True) & (df['Broken Now'] == False) & (df['Original'] != df['Fixed'])].shape[0]
    num_fixed_true_to_false = df[
        (df['Broken Originally'] == True) & (df['Broken Now'] == False) & (df['Original'] == True) & (
                    df['Fixed'] == False)].shape[0]
    num_fixed_false_to_true = df[
        (df['Broken Originally'] == True) & (df['Broken Now'] == False) & (df['Original'] == False) & (
                    df['Fixed'] == True)].shape[0]

    print('num_fixed', num_fixed)
    print('num_fixed_changed_status', num_fixed_changed_status)
    print('num_fixed_true_to_false', num_fixed_true_to_false)
    print('num_fixed_false_to_true', num_fixed_false_to_true)

    # Calculate counts
    counts_original = [df[condition].shape[0] for condition in conditions]
    counts_fixed = [df[condition].shape[0] for condition in conditions_fixed]

    percent_original = [count / len(df) * 100 for count in counts_original]
    percent_fixed = [count / len(df) * 100 for count in counts_fixed]

    # Data preparation for looped plotting
    data = {
        'Original': percent_original,
        'Fixed': percent_fixed
    }

    counts_data = {
        'Original': counts_original,
        'Fixed': counts_fixed
    }

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    bar_width = 0.4

    for i, category in enumerate(['Original', 'Fixed']):
        bottom = 0
        for j, color in enumerate(colors.values()):
            # Draw the bar
            bar = ax.bar(category, data[category][j], color=color, bottom=bottom, width=bar_width)

            # Calculate the center of the bar segment
            bar_center = bottom + data[category][j] / 2
            if j == 3 and data[category][j] < 5:
                bar_center = (2 * bottom + bar_center) / 3
                # bar_center = bottom

            if data[category][j] != 0:
                ax.text(
                    i, bar_center, f'{counts_data[category][j]}', ha='center', va='center', color='white', fontsize=10
                )
            bottom += data[category][j]

    ax.set_ylabel('Percentage')
    ax.set_title('Original vs Fixed Status')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Fixed'])

    # Custom legend
    legend_elements = [
        Patch(facecolor=colors['pass'], label='Pass'),
        Patch(facecolor=colors['pass_broken'], label='Pass (Broken)'),
        Patch(facecolor=colors['fail'], label='Fail'),
        Patch(facecolor=colors['fail_broken'], label='Fail (Broken)')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'../../results/results_and_status_{scenario}.pdf', dpi=300, bbox_inches='tight')
    if SHOW:
        plt.show()


def main():
    all_original_os = "2024-08-05-15-10-54"
    final_fixed = "2024-08-25-21-34-31"

    scenarios = {
        'dev': ('dev_samples.csv', (all_original_os, "Original"), (final_fixed, "Fixed")),
        'std': ('std_samples.csv', (all_original_os, "Original"), (final_fixed, "Fixed")),
    }
    merged_frames = []
    for scenario, args in scenarios.items():
        print(scenario)
        df_sorted = read_samples_csv(args[0])

        # Individual samples
        write_samples_table(df_sorted, scenario)
        create_stacked_column_results(df_sorted, scenario, args[1], args[2])

        # Summaries
        merged_df_sorted = create_summary_table(df_sorted)
        write_count_table(merged_df_sorted, scenario, scenario == 'std')
        create_pie_chart(merged_df_sorted, scenario)
        merged_frames.append(merged_df_sorted)

    create_stacked_bar_chart(merged_frames, list(scenarios.keys()))


if __name__ == '__main__':
    main()
