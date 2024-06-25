import os
import shutil


def main():
    all_original_os = "2024-08-05-15-10-54"
    final_fixed = "2024-08-25-21-34-31"

    conf_final_fixed = "2024-08-27-11-41-48"
    conf_final_logits = "2024-08-28-15-49-29"
    conf_final_logits_scaled = "2024-08-28-15-49-29_scaled"
    conf_final_fixed_gtp35_base = "2024-08-28-19-22-48"
    conf_final_fixed_gtp4_turbo_base = "2024-08-29-12-11-11"
    conf_final_fixed_scaled = "2024-08-27-11-41-48_scaled"

    outputs_to_parse = [
        conf_final_fixed, conf_final_logits, conf_final_fixed_gtp35_base, conf_final_fixed_gtp4_turbo_base
    ]


    names = {
        # all_original_os: 'original',
        # final_fixed: 'fixed',
        conf_final_fixed: 'verb',
        conf_final_fixed_scaled: 'verb-t',
        conf_final_logits: 'logits',
        conf_final_logits_scaled: 'logits-t',
        conf_final_fixed_gtp35_base: 'verb_gtp35_base',
        conf_final_fixed_gtp4_turbo_base: 'verb_gtp4turbo_base',
    }

    figures = [
        'roc_metric_all.pdf',
        'calibration_all.pdf',
    ]
    for run, name in names.items():
        print(run)
        for fig in figures:
            shutil.copyfile(
                src=os.path.join('outputs', run, 'gpt-4-turbo/os-std/analysis', fig),
                dst=os.path.join('results', f'{fig.split(".")[0]}_{name}.pdf')
            )

    names = {
        conf_final_fixed: 'verb',
        conf_final_logits_scaled: 'logits-t',
    }
    for run, name in names.items():
        models = next(os.walk(os.path.join('outputs', run)))[1]
        for model in models:
            for fig in figures:
                shutil.copyfile(
                    src=os.path.join('outputs', run, f'{model}/os-std/analysis', fig),
                    dst=os.path.join('results', f'{fig.split(".")[0]}_{model}_{name}.pdf')
                )
            for status in ['failure', 'success']:
                shutil.copyfile(
                    src=os.path.join('outputs', run, f'{model}/os-std/analysis', f'iterations_{status}.pdf'),
                    dst=os.path.join('results', f'iterations_{status}_{model}_{name}.pdf')
                )


if __name__ == '__main__':
    main()
