"""Check published validation summaries and pooled analysis counts."""
import json
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def main():
    runs = read('validation_runs.json')['runs']
    metrics = ['Has0_acc_2', 'Non0_acc_2', 'Has0_F1', 'Non0_F1', 'Acc_7', 'MAE', 'Corr']
    expected = {
        ('MOSI', 'CLGSI'): ['82.82±0.91', '85.96±0.96', '82.95±0.91', '85.98±0.98', '45.12±2.63', '0.692±0.012', '0.807±0.004'],
        ('MOSI', 'LCTW'): ['85.74±0.67', '88.12±0.53', '85.77±0.68', '88.07±0.56', '43.52±1.82', '0.700±0.005', '0.807±0.001'],
        ('MOSEI', 'CLGSI'): ['81.92±0.39', '85.74±0.39', '82.46±0.29', '85.70±0.43', '53.79±1.39', '0.514±0.007', '0.752±0.008'],
        ('MOSEI', 'LCTW'): ['82.45±0.88', '85.19±0.66', '82.86±0.72', '85.09±0.74', '54.05±1.01', '0.518±0.005', '0.751±0.004'],
    }
    assert len(runs) == 12
    for (dataset, method), reference in expected.items():
        selected = [r for r in runs if (r['dataset'], r['method']) == (dataset, method)]
        assert sorted(r['seed'] for r in selected) == [10111, 10112, 10113]
        shown = []
        for metric in metrics:
            scale, digits = (1, 3) if metric in ('MAE', 'Corr') else (100, 2)
            values = [r['metrics'][metric] * scale for r in selected]
            shown.append(f'{mean(values):.{digits}f}±{stdev(values):.{digits}f}')
        assert shown == reference, (dataset, method, shown, reference)
        print(dataset, method, dict(zip(metrics, shown)))

    counts = read('analysis_counts.json')['runs']
    expected_counts = {
        'MOSI': (687, 32, 26, 6, 6, 14, 67, 19, 3, 10532, 107856),
        'MOSEI': (5613, 472, 251, 221, 54, -24, 1745, 226, 191, 220617, 783648),
    }
    keys = ['validation_observations', 'sign_flips', 'error_to_correct', 'correct_to_error',
            'net_zero_label', 'net_nonzero_label', 'near_boundary_observations',
            'near_error_to_correct', 'near_correct_to_error',
            'training_gate_positive', 'training_observations']
    assert len(counts) == 6
    for dataset, reference in expected_counts.items():
        selected = [r for r in counts if r['dataset'] == dataset]
        assert sorted(r['seed'] for r in selected) == [10111, 10112, 10113]
        sums = tuple(sum(r[k] for r in selected) for k in keys)
        assert sums == reference, (dataset, sums, reference)
        for r in selected:
            assert r['sign_flips'] == r['error_to_correct'] + r['correct_to_error']
            assert r['error_to_correct'] - r['correct_to_error'] == r['net_zero_label'] + r['net_nonzero_label']
            assert 0 <= r['training_gate_positive'] <= r['training_observations']
        total = dict(zip(keys, sums))
        gain = total['error_to_correct'] - total['correct_to_error']
        gate = total['training_gate_positive'] / total['training_observations'] * 100
        print(dataset, f'paired accuracy gain {gain / total["validation_observations"] * 100:.2f} pp;', f'gate positive {gate:.2f}%')

    protocol = read('training_protocol.json')
    for dataset, config in protocol['datasets'].items():
        assert sum(config['batch_sizes']) == config['train_samples']
        assert len(config['batch_sizes']) * protocol['max_epochs'] == config['scheduler_steps']
        assert config['warmup_steps'] * 10 == config['scheduler_steps']
    print('PASS: validation summaries, pooled counts and schedule arithmetic. No training was performed.')


if __name__ == '__main__':
    main()
