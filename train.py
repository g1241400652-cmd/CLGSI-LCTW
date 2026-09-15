"""Train CLGSI or CLGSI with LCTW and evaluate the selected checkpoint."""
import argparse
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], required=True)
    parser.add_argument('--data', type=Path, required=True, help='Processed unaligned dataset pickle')
    parser.add_argument('--bert', type=Path, required=True, help='Local bert-base-uncased directory')
    parser.add_argument('--method', choices=['clgsi', 'lctw'], default='lctw')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10111, 10112, 10113])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forward(model, batch, device):
    return model(batch['text'].to(device),
                 (batch['audio'].to(device), batch['audio_lengths'].to(device)),
                 (batch['vision'].to(device), batch['vision_lengths'].to(device)))


def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(forward(model, batch, device)['M'].view(-1).cpu())
            labels.append(batch['labels']['M'].view(-1).cpu())
    return regression_metrics(torch.cat(predictions).numpy(), torch.cat(labels).numpy())


def optimizer_for(model, config):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert = list(model.Model.text_model.named_parameters())
    groups = [
        dict(params=[p for n, p in bert if not any(k in n for k in no_decay)],
             lr=config.learning_rate_bert, weight_decay=config.weight_decay_bert),
        dict(params=[p for n, p in bert if any(k in n for k in no_decay)],
             lr=config.learning_rate_bert, weight_decay=0.0),
        dict(params=model.Model.audio_model.parameters(), lr=config.learning_rate_audio,
             weight_decay=config.weight_decay_audio),
        dict(params=model.Model.video_model.parameters(), lr=config.learning_rate_video,
             weight_decay=config.weight_decay_video),
        dict(params=[p for n, p in model.Model.named_parameters()
                     if not any(k in n for k in ('text_model', 'audio_model', 'video_model'))],
             lr=config.learning_rate_other, weight_decay=config.weight_decay_other),
    ]
    return torch.optim.AdamW(groups)


class TailMergeBatchSampler:
    """Merge the final partial batch into the preceding full batch."""

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        if batch_size <= 0 or len(sampler) < batch_size:
            raise ValueError('Tail merge requires a positive batch size and one full batch')

    def __iter__(self):
        indices = list(iter(self.sampler))
        full, tail = divmod(len(indices), self.batch_size)
        end = (full - 1) * self.batch_size if tail else len(indices)
        for start in range(0, end, self.batch_size):
            yield indices[start:start + self.batch_size]
        if tail:
            yield indices[end:]

    def __len__(self):
        return len(self.sampler) // self.batch_size


def make_loaders(datasets, dataset, batch_size, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    loaders = {split: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
               for split, ds in datasets.items() if split != 'train'}
    if dataset == 'mosi':
        sampler = torch.utils.data.RandomSampler(datasets['train'], generator=generator)
        batches = TailMergeBatchSampler(sampler, batch_size)
        loaders['train'] = DataLoader(datasets['train'], batch_sampler=batches,
                                     num_workers=0, generator=generator)
    else:
        loaders['train'] = DataLoader(datasets['train'], batch_size=batch_size,
                                     shuffle=True, num_workers=0, generator=generator)
    return loaders


def scheduler_for(optimizer, train_loader, schedule_epochs):
    steps = len(train_loader) * schedule_epochs
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * steps,
                                          num_training_steps=steps)


def run_seed(args, datasets, seed):
    dest = args.output / args.dataset / args.method / str(seed)
    dest.mkdir(parents=True, exist_ok=False)
    base = SimpleNamespace(modelName='clgsi', datasetName=args.dataset, train_mode='regression')
    config = ConfigRegression(base).get_config()
    config.device = torch.device(args.device)
    train_data = datasets['train']
    config.train_samples = len(train_data)
    config.seq_lens = (train_data.text.shape[2], train_data.audio.shape[1], train_data.vision.shape[1])
    expected = config.feature_dims[1:]
    if (train_data.audio.shape[2], train_data.vision.shape[2]) != expected:
        raise ValueError('Dataset feature dimensions do not match the selected CLGSI configuration')
    config.warm_up_epochs = 40
    config.early_stop = 8
    seed_all(seed)
    model = AMIO(config).to(config.device)
    # Both methods start from the same seeded initialization and training RNG.
    seed_all(seed)
    loaders = make_loaders(datasets, args.dataset, config.batch_size, seed)
    # Match the label initialization pass of the paired training procedure.
    for _ in DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0):
        pass
    optimizer = optimizer_for(model, config)
    scheduler = scheduler_for(optimizer, loaders['train'], config.warm_up_epochs)
    loss_class = (make_lctw_loss_class(contrastive_loss, torch, config.gamma)
                  if args.method == 'lctw' else contrastive_loss)
    best_mae, best_epoch = float('inf'), 0
    checkpoint = dest / 'best.pt'
    with (dest / 'config.json').open('w', encoding='utf-8') as handle:
        json.dump(dict(config, seed=seed, method=args.method, max_epochs=40), handle,
                  indent=2, default=str)
    for epoch in range(1, 41):
        model.train()
        for batch in loaders['train']:
            optimizer.zero_grad()
            outputs = forward(model, batch, config.device)
            target = batch['labels']['M'].to(config.device).view(-1)
            task_loss = torch.mean(torch.abs(outputs['M'].view(-1) - target))
            auxiliary = loss_class(args.dataset, config.device, config.dividing_line)(outputs, target)
            loss = task_loss + config.gamma * auxiliary
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
        validation = evaluate(model, loaders['valid'], config.device)
        rounded_mae = round(np.float32(validation['MAE']), 4)
        if rounded_mae < best_mae:
            best_mae, best_epoch = rounded_mae, epoch
            torch.save(model.cpu().state_dict(), checkpoint)
            model.to(config.device)
        print(json.dumps(dict(seed=seed, epoch=epoch, validation=validation)), flush=True)
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=config.device, weights_only=True))
    result = dict(seed=seed, selected_epoch=best_epoch,
                  validation=evaluate(model, loaders['valid'], config.device),
                  test=evaluate(model, loaders['test'], config.device))
    (dest / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    args = parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['CLGSI_BERT_BASE_UNCASED_SNAPSHOT'] = str(args.bert.resolve())
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from config.config_regression import ConfigRegression
    from models.AMIO import AMIO
    from models.contrastive_loss import contrastive_loss
    from data.load_data import load_splits
    from metrics import regression_metrics
    from lctw import make_lctw_loss_class
    datasets = load_splits(args.data)
    results = [run_seed(args, datasets, seed) for seed in args.seeds]
    means = {k: float(np.mean([r['test'][k] for r in results])) for k in results[0]['test']}
    (args.output / args.dataset / args.method / 'mean_metrics.json').write_text(
        json.dumps(dict(seeds=args.seeds, test_mean=means), indent=2), encoding='utf-8')
    print(json.dumps(dict(test_mean=means), indent=2))
