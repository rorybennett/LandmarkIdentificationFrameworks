"""
Create deterministic 5-fold train, test, and validation lists from one mark-list file.

Each fold is approximately 80% training, 10% testing, and 10% validation.
"""

import csv
import random
import re
import shutil
from pathlib import Path


NUM_FOLDS = 5
SEED = 42

MARK_LIST_PATH = Path(r'D:\Datasets\IPV\OriginalData\doctors_resampled_transverseMarkList.txt')
OUTPUT_DIR = Path(r'D:\GeneratedFiles\IPV\NetworkStudy\folds_network_study')

CLEAN_OUTPUT_DIR = False
SORT_OUTPUT_FILES = True
WRITE_SUMMARY_CSV = True
WRITE_MEMBERSHIP_CSV = True

TRAIN_PREFIX = 'train'
TEST_PREFIX = 'test'
VAL_PREFIX = 'val'


def natural_key(value):
    """Create a natural sorting key, so A2 comes before A10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


def read_sample_ids(mark_list_path):
    """Read sample IDs from the first column of a mark-list file."""
    sample_ids = []

    with open(mark_list_path, 'r', encoding='utf-8') as mark_file:
        for line in mark_file:
            line = line.strip()

            if not line:
                continue

            image_name = line.split()[0]
            sample_ids.append(Path(image_name).stem)

    return sample_ids


def check_duplicates(sample_ids, source_name):
    """Stop execution if duplicated sample IDs are present."""
    seen = set()
    duplicates = set()

    for sample_id in sample_ids:
        if sample_id in seen:
            duplicates.add(sample_id)

        seen.add(sample_id)

    if duplicates:
        duplicate_text = ', '.join(sorted(duplicates, key=natural_key))
        raise ValueError(f'Duplicate sample IDs found in {source_name}: {duplicate_text}')


def load_unique_sample_ids(mark_list_path):
    """Load, validate, and naturally sort sample IDs before shuffling."""
    sample_ids = read_sample_ids(mark_list_path)
    check_duplicates(sample_ids, str(mark_list_path))

    if not sample_ids:
        raise ValueError(f'No sample IDs found in {mark_list_path}')

    return sorted(sample_ids, key=natural_key)


def make_balanced_chunks(sample_ids, num_chunks):
    """Split shuffled sample IDs into balanced chunks."""
    if num_chunks < 1:
        raise ValueError('num_chunks must be at least 1.')

    if len(sample_ids) < num_chunks:
        raise ValueError(f'Not enough samples to create {num_chunks} chunks.')

    chunks = [[] for _ in range(num_chunks)]

    for index, sample_id in enumerate(sample_ids):
        chunks[index % num_chunks].append(sample_id)

    return chunks


def create_folds(sample_ids, num_folds):
    """Create 5 folds by pairing 10 balanced chunks into test and validation sets."""
    if num_folds != 5:
        raise ValueError('This script is configured for 5 folds to produce an 80/10/10 split.')

    chunks = make_balanced_chunks(sample_ids, num_chunks=num_folds * 2)
    folds = []

    for fold_index in range(num_folds):
        test_ids = chunks[fold_index * 2]
        val_ids = chunks[(fold_index * 2) + 1]
        holdout_ids = set(test_ids) | set(val_ids)
        train_ids = [sample_id for sample_id in sample_ids if sample_id not in holdout_ids]

        folds.append({
            'fold': fold_index + 1,
            'train': train_ids,
            'test': test_ids,
            'val': val_ids
        })

    return folds


def validate_folds(folds, all_sample_ids):
    """Check that train, test, and validation splits are valid for every fold."""
    all_sample_set = set(all_sample_ids)

    for fold in folds:
        train_set = set(fold['train'])
        test_set = set(fold['test'])
        val_set = set(fold['val'])

        if train_set & test_set:
            raise ValueError(f'Fold {fold["fold"]} has train/test overlap.')

        if train_set & val_set:
            raise ValueError(f'Fold {fold["fold"]} has train/val overlap.')

        if test_set & val_set:
            raise ValueError(f'Fold {fold["fold"]} has test/val overlap.')

        if train_set | test_set | val_set != all_sample_set:
            raise ValueError(f'Fold {fold["fold"]} does not cover the full sample set.')

    holdout_ids = []

    for fold in folds:
        holdout_ids.extend(fold['test'])
        holdout_ids.extend(fold['val'])

    if set(holdout_ids) != all_sample_set:
        raise ValueError('The combined test and validation sets do not cover all samples.')

    if len(holdout_ids) != len(set(holdout_ids)):
        raise ValueError('A sample appears in more than one held-out split across folds.')


def prepare_output_dir(output_dir, clean_output_dir):
    """Create the output directory and optionally remove old fold files."""
    if clean_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)


def sorted_for_output(sample_ids):
    """Sort output files if configured to do so."""
    return sorted(sample_ids, key=natural_key) if SORT_OUTPUT_FILES else sample_ids


def write_sample_list(path, sample_ids):
    """Write one sample ID per line."""
    with open(path, 'w', encoding='utf-8') as output_file:
        for sample_id in sorted_for_output(sample_ids):
            output_file.write(f'{sample_id}\n')


def write_fold_files(output_dir, folds):
    """Write train, test, and validation list files for every fold."""
    for fold in folds:
        fold_index = fold['fold']

        write_sample_list(output_dir / f'{TRAIN_PREFIX}_f{fold_index}.txt', fold['train'])
        write_sample_list(output_dir / f'{TEST_PREFIX}_f{fold_index}.txt', fold['test'])
        write_sample_list(output_dir / f'{VAL_PREFIX}_f{fold_index}.txt', fold['val'])


def write_summary_csv(output_dir, folds, total_count):
    """Write per-fold split counts and fractions."""
    summary_path = output_dir / 'fold_summary.csv'

    with open(summary_path, 'w', newline='', encoding='utf-8') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['fold', 'train_count', 'test_count', 'val_count', 'train_fraction', 'test_fraction', 'val_fraction'])

        for fold in folds:
            train_count = len(fold['train'])
            test_count = len(fold['test'])
            val_count = len(fold['val'])

            writer.writerow([
                fold['fold'],
                train_count,
                test_count,
                val_count,
                round(train_count / total_count, 4),
                round(test_count / total_count, 4),
                round(val_count / total_count, 4)
            ])


def write_membership_csv(output_dir, folds):
    """Write a long-format file showing each sample's split assignment."""
    membership_path = output_dir / 'fold_membership.csv'

    with open(membership_path, 'w', newline='', encoding='utf-8') as membership_file:
        writer = csv.writer(membership_file)
        writer.writerow(['fold', 'split', 'sample_id'])

        for fold in folds:
            for split_name in ('train', 'test', 'val'):
                for sample_id in sorted_for_output(fold[split_name]):
                    writer.writerow([fold['fold'], split_name, sample_id])


def print_summary(output_dir, folds, total_count):
    """Print split counts to the terminal."""
    print('======================================================================================')
    print(f'Fold list output directory: {output_dir}')
    print(f'Total samples: {total_count}')
    print(f'Seed: {SEED}')
    print('--------------------------------------------------------------------------------------')

    for fold in folds:
        train_count = len(fold['train'])
        test_count = len(fold['test'])
        val_count = len(fold['val'])

        print(
            f'Fold {fold["fold"]}: '
            f'train={train_count} ({train_count / total_count:.2%}), '
            f'test={test_count} ({test_count / total_count:.2%}), '
            f'val={val_count} ({val_count / total_count:.2%})'
        )

    print('======================================================================================')


def create_fold_lists(mark_list_path, output_dir, num_folds, seed):
    """Create deterministic fold-list files from one mark list."""
    sample_ids = load_unique_sample_ids(mark_list_path)

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    folds = create_folds(sample_ids, num_folds=num_folds)
    validate_folds(folds, sample_ids)

    prepare_output_dir(output_dir, clean_output_dir=CLEAN_OUTPUT_DIR)
    write_fold_files(output_dir, folds)

    if WRITE_SUMMARY_CSV:
        write_summary_csv(output_dir, folds, total_count=len(sample_ids))

    if WRITE_MEMBERSHIP_CSV:
        write_membership_csv(output_dir, folds)

    print_summary(output_dir, folds, total_count=len(sample_ids))


def main():
    """Run fold generation using the config block."""
    create_fold_lists(mark_list_path=MARK_LIST_PATH, output_dir=OUTPUT_DIR, num_folds=NUM_FOLDS, seed=SEED)


if __name__ == '__main__':
    main()