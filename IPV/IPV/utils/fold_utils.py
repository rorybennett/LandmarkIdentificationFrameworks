"""Repeated k-fold discovery, validation, and fingerprint helpers for IPV."""

import hashlib
import re
from pathlib import Path


REPETITION_DIR_PATTERN = re.compile(r'^repetition_(\d+)$')
TRAINING_LIST_PATTERN = re.compile(r'^training_f(\d+)\.txt$')
VALIDATION_LIST_PATTERN = re.compile(r'^val_f(\d+)\.txt$')
SPLIT_FILE_PREFIXES = {'training': 'training', 'validation': 'val'}


def natural_key(value):
    """Create a natural sorting key, so A2 comes before A10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(value))]


def canonical_sample_name(value):
    """Return the canonical stem used to compare split memberships."""
    return Path(str(value).split()[0]).stem


def get_repetition_dir(fold_lists_path, repetition):
    """Return the directory containing one repetition's fold lists."""
    return Path(fold_lists_path) / f'repetition_{int(repetition)}'


def discover_repetition_numbers(fold_lists_path):
    """Return contiguous repetition numbers with identical fold collections."""
    fold_lists_path = Path(fold_lists_path)

    if not fold_lists_path.is_dir():
        raise ValueError(f'fold_lists_path does not exist or is not a directory: {fold_lists_path}')

    repetitions = sorted({int(match.group(1)) for path in fold_lists_path.iterdir()
                          if path.is_dir() and (match := REPETITION_DIR_PATTERN.fullmatch(path.name))})

    if not repetitions:
        raise ValueError(f'No repetition_N directories found in {fold_lists_path}')

    expected = list(range(1, repetitions[-1] + 1))

    if repetitions != expected:
        raise ValueError(f'Repetition directories must be contiguous from repetition_1. Found {repetitions}, expected {expected}')

    fold_sets = {repetition: discover_fold_numbers(fold_lists_path, repetition) for repetition in repetitions}
    reference = fold_sets[repetitions[0]]

    if any(folds != reference for folds in fold_sets.values()):
        raise ValueError(f'Every repetition must contain identical contiguous fold numbers. Found: {fold_sets}')

    return repetitions


def discover_fold_numbers(fold_lists_path, repetition):
    """Return contiguous fold numbers and require training/validation pairs."""
    repetition_dir = get_repetition_dir(fold_lists_path, repetition)

    if not repetition_dir.is_dir():
        raise ValueError(f'Repetition directory does not exist: {repetition_dir}')

    training_folds = []
    validation_folds = []

    for path in repetition_dir.iterdir():
        if not path.is_file():
            continue

        training_match = TRAINING_LIST_PATTERN.fullmatch(path.name)
        validation_match = VALIDATION_LIST_PATTERN.fullmatch(path.name)

        if training_match:
            training_folds.append(int(training_match.group(1)))

        if validation_match:
            validation_folds.append(int(validation_match.group(1)))

    training_folds = sorted(set(training_folds))
    validation_folds = sorted(set(validation_folds))

    if not training_folds:
        raise ValueError(f'No training_fN.txt files found in {repetition_dir}')

    expected = list(range(1, training_folds[-1] + 1))

    if training_folds != expected:
        raise ValueError(f'Fold files in {repetition_dir} must be contiguous from training_f1.txt. Found {training_folds}, expected {expected}')

    if validation_folds != training_folds:
        raise ValueError(f'Training and validation fold numbers in {repetition_dir} must match exactly. '
                         f'Found training={training_folds}, validation={validation_folds}.')

    return training_folds


def get_split_file_path(fold_lists_path, repetition, split_name, fold):
    """Return one repetition/fold split-list path."""
    split_name = str(split_name).lower()

    if split_name not in SPLIT_FILE_PREFIXES:
        raise ValueError(f'Unknown split name: {split_name}. Expected one of {tuple(SPLIT_FILE_PREFIXES)}.')

    return get_repetition_dir(fold_lists_path, repetition) / f'{SPLIT_FILE_PREFIXES[split_name]}_f{int(fold)}.txt'


def read_split_names(fold_lists_path, repetition, split_name, fold):
    """Read canonical sample names from one split list."""
    split_path = get_split_file_path(fold_lists_path, repetition, split_name, fold)

    if not split_path.is_file():
        raise FileNotFoundError(f'Split file not found: {split_path}')

    names = [canonical_sample_name(line.strip()) for line in split_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    if not names:
        raise ValueError(f'Split file is empty: {split_path}')

    return names


def validate_split_duplicates(split_name, names, repetition, fold):
    """Reject duplicate canonical sample names in one split."""
    seen = set()
    duplicates = set()

    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    if duplicates:
        duplicate_text = ', '.join(sorted(duplicates, key=natural_key))
        raise ValueError(f'Repetition {repetition}, fold {fold} {split_name} split contains duplicate sample ID(s): {duplicate_text}')


def validate_fold_split(fold_lists_path, repetition, fold):
    """Validate one training/validation pair and return its membership sets."""
    split_names = {split: read_split_names(fold_lists_path, repetition, split, fold) for split in SPLIT_FILE_PREFIXES}
    split_sets = {}

    for split, names in split_names.items():
        validate_split_duplicates(split, names, repetition, fold)
        split_sets[split] = set(names)

    overlap = split_sets['training'] & split_sets['validation']

    if overlap:
        overlap_text = ', '.join(sorted(overlap, key=natural_key))
        raise ValueError(f'Repetition {repetition}, fold {fold} has training/validation overlap: {overlap_text}')

    return split_sets


def validate_repeated_kfold_lists(fold_lists_path):
    """Validate the complete repeated k-fold collection."""
    repetitions = discover_repetition_numbers(fold_lists_path)
    folds = discover_fold_numbers(fold_lists_path, repetitions[0])
    reference_samples = None

    for repetition in repetitions:
        repetition_samples = None
        validation_occurrences = []

        for fold in folds:
            split_sets = validate_fold_split(fold_lists_path, repetition, fold)
            fold_samples = split_sets['training'] | split_sets['validation']

            if repetition_samples is None:
                repetition_samples = fold_samples
            elif fold_samples != repetition_samples:
                raise ValueError(f'Repetition {repetition}, fold {fold} does not cover the same dataset as the other folds.')

            if split_sets['training'] != fold_samples - split_sets['validation']:
                raise ValueError(f'Repetition {repetition}, fold {fold} training split is not the complement of validation.')

            validation_occurrences.extend(split_sets['validation'])

        if len(validation_occurrences) != len(set(validation_occurrences)):
            raise ValueError(f'Repetition {repetition} uses at least one sample as validation more than once.')

        if set(validation_occurrences) != repetition_samples:
            raise ValueError(f'Repetition {repetition} does not use every sample as validation exactly once.')

        if reference_samples is None:
            reference_samples = repetition_samples
        elif repetition_samples != reference_samples:
            raise ValueError(f'Repetition {repetition} does not contain the same dataset as repetition {repetitions[0]}.')

    return {'repetitions': repetitions, 'folds': folds, 'sample_count': len(reference_samples)}


def calculate_fold_collection_sha256(fold_lists_path, repetition_numbers=None, fold_numbers=None):
    """Hash every active split-list path and byte sequence."""
    fold_lists_path = Path(fold_lists_path)
    repetition_numbers = repetition_numbers or discover_repetition_numbers(fold_lists_path)
    fold_numbers = fold_numbers or discover_fold_numbers(fold_lists_path, repetition_numbers[0])
    digest = hashlib.sha256()

    for repetition in repetition_numbers:
        for fold in fold_numbers:
            for split_name in ('training', 'validation'):
                split_path = get_split_file_path(fold_lists_path, repetition, split_name, fold)
                digest.update(split_path.relative_to(fold_lists_path).as_posix().encode('utf-8'))
                digest.update(b'\0')
                digest.update(split_path.read_bytes())
                digest.update(b'\0')

    return digest.hexdigest()
