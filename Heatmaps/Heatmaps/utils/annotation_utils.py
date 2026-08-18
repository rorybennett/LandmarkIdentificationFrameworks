"""Annotation parsing and exact landmark-count validation helpers."""

import re
from pathlib import Path


POINT_PATTERN = re.compile(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)')


def read_mark_list(mark_list_file):
    """Read every mark-list row without silently changing its landmark count."""
    mark_list_file = Path(mark_list_file)

    if not mark_list_file.is_file():
        raise FileNotFoundError(f'Mark-list file not found: {mark_list_file}')

    records = {}

    with open(mark_list_file, 'r', encoding='utf-8') as mark_handle:
        for line_number, line in enumerate(mark_handle, start=1):
            line = line.strip()

            if not line:
                continue

            image_name = line.split()[0]
            points = [(float(x), float(y)) for x, y in POINT_PATTERN.findall(line)]
            sample_stem = Path(image_name).stem

            if sample_stem in records:
                raise ValueError(f'Duplicate patient/sample {sample_stem!r} in annotation file {mark_list_file} at line {line_number}.')

            records[sample_stem] = {'image_name': image_name, 'points': points, 'line_number': line_number}

    if not records:
        raise ValueError(f'No valid mark-list rows found in {mark_list_file}')

    return records


def validate_annotation_point_count(mark_record, expected_points, sample_name, mark_list_file, repetition=None, fold=None, split_name=None,
                                    training_context=False):
    """Require an exact landmark count and report complete dataset context."""
    actual_points = len(mark_record['points'])
    expected_points = int(expected_points)

    if actual_points == expected_points:
        return

    difference = abs(actual_points - expected_points)
    difference_label = 'missing' if actual_points < expected_points else 'extra'
    location_parts = []

    if repetition is not None:
        location_parts.append(f'repetition {int(repetition)}')

    if fold is not None:
        location_parts.append(f'fold {int(fold)}')

    if split_name is not None:
        location_parts.append(f'{split_name} split')

    location = f' for {", ".join(location_parts)}' if location_parts else ''
    message = (f"Dataset validation failed{location}: patient/sample {str(sample_name)!r} in annotation file '{Path(mark_list_file)}' "
               f'(line {mark_record["line_number"]}) has {actual_points} landmark point(s); exactly {expected_points} are required '
               f'({difference} {difference_label}).')

    if training_context:
        message += ' Training cancelled; existing outputs were not removed.'

    raise ValueError(message)


def resolve_mark_record(sample_name, mark_records):
    """Match a fold-list sample name to a mark-list record."""
    sample_stem = Path(str(sample_name)).stem

    if sample_stem in mark_records:
        return sample_stem, mark_records[sample_stem]

    raise KeyError(f'Sample {sample_name} was not found in the mark list.')
