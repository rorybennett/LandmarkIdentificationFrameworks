"""Regression tests for exact annotation landmark-count validation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / 'Heatmaps' / 'utils' / 'annotation_utils.py'
SPEC = importlib.util.spec_from_file_location('heatmap_annotation_utils', MODULE_PATH)
annotation_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotation_utils)


class AnnotationValidationTests(unittest.TestCase):
    def read_record(self, row):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        mark_list = Path(temporary_dir.name) / 'annotations.txt'
        mark_list.write_text(f'{row}\n', encoding='utf-8')
        records = annotation_utils.read_mark_list(mark_list)
        return mark_list, records['patient_7']

    def test_exact_landmark_count_is_accepted(self):
        mark_list, record = self.read_record('patient_7.png (1, 2) (3, 4)')
        annotation_utils.validate_annotation_point_count(record, expected_points=2, sample_name='patient_7', mark_list_file=mark_list,
                                                         repetition=2, fold=3, split_name='validation', training_context=True)

    def test_missing_landmark_message_contains_full_context(self):
        mark_list, record = self.read_record('patient_7.png (1, 2)')

        with self.assertRaises(ValueError) as context:
            annotation_utils.validate_annotation_point_count(record, expected_points=2, sample_name='patient_7', mark_list_file=mark_list,
                                                             repetition=2, fold=3, split_name='validation', training_context=True)

        message = str(context.exception)
        self.assertIn('repetition 2, fold 3, validation split', message)
        self.assertIn("patient/sample 'patient_7'", message)
        self.assertIn(str(mark_list), message)
        self.assertIn('(line 1)', message)
        self.assertIn('(1 missing)', message)
        self.assertIn('Training cancelled; existing outputs were not removed.', message)

    def test_extra_landmark_message_contains_full_context(self):
        mark_list, record = self.read_record('patient_7.png (1, 2) (3, 4) (5, 6)')

        with self.assertRaises(ValueError) as context:
            annotation_utils.validate_annotation_point_count(record, expected_points=2, sample_name='patient_7', mark_list_file=mark_list,
                                                             repetition=1, fold=4, split_name='training', training_context=True)

        message = str(context.exception)
        self.assertIn('repetition 1, fold 4, training split', message)
        self.assertIn("patient/sample 'patient_7'", message)
        self.assertIn(str(mark_list), message)
        self.assertIn('(line 1)', message)
        self.assertIn('(1 extra)', message)
        self.assertIn('Training cancelled; existing outputs were not removed.', message)

    def test_malformed_or_unmatched_point_is_counted_as_missing(self):
        mark_list, record = self.read_record('patient_7.png (1, 2) malformed')

        with self.assertRaisesRegex(ValueError, r'\(1 missing\)'):
            annotation_utils.validate_annotation_point_count(record, expected_points=2, sample_name='patient_7', mark_list_file=mark_list)

    def test_fold_all_is_included_in_error_context(self):
        mark_list, record = self.read_record('patient_7.png (1, 2)')

        with self.assertRaisesRegex(ValueError, r'repetition 1, fold all, validation split'):
            annotation_utils.validate_annotation_point_count(record, expected_points=2, sample_name='patient_7', mark_list_file=mark_list,
                                                             repetition=1, fold='all', split_name='validation', training_context=True)


if __name__ == '__main__':
    unittest.main()
