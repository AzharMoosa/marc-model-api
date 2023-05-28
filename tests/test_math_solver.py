import unittest
from app.main import MathSolver


class TestMathSolver(unittest.TestCase):
    def test_is_correct_true_extraction_lower(self):
        TEXT = "is_correct: true"
        self.assertTrue(MathSolver.extract_correct_boolean(TEXT))

    def test_is_correct_true_extraction_upper(self):
        TEXT = "is_correct: True"
        self.assertTrue(MathSolver.extract_correct_boolean(TEXT))

    def test_is_correct_true_extraction_mixed(self):
        TEXT = "is_correct: TrUe"
        self.assertTrue(MathSolver.extract_correct_boolean(TEXT))

    def test_is_correct_false_extraction_lower(self):
        TEXT = "is_correct: false"
        self.assertFalse(MathSolver.extract_correct_boolean(TEXT))

    def test_is_correct_false_extraction_upper(self):
        TEXT = "is_correct: False"
        self.assertFalse(MathSolver.extract_correct_boolean(TEXT))

    def test_is_correct_false_extraction_mixed(self):
        TEXT = "is_correct: FaLsE"
        self.assertFalse(MathSolver.extract_correct_boolean(TEXT))


if __name__ == "__main__":
    unittest.main()
