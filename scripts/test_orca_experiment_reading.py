"""Test comment-only source annotation without importing training dependencies."""

import ast
import unittest

from package_orca_experiment_reading import NOTES, ROOT, annotate, code_tokens


class ReadingPackageTest(unittest.TestCase):
    def test_all_real_sources_unchanged(self):
        for filename, (title, notes) in NOTES.items():
            with self.subTest(filename=filename):
                source = (ROOT / "scripts" / filename).read_text()
                result = annotate(source, filename, title, notes)
                self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(result)))
                self.assertEqual(code_tokens(source), code_tokens(result))
                self.assertEqual(result.count("# 导读："), len(notes))

    def test_shebang_preserved(self):
        source = "#!/usr/bin/env python3\nx = 1\n"
        result = annotate(source, "example.py", "Example", [("x =", "A comment")])
        self.assertTrue(result.startswith("#!/usr/bin/env python3\n"))
        self.assertEqual(code_tokens(source), code_tokens(result))

    def test_ambiguous_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            annotate("x = 1\nx = 2\n", "example.py", "Example", [("x =", "Comment")])

    def test_missing_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            annotate("x = 1\n", "example.py", "Example", [("y =", "Comment")])


if __name__ == "__main__":
    unittest.main()
