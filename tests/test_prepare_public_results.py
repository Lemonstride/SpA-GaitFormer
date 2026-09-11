from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_public_results.py"
SPEC = importlib.util.spec_from_file_location("prepare_public_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicResultsExportTest(unittest.TestCase):
    def test_rewrite_text_removes_private_ids_and_private_path(self) -> None:
        source = "/private/storage/data/S01/walk,S11\n"
        result = MODULE.rewrite_text(
            source,
            {"S01": "P-AAA11111", "S11": "P-BBB22222"},
            ["/private/storage"],
        )

        self.assertEqual(result, "${PRIVATE_ROOT_1}/data/P-AAA11111/walk,P-BBB22222\n")
        MODULE.assert_public_text(Path("manifest.csv"), result)

    def test_subject_pattern_does_not_rewrite_longer_tokens(self) -> None:
        source = "S01 S010 XS01 S01X"
        result = MODULE.rewrite_text(source, {"S01": "P-AAA11111"})

        self.assertEqual(result, "P-AAA11111 S010 XS01 S01X")


if __name__ == "__main__":
    unittest.main()
