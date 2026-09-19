import unittest

import numpy as np

from src.experiments.multibench.mosei_alignment_audit import audit_id_sets, processed_id


class MoseiAlignmentAuditTest(unittest.TestCase):
    def test_processed_id_preserves_timestamp_identity(self):
        self.assertEqual(processed_id(np.asarray(["video", "1.0", "2.0"])), "video|1.0|2.0")

    def test_audit_detects_duplicates_and_unmatched_ids(self):
        processed = {"train": {"id": np.asarray([["v", "0", "1"], ["v", "0", "1"]])}, "valid": {"id": np.empty((0, 3), dtype="U1")}, "test": {"id": np.empty((0, 3), dtype="U1")}}
        official = {"train": {"id": ["v[0]", "v[1]"]}, "valid": {"id": []}, "test": {"id": []}}
        result = audit_id_sets(processed, official)
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["duplicate_ids"]["processed"], ["v|0|1"])
        self.assertEqual(result["by_split"]["train"]["official_count"], 2)


if __name__ == "__main__":
    unittest.main()
