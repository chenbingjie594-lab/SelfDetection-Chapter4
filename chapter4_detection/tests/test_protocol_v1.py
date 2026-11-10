import json
import unittest
from pathlib import Path


AUDIT = Path("/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/data_audit.json")


class ProtocolV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not AUDIT.exists():
            raise unittest.SkipTest("frozen chapter4_real_only_v1 is not available")
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_gate_and_real_only_counts(self):
        self.assertEqual(self.audit["status"], "PASS")
        self.assertEqual(self.audit["synthetic_images"], 0)
        self.assertEqual(sum(v["images"] for v in self.audit["summary"].values()), 184)
        self.assertFalse(self.audit["unknown_provenance"])

    def test_source_groups_do_not_cross_splits(self):
        groups = {}
        for row in self.audit["records"]:
            groups.setdefault(row["source_group"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertFalse(self.audit["cross_split_source_groups"])

    def test_known_siblings_are_together(self):
        by_name = {row["image_name"]: row for row in self.audit["records"]}
        for stem in ("img_1020_pre", "img_9509_pre"):
            left, right = by_name[f"{stem}_part_0.jpg"], by_name[f"{stem}_part_1.jpg"]
            self.assertEqual(left["source_group"], right["source_group"])
            self.assertEqual(left["split"], right["split"])


if __name__ == "__main__":
    unittest.main()
