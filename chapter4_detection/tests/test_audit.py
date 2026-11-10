import unittest

from chapter4_detection.scripts.audit_data import canonical_image_id, parse_labels


class AuditTests(unittest.TestCase):
    def test_known_siblings_share_canonical_id(self):
        self.assertEqual(canonical_image_id("img_1020_pre_part_0.jpg"), canonical_image_id("img_1020_pre_part_1.jpg"))


if __name__ == "__main__":
    unittest.main()
