import unittest

from business_entity_resolution.normalization import (
    address_views,
    name_views,
    normalize_text,
    numeric_address_views,
)


class NormalizationTests(unittest.TestCase):
    def test_name_views_preserve_full_and_core(self):
        views = name_views("A&B Technologies Pvt. Ltd.")
        self.assertEqual(views["name_norm_full"], "a and b technologies pvt ltd")
        self.assertEqual(views["name_core_ordered"], "a b technologies")
        self.assertEqual(views["name_core_sorted"], "a|b|technologies")

    def test_devanagari_combining_marks_remain_in_words(self):
        self.assertEqual(
            normalize_text("राम मार्केटिंग प्राइवेट लिमिटेड"),
            "राम मार्केटिंग प्राइवेट लिमिटेड",
        )

    def test_numeric_signature_and_explicit_india_pin(self):
        views = numeric_address_views(
            "Flat 4B, Plot 27, 3rd Floor, Sector 18, Road 5, PIN 560001",
            "India",
        )
        self.assertEqual(views.tokens_ordered, ("4B", "27", "3", "18", "5"))
        self.assertEqual(views.signature_ordered, "4B|27|3|18|5")
        self.assertEqual(views.signature_sorted, "18|27|3|4B|5")
        self.assertEqual(views.postal_code.value, "560001")
        self.assertTrue(views.postal_code.explicit)

    def test_compound_identifiers_and_unicode_digits(self):
        views = numeric_address_views("Plot B-७८/१, Unit G-3/571, 5A1", "India")
        self.assertEqual(views.tokens_ordered, ("B78/1", "G3/571", "5A1"))

    def test_explicit_survey_number_is_not_postal(self):
        views = numeric_address_views("Survey No 560001, Village Example", "India")
        self.assertIsNone(views.postal_code)
        self.assertIn("560001", views.tokens_ordered)

    def test_trailing_survey_number_is_not_postal(self):
        views = numeric_address_views("Village Example, Survey 560001", "India")
        self.assertIsNone(views.postal_code)
        self.assertIn("560001", views.tokens_ordered)

    def test_literal_null_removed_from_normalized_address(self):
        views = address_views("12 Main Road, null, Jaipur", "India")
        self.assertNotIn("null", views["address_norm_full"].split())
        self.assertNotIn("null", views["address_tokens_useful"])


if __name__ == "__main__":
    unittest.main()
