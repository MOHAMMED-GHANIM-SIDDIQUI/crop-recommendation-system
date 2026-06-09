import pytest

from crop_recommender.predictor import crop_name_from_label, validate_feature_values


def test_crop_name_from_label_maps_known_crop():
    assert crop_name_from_label(1) == "Rice"
    assert crop_name_from_label(22) == "Coffee"


def test_crop_name_from_label_handles_unknown_label():
    assert crop_name_from_label(999) == "Unknown crop"


def test_validate_feature_values_requires_seven_features():
    with pytest.raises(ValueError, match="7"):
        validate_feature_values([1, 2, 3])


def test_validate_feature_values_requires_numeric_values():
    with pytest.raises(TypeError, match="numeric"):
        validate_feature_values([1, 2, 3, 4, 5, 6, "bad"])
