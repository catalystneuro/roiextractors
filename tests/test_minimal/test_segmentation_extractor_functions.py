import pytest
import numpy as np

from roiextractors.segmentationextractor import (
    convert_image_masks_to_pixel_masks,
    convert_pixel_masks_to_image_masks,
    get_default_roi_locations_from_image_masks,
)


@pytest.fixture(scope="module")
def rng():
    seed = 1728084845  # int(datetime.datetime.now().timestamp()) at the time of writing
    return np.random.default_rng(seed=seed)


@pytest.fixture(scope="function")
def image_masks(rng):
    return rng.random((3, 3, 3))


def test_convert_image_masks_to_pixel_masks(image_masks):
    pixel_masks = convert_image_masks_to_pixel_masks(image_masks=image_masks)
    for i, pixel_mask in enumerate(pixel_masks):
        assert pixel_mask.shape == (image_masks.shape[0] * image_masks.shape[1], 3)
        for row, column, wt in pixel_mask:
            assert row == int(row)
            assert column == int(column)
            assert image_masks[int(row), int(column), i] == wt


def test_convert_image_masks_to_pixel_masks_with_zeros(image_masks):
    image_masks[0, 0, 0] = 0
    pixel_masks = convert_image_masks_to_pixel_masks(image_masks=image_masks)
    assert pixel_masks[0].shape == (image_masks.shape[0] * image_masks.shape[1] - 1, 3)
    for i, pixel_mask in enumerate(pixel_masks):
        for row, column, wt in pixel_mask:
            assert row == int(row)
            assert column == int(column)
            assert image_masks[int(row), int(column), i] == wt


def test_convert_image_masks_to_pixel_masks_all_zeros(image_masks):
    image_masks = np.zeros(image_masks.shape)
    pixel_masks = convert_image_masks_to_pixel_masks(image_masks=image_masks)
    for pixel_mask in pixel_masks:
        assert pixel_mask.shape == (0, 3)


def test_convert_pixel_masks_to_image_masks(image_masks):
    pixel_masks = []
    for i in range(image_masks.shape[2]):
        image_mask = image_masks[:, :, i]
        locs = np.where(image_mask > 0)
        pix_values = image_mask[image_mask > 0]
        pixel_masks.append(np.vstack((locs[0], locs[1], pix_values)).T)

    image_masks = convert_pixel_masks_to_image_masks(pixel_masks=pixel_masks, image_shape=image_masks.shape[:2])
    for i in range(image_masks.shape[2]):
        image_mask = image_masks[:, :, i]
        indices = np.ndindex(image_mask.shape)
        for row, column in indices:
            pixel_mask_mask = np.logical_and(pixel_masks[i][:, 0] == row, pixel_masks[i][:, 1] == column)
            assert image_mask[row, column] == pixel_masks[i][pixel_mask_mask, 2]


def test_convert_pixel_masks_to_image_masks_with_zeros(image_masks):
    pixel_masks = []
    for i in range(image_masks.shape[2]):
        image_mask = image_masks[:, :, i]
        locs = np.where(image_mask > 0)
        pix_values = image_mask[image_mask > 0]
        pixel_masks.append(np.vstack((locs[0], locs[1], pix_values)).T)

    pixel_masks[0] = pixel_masks[0][1:]
    image_masks = convert_pixel_masks_to_image_masks(pixel_masks=pixel_masks, image_shape=image_masks.shape[:2])
    for i in range(image_masks.shape[2]):
        image_mask = image_masks[:, :, i]
        indices = np.ndindex(image_mask.shape)
        for row, column in indices:
            pixel_mask_mask = np.logical_and(pixel_masks[i][:, 0] == row, pixel_masks[i][:, 1] == column)
            if i == 0 and row == 0 and column == 0:
                assert np.all(np.logical_not(pixel_mask_mask))
            else:
                assert image_mask[row, column] == pixel_masks[i][pixel_mask_mask, 2]


def test_convert_pixel_masks_to_image_masks_all_zeros(image_masks):
    pixel_masks = [np.zeros((0, 0)) for _ in range(image_masks.shape[2])]
    output_image_masks = convert_pixel_masks_to_image_masks(pixel_masks=pixel_masks, image_shape=image_masks.shape[:2])
    assert output_image_masks.shape == image_masks.shape
    for image_mask in output_image_masks:
        assert np.all(image_mask == 0)


def test_convert_masks_roundtrip(image_masks):
    pixel_masks = convert_image_masks_to_pixel_masks(image_masks=image_masks)
    output_image_masks = convert_pixel_masks_to_image_masks(pixel_masks=pixel_masks, image_shape=image_masks.shape[:2])
    np.testing.assert_array_equal(image_masks, output_image_masks)


def test_get_default_roi_locations_from_image_masks():
    image_masks = np.zeros((3, 3, 3))
    image_masks[0, 0, 0] = 1
    image_masks[1, 1, 1] = 1
    image_masks[2, 2, 2] = 1
    roi_locations = get_default_roi_locations_from_image_masks(image_masks=image_masks)
    expected_roi_locations = np.array([[0, 0], [1, 1], [2, 2]]).T
    np.testing.assert_array_equal(roi_locations, expected_roi_locations)


def test_get_default_roi_locations_from_image_masks_tie1():
    image_masks = np.zeros((3, 3, 3))
    image_masks[0, 0, 0] = 1
    image_masks[0, 1, 0] = 1
    image_masks[1, 1, 1] = 1
    image_masks[2, 2, 2] = 1
    roi_locations = get_default_roi_locations_from_image_masks(image_masks=image_masks)
    expected_roi_locations = np.array([[0, 0], [1, 1], [2, 2]]).T
    np.testing.assert_array_equal(roi_locations, expected_roi_locations)


def test_get_default_roi_locations_from_image_masks_tie2():
    image_masks = np.zeros((3, 3, 3))
    image_masks[0, 0, 0] = 1
    image_masks[0, 1, 0] = 1
    image_masks[1, 1, 0] = 1
    image_masks[1, 1, 1] = 1
    image_masks[2, 2, 2] = 1
    roi_locations = get_default_roi_locations_from_image_masks(image_masks=image_masks)
    expected_roi_locations = np.array([[0, 1], [1, 1], [2, 2]]).T
    np.testing.assert_array_equal(roi_locations, expected_roi_locations)
