import numpy as np
import pytest

from cvt.data.riders import merge_riders

RIDER, MOTOR, BIKE, PED = 2, 10, 3, 1


def test_rider_inside_motor_is_folded_into_union():
    boxes = np.array([[10, 10, 30, 24], [14, 8, 22, 18]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, RIDER]), RIDER, (BIKE, MOTOR))
    np.testing.assert_array_equal(keep, [True, False])
    np.testing.assert_array_equal(labels, [MOTOR])
    np.testing.assert_allclose(out, [[10, 8, 30, 24]])


def test_rider_mostly_outside_is_kept():
    boxes = np.array([[0, 0, 10, 10], [7, 0, 17, 10]], np.float32)  # 30% overlap
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, RIDER]), RIDER, (BIKE, MOTOR))
    assert keep.all() and len(out) == 2


def test_two_riders_on_one_motor_both_merge():
    boxes = np.array([[0, 0, 20, 20], [2, 2, 8, 8], [10, 10, 18, 18]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, RIDER, RIDER]), RIDER, (MOTOR,))
    np.testing.assert_array_equal(keep, [True, False, False])
    np.testing.assert_allclose(out, [[0, 0, 20, 20]])


def test_rider_goes_to_vehicle_covering_most_of_it():
    boxes = np.array([[0, 0, 10, 10], [6, 0, 30, 10], [5, 0, 15, 10]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([BIKE, MOTOR, RIDER]), RIDER, (BIKE, MOTOR))
    np.testing.assert_allclose(out, [[0, 0, 10, 10], [5, 0, 30, 10]])
    np.testing.assert_array_equal(keep, [True, True, False])
    np.testing.assert_array_equal(labels, [BIKE, MOTOR])


def test_pedestrians_and_empty_inputs_untouched():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 5, 5]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, PED]), RIDER, (MOTOR,))
    assert keep.all()
    out, labels, keep, _ = merge_riders(np.zeros((0, 4)), np.zeros(0, int), RIDER, (MOTOR,))
    assert out.shape == (0, 4) and keep.shape == (0,)


CAR = 4


def test_rider_inside_non_rideable_vehicle_is_kept():
    boxes = np.array([[0, 0, 20, 20], [2, 2, 8, 8]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([CAR, RIDER]), RIDER, (BIKE, MOTOR))
    assert keep.all()


def test_scores_low_confidence_vehicle_cannot_absorb_rider():
    boxes = np.array([[10, 10, 30, 24], [14, 8, 22, 18]], np.float32)
    out, labels, keep, scores = merge_riders(
        boxes, np.array([MOTOR, RIDER]), RIDER, (MOTOR,),
        scores=np.array([0.05, 0.8]), min_vehicle_score=0.25,
    )
    assert keep.all()
    np.testing.assert_allclose(scores, [0.05, 0.8])


def test_scores_merged_vehicle_keeps_max_and_keep_filters_parallel_arrays():
    boxes = np.array([[10, 10, 30, 24], [14, 8, 22, 18], [50, 50, 60, 60]], np.float32)
    raw_scores = np.array([0.4, 0.9, 0.3])
    out, labels, keep, scores = merge_riders(
        boxes, np.array([MOTOR, RIDER, PED]), RIDER, (MOTOR,),
        scores=raw_scores, min_vehicle_score=0.25,
    )
    np.testing.assert_array_equal(keep, [True, False, True])
    np.testing.assert_allclose(scores, [0.9, 0.3])
    np.testing.assert_allclose(raw_scores[keep], [0.4, 0.3])  # caller-side filtering also works


def test_without_scores_returns_none():
    *_, scores = merge_riders(np.zeros((0, 4)), np.zeros(0, int), RIDER, (MOTOR,))
    assert scores is None


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="boxes vs"):
        merge_riders(np.zeros((2, 4)), np.array([MOTOR]), RIDER, (MOTOR,))
    with pytest.raises(ValueError, match="must not also be"):
        merge_riders(np.zeros((1, 4)), np.array([MOTOR]), RIDER, (RIDER, MOTOR))


def test_band_rider_with_feet_on_vehicle_is_merged():
    # 40% inside a small bike box, feet (bottom-centre 15, 20) on the bike: an upright rider
    boxes = np.array([[10, 12, 20, 22], [10, 0, 20, 20]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, RIDER]), RIDER, (MOTOR,))
    np.testing.assert_array_equal(keep, [True, False])
    np.testing.assert_allclose(out, [[10, 0, 20, 22]])


def test_band_person_with_feet_off_vehicle_is_kept():
    # 30% inside, but feet (bottom-centre 22, 10) are beside the bike: a bystander
    boxes = np.array([[0, 0, 20, 20], [17, 0, 27, 10]], np.float32)
    out, labels, keep, _ = merge_riders(boxes, np.array([MOTOR, RIDER]), RIDER, (MOTOR,))
    assert keep.all()
