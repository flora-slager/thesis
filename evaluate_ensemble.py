import numpy as np
from keras_retinanet.utils.compute_overlap import compute_overlap
from keras_retinanet.utils.eval import _get_detections, _get_annotations, _compute_ap

# Combine predictions of two models:

def combine_all_detections(detections, iou_threshold):
    """
    Input: m by n_i by 5, where m is the number of models and n_i is the number
    of detections from model i

    Output: n by 5
    """
    if len(detections) == 1:
        return detections[0]
    else:
        resulting_detections = []
        for reference_detection in detections[0]:
            if len(detections[1]) == 0:
                resulting_detections.append(reference_detection)
                continue
            overlaps = compute_overlap(np.expand_dims(reference_detection, axis=0),
                                       detections[1])
            if len(overlaps) == 0:
                resulting_detections.append(reference_detection)
                continue
            best_index = np.argmax(overlaps, axis=1)
            # print("Assigned annotation:")
            # print(assigned_annotation)
            max_overlap = overlaps[0, best_index][0]
            if max_overlap > 0 or max_overlap >= iou_threshold:
                combined_box = _combine_bounding_boxes(reference_detection,
                                                       detections[1][best_index][0])
                resulting_detections.append(combined_box)
        return resulting_detections

def combine_detections(all_detections, image_number, label, iou_threshold):
    """
    all_detections = [
        detections_from_model
        for model in all_models
    ]
    :param all_detections:
    :param image_number:
    :param label:
    :return:
    """
    if len(all_detections) == 1:
        return all_detections[0][0]

    resulting_detections = []
    for detection in all_detections[0][image_number][label]:
        overlaps = compute_overlap(np.expand_dims(detection, axis=0), all_detections[1][image_number][label])
        assigned_annotation = np.argmax(overlaps, axis=1)
        # print("Assigned annotation:")
        # print(assigned_annotation)
        max_overlap = overlaps[0, assigned_annotation][0]
        # print("Max overlap:")
        # print(max_overlap)
        if max_overlap > 0 or max_overlap >= iou_threshold:
            combined_box = _combine_bounding_boxes(detection, all_detections[1][image_number][label][assigned_annotation[0]])
            resulting_detections.append(combined_box)

    return resulting_detections

def _combine_bounding_boxes(detection, best_overlap_detection):
    result = np.mean([detection, best_overlap_detection], axis=0)
    return result

def evaluate_dual_memory_model(
        generator,
        models,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    detections_per_model = []
    all_inferences = []

    for model in models:
        current_detections, current_inferences = _get_detections(generator, model, score_threshold=score_threshold,
                                                                 max_detections=max_detections, save_path=save_path)
        detections_per_model.append(current_detections)
        all_inferences.append(current_inferences)
    average_precisions = {}
    all_annotations = _get_annotations(generator)

    # all_detections = detections_per_model[0]

    # process detections and annotations
    final_detections = {}
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = [all_detections[i][label] for all_detections in detections_per_model]
            detections = combine_all_detections(detections, iou_threshold)
            key = f"label={label}, instance_index={i}"
            final_detections[key] = [[d.tolist() for d in detection]
                                            for detection in detections]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # Sort by score:
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # Compute false positives and true positives:
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # Compute recall and precision:
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # Compute average precision:
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    # Inference time:
    inference_time = np.sum(all_inferences) / generator.size()

    return average_precisions, inference_time, detections_per_model, final_detections