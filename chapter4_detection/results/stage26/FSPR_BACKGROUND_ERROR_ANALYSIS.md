# FSPR background error analysis

The Stage25 any-class-IoU<0.10 definition is used uniformly.

- E0: {'background_style_prediction': 6, 'stage24_legacy_sameclass_iou_lt010': 10, 'duplicate_prediction': 4, 'same_class_localization_error': 17, 'class_confusion': 4}.
- F0: {'background_style_prediction': 21252, 'stage24_legacy_sameclass_iou_lt010': 21377, 'class_confusion': 74, 'duplicate_prediction': 187, 'same_class_localization_error': 424, 'unconfirmed_overlap_candidate': 51}.
- F1: {'class_confusion': 47, 'stage24_legacy_sameclass_iou_lt010': 23344, 'unconfirmed_overlap_candidate': 103, 'background_style_prediction': 23194, 'duplicate_prediction': 235, 'same_class_localization_error': 600}.
- F2: {'background_style_prediction': 22344, 'stage24_legacy_sameclass_iou_lt010': 22508, 'class_confusion': 44, 'unconfirmed_overlap_candidate': 120, 'same_class_localization_error': 573, 'duplicate_prediction': 243}.
