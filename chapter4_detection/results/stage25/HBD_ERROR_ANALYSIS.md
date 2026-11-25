# HBD unified error analysis

One fixed post-NMS attribution is used for all seven models. It cannot prove the exact NMS suppressor and therefore does not reuse Stage12 A/B/C labels. Priority is one-to-one correct match, duplicate, class confusion, same-class localization error, background style, then unconfirmed overlap.

- E0: {'background_style_prediction': 6, 'stage24_legacy_sameclass_iou_lt010': 10, 'duplicate_prediction': 4, 'same_class_localization_error': 17, 'class_confusion': 4}
- M2: {'same_class_localization_error': 16, 'duplicate_prediction': 25, 'background_style_prediction': 24, 'stage24_legacy_sameclass_iou_lt010': 28, 'class_confusion': 4}
- M3: {'same_class_localization_error': 16, 'duplicate_prediction': 24, 'background_style_prediction': 30, 'stage24_legacy_sameclass_iou_lt010': 34, 'class_confusion': 4}
- H1: {'duplicate_prediction': 28, 'same_class_localization_error': 14, 'background_style_prediction': 60, 'stage24_legacy_sameclass_iou_lt010': 62, 'class_confusion': 2}
- H2: {'duplicate_prediction': 25, 'same_class_localization_error': 17, 'background_style_prediction': 39, 'stage24_legacy_sameclass_iou_lt010': 42, 'class_confusion': 3}
- H3: {'duplicate_prediction': 24, 'same_class_localization_error': 17, 'background_style_prediction': 35, 'stage24_legacy_sameclass_iou_lt010': 40, 'class_confusion': 5}
- H4: {'same_class_localization_error': 20, 'duplicate_prediction': 29, 'background_style_prediction': 80, 'stage24_legacy_sameclass_iou_lt010': 84, 'class_confusion': 4}
