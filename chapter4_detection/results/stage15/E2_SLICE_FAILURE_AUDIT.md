# Historical E2 slice failure audit

## Geometry and labels

The historical protocol used 1024×1024 native-scale tiles, overlap 128, five tiles per 4096×1024 image, and `min_visibility=0.5`. Training labels were clipped to tile bounds; validation restored tile predictions to original coordinates and applied class-aware full-image NMS at IoU 0.7.

| Audit item | Result |
|---|---:|
| Training images / tiles | 138 / 690 |
| GT | 174 |
| GT fully contained in at least one tile | 174 (100%) |
| GT fully contained in multiple tiles | 59 |
| GT intersecting any tile boundary | 7 |
| GT with a retained partial label | 3 |
| GT with a dropped partial intersection | 4 |
| Labels written across tiles | 236 |
| Empty/background tile ratio | 71.45% |

The source manifest carries `canonical_image_id` and `source_group` into every tile. All tiles of a source remain in its training split; the untouched historical validation split is symlinked, so no new source-group crossing was found.

## Confirmed protocol differences from E0

- E2 presents 690 tiles per epoch instead of 138 full images, with 71.45% empty tiles and duplicated supervision for 59 GTs.
- Native tiles remove most full-width context and change Mosaic geometry.
- E2 completed 150 epochs, used fixed `last.pt`, disabled training validation, and recorded 1,723 optimizer-update attempts.
- Historical E0 stopped at epoch 76, selected validation `best.pt`, and recorded 184 optimizer-update attempts.
- E2 used imgsz1024; E0 used square imgsz1536.

Therefore historical E2 is not a one-variable resolution ablation. Its failure cannot be attributed to missed tile coverage: every GT had a full tile. Plausible contributors are background-dominated sampling, duplicated GT exposure, lost global context, very different update budget, and checkpoint protocol. Their individual causal effects remain untested.
