# Stage16 final summary

1. **Expert review:** 0/25 samples have an expert conclusion; all remain pending. The two low XML/YOLO agreement cases are one-pixel boundary-edge differences and remain pending.
2. **UNKNOWN recovery:** 25/35 XML-UNKNOWN images have high-confidence historical reconstruction evidence; 10/35 remain UNKNOWN. Original XML is still absent for all 35.
3. **Current split isolation:** the 138/46 split has zero source-group and zero canonical-ID intersections. This supports the current group-isolation protocol, subject to the 10 unresolved sources.
4. **Historical overlap:** exact current-image copies occur across detector, generator-study, real-control and augmentation directories, including current validation images. Such directories are not independent evaluation sources. The exact historical 136/48 membership could not be identified from an auditable manifest, so no stronger claim is made.
5. **Independent test:** `NO_VERIFIED_INDEPENDENT_TEST_SET`; 0/777 test-labelled candidates passed all criteria.
6. **What existing experiments support:** controlled comparisons inside the frozen development protocol and hypotheses about resolution/localization mechanisms.
7. **What they do not support:** external/general population performance, definitive ranking on an unseen production batch, or annotation-correctness claims for the 25 pending cases.

No training was run, no model was changed, and no image, YOLO label, XML, checkpoint or historical result was overwritten.

