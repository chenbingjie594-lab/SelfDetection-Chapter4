# Literature Mechanism Analysis

## Scope and access

The three requested papers were checked by exact DOI. The FDFNet publisher page exposed substantial method text and a PDF endpoint, while the PDF endpoint itself returned HTTP 403 in this environment. The PST record exposed the peer-reviewed abstract and bibliographic record; IEEE full text was access-controlled. LWMS-Net exposed the publisher abstract/highlights but not the purchased full text. Consequently, implementation details below are separated into **source-confirmed facts** and **mechanistic interpretation**; inaccessible equations are not invented.

## Wavelet-Guided Promotion-Suppression Transformer (TIP 2023)

Source-confirmed: PST addresses pixel-level surface-defect segmentation. Its promotion stage uses Haar features to increase sensitivity to uncommon texture/edge detail. Because background noise is also high frequency, its suppression stage uses a quadratic feature-fusion module and the stated noise properties of independence and attenuation to compare feature similarities/differences. The paper reports DAGM, MT and CRACK500, not HBB detection or Plastic_Bomo ([PubMed record](https://pubmed.ncbi.nlm.nih.gov/37490374/), [DOI](https://doi.org/10.1109/TIP.2023.3293770)).

The transferable principle is not “high frequency equals defect,” but promotion followed by conditioned suppression. A fixed orthonormal Haar transform decomposes a 2×2 cell into LL/LH/HL/HH bands; energy is conserved up to boundary handling. For HBB detection, pixel-level masks and the original transformer decoder are inappropriate: a box includes normal pixels, and the dataset is too small to justify a global transformer branch.

## FDFNet (Digital Signal Processing 2025)

Source-confirmed: FDFNet is a YOLOv9-based detector. It combines CLAHE, SPD-Conv and a two-branch Differentiated Fusion module intended to process similar and different components of multi-scale features. SPD rearranges spatial samples into channels before convolution to reduce information loss during downsampling. The paper targets rolled-steel datasets and does not establish gains on Plastic_Bomo ([publisher article](https://www.sciencedirect.com/science/article/pii/S1051200425004543), [DOI](https://doi.org/10.1016/j.dsp.2025.105432)).

The useful principle is explicit cross-scale similarity/difference after spatial and channel alignment. CLAHE is rejected here because it would change the frozen input protocol and can amplify texture; replacing all downsampling with SPD-Conv is also too broad for a controlled Stage8 comparison.

## LWMS-Net (Measurement 2025)

Source-confirmed: LWMS-Net proposes a multi-wavelet/multi-scale backbone and a feature-difference enhancement module, with the stated objective of balancing detailed/contextual representation and speed. Its highlights specifically mention complex defect edges and three evaluation datasets ([publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0263224125007523), [DOI](https://doi.org/10.1016/j.measurement.2025.117393)).

Legendre multi-wavelets provide multiple basis functions and vanishing moments, but directly replacing the backbone would introduce many uncontrolled variables and a larger hypothesis space. With only 138 training images, fixed Haar is the safer diagnostic transform; a learned multi-wavelet basis is not justified before a simpler frequency mechanism succeeds.

## Differentiated design for Plastic_Bomo

SFDFM is deliberately narrower than the cited networks:

1. It operates once at Neck P3 rather than replacing the backbone or adding P2 detection.
2. Fixed Haar supplies directional detail without learnable transform drift.
3. An aligned P4 projection supplies cross-scale difference `|P3-P4↑|` and consistency `P3·P4↑`.
4. A learned gate conditions high-frequency detail on spatial cross-scale relations; high response is not automatically promoted.
5. The original loss, P4/P5 paths, detection scales and frozen data protocol remain unchanged.

This is a new controlled combination for this project, not a claim that the underlying Haar, difference or gating operations are novel individually.
