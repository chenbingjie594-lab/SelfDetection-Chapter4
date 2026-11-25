# HBD implementation audit

Only the P3 classification adapter is deployed. Backbone, neck, original classification towers, all regression towers and P4/P5 outputs remain frozen. Each prediction audit must show zero P3/P4/P5 box-logit change and zero P4/P5 class-logit change. The training-only projection is saved for audit but not installed during inference. Cached patches use zero padding plus a spatial validity mask after each classifier spatial block.
