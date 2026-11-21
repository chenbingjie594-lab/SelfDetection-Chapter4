# CIR crop alignment audit

The low route exactly reproduces the content resize 4096×1024→1536×384 and resizes that content back to 4096×1024 without letterbox padding. High and low crops share the identical prediction-derived ROI coordinates, replicate padding, 128×128 resize, BGR→RGB conversion, and [0,1] normalization. `alignment/crop_alignment.json` and paired Flash/Black plates confirm equal shapes and coordinates. GT is absent from ROI construction.
