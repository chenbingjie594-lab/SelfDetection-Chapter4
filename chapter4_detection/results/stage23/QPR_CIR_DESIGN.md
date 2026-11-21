# QPR-CIR design

A frozen Stage22 CIR-C residual is multiplied by a learned scalar gate. Gate-Geo uses 13 deployable geometry/residual features. Gate-Full uses the same 13 plus shared CIR high, low and absolute-difference feature summaries (205 inputs total). Both use a 32-unit SiLU MLP and sigmoid output initialized to g=0.5. GT-derived quality is training-only.
