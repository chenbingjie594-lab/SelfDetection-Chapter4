# Model Selection Recommendation

**Provisional architecture choice: A1 (difference + gate), with E3 retained as the full research comparison—not as a proven superior final model.**

E3 has the largest validation mean among the mature variants, but E3−E0 (+0.0082) and E3−A1 (+0.0061) bootstrap intervals cross zero. The similarity-reference branch has no robust independent gain, whereas removing the gate during training is consistently harmful. A1 is therefore the more parsimonious statement of the supported mechanism. Its AP75 advantage over E3 is also uncertain, so it must not be described as accuracy-superior. E4 remains exploratory only.

Current evidence supports a thesis claim that background-difference gating changes training behavior and can improve high-IoU localization on this development split. It does not yet support a strong generalization claim. The highest-priority missing evidence is a provenance-isolated, never-used real test cohort (or externally collected data), followed by a preregistered replication with more source groups.
