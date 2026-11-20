# Annotation provenance audit

The frozen protocol contains 184 real images: 138 train and 46 validation. Stage10 could trace 149 annotations to original XML and reported 35 XML-UNKNOWN records.

Stage16 separated “original XML exists” from “image source can be recovered”:

| Stage10 XML status | Stage16 provenance result | Count |
|---|---|---:|
| TRACED | CONFIRMED_ORIGINAL_XML | 149 |
| UNKNOWN | CONFIRMED_HISTORICAL_RECONSTRUCTION | 25 |
| UNKNOWN | UNKNOWN | 10 |

The 25 recovered records are `blackinst_*` exports supported by the historical `black_exports.json` evidence and recorded original-source SHA256. This does **not** recreate an original XML annotation. Ten records (`blackinst_6,14,18,21,25,29,31,33,35,38`) only have a basename-emission explanation and no retained original bytes/XML, so they remain UNKNOWN.

Per-image evidence is in `DATASET_LINEAGE.csv`; the focused list is `unknown_35_provenance.csv`. No conclusion is based on the `blackinst` filename alone.

