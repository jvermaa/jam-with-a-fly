"""Shiu taste validation, step 1 (Phase S, D3): can sugar GRNs, bitter GRNs and MN9 be found by annotation?

No simulation is run. Shiu et al. 2024 validated their model by activating
sugar-sensing gustatory receptor neurons and reading motor neuron MN9
(proboscis extension). To repeat that here the three populations must be
identifiable from the MaleCNS annotations alone; this script searches every
text column for them and records exactly what was searched and found. It does
not guess a mapping from an unlabelled type to a taste.

Real vs. chosen: annotations are REAL (MaleCNS v1.0). The search terms are CHOSEN.

Usage (from the repo root):  python -m src.taste_search
"""

import json
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.provenance import provenance

ANNOTATIONS = ROOT / "data" / "body-annotations.feather"
TEXT_COLUMNS = ["type", "flywireType", "instance", "hemibrainType", "supertype", "mancType", "class", "subclass",
                "synonyms", "matchingNotes", "entryNerve", "exitNerve", "receptorType"]
PATTERNS = {
    "sugar": r"sugar|sweet|Gr5a|Gr64",
    "bitter": r"bitter|Gr66|Gr33",
    "taste_general": r"GRN|gustat|taste|labell|pharyn",
    "MN9": r"\bMN9\b",
}


def search(ann, pattern):
    """{column: {value: count}} for every text column value matching pattern (case-insensitive)."""
    out = {}
    rx = re.compile(pattern, re.IGNORECASE)
    for col in TEXT_COLUMNS:
        s = ann[col].dropna().astype(str)
        hit = s[s.str.contains(rx)]
        if len(hit):
            out[col] = {str(k): int(v) for k, v in hit.value_counts().items()}
    return out


def main():
    ann = pd.read_feather(ANNOTATIONS)
    ann = ann[(ann.status == "Traced") & (ann.statusLabel != "Glia")]
    found = {name: search(ann, pat) for name, pat in PATTERNS.items()}

    sensory = ann[ann.superclass.astype(str).str.contains("sensory")]
    sugar_sensory = search(sensory, PATTERNS["sugar"])
    bitter_sensory = search(sensory, PATTERNS["bitter"])
    gust = ann[ann["class"] == "gustatory"]
    labellar = gust[gust.subclass == "labellar bristle"]
    mn9 = ann[ann.type == "MN9"]

    result = {
        "phase": "S",
        "step": "D3",
        **provenance(None),
        "seed_note": "no random numbers used; no simulation run",
        "neurons_searched": len(ann),
        "columns_searched": TEXT_COLUMNS,
        "patterns": PATTERNS,
        "matches_all_neurons": found,
        "matches_among_sensory_neurons": {"sugar": sugar_sensory, "bitter": bitter_sensory},
        "gustatory_class": {
            "neurons": len(gust),
            "by_subclass": {str(k): int(v) for k, v in gust.subclass.fillna("(none)").value_counts().items()},
            "labellar_bristle_types": {str(k): int(v) for k, v in labellar.type.value_counts().items()},
            "receptorType_values_in_dataset": {str(k): int(v) for k, v in ann.receptorType.dropna().value_counts().items()},
        },
        "mn9": {"found_by_type": len(mn9) > 0, "neurons": len(mn9), "instances": sorted(mn9.instance.astype(str).tolist()),
                "superclass": sorted(set(mn9.superclass.astype(str)))},
        "sugar_grns_found_by_annotation": bool(sugar_sensory),
        "bitter_grns_found_by_annotation": bool(bitter_sensory),
        "d3_can_run": bool(sugar_sensory) and bool(bitter_sensory) and len(mn9) > 0,
        "note": "Gustatory sensory neurons are typed (e.g. labellar LB1a-LB4b) but no column says which type senses sugar or "
                "bitter. The only 'sugar'/'bitter' strings are synonyms on second-order (non-sensory) neurons. No taste is "
                "assigned to an unlabelled type here.",
    }
    out = ROOT / "results" / "taste_search.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("sugar (all neurons):", found["sugar"])
    print("bitter (all neurons):", found["bitter"])
    print("sugar among sensory:", sugar_sensory, "| bitter among sensory:", bitter_sensory)
    print("labellar types:", result["gustatory_class"]["labellar_bristle_types"])
    print("MN9:", result["mn9"], "| D3 can run:", result["d3_can_run"], f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
