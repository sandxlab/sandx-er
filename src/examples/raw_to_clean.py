"""
From raw data to clean entities — SandX Entity Resolution.

Realistic scenario: a vendor database with 24 records representing
6 underlying companies, each duplicated with typical real-world noise
(abbreviations, punctuation differences, hyphenation, word-boundary
splits, missing suffixes, address format variation).

Install:
    pip install sandx-er

Run:
    python -m examples.raw_to_clean
"""

from __future__ import annotations

import time

import pandas as pd

from sandx_er import EntityResolver

# ---------------------------------------------------------------------------
# Dataset: 6 vendors × 4 noisy records each = 24 raw records
# ---------------------------------------------------------------------------

RAW_RECORDS = [
    # ── Meridian Health Solutions ────────────────────────────────────────
    {"id": "v01", "name": "Meridian Health Solutions",       "city": "Boston, MA"},
    {"id": "v02", "name": "Meridian Health Solutions Inc.",  "city": "Boston MA"},
    {"id": "v03", "name": "Meridian Health Soln. LLC",       "city": "Boston"},
    {"id": "v04", "name": "Meridian Health Solution",        "city": "Boston, MA"},

    # ── BioCore Analytics ────────────────────────────────────────────────
    {"id": "v05", "name": "BioCore Analytics Inc.",          "city": "San Diego, CA"},
    {"id": "v06", "name": "Bio-Core Analytics",              "city": "San Diego CA"},
    {"id": "v07", "name": "Biocore Analytics LLC",           "city": "San Diego"},
    {"id": "v08", "name": "Biocore Analytics",               "city": "San Diego, CA"},

    # ── DataVault Systems ────────────────────────────────────────────────
    {"id": "v09", "name": "DataVault Systems Corp.",         "city": "Austin, TX"},
    {"id": "v10", "name": "Data Vault Systems",              "city": "Austin TX"},
    {"id": "v11", "name": "DataVault Systems",               "city": "Austin"},
    {"id": "v12", "name": "DataVault Sys.",                  "city": "Austin, TX"},

    # ── CloudPeak Infrastructure ─────────────────────────────────────────
    {"id": "v13", "name": "CloudPeak Infrastructure",        "city": "Seattle, WA"},
    {"id": "v14", "name": "CloudPeak Infrastructure Inc.",   "city": "Seattle WA"},
    {"id": "v15", "name": "Cloud Peak Infrastructure",       "city": "Seattle"},
    {"id": "v16", "name": "Cloudpeak Infra.",                "city": "Seattle, WA"},

    # ── Nexus Financial Group ────────────────────────────────────────────
    {"id": "v17", "name": "Nexus Financial Group",           "city": "New York, NY"},
    {"id": "v18", "name": "Nexus Financial Grp.",            "city": "New York NY"},
    {"id": "v19", "name": "Nexus Financial Group Inc.",      "city": "New York"},
    {"id": "v20", "name": "Nexus Fin. Group",                "city": "NYC"},

    # ── Vertex Research Labs ─────────────────────────────────────────────
    {"id": "v21", "name": "Vertex Research Labs",            "city": "Cambridge, MA"},
    {"id": "v22", "name": "Vertex Research Laboratories",    "city": "Cambridge MA"},
    {"id": "v23", "name": "Vertex Research Labs Inc.",       "city": "Cambridge"},
    {"id": "v24", "name": "Vertex Res. Labs",                "city": "Cambridge, MA"},
]

W = 62  # column separator width
SEP   = "=" * W
RULE  = "-" * W


def _print_raw(df: pd.DataFrame) -> None:
    print(f"  {'ID':<5}  {'NAME':<38}  CITY")
    print("  " + RULE)
    for row in df.itertuples():
        print(f"  {row.Index:<5}  {row.name:<38}  {row.city}")


def _shortest(names: list[str]) -> str:
    return min(names, key=len)


def run() -> None:
    print()
    print("  " + SEP)
    print("   SandX Entity Resolution  --  Raw to Clean")
    print("  " + SEP)
    print(f"  {len(RAW_RECORDS)} raw records  .  6 underlying vendors  "
          f".  real-world name/address noise")
    print()

    df = pd.DataFrame(RAW_RECORDS).set_index("id")

    # ── Show raw input ───────────────────────────────────────────────────
    print("  RAW INPUT")
    print("  " + RULE)
    _print_raw(df)
    print()

    # ── Noise types present in the dataset ──────────────────────────────
    print("  Noise types covered:")
    noise_types = [
        ("Punctuation",       '"Boston, MA"  -> "Boston MA"  -> "Boston"'),
        ("Abbreviation",      '"Solutions"   -> "Soln."  /  "Systems" -> "Sys."'),
        ("Suffix variation",  '"Inc."  /  "LLC"  /  "Corp."  / (none)'),
        ("Hyphenation",       '"BioCore"  -> "Bio-Core"'),
        ("Case",              '"BioCore"  -> "Biocore"'),
        ("Word boundary",     '"DataVault"  -> "Data Vault"'),
        ("Address short",     '"New York, NY"  -> "NYC"'),
    ]
    for label, example in noise_types:
        print(f"    {label:<20}  {example}")
    print()

    # ── Resolve ──────────────────────────────────────────────────────────
    er = EntityResolver(
        blocking="lsh",
        similarity="jaccard",
        clustering="connected_components",
        threshold=0.30,
    )

    t0 = time.perf_counter()
    result = er.resolve(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    multi = [c for c in result.clusters if c.size > 1]
    multi.sort(key=lambda c: c.record_ids[0])

    # ── Show resolved entities ────────────────────────────────────────────
    print("  RESOLVED ENTITIES")
    print("  " + RULE)
    print(f"  {'ENTITY':<34}  {'CONF':>5}  {'SIZE':>4}  RECORDS")
    print("  " + RULE)

    for cluster in multi:
        names = [df.loc[r, "name"] for r in cluster.record_ids if r in df.index]
        label = _shortest(names)
        members = "  ".join(cluster.record_ids)
        print(f"  {label:<34}  {cluster.confidence:>5.2f}  {cluster.size:>4}  [{members}]")

    singletons = [c for c in result.clusters if c.size == 1]
    if singletons:
        print(f"\n  Unresolved singletons: {len(singletons)}")

    print()
    print("  " + SEP)
    print(f"   {result.n_records} raw records  ->  {len(multi)} resolved entities"
          f"  [{elapsed_ms:.0f} ms]")
    print("  " + SEP)
    print()

    # ── Show the 3-line API that did all of this ─────────────────────────
    print("  What ran:")
    print()
    print("    from sandx_er import EntityResolver")
    print()
    print("    er     = EntityResolver(blocking='lsh',")
    print("                            similarity='jaccard', threshold=0.30)")
    print("    result = er.resolve(df)        # df: pandas DataFrame of raw records")
    print("    for c in result.clusters:")
    print("        print(c.canonical_id[:8], c.size, c.confidence)")
    print()
    print("  Upgrade to embedding-based matching for higher recall on noisy text:")
    print("    pip install sandx-embed")
    print("    er = EntityResolver(blocking='embedding',")
    print("                        similarity='embedding', threshold=0.85)")
    print()


if __name__ == "__main__":
    run()
