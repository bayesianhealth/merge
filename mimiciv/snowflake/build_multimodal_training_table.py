from deterioration_config import TS_TABLE, NOTES_TABLE, MULTIMODAL_TABLE, SEQ_LEN


def build_sql() -> str:
    return f"""
CREATE OR REPLACE TABLE {MULTIMODAL_TABLE}
CLUSTER BY (split) AS
WITH notes_agg AS (
    SELECT hadm_id, obs_tsp,
           ARRAY_AGG(OBJECT_CONSTRUCT(
               'slot', {SEQ_LEN - 1} - hours_before_obs,
               'emb',  note_embedding
           )) AS notes_packed
    FROM {NOTES_TABLE}
    WHERE hours_before_obs BETWEEN 0 AND {SEQ_LEN - 1}
    GROUP BY hadm_id, obs_tsp
)
SELECT t.*, n.notes_packed
FROM {TS_TABLE} t
LEFT JOIN notes_agg n
  ON t.hadm_id = n.hadm_id AND t.tsp = n.obs_tsp
"""


def run(session):
    print(f"Phase 3  building {MULTIMODAL_TABLE} (single pre-join, no runtime joins) ...")
    session.sql(build_sql()).collect()
    stats = session.sql(f"""
        SELECT split,
               COUNT(*) AS n,
               COUNT(notes_packed) AS n_with_notes,
               AVG(label) AS prevalence
        FROM {MULTIMODAL_TABLE}
        GROUP BY split ORDER BY split
    """).collect()
    print(f"Done. {MULTIMODAL_TABLE} written:")
    for r in stats:
        frac = r["N_WITH_NOTES"] / max(r["N"], 1) * 100
        print(f"  {r['SPLIT']}: n={r['N']:,} with_notes={r['N_WITH_NOTES']:,} "
              f"({frac:.1f}%) prevalence={r['PREVALENCE']:.4f}")


if __name__ == "__main__":
    from snowflake_utils import get_snowpark_session
    run(get_snowpark_session())
