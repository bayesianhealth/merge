CREATE OR REPLACE TABLE TEST.SILVER.DETERIORATION_RADIOLOGY (
    note_id     VARCHAR,
    subject_id  NUMBER(38,0),
    hadm_id     NUMBER(38,0),
    note_type   VARCHAR,
    note_seq    NUMBER(38,0),
    charttime   TIMESTAMP_NTZ,
    storetime   TIMESTAMP_NTZ,
    text        VARCHAR
);

COPY INTO TEST.SILVER.DETERIORATION_RADIOLOGY
    FROM @"TEST"."SILVER"."UDTF_FEATURE_STAGE"/radiology.csv.gz
    FILE_FORMAT = (
        TYPE = CSV
        COMPRESSION = GZIP
        FIELD_DELIMITER = ','
        SKIP_HEADER = 1
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        ESCAPE_UNENCLOSED_FIELD = NONE
        EMPTY_FIELD_AS_NULL = TRUE
        NULL_IF = ('', 'NULL')
        MULTI_LINE = TRUE
    )
    ON_ERROR = CONTINUE;

SELECT COUNT(*) AS n_notes,
       COUNT(DISTINCT hadm_id) AS n_admissions,
       MIN(charttime) AS min_charttime,
       MAX(charttime) AS max_charttime
FROM TEST.SILVER.DETERIORATION_RADIOLOGY;
