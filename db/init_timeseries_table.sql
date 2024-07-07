-- init_timeseries_table.sql
CREATE TABLE timeseries (
    ts TIMESTAMPTZ NOT NULL,
    load FLOAT,
    grid_operator VARCHAR(255),
    grid_area VARCHAR(255),
    day_ahead_temp FLOAT,
    PRIMARY KEY (ts, grid_operator, grid_area)
);

-- Add an index on ts for efficient querying by timestamp alone
CREATE INDEX idx_timeseries_ts ON timeseries (ts);
