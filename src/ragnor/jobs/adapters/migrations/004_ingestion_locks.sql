CREATE TABLE IF NOT EXISTS ragnor_ingestion_locks (
    lock_key text PRIMARY KEY,
    generation integer NOT NULL DEFAULT 0
);
