CREATE TABLE IF NOT EXISTS ragnor_embedding_jobs (
        job_id text PRIMARY KEY,
        scope text NOT NULL,
        idempotency_key text NOT NULL,
        payload jsonb NOT NULL,
        status text NOT NULL DEFAULT 'pending',
        retry_count integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 4,
        next_retry_at timestamptz,
        locked_at timestamptz,
        lease_expires_at timestamptz,
        locked_by text,
        claim_generation bigint NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        status_detail text,
        UNIQUE(scope, idempotency_key)
    );

CREATE INDEX IF NOT EXISTS ragnor_job_claim_idx
       ON ragnor_embedding_jobs(scope, status, next_retry_at, created_at, job_id);

CREATE INDEX IF NOT EXISTS ragnor_job_lease_idx
       ON ragnor_embedding_jobs(scope, status, lease_expires_at);
