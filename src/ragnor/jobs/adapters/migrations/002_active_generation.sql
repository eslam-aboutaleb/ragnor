CREATE TABLE IF NOT EXISTS ragnor_active_generations (
        scope text NOT NULL,
        document_id text NOT NULL,
        revision bigint NOT NULL,
        generation_key text NOT NULL,
        activated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY(scope, document_id)
    );
