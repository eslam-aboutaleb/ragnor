CREATE TABLE IF NOT EXISTS ragnor_generation_reservations (
    scope text NOT NULL,
    generation_key text NOT NULL,
    content_digest text NOT NULL,
    PRIMARY KEY(scope, generation_key)
);
