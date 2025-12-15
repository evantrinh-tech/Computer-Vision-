-- Migration: Initial Schema for Traffic Incident Detection System
-- Database: PostgreSQL
-- Created: 2024

-- ============================================
-- TABLE: incidents
-- ============================================
CREATE TABLE IF NOT EXISTS incidents (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    camera_id VARCHAR(100),
    location VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    incident_type VARCHAR(50),
    severity VARCHAR(20) DEFAULT 'medium',
    confidence_score FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    threshold FLOAT NOT NULL,
    rule_version VARCHAR(50),
    confirmation_method VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'detected',
    image_path TEXT,
    video_path TEXT,
    media_storage_type VARCHAR(20) DEFAULT 'local',
    media_url TEXT,
    metadata JSONB,
    latency_ms FLOAT,
    processing_time_ms FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Indexes for incidents
CREATE INDEX IF NOT EXISTS idx_incident_timestamp ON incidents(timestamp);
CREATE INDEX IF NOT EXISTS idx_incident_camera_id ON incidents(camera_id);
CREATE INDEX IF NOT EXISTS idx_incident_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incident_timestamp_camera ON incidents(timestamp, camera_id);
CREATE INDEX IF NOT EXISTS idx_incident_status_timestamp ON incidents(status, timestamp);
CREATE INDEX IF NOT EXISTS idx_incident_type_timestamp ON incidents(incident_type, timestamp);

-- ============================================
-- TABLE: predictions
-- ============================================
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction BOOLEAN NOT NULL,
    probability FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    camera_id VARCHAR(100),
    frame_number INTEGER,
    timestamp TIMESTAMP NOT NULL,
    processing_time_ms FLOAT,
    latency_ms FLOAT,
    ground_truth BOOLEAN,
    is_correct BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for predictions
CREATE INDEX IF NOT EXISTS idx_prediction_incident_id ON predictions(incident_id);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_camera_id ON predictions(camera_id);
CREATE INDEX IF NOT EXISTS idx_prediction_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp_camera ON predictions(timestamp, camera_id);
CREATE INDEX IF NOT EXISTS idx_prediction_model_timestamp ON predictions(model_name, timestamp);

-- ============================================
-- TABLE: model_runs
-- ============================================
CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    mlflow_run_id VARCHAR(100) UNIQUE,
    experiment_name VARCHAR(100),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    training_config JSONB,
    hyperparameters JSONB,
    train_metrics JSONB,
    val_metrics JSONB,
    test_metrics JSONB,
    n_train_samples INTEGER,
    n_val_samples INTEGER,
    n_test_samples INTEGER,
    model_path TEXT,
    artifacts_path TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for model_runs
CREATE INDEX IF NOT EXISTS idx_model_run_mlflow_run_id ON model_runs(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_model_run_experiment_name ON model_runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_model_run_started_at ON model_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_model_run_experiment_started ON model_runs(experiment_name, started_at);

-- ============================================
-- TABLE: alerts
-- ============================================
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    recipient VARCHAR(255),
    title VARCHAR(255),
    message TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    sent_at TIMESTAMP,
    read_at TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alert_incident_id ON alerts(incident_id);
CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alert_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_status_created ON alerts(status, created_at);

-- ============================================
-- TABLE: incident_media
-- ============================================
CREATE TABLE IF NOT EXISTS incident_media (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    media_type VARCHAR(20) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    mime_type VARCHAR(100),
    storage_type VARCHAR(20) NOT NULL DEFAULT 'local',
    storage_bucket VARCHAR(255),
    storage_key VARCHAR(500),
    signed_url TEXT,
    public_url TEXT,
    width INTEGER,
    height INTEGER,
    duration_seconds FLOAT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for incident_media
CREATE INDEX IF NOT EXISTS idx_media_incident_id ON incident_media(incident_id);
CREATE INDEX IF NOT EXISTS idx_media_incident_type ON incident_media(incident_id, media_type);
CREATE INDEX IF NOT EXISTS idx_media_created_at ON incident_media(created_at);

-- ============================================
-- PARTITIONING (Optional, for large-scale production)
-- ============================================
-- Partition incidents table by month (for PostgreSQL 10+)
-- CREATE TABLE incidents_2024_01 PARTITION OF incidents
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- CREATE TABLE incidents_2024_02 PARTITION OF incidents
--     FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... (tạo partition cho từng tháng)

-- ============================================
-- EXAMPLE QUERIES
-- ============================================

-- Query 1: Top cameras có nhiều false alarm nhất
-- SELECT 
--     camera_id,
--     COUNT(*) as false_alarm_count,
--     COUNT(*) * 100.0 / (SELECT COUNT(*) FROM incidents WHERE camera_id = i.camera_id) as false_alarm_rate
-- FROM incidents i
-- WHERE status = 'false_alarm'
-- GROUP BY camera_id
-- ORDER BY false_alarm_count DESC
-- LIMIT 10;

-- Query 2: FAR theo ngày
-- SELECT 
--     DATE(timestamp) as date,
--     COUNT(*) FILTER (WHERE status = 'false_alarm') as false_alarms,
--     COUNT(*) as total_incidents,
--     COUNT(*) FILTER (WHERE status = 'false_alarm') * 100.0 / COUNT(*) as far_percentage
-- FROM incidents
-- WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
-- GROUP BY DATE(timestamp)
-- ORDER BY date DESC;

-- Query 3: Model performance comparison
-- SELECT 
--     model_version,
--     COUNT(*) as total_predictions,
--     AVG(probability) as avg_confidence,
--     COUNT(*) FILTER (WHERE is_correct = true) * 100.0 / COUNT(*) as accuracy
-- FROM predictions
-- WHERE ground_truth IS NOT NULL
-- GROUP BY model_version
-- ORDER BY accuracy DESC;

-- Query 4: MTTD (Mean Time To Detection) - cần join với ground truth data
-- SELECT 
--     AVG(EXTRACT(EPOCH FROM (i.timestamp - gt.incident_start_time))) as mttd_seconds
-- FROM incidents i
-- JOIN ground_truth_incidents gt ON i.id = gt.incident_id
-- WHERE i.status = 'confirmed'
--   AND gt.incident_start_time IS NOT NULL;

