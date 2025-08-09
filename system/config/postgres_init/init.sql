-- NIS Protocol v3 Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS nis_agents;
CREATE SCHEMA IF NOT EXISTS nis_monitoring;
CREATE SCHEMA IF NOT EXISTS nis_infrastructure;

-- Create basic tables for agents
CREATE TABLE IF NOT EXISTS nis_agents.agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type VARCHAR(50) NOT NULL,
    session_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for monitoring
CREATE TABLE IF NOT EXISTS nis_monitoring.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agent_sessions_type ON nis_agents.agent_sessions(agent_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON nis_monitoring.performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON nis_monitoring.performance_metrics(timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA nis_agents TO nis_user;
GRANT ALL PRIVILEGES ON SCHEMA nis_monitoring TO nis_user;
GRANT ALL PRIVILEGES ON SCHEMA nis_infrastructure TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_agents TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_monitoring TO nis_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nis_infrastructure TO nis_user;
