-- Initialize database with tables and sample data

-- Create tables
CREATE TABLE IF NOT EXISTS meter_readings (
    id SERIAL PRIMARY KEY,
    meter_id VARCHAR(50) NOT NULL,
    reading_date DATE NOT NULL,
    consumption DECIMAL(10, 2) NOT NULL,
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    location_name VARCHAR(100)
);

-- Insert sample data
INSERT INTO meter_readings (meter_id, reading_date, consumption, latitude, longitude, location_name)
VALUES
    ('M001', '2025-01-05', 25.5, -1.234567, 36.789012, 'Location 1'),
    ('M001', '2025-01-15', 28.7, -1.234567, 36.789012, 'Location 1'),
    ('M001', '2025-01-25', 26.3, -1.234567, 36.789012, 'Location 1'),
    ('M001', '2025-02-05', 24.8, -1.234567, 36.789012, 'Location 1'),
    ('M002', '2025-01-05', 32.1, -1.345678, 36.890123, 'Location 2'),
    ('M002', '2025-01-15', 35.6, -1.345678, 36.890123, 'Location 2'),
    ('M002', '2025-01-25', 31.9, -1.345678, 36.890123, 'Location 2'),
    ('M002', '2025-02-05', 33.2, -1.345678, 36.890123, 'Location 2'),
    ('M003', '2025-01-05', 18.3, -1.456789, 36.901234, 'Location 3'),
    ('M003', '2025-01-15', 19.5, -1.456789, 36.901234, 'Location 3'),
    ('M003', '2025-01-25', 21.2, -1.456789, 36.901234, 'Location 3'),
    ('M003', '2025-02-05', 17.8, -1.456789, 36.901234, 'Location 3'),
    ('M004', '2025-01-05', 42.7, -1.567890, 37.012345, 'Location 4'),
    ('M004', '2025-01-15', 45.3, -1.567890, 37.012345, 'Location 4'),
    ('M004', '2025-01-25', 48.9, -1.567890, 37.012345, 'Location 4'), -- Anomaly (high)
    ('M004', '2025-02-05', 44.1, -1.567890, 37.012345, 'Location 4'),
    ('M005', '2025-01-05', 15.4, -1.678901, 37.123456, 'Location 5'),
    ('M005', '2025-01-15', 3.8, -1.678901, 37.123456, 'Location 5'),  -- Anomaly (low)
    ('M005', '2025-01-25', 16.7, -1.678901, 37.123456, 'Location 5'),
    ('M005', '2025-02-05', 14.9, -1.678901, 37.123456, 'Location 5');

-- Create view for aggregated consumption by dekad
CREATE OR REPLACE VIEW dekad_consumption AS
SELECT 
    CASE
        WHEN EXTRACT(DAY FROM reading_date) <= 10 THEN 
            DATE_TRUNC('month', reading_date) + INTERVAL '0 days'
        WHEN EXTRACT(DAY FROM reading_date) <= 20 THEN 
            DATE_TRUNC('month', reading_date) + INTERVAL '10 days'
        ELSE 
            DATE_TRUNC('month', reading_date) + INTERVAL '20 days'
    END AS dekad_period,
    SUM(consumption) as total_consumption,
    AVG(consumption) as avg_consumption,
    COUNT(*) as reading_count
FROM meter_readings
GROUP BY dekad_period
ORDER BY dekad_period;