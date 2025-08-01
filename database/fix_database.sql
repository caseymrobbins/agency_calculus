-- Fix Database Permissions for Agency Calculus
-- Run this script as PostgreSQL superuser (usually 'postgres')

-- Connect to the agency_monitor database
\c agency_monitor;

-- Replace 'your_username' with your actual database username
-- You can find your username with: SELECT current_user;

-- Grant all privileges on the database
GRANT ALL PRIVILEGES ON DATABASE agency_monitor TO your_username;

-- Grant all privileges on all existing tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_username;

-- Grant all privileges on all existing sequences (for auto-increment columns)
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_username;

-- Grant default privileges for future tables and sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO your_username;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO your_username;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO your_username;

-- Grant create privilege on schema (to create new tables if needed)
GRANT CREATE ON SCHEMA public TO your_username;

-- Show current permissions (optional - for verification)
\dp

-- Show table ownership (optional - for verification)  
SELECT 
    schemaname,
    tablename,
    tableowner,
    hasinsert,
    hasselect,a
    hasupdate,
    hasdelete
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- If you need to change table ownership (uncomment if needed):
-- ALTER TABLE countries OWNER TO your_username;
-- ALTER TABLE indicators OWNER TO your_username;
-- ALTER TABLE observations OWNER TO your_username;
-- ALTER TABLE agency_scores OWNER TO your_username;

NOTIFY successful_permissions_fix;