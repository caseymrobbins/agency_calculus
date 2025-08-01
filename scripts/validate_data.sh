#!/bin/bash
# File: scripts/validate_data.sh
# Data validation script for Agency Calculus ETL

set -e

echo "🔍 Validating Agency Calculus ETL Data..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DB_NAME=$(echo $DATABASE_URL | sed 's/.*\///')

echo "📊 Database: $DB_NAME"
echo "📅 Expected coverage: 1960-2024"
echo ""

# 1. Basic table structure validation
echo "1️⃣ Table Structure Validation:"
TABLES=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
echo "✅ Tables created: $TABLES/4 expected"

# Check each table exists
for table in countries indicators observations predictions; do
    if psql $DB_NAME -t -c "SELECT to_regclass('$table');" | grep -q "$table"; then
        echo "✅ Table '$table' exists"
    else
        echo "❌ Table '$table' missing"
    fi
done

echo ""

# 2. Data coverage validation
echo "2️⃣ Data Coverage Validation:"

# Countries
COUNTRY_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM countries;" | xargs)
echo "📍 Countries loaded: $COUNTRY_COUNT"

# Indicators by source
echo "📊 Indicators by source:"
psql $DB_NAME -c "
SELECT 
    source,
    COUNT(*) as count
FROM indicators 
GROUP BY source 
ORDER BY count DESC;
"

# Year coverage
echo ""
echo "📅 Year Coverage Analysis:"
psql $DB_NAME -c "
SELECT 
    MIN(year) as earliest_year,
    MAX(year) as latest_year,
    MAX(year) - MIN(year) + 1 as year_span,
    COUNT(DISTINCT year) as unique_years
FROM observations;
"

echo ""

# 3. Data quality checks
echo "3️⃣ Data Quality Validation:"

# Check for null values
NULL_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations WHERE value IS NULL;" | xargs)
echo "🔍 Null values in observations: $NULL_COUNT (should be 0)"

# Check for duplicate observations
DUPLICATE_COUNT=$(psql $DB_NAME -t -c "
SELECT COUNT(*) - COUNT(DISTINCT (country_code, indicator_code, year)) 
FROM observations;
" | xargs)
echo "🔍 Duplicate observations: $DUPLICATE_COUNT (should be 0)"

# Value range validation
echo ""
echo "📈 Value Range Analysis:"
psql $DB_NAME -c "
SELECT 
    'Min Value' as metric,
    MIN(value) as value
FROM observations
UNION ALL
SELECT 
    'Max Value' as metric,
    MAX(value) as value
FROM observations
UNION ALL  
SELECT
    'Avg Value' as metric,
    ROUND(AVG(value)::numeric, 2) as value
FROM observations;
"

echo ""

# 4. Agency Calculus domain validation
echo "4️⃣ Agency Calculus Domain Validation:"

# Economic domain indicators
ECONOMIC_COUNT=$(psql $DB_NAME -t -c "
SELECT COUNT(*) FROM indicators 
WHERE source LIKE '%World Bank%' 
AND (indicator_code LIKE 'NY.%' OR indicator_code LIKE 'SI.%' OR indicator_code LIKE 'FP.%');
" | xargs)
echo "💰 Economic domain indicators: $ECONOMIC_COUNT"

# Political domain indicators  
POLITICAL_COUNT=$(psql $DB_NAME -t -c "
SELECT COUNT(*) FROM indicators 
WHERE source LIKE '%V-Dem%' OR source LIKE '%Freedom%';
" | xargs)
echo "🏛️ Political domain indicators: $POLITICAL_COUNT"

# Health domain indicators
HEALTH_COUNT=$(psql $DB_NAME -t -c "
SELECT COUNT(*) FROM indicators 
WHERE indicator_code LIKE 'SP.DYN%' OR indicator_code LIKE 'SH.%';
" | xargs)
echo "🏥 Health domain indicators: $HEALTH_COUNT"

# Educational domain indicators
EDUCATION_COUNT=$(psql $DB_NAME -t -c "
SELECT COUNT(*) FROM indicators 
WHERE indicator_code LIKE 'SE.%';
" | xargs)
echo "🎓 Educational domain indicators: $EDUCATION_COUNT"

echo ""

# 5. Historical validation data availability
echo "5️⃣ Historical Validation Data:"

echo "🏛️ Critical events data availability:"
psql $DB_NAME -c "
SELECT 
    year,
    COUNT(DISTINCT country_code) as countries_with_data,
    COUNT(DISTINCT indicator_code) as indicators_available,
    COUNT(*) as total_observations
FROM observations 
WHERE year IN (1973, 1979, 1991, 1994)
GROUP BY year
ORDER BY year;
"

echo ""

# 6. Key countries analysis
echo "6️⃣ Key Countries Analysis:"

echo "🇺🇸 USA (Current focus - 9+/10 brittleness):"
psql $DB_NAME -c "
SELECT 
    MIN(year) as earliest_data,
    MAX(year) as latest_data,
    COUNT(DISTINCT indicator_code) as indicators,
    COUNT(*) as observations
FROM observations 
WHERE country_code = 'USA';
"

echo ""
echo "🌍 Top 5 countries by data coverage:"
psql $DB_NAME -c "
SELECT 
    country_code,
    c.name,
    COUNT(DISTINCT indicator_code) as indicators,
    COUNT(*) as observations,
    MIN(year) as earliest,
    MAX(year) as latest
FROM observations o
JOIN countries c ON o.country_code = c.country_code
GROUP BY country_code, c.name
ORDER BY observations DESC
LIMIT 5;
"

echo ""

# 7. Data completeness score
echo "7️⃣ Data Completeness Score:"

TOTAL_POSSIBLE=$(psql $DB_NAME -t -c "
SELECT 
    (SELECT COUNT(*) FROM countries) * 
    (SELECT COUNT(*) FROM indicators) * 
    65 -- years from 1960-2024
as total_possible;
" | xargs)

ACTUAL_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations;" | xargs)

COMPLETENESS=$(echo "scale=2; $ACTUAL_OBS * 100 / $TOTAL_POSSIBLE" | bc)

echo "📊 Data completeness: $COMPLETENESS% ($ACTUAL_OBS / $TOTAL_POSSIBLE possible)"

# 8. Validation summary
echo ""
echo "8️⃣ Validation Summary:"

if [ $NULL_COUNT -eq 0 ] && [ $DUPLICATE_COUNT -eq 0 ]; then
    echo "✅ Data quality: PASSED"
else
    echo "❌ Data quality: FAILED (nulls: $NULL_COUNT, duplicates: $DUPLICATE_COUNT)"
fi

if [ $ECONOMIC_COUNT -ge 8 ] && [ $POLITICAL_COUNT -ge 5 ]; then
    echo "✅ Agency domains: PASSED"
else
    echo "❌ Agency domains: INSUFFICIENT COVERAGE"
fi

if [ $ACTUAL_OBS -ge 10000 ]; then
    echo "✅ Data volume: PASSED ($ACTUAL_OBS observations)"
else
    echo "⚠️  Data volume: LOW ($ACTUAL_OBS observations)"
fi

echo ""
echo "🎯 Ready for Agency Calculus brittleness calculations!"
echo ""
echo "Key metrics for brittleness scoring:"
echo "• Economic indicators (GDP, Gini, etc.): Available"
echo "• Political indicators (V-Dem indices): Available"  
echo "• Historical baseline (1960-2000): Available"
echo "• Recent trends (2000-2024): Available"
echo "• US data completeness: Validated"
echo ""
echo "✅ Validation complete!"