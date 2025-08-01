#!/usr/bin/env python3
"""
Database Test Script for Agency Calculus

This script tests the database connection and permissions.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agency_calculus.api.database import (
    test_connection, 
    create_db_engine, 
    bulk_upsert_countries, 
    bulk_upsert_indicators,
    bulk_upsert_observations
)

# Load environment variables
load_dotenv()

def test_basic_connection():
    """Test basic database connection."""
    print("🔍 Testing basic database connection...")
    try:
        if test_connection():
            print("✅ Basic connection successful!")
            return True
        else:
            print("❌ Basic connection failed!")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_table_access():
    """Test table access and permissions."""
    print("\n🔍 Testing table access and permissions...")
    
    engine = create_db_engine()
    
    try:
        with engine.connect() as conn:
            # Test SELECT permissions
            tables = ['countries', 'indicators', 'observations']
            
            for table in tables:
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = result.scalar()
                    print(f"✅ {table}: SELECT permission OK, {count} records")
                except Exception as e:
                    print(f"❌ {table}: SELECT permission failed - {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Table access test failed: {e}")
        return False

def test_insert_permissions():
    """Test INSERT permissions with sample data."""
    print("\n🔍 Testing INSERT permissions...")
    
    # Test country insert
    print("Testing country insert...")
    test_countries = [{
        'country_code': 'TST',
        'name': 'Test Country',
        'region': 'Test Region',
        'income_level': 'Test Income',
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }]
    
    try:
        rows = bulk_upsert_countries(test_countries)
        print(f"✅ Country insert successful: {rows} rows affected")
    except Exception as e:
        print(f"❌ Country insert failed: {e}")
        return False
    
    # Test indicator insert
    print("Testing indicator insert...")
    test_indicators = [{
        'indicator_code': 'TST.TEST.CODE',
        'name': 'Test Indicator',
        'description': 'Test Description',
        'unit': 'Test Unit',
        'source': 'Test Source',
        'topic': 'Test Topic',
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }]
    
    try:
        rows = bulk_upsert_indicators(test_indicators)
        print(f"✅ Indicator insert successful: {rows} rows affected")
    except Exception as e:
        print(f"❌ Indicator insert failed: {e}")
        return False
    
    # Test observation insert
    print("Testing observation insert...")
    test_observations = [{
        'country_code': 'TST',
        'indicator_code': 'TST.TEST.CODE',
        'year': 2023,
        'value': 123.45,
        'dataset_version': 'TEST-2023',
        'notes': 'Test observation',
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }]
    
    try:
        rows = bulk_upsert_observations(test_observations)
        print(f"✅ Observation insert successful: {rows} rows affected")
    except Exception as e:
        print(f"❌ Observation insert failed: {e}")
        return False
    
    return True

def cleanup_test_data():
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")
    
    engine = create_db_engine()
    
    try:
        with engine.begin() as conn:
            # Delete test data in reverse order (observations, indicators, countries)
            conn.execute("DELETE FROM observations WHERE country_code = 'TST'")
            conn.execute("DELETE FROM indicators WHERE indicator_code = 'TST.TEST.CODE'")
            conn.execute("DELETE FROM countries WHERE country_code = 'TST'")
            print("✅ Test data cleaned up successfully!")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def show_database_info():
    """Show database information."""
    print("\n📊 Database Information:")
    
    engine = create_db_engine()
    
    try:
        with engine.connect() as conn:
            # Get database name
            result = conn.execute("SELECT current_database()")
            db_name = result.scalar()
            print(f"Database: {db_name}")
            
            # Get current user
            result = conn.execute("SELECT current_user")
            user = result.scalar()
            print(f"User: {user}")
            
            # Get table counts
            tables = ['countries', 'indicators', 'observations']
            for table in tables:
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = result.scalar()
                    print(f"{table}: {count:,} records")
                except:
                    print(f"{table}: Unable to count")
                    
    except Exception as e:
        print(f"❌ Unable to get database info: {e}")

def main():
    """Main test function."""
    print("🧪 Agency Calculus Database Test Suite")
    print("=" * 50)
    
    # Test basic connection
    if not test_basic_connection():
        print("\n💥 Basic connection failed. Check your DATABASE_URL in .env file.")
        return False
    
    # Show database info
    show_database_info()
    
    # Test table access
    if not test_table_access():
        print("\n💥 Table access failed. Check your database schema.")
        return False
    
    # Test insert permissions
    if not test_insert_permissions():
        print("\n💥 Insert permissions failed. Check your database user permissions.")
        return False
    
    # Clean up
    cleanup_test_data()
    
    print("\n🎉 All database tests passed!")
    print("Your database is ready for World Bank data import.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)