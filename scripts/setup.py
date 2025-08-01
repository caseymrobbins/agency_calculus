#!/usr/bin/env python3
"""
Agency Calculus Setup Script

This script automates the setup process for the Agency Calculus World Bank data import.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        'etl',
        'config', 
        'logs',
        'models',
        'data',
        'scripts',
        'api',
        'database'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Need Python 3.8+")
        return False

def check_postgresql():
    """Check if PostgreSQL is installed and running."""
    print("🐘 Checking PostgreSQL...")
    
    # Check if psql is available
    if not shutil.which('psql'):
        print("❌ PostgreSQL client (psql) not found. Please install PostgreSQL.")
        return False
    
    # Try to connect to PostgreSQL
    result = run_command('psql --version', 'Getting PostgreSQL version', check=False)
    if result:
        print("✅ PostgreSQL client is available")
        return True
    else:
        print("❌ PostgreSQL client check failed")
        return False

def setup_virtual_environment():
    """Set up Python virtual environment."""
    print("🏠 Setting up virtual environment...")
    
    venv_path = Path('venv')
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command(f'{sys.executable} -m venv venv', 'Creating virtual environment'):
        return False
    
    print("✅ Virtual environment created successfully")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = 'venv\\Scripts\\pip'
    else:  # Unix-like
        pip_path = 'venv/bin/pip'
    
    # Upgrade pip first
    if not run_command(f'{pip_path} install --upgrade pip', 'Upgrading pip'):
        return False
    
    # Install requirements
    if Path('requirements.txt').exists():
        if not run_command(f'{pip_path} install -r requirements.txt', 'Installing requirements'):
            return False
    else:
        # Install essential packages manually
        essential_packages = [
            'pandas>=1.5.0',
            'numpy>=1.21.0', 
            'sqlalchemy>=1.4.0',
            'psycopg2-binary>=2.9.0',
            'python-dotenv>=0.19.0',
            'wbgapi>=1.0.12',
            'pyyaml>=6.0',
            'requests>=2.28.0'
        ]
        
        for package in essential_packages:
            if not run_command(f'{pip_path} install "{package}"', f'Installing {package}'):
                return False
    
    print("✅ All dependencies installed successfully")
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    print("⚙️ Setting up environment configuration...")
    
    env_path = Path('.env')
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    # Create a template .env file
    env_template = """# Database Configuration
# Replace with your actual database credentials
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/agency_monitor

# World Bank API Configuration
WB_API_BASE_URL=https://api.worldbank.org/v2
WB_API_TIMEOUT=30

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/etl.log

# ETL Configuration
BATCH_SIZE=100
MAX_RETRIES=3

# API Configuration (for the FastAPI server)
API_KEY=your_secret_api_key_here
API_HOST=0.0.0.0
API_PORT=8000

# Development settings
DEBUG=False
TESTING=False
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("⚠️  Please edit .env with your actual database credentials!")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_setup():
    """Test the setup."""
    print("🧪 Testing setup...")
    
    # Test if we can import the main modules
    try:
        sys.path.append('.')
        from agency_calculus.api.database import test_connection
        print("✅ Database module import successful")
        return True
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("=" * 50)
    print("1. Edit the .env file with your actual database credentials:")
    print("   - Set DATABASE_URL with your PostgreSQL connection string")
    print("   - Example: postgresql://username:password@localhost:5432/agency_monitor")
    print("")
    print("2. Create the PostgreSQL database:")
    print("   createdb agency_monitor")
    print("")
    print("3. Grant database permissions (as PostgreSQL superuser):")
    print("   psql -U postgres -c \"GRANT ALL PRIVILEGES ON DATABASE agency_monitor TO your_username;\"")
    print("")
    print("4. Apply the database schema:")
    print("   psql -U your_username -d agency_monitor -f database/schema.sql")
    print("")
    print("5. Test the database connection:")
    print("   python scripts/test_database.py")
    print("")
    print("6. Run the World Bank ETL (dry run first):")
    print("   python etl/etl_world_bank.py --dry-run")
    print("   python etl/etl_world_bank.py")
    print("")
    print("💡 Tip: Activate the virtual environment first:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")

def main():
    """Main setup function."""
    print("🚀 Agency Calculus Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check PostgreSQL
    if not check_postgresql():
        print("⚠️  PostgreSQL check failed. Please install PostgreSQL and try again.")
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Test setup
    if not test_setup():
        return False
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        sys.exit(0)