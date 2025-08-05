"""
Test ERA5 data pipeline connection and basic functionality.
Downloads a small test dataset to verify credentials and setup.
"""

import logging
from pathlib import Path
from era5_pipeline import ERA5Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_era5_connection():
    """Test ERA5 connection with minimal data download."""
    try:
        # Initialize pipeline
        logger.info("Initializing ERA5 pipeline...")
        pipeline = ERA5Pipeline(data_dir='data/test')
        
        # Download small test dataset (1 day of data)
        logger.info("Testing data download...")
        test_data = pipeline.download_validation_data(2024, 1)
        
        # Verify data
        logger.info("\nVerifying downloaded data:")
        for dataset_name, dataset in test_data.items():
            logger.info(f"\n{dataset_name.upper()} DATASET:")
            logger.info(f"Variables: {list(dataset.variables)}")
            logger.info(f"Time range: {dataset.time.values[0]} to {dataset.time.values[-1]}")
            logger.info(f"Data shape: {dataset.dims}")
        
        logger.info("\n✅ ERA5 pipeline test successful!")
        logger.info("✅ Credentials verified")
        logger.info("✅ Data download working")
        logger.info("✅ NetCDF processing OK")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ERA5 pipeline test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_era5_connection()