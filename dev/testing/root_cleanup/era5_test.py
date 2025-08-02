"""
Simple ERA5 API connection test.
Run this directly from command line: python era5_test.py
"""

import os
import requests
import json
from dotenv import load_dotenv

def test_era5_connection():
    print("🔍 Testing ERA5 API connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('CDS_API_KEY', '')
    if ':' in api_key:
        print("⚠️  Removing UID prefix from API key...")
        api_key = api_key.split(':')[1]
    
    if not api_key:
        print("❌ ERROR: CDS_API_KEY not found in .env file")
        return False
    
    # Set up API configuration
    api_url = 'https://cds.climate.copernicus.eu/api/retrieve/v1/processes'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Test API connection
    try:
        print("🔄 Testing API connection...")
        # Just request a small amount of data (1 variable, 1 day, 1 time)
        url = f"{api_url}/reanalysis-era5-single-levels/execution"
        data = {
            'inputs': {
                'product_type': ['reanalysis'],
                'data_format': 'netcdf',
                'download_format': 'zip',
                'variable': ['2m_temperature'],
                'year': ['2024'],
                'month': ['01'],
                'day': ['01'],
                'time': ['12:00'],
                'area': [90, -180, -90, 180]
            }
        }
        
        print("📤 Sending API request...")
        response = requests.post(url, json=data, headers=headers)
        
        # Check response
        if response.status_code == 200:
            job_id = response.json().get('job_id')
            if job_id:
                print(f"✅ API connection successful! Job ID: {job_id}")
                print("🔄 Request submitted successfully. You can now check job status at:")
                print(f"   {api_url}/reanalysis-era5-single-levels/jobs/{job_id}")
                return True
            else:
                print(f"❌ API response missing job_id: {response.json()}")
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
    
    return False

if __name__ == "__main__":
    test_era5_connection()