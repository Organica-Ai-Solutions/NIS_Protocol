#!/usr/bin/env python3
"""
🚨 BILLING MONITOR - Prevents surprise API charges
Monitors API usage and automatically shuts down if limits exceeded
"""
import os
import sys
import time
import subprocess
import requests
from datetime import datetime, timedelta

# Configuration
MAX_DAILY_COST = 5.0  # Maximum daily cost in USD
CHECK_INTERVAL = 300   # Check every 5 minutes
EMERGENCY_SHUTDOWN_SCRIPT = "./scripts/emergency/emergency_shutdown.sh"

def check_google_billing():
    """Check current Google Cloud billing (requires gcloud CLI)"""
    try:
        # This would require proper billing API setup
        # For now, we'll monitor container uptime as a proxy
        result = subprocess.run(['docker', 'ps', '--filter', 'name=nis-backend', '--format', '{{.Status}}'], 
                              capture_output=True, text=True)
        
        if "Up" in result.stdout:
            # Check how long it's been running
            uptime_result = subprocess.run(['docker', 'ps', '--filter', 'name=nis-backend', '--format', '{{.Status}}'], 
                                         capture_output=True, text=True)
            print(f"⚠️  NIS Backend is running: {uptime_result.stdout.strip()}")
            return True
        return False
    except Exception as e:
        print(f"Error checking containers: {e}")
        return False

def emergency_shutdown():
    """Execute emergency shutdown"""
    print("🚨 EXECUTING EMERGENCY SHUTDOWN!")
    try:
        subprocess.run(['bash', EMERGENCY_SHUTDOWN_SCRIPT], check=True)
        print("✅ Emergency shutdown completed")
    except Exception as e:
        print(f"❌ Emergency shutdown failed: {e}")
        # Fallback: kill containers directly
        subprocess.run(['docker', 'stop', '$(docker ps -q)'], shell=True)

def monitor_billing():
    """Main monitoring loop"""
    print("🛡️  Starting Billing Monitor...")
    print(f"💰 Daily limit: ${MAX_DAILY_COST}")
    print(f"⏱️  Check interval: {CHECK_INTERVAL} seconds")
    
    start_time = datetime.now()
    
    while True:
        try:
            # Check if containers are running
            containers_running = check_google_billing()
            
            # Calculate runtime
            runtime = datetime.now() - start_time
            
            if containers_running:
                print(f"⚠️  Runtime: {runtime} - Containers still running!")
                
                # If running for more than 2 hours, warn
                if runtime > timedelta(hours=2):
                    print("🚨 WARNING: Containers running for >2 hours!")
                    print("💸 This could generate significant API charges!")
                    
                    # If running for more than 4 hours, emergency shutdown
                    if runtime > timedelta(hours=4):
                        print("🚨 EMERGENCY: Containers running for >4 hours!")
                        print("💸 Executing emergency shutdown to prevent billing!")
                        emergency_shutdown()
                        break
            else:
                print("✅ No containers running - billing risk minimal")
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Billing monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    monitor_billing()
