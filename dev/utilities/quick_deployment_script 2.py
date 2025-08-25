#!/usr/bin/env python3
"""
Quick Model Deployment Script for Next Session
Optimized for fast deployment and testing with cost controls
"""

import subprocess
import json
import time
import sys

class QuickDeployment:
    def __init__(self):
        self.instance_id = None
        self.public_ip = None
        self.start_time = time.time()
        
    def deploy_instance(self, instance_type="g5.xlarge"):
        """Deploy AWS instance with Ollama ready for models"""
        print(f"🚀 Quick Deployment Starting...")
        print(f"💰 Instance Type: {instance_type}")
        print(f"📊 Expected Cost: ~${'0.60' if 'xlarge' in instance_type else '1.20'}/hour")
        
        user_data = '''#!/bin/bash
yum update -y
yum install -y docker git htop
curl -fsSL https://ollama.ai/install.sh | sh
systemctl start docker ollama
systemctl enable docker ollama
echo "Instance ready for model deployment" > /tmp/ready.txt
echo "$(date): Instance initialization complete" >> /var/log/deployment.log
'''
        
        cmd = [
            'aws', 'ec2', 'run-instances',
            '--image-id', 'ami-0c12c782c6284b66c',
            '--count', '1',
            '--instance-type', instance_type,
            '--key-name', 'organica-k2-limited-key',
            '--security-group-ids', 'sg-027be650fc4f5a5f4',
            '--region', 'us-east-1',
            '--tag-specifications', f'ResourceType=instance,Tags=[{{Key=Name,Value=organica-model-test-{int(time.time())}}}]',
            '--user-data', user_data
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                self.instance_id = data['Instances'][0]['InstanceId']
                print(f"✅ Instance launched: {self.instance_id}")
                return True
            else:
                print(f"❌ Launch failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def wait_for_running(self):
        """Wait for instance to be running and get IP"""
        print("⏳ Waiting for instance to start...")
        
        for i in range(30):  # 5 minutes max
            cmd = [
                'aws', 'ec2', 'describe-instances',
                '--instance-ids', self.instance_id,
                '--query', 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]',
                '--output', 'text',
                '--region', 'us-east-1'
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    state, ip = result.stdout.strip().split('\t')
                    if state == 'running' and ip != 'None':
                        self.public_ip = ip
                        print(f"✅ Instance running: {ip}")
                        return True
                    elif state == 'running':
                        print(f"🔄 Running but no IP yet...")
                    else:
                        print(f"🔄 State: {state}")
            except Exception as e:
                print(f"⚠️ Check failed: {e}")
            
            time.sleep(10)
        
        print("❌ Timeout waiting for instance")
        return False
    
    def test_readiness(self):
        """Test if instance is ready for model deployment"""
        if not self.public_ip:
            return False
            
        print("🔍 Testing instance readiness...")
        
        # Test SSH connectivity
        ssh_cmd = [
            'ssh', '-i', '~/.ssh/organica-k2-limited-key.pem',
            '-o', 'ConnectTimeout=10',
            '-o', 'StrictHostKeyChecking=no',
            f'ec2-user@{self.public_ip}',
            'cat /tmp/ready.txt'
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and 'ready' in result.stdout:
                print("✅ Instance ready for model deployment!")
                return True
            else:
                print("⏳ Still initializing...")
                return False
        except Exception as e:
            print(f"⚠️ SSH test failed: {e}")
            return False
    
    def deploy_model(self, model_name="gpt-oss:120b"):
        """Deploy specified model to the instance"""
        if not self.public_ip:
            print("❌ No instance IP available")
            return False
            
        print(f"📦 Deploying model: {model_name}")
        
        # For now, we'll assume the model needs to be pulled
        # In practice, you might copy from local Ollama
        ssh_cmd = [
            'ssh', '-i', '~/.ssh/organica-k2-limited-key.pem',
            f'ec2-user@{self.public_ip}',
            f'ollama pull {model_name}'
        ]
        
        try:
            print("⏳ This may take 10-30 minutes depending on model size...")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            if result.returncode == 0:
                print(f"✅ Model {model_name} deployed successfully!")
                return True
            else:
                print(f"❌ Model deployment failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Model deployment timed out")
            return False
        except Exception as e:
            print(f"❌ Model deployment error: {e}")
            return False
    
    def terminate_instance(self):
        """Terminate instance to save costs"""
        if not self.instance_id:
            print("No instance to terminate")
            return
            
        elapsed = (time.time() - self.start_time) / 3600
        cost = elapsed * (0.60 if 'xlarge' in str(self.instance_id) else 1.20)
        
        print(f"💰 Session time: {elapsed:.2f} hours")
        print(f"💰 Estimated cost: ${cost:.2f}")
        print(f"🛑 Terminating instance: {self.instance_id}")
        
        cmd = ['aws', 'ec2', 'terminate-instances', '--instance-ids', self.instance_id, '--region', 'us-east-1']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Instance termination initiated")
            else:
                print(f"⚠️ Termination failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Termination error: {e}")

def main():
    """Main deployment workflow"""
    if len(sys.argv) > 1:
        instance_type = sys.argv[1]
    else:
        instance_type = "g5.xlarge"  # Default to smaller instance
    
    deployer = QuickDeployment()
    
    try:
        # Deploy instance
        if not deployer.deploy_instance(instance_type):
            print("❌ Instance deployment failed")
            return
        
        # Wait for running state
        if not deployer.wait_for_running():
            print("❌ Instance failed to start")
            deployer.terminate_instance()
            return
        
        # Wait for readiness
        ready = False
        for i in range(6):  # Wait up to 6 minutes for initialization
            if deployer.test_readiness():
                ready = True
                break
            time.sleep(60)
        
        if not ready:
            print("❌ Instance not ready after 6 minutes")
            deployer.terminate_instance()
            return
        
        print("\n🎯 Instance ready for model deployment!")
        print(f"📍 SSH: ssh -i ~/.ssh/organica-k2-limited-key.pem ec2-user@{deployer.public_ip}")
        print(f"🌐 IP: {deployer.public_ip}")
        print(f"🆔 ID: {deployer.instance_id}")
        
        # Ask user what to do next
        print("\nOptions:")
        print("1. Deploy GPT OSS 120B model")
        print("2. Deploy Kimi K2 model") 
        print("3. Keep running for manual testing")
        print("4. Terminate now")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            deployer.deploy_model("gpt-oss:120b")
        elif choice == "2":
            deployer.deploy_model("huihui_ai/kimi-k2:latest")
        elif choice == "3":
            print("Instance running - remember to terminate when done!")
            print(f"Terminate with: aws ec2 terminate-instances --instance-ids {deployer.instance_id} --region us-east-1")
            return
        
        # Always terminate unless user chose to keep running
        if choice != "3":
            input("Press Enter to terminate instance...")
            deployer.terminate_instance()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted - terminating instance for safety")
        deployer.terminate_instance()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        deployer.terminate_instance()

if __name__ == "__main__":
    main()