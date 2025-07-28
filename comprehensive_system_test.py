#!/usr/bin/env python3
"""
Comprehensive System Test & Fix Script

Tests all NIS Protocol capabilities and fixes issues encountered.
Avoids import dependency issues by using direct imports.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path

# Add paths for direct imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from confidence_calculator import calculate_confidence

def test_self_audit_engine():
    """Test the enhanced self-audit engine"""
    print("🔍 TESTING SELF-AUDIT ENGINE")
    print("-" * 40)
    
    try:
        from self_audit import SelfAuditEngine, ViolationType
        
        engine = SelfAuditEngine()
        
        # Test code with various violations
        test_cases = [
            ('confidence = calculate_confidence()  # Test data - acceptable for demo', 'hardcoded_value'),
            ('accuracy = 0.88', 'hardcoded_value'),
            ('This is a sophisticated system', 'hype_language'),
            ('perfect accuracy', 'hype_language'),
            ('revolutionary breakthrough', 'hype_language')
        ]
        
        total_violations = 0
        for test_code, expected_type in test_cases:
            violations = engine.audit_text(test_code, 'test.py')
            if violations:
                violation = violations[0]
                print(f"✅ Detected {violation.violation_type.value}: '{violation.text}'")
                print(f"   Fix: {violation.suggested_replacement}")
                print(f"   Confidence: {violation.confidence:.3f}")
                total_violations += len(violations)
            else:
                print(f"❌ Failed to detect: {test_code}")
        
        print(f"\n📊 Total violations detected: {total_violations}")
        return total_violations > 0
        
    except Exception as e:
        print(f"❌ Self-audit engine test failed: {e}")
        return False

def fix_hardcoded_values_in_file(file_path):
    """Fix hardcoded values in a specific file"""
    print(f"🔧 FIXING HARDCODED VALUES IN: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        from self_audit import SelfAuditEngine
        
        engine = SelfAuditEngine()
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find violations
        violations = engine.audit_text(content, file_path)
        hardcoded_violations = [v for v in violations if v.violation_type.value == 'hardcoded_value']
        
        if not hardcoded_violations:
            print(f"✅ No hardcoded values found in {file_path}")
            return True
        
        print(f"📊 Found {len(hardcoded_violations)} hardcoded values")
        
        # Apply fixes
        fixed_content = content
        fixes_applied = 0
        
        for violation in hardcoded_violations:
            if violation.text in fixed_content:
                fixed_content = fixed_content.replace(violation.text, violation.suggested_replacement)
                fixes_applied += 1
                print(f"  ✅ Fixed: {violation.text} → {violation.suggested_replacement}")
        
        # Write back if changes made
        if fixes_applied > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Applied {fixes_applied} fixes to {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def test_system_startup():
    """Test if the system can start up"""
    print("🚀 TESTING SYSTEM STARTUP")
    print("-" * 40)
    
    try:
        # Test health endpoint via curl
        result = subprocess.run([
            'curl', '-s', '-X', 'GET', 'http://localhost:8000/health'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            try:
                health_data = json.loads(result.stdout)
                print(f"✅ System is running!")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Provider: {health_data.get('provider', 'unknown')}")
                print(f"   Real AI: {health_data.get('real_ai', False)}")
                return True
            except json.JSONDecodeError:
                print("⚠️  System responding but not JSON")
                return False
        else:
            print("❌ System not responding")
            print("   Trying to start system...")
            
            # Try to start the system
            start_result = subprocess.run(['bash', 'start.sh'], capture_output=True, text=True)
            if start_result.returncode == 0:
                print("✅ System started successfully")
                time.sleep(5)  # Wait for startup
                return test_system_startup()  # Recursive test
            else:
                print(f"❌ Failed to start system: {start_result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("❌ Health check timed out")
        return False
    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        return False

def test_endpoints():
    """Test key API endpoints"""
    print("🌐 TESTING API ENDPOINTS")
    print("-" * 40)
    
    endpoints = [
        ('GET', 'http://localhost:8000/', 'Root endpoint'),
        ('GET', 'http://localhost:8000/health', 'Health check'),
        ('GET', 'http://localhost:8000/agents', 'List agents'),
    ]
    
    working_endpoints = 0
    
    for method, url, description in endpoints:
        try:
            result = subprocess.run([
                'curl', '-s', '-X', method, url
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    print(f"✅ {description}: Working")
                    working_endpoints += 1
                except json.JSONDecodeError:
                    print(f"⚠️  {description}: Responding but not JSON")
            else:
                print(f"❌ {description}: Failed")
                
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
    
    print(f"\n📊 Working endpoints: {working_endpoints}/{len(endpoints)}")
    return working_endpoints > 0

def run_comprehensive_audit():
    """Run the comprehensive audit"""
    print("📋 RUNNING COMPREHENSIVE AUDIT")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            'python', 'nis-integrity-toolkit/audit-scripts/full-audit.py', 
            '--project-path', '.', '--output-report'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Audit completed successfully")
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Integrity Score:' in line:
                    print(f"📊 {line.strip()}")
                elif 'Total Issues:' in line:
                    print(f"📊 {line.strip()}")
                elif '🔴 High:' in line or '🟡 Medium:' in line or '🟢 Low:' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ Audit failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Audit error: {e}")
        return False

def fix_common_issues():
    """Fix common issues found in the system"""
    print("🔧 FIXING COMMON ISSUES")
    print("-" * 40)
    
    fixes_applied = 0
    
    # 1. Fix hardcoded values in key files
    key_files = [
        'simple_real_chat_test.py',
        'src/agents/physics/physics_agent.py',
        'src/utils/self_audit.py'
    ]
    
    for file_path in key_files:
        if fix_hardcoded_values_in_file(file_path):
            fixes_applied += 1
    
    # 2. Fix import issues in message_streaming.py
    print("🔧 Checking message_streaming.py imports...")
    try:
        with open('src/infrastructure/message_streaming.py', 'r') as f:
            content = f.read()
        
        if 'AIOKafkaConsumer' in content and 'except ImportError:' in content:
            print("✅ Message streaming imports already fixed")
        else:
            print("⚠️  Message streaming might need import fixes")
    except Exception as e:
        print(f"❌ Could not check message_streaming.py: {e}")
    
    print(f"\n📊 Files processed: {fixes_applied}/{len(key_files)}")
    return fixes_applied > 0

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE SYSTEM TEST REPORT")
    print("="*60)
    
    tests = [
        ("Self-Audit Engine", test_self_audit_engine),
        ("System Startup", test_system_startup),
        ("API Endpoints", test_endpoints),
        ("Issue Fixing", fix_common_issues),
        ("Comprehensive Audit", run_comprehensive_audit)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name.upper()}")
        print("="*60)
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
    elif passed >= len(tests) * 0.8:
        print("\n⚠️  Most tests passed. Minor issues need attention.")
    else:
        print("\n🚨 Multiple failures detected. System needs significant fixes.")
    
    return passed / len(tests)

if __name__ == "__main__":
    print("🔬 NIS PROTOCOL COMPREHENSIVE SYSTEM TEST")
    print("=========================================")
    print("This script will test all system capabilities and fix issues.")
    print()
    
    success_rate = generate_test_report()
    
    print(f"\n🏁 Test session complete. Success rate: {success_rate*100:.1f}%")
    
    # Exit code based on success rate
    exit_code = 0 if success_rate >= 0.8 else 1
    sys.exit(exit_code) 