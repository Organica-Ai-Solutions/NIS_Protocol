#!/usr/bin/env python3
"""
🚨 NIS Anti-Simulation Validator
CRITICAL INTEGRITY TOOL: Prevents simulation-based fake training results

This tool was created after discovering major integrity violations where training
scripts were simulating results instead of processing real data.

DETECTS:
- time.sleep() patterns in training scripts
- np.random fake metric generation  
- Missing real data file processing
- Hardcoded "success" results
- Fake progress indicators
- Missing actual ML training calls

ENFORCES:
- Real data file verification
- Actual processing time requirements
- Evidence of real computation
- Honest failure reporting
"""

import os
import re
import ast
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

class AntiSimulationValidator:
    """Validates training scripts are processing real data, not simulating results"""
    
    # Patterns that indicate simulation instead of real training
    SIMULATION_PATTERNS = {
        'sleep_simulation': [
            r'time\.sleep\s*\(\s*[0-9.]+\s*\)',
            r'sleep\s*\(\s*[0-9.]+\s*\)',
            r'time\.sleep\s*\(\s*\d+\s*\)',  # Fixed sleep durations
        ],
        'fake_random_metrics': [
            r'np\.random\.uniform\s*\(\s*0\.\d+\s*,\s*0\.\d+\s*\)',
            r'random\.uniform\s*\(\s*0\.\d+\s*,\s*0\.\d+\s*\)',
            r'np\.random\.random\s*\(\s*\)\s*\*\s*0\.\d+',
            r'fake_.*score\s*=',
            r'simulated_.*=',
        ],
        'hardcoded_success': [
            r'kan_interpretability\s*=\s*0\.\d+',
            r'physics_compliance\s*=\s*0\.9\d+',
            r'gll_score\s*=\s*-?\d+\.\d+',
            r'success.*=.*True',
            r'training_successful\s*=\s*True',
        ],
        'missing_real_processing': [
            r'# TODO: implement real training',
            r'# SIMULATION ONLY',
            r'# Fake.*implementation',
            r'pass\s*#.*simulation',
        ]
    }
    
    # Required patterns for legitimate training
    REQUIRED_REAL_PATTERNS = {
        'data_loading': [
            r'\.parquet',
            r'pd\.read_parquet',
            r'\.read_csv',
            r'load_dataset',
            r'DataLoader',
        ],
        'real_processing': [
            r'\.fit\(',
            r'\.train\(',
            r'optimizer\.step',
            r'loss\.backward',
            r'model\.forward',
            r'torch\.',
            r'sklearn\.',
        ],
        'file_verification': [
            r'os\.path\.exists',
            r'Path\(.*\)\.exists',
            r'file.*exists',
            r'\.stat\(',
        ]
    }
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'simulation_violations': [],
            'suspicious_files': [],
            'clean_files': [],
            'total_score': 0,
            'max_score': 100
        }
        
    def validate_all_training_scripts(self) -> Dict:
        """Validate all training scripts for simulation patterns"""
        
        print("🚨 NIS Anti-Simulation Validator")
        print("=" * 60)
        print(f"📁 Project: {self.project_path}")
        print(f"🎯 Mission: Detect fake training simulation patterns")
        print()
        
        # Find all training scripts
        training_scripts = self._find_training_scripts()
        
        print(f"🔍 Found {len(training_scripts)} training scripts")
        print()
        
        for script in training_scripts:
            self._validate_training_script(script)
            
        self._calculate_integrity_score()
        self._generate_report()
        
        return self.results
        
    def _find_training_scripts(self) -> List[Path]:
        """Find all potential training scripts"""
        
        training_patterns = [
            '**/train*.py',
            '**/training*.py',
            '**/*train*.py',
            '**/*training*.py',
            '**/agi*.py',
            '**/simple*.py',
        ]
        
        scripts = set()
        for pattern in training_patterns:
            scripts.update(self.project_path.glob(pattern))
            
        # Filter utilities directory specifically
        utilities_scripts = list(self.project_path.glob('utilities/*.py'))
        scripts.update(utilities_scripts)
        
        return sorted(list(scripts))
        
    def _validate_training_script(self, script_path: Path):
        """Validate individual training script for simulation patterns"""
        
        print(f"🔍 Validating: {script_path.name}")
        
        try:
            content = script_path.read_text(encoding='utf-8')
            
            # Check for simulation patterns
            violations = self._check_simulation_patterns(content, script_path)
            
            # Check for required real patterns
            real_indicators = self._check_real_patterns(content, script_path)
            
            # Calculate script score
            script_score = self._calculate_script_score(violations, real_indicators)
            
            # Determine status
            if violations:
                self.results['suspicious_files'].append({
                    'file': str(script_path),
                    'violations': violations,
                    'real_indicators': real_indicators,
                    'score': script_score
                })
                print(f"   ⚠️  SUSPICIOUS: {len(violations)} violations found")
                for violation in violations:
                    print(f"      🚨 {violation['type']}: {violation['pattern']}")
            else:
                self.results['clean_files'].append({
                    'file': str(script_path),
                    'real_indicators': real_indicators,
                    'score': script_score
                })
                print(f"   ✅ CLEAN: {len(real_indicators)} real processing indicators")
                
        except Exception as e:
            print(f"   ❌ ERROR reading {script_path.name}: {str(e)}")
            
        print()
        
    def _check_simulation_patterns(self, content: str, script_path: Path) -> List[Dict]:
        """Check for simulation patterns that indicate fake training"""
        
        violations = []
        
        for category, patterns in self.SIMULATION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    violations.append({
                        'type': category,
                        'pattern': match.group(0),
                        'line': line_num,
                        'severity': self._get_violation_severity(category)
                    })
                    
        return violations
        
    def _check_real_patterns(self, content: str, script_path: Path) -> List[Dict]:
        """Check for patterns indicating real data processing"""
        
        indicators = []
        
        for category, patterns in self.REQUIRED_REAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    indicators.append({
                        'type': category,
                        'pattern': match.group(0),
                        'line': line_num
                    })
                    
        return indicators
        
    def _get_violation_severity(self, category: str) -> str:
        """Get severity level for violation category"""
        
        severity_map = {
            'sleep_simulation': 'CRITICAL',
            'fake_random_metrics': 'CRITICAL', 
            'hardcoded_success': 'HIGH',
            'missing_real_processing': 'MEDIUM'
        }
        
        return severity_map.get(category, 'LOW')
        
    def _calculate_script_score(self, violations: List[Dict], real_indicators: List[Dict]) -> int:
        """Calculate integrity score for individual script"""
        
        base_score = 100
        
        # Deduct for violations
        for violation in violations:
            if violation['severity'] == 'CRITICAL':
                base_score -= 30
            elif violation['severity'] == 'HIGH':
                base_score -= 20
            elif violation['severity'] == 'MEDIUM':
                base_score -= 10
            else:
                base_score -= 5
                
        # Add for real indicators
        for indicator in real_indicators:
            base_score += 5
            
        return max(0, min(100, base_score))
        
    def _calculate_integrity_score(self):
        """Calculate overall integrity score"""
        
        total_files = len(self.results['suspicious_files']) + len(self.results['clean_files'])
        
        if total_files == 0:
            self.results['total_score'] = 0
            return
            
        # Calculate weighted score
        clean_score = len(self.results['clean_files']) * 100
        suspicious_penalty = 0
        
        for file_data in self.results['suspicious_files']:
            # Critical penalty for simulation violations
            critical_violations = sum(1 for v in file_data['violations'] if v['severity'] == 'CRITICAL')
            suspicious_penalty += critical_violations * 50
            
        total_possible = total_files * 100
        actual_score = max(0, clean_score - suspicious_penalty)
        
        self.results['total_score'] = int((actual_score / total_possible) * 100)
        
    def _generate_report(self):
        """Generate comprehensive validation report"""
        
        print("📊 ANTI-SIMULATION VALIDATION REPORT")
        print("=" * 60)
        print(f"🎯 Overall Integrity Score: {self.results['total_score']}/100")
        print()
        
        if self.results['suspicious_files']:
            print("🚨 SUSPICIOUS FILES (POTENTIAL SIMULATION):")
            print("-" * 50)
            for file_data in self.results['suspicious_files']:
                print(f"📁 {Path(file_data['file']).name}")
                print(f"   Score: {file_data['score']}/100")
                
                critical_violations = [v for v in file_data['violations'] if v['severity'] == 'CRITICAL']
                if critical_violations:
                    print("   🚨 CRITICAL VIOLATIONS:")
                    for violation in critical_violations:
                        print(f"      Line {violation['line']}: {violation['type']} - {violation['pattern']}")
                        
                print()
                
        if self.results['clean_files']:
            print("✅ CLEAN FILES (REAL PROCESSING DETECTED):")
            print("-" * 50)
            for file_data in self.results['clean_files']:
                print(f"📁 {Path(file_data['file']).name} (Score: {file_data['score']}/100)")
                
            print()
            
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        if self.results['total_score'] < 80:
            print("🚨 URGENT: Review suspicious files immediately")
            print("🔧 Remove all time.sleep() simulation patterns")
            print("🔧 Replace np.random metrics with real computation")
            print("🔧 Implement actual data loading and processing")
            
        if self.results['total_score'] >= 80:
            print("✅ Good integrity score - minimal simulation detected")
            print("🔧 Continue monitoring for future violations")
            
        print("\n📝 Report saved to: nis-integrity-toolkit/reports/anti_simulation_report.json")
        
        # Save detailed report
        report_path = self.project_path / "nis-integrity-toolkit" / "reports" / "anti_simulation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    """Run anti-simulation validation"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='NIS Anti-Simulation Validator')
    parser.add_argument('--project-path', default='.', help='Project path to validate')
    
    args = parser.parse_args()
    
    validator = AntiSimulationValidator(args.project_path)
    results = validator.validate_all_training_scripts()
    
    # Exit code based on integrity score
    if results['total_score'] < 70:
        print("\n❌ VALIDATION FAILED - Critical simulation violations detected")
        exit(1)
    elif results['total_score'] < 90:
        print("\n⚠️  VALIDATION WARNING - Some suspicious patterns detected")
        exit(0)
    else:
        print("\n✅ VALIDATION PASSED - No significant simulation violations")
        exit(0)

if __name__ == "__main__":
    main() 