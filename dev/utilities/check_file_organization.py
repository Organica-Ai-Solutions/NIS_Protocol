#!/usr/bin/env python3
"""
🔍 NIS Protocol File Organization Compliance Checker

Ensures that files are organized according to the strict file organization rules.
Helps maintain a clean, professional root directory structure.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Root directory allowed files (must match FILE_ORGANIZATION_RULES.md)
ALLOWED_ROOT_FILES = {
    'main.py',
    'requirements.txt',
    'docker-compose.yml', 
    'Dockerfile',
    'README.md',
    'LICENSE',
    '.gitignore',
    '.dockerignore',
    '.gitattributes',
    '.cursorrules',
    '.nojekyll',
    'CNAME',
    'start.sh',
    'stop.sh',
    'reset.sh',
    'NIS_Protocol_v3_COMPLETE_Postman_Collection.json',
    '.env',
    '.env.example'
}

# Allowed root directories
ALLOWED_ROOT_DIRS = {
    'dev', 'src', 'system', 'scripts', 'logs', 'cache', 'data',
    'config', 'benchmarks', 'assets', 'static', 'models',
    'monitoring', 'private', 'backup', '.git', '.github', '.venv',
    'endpoint_fixes_backup'  # Temporary backup directory
}

# File patterns that should never be in root
PROHIBITED_PATTERNS = [
    'test_*.py',
    '*_test.py', 
    '*_utility.py',
    '*_helper.py',
    '*_report.json',
    '*_summary.md',
    '*.tmp',
    '*~',
    '*.backup',
    '*_backup.*',
    '*.log',
    'fix_*.py',
    'generate_*.py',
    'commit_message.txt',
    'server.log'
]

class FileOrganizationChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.violations = []
        self.warnings = []
        self.compliance_score = 0
        
    def check_compliance(self) -> Dict:
        """Run complete file organization compliance check"""
        print("🔍 Running NIS Protocol File Organization Compliance Check...")
        print(f"📂 Project Root: {self.project_root}")
        print("=" * 70)
        
        # Check root directory compliance
        self._check_root_directory()
        
        # Check file placement compliance
        self._check_file_placement()
        
        # Calculate compliance score
        self._calculate_compliance_score()
        
        # Generate report
        return self._generate_report()
    
    def _check_root_directory(self):
        """Check that only allowed files are in root directory"""
        print("📋 Checking root directory compliance...")
        
        root_items = list(self.project_root.iterdir())
        
        for item in root_items:
            if item.is_file():
                if item.name not in ALLOWED_ROOT_FILES:
                    self.violations.append({
                        'type': 'ROOT_FILE_VIOLATION',
                        'file': item.name,
                        'location': str(item),
                        'severity': 'HIGH',
                        'message': f"File '{item.name}' is not allowed in root directory",
                        'suggested_location': self._suggest_file_location(item.name)
                    })
                    print(f"  ❌ {item.name} - NOT ALLOWED IN ROOT")
                else:
                    print(f"  ✅ {item.name} - OK")
                    
            elif item.is_dir():
                if item.name not in ALLOWED_ROOT_DIRS:
                    self.warnings.append({
                        'type': 'ROOT_DIR_WARNING',
                        'directory': item.name,
                        'location': str(item),
                        'severity': 'MEDIUM',
                        'message': f"Directory '{item.name}' may not belong in root",
                        'action': 'Review if this directory should be reorganized'
                    })
                    print(f"  ⚠️  {item.name}/ - REVIEW NEEDED")
                else:
                    print(f"  ✅ {item.name}/ - OK")
    
    def _check_file_placement(self):
        """Check that files are in correct subdirectories"""
        print("\n📁 Checking file placement compliance...")
        
        # Check for misplaced test files
        self._check_test_files()
        
        # Check for misplaced utility files
        self._check_utility_files()
        
        # Check for misplaced reports
        self._check_report_files()
        
        # Check for temporary files
        self._check_temporary_files()
    
    def _check_test_files(self):
        """Check that test files are in proper testing directories"""
        test_files = []
        skip_dirs = {'.venv', 'venv', 'env', 'models/bitnet/venv', '.git', '__pycache__', 'node_modules'}
        
        # Find test files outside of dev/testing/
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.project_root)
            relative_root_str = str(relative_root).replace('\\', '/')
            
            # Skip virtual environments and external dependencies
            if any(skip_dir in relative_root_str for skip_dir in skip_dirs):
                continue
                
            # Skip the proper testing directories
            if 'dev/testing' in relative_root_str or 'tests' in relative_root_str:
                continue
                
            for file in files:
                if (file.startswith('test_') and file.endswith('.py')) or \
                   (file.endswith('_test.py')):
                    file_path = Path(root) / file
                    test_files.append(file_path)
        
        for test_file in test_files:
            relative_path = test_file.relative_to(self.project_root)
            relative_str = str(relative_path).replace('\\', '/')
            
            if relative_str.startswith('.'):
                continue  # Skip hidden directories
                
            # Only flag files in our project directories
            if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/']):
                continue
                
            self.violations.append({
                'type': 'MISPLACED_TEST_FILE',
                'file': test_file.name,
                'location': str(relative_path),
                'severity': 'HIGH',
                'message': f"Test file '{test_file.name}' should be in dev/testing/",
                'suggested_location': 'dev/testing/root_cleanup/'
            })
            print(f"  ❌ {relative_path} - Should be in dev/testing/")
    
    def _check_utility_files(self):
        """Check that utility files are in proper directories"""
        utility_patterns = ['*_utility.py', 'fix_*.py', 'generate_*.py', '*_helper.py']
        skip_dirs = {'.venv', 'venv', 'env', 'models/bitnet/venv', '.git', '__pycache__', 'node_modules'}
        
        for pattern in utility_patterns:
            for file_path in self.project_root.rglob(pattern):
                relative_path = file_path.relative_to(self.project_root)
                relative_str = str(relative_path).replace('\\', '/')
                
                # Skip virtual environments and external dependencies
                if any(skip_dir in relative_str for skip_dir in skip_dirs):
                    continue
                
                # Skip if already in proper location
                if relative_str.startswith('dev/utilities') or \
                   relative_str.startswith('system/utilities'):
                    continue
                    
                # Skip hidden directories
                if relative_str.startswith('.'):
                    continue
                
                # Only flag files in our project directories
                if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/']):
                    continue
                
                self.violations.append({
                    'type': 'MISPLACED_UTILITY_FILE',
                    'file': file_path.name,
                    'location': str(relative_path),
                    'severity': 'MEDIUM',
                    'message': f"Utility file '{file_path.name}' should be in utilities directory",
                    'suggested_location': 'dev/utilities/ or system/utilities/'
                })
                print(f"  ❌ {relative_path} - Should be in utilities/")
    
    def _check_report_files(self):
        """Check that report files are in proper directories"""
        report_patterns = ['*_report.json', '*_summary.md', 'audit-*.json']
        skip_dirs = {'.venv', 'venv', 'env', 'models/bitnet/venv', '.git', '__pycache__', 'node_modules'}
        
        for pattern in report_patterns:
            for file_path in self.project_root.rglob(pattern):
                relative_path = file_path.relative_to(self.project_root)
                relative_str = str(relative_path).replace('\\', '/')
                
                # Skip virtual environments and external dependencies
                if any(skip_dir in relative_str for skip_dir in skip_dirs):
                    continue
                
                # Skip if already in proper location
                if relative_str.startswith('system/utilities/reports'):
                    continue
                    
                # Skip hidden directories
                if relative_str.startswith('.'):
                    continue
                
                # Only flag files in our project directories
                if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/']):
                    continue
                
                self.violations.append({
                    'type': 'MISPLACED_REPORT_FILE',
                    'file': file_path.name,
                    'location': str(relative_path),
                    'severity': 'MEDIUM',
                    'message': f"Report file '{file_path.name}' should be in reports directory",
                    'suggested_location': 'system/utilities/reports/'
                })
                print(f"  ❌ {relative_path} - Should be in system/utilities/reports/")
    
    def _check_temporary_files(self):
        """Check for temporary files that should be cleaned up"""
        temp_patterns = ['*.tmp', '*~', 'temp_*', '*.backup']
        
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                relative_path = file_path.relative_to(self.project_root)
                
                # Skip if in proper temp location
                if str(relative_path).startswith('cache/') or \
                   str(relative_path).startswith('.git/'):
                    continue
                
                self.violations.append({
                    'type': 'TEMPORARY_FILE',
                    'file': file_path.name,
                    'location': str(relative_path),
                    'severity': 'LOW',
                    'message': f"Temporary file '{file_path.name}' should be cleaned up",
                    'suggested_location': 'DELETE or move to cache/'
                })
                print(f"  ⚠️  {relative_path} - Temporary file should be cleaned")
    
    def _suggest_file_location(self, filename: str) -> str:
        """Suggest proper location for a misplaced file"""
        if filename.startswith('test_') or filename.endswith('_test.py'):
            return 'dev/testing/root_cleanup/'
        elif any(pattern in filename for pattern in ['utility', 'fix_', 'generate_']):
            return 'dev/utilities/'
        elif filename.endswith('_report.json') or filename.endswith('_summary.md'):
            return 'system/utilities/reports/'
        elif filename.endswith('.log'):
            return 'logs/archived/'
        elif filename.endswith('.md') and filename != 'README.md':
            return 'system/docs/ or dev/documentation/'
        elif filename.startswith('install_') or filename.endswith('.sh'):
            return 'scripts/'
        else:
            return 'Review FILE_ORGANIZATION_RULES.md for guidance'
    
    def _calculate_compliance_score(self):
        """Calculate overall compliance score"""
        total_issues = len(self.violations) + len(self.warnings)
        
        if total_issues == 0:
            self.compliance_score = 100
        else:
            # Weight violations more heavily than warnings
            violation_weight = len(self.violations) * 2
            warning_weight = len(self.warnings) * 1
            
            # Calculate score (max deduction of 100 points)
            deduction = min(100, violation_weight * 5 + warning_weight * 2)
            self.compliance_score = max(0, 100 - deduction)
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'compliance_score': self.compliance_score,
            'summary': {
                'total_violations': len(self.violations),
                'total_warnings': len(self.warnings),
                'high_severity': len([v for v in self.violations if v['severity'] == 'HIGH']),
                'medium_severity': len([v for v in self.violations if v['severity'] == 'MEDIUM']),
                'low_severity': len([v for v in self.violations if v['severity'] == 'LOW'])
            },
            'violations': self.violations,
            'warnings': self.warnings,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if len(self.violations) > 0:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Move violated files to proper locations")
        
        if any(v['type'] == 'ROOT_FILE_VIOLATION' for v in self.violations):
            recommendations.append("📁 Clean root directory by moving prohibited files")
            
        if any(v['type'] == 'MISPLACED_TEST_FILE' for v in self.violations):
            recommendations.append("🧪 Move all test files to dev/testing/ directories")
            
        if any(v['type'] == 'TEMPORARY_FILE' for v in self.violations):
            recommendations.append("🧹 Clean up temporary files and backup files")
            
        if len(self.warnings) > 0:
            recommendations.append("⚠️  Review warned items for potential reorganization")
            
        if self.compliance_score < 80:
            recommendations.append("📋 Review system/docs/FILE_ORGANIZATION_RULES.md for guidance")
            
        if len(recommendations) == 0:
            recommendations.append("🎉 Excellent! File organization is fully compliant")
            
        return recommendations

def main():
    """Main execution function"""
    checker = FileOrganizationChecker()
    report = checker.check_compliance()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 COMPLIANCE REPORT SUMMARY")
    print("=" * 70)
    print(f"🎯 Compliance Score: {report['compliance_score']}/100")
    print(f"❌ Total Violations: {report['summary']['total_violations']}")
    print(f"⚠️  Total Warnings: {report['summary']['total_warnings']}")
    
    if report['compliance_score'] >= 95:
        print("🎉 EXCELLENT: Outstanding file organization!")
    elif report['compliance_score'] >= 80:
        print("✅ GOOD: File organization mostly compliant")
    elif report['compliance_score'] >= 60:
        print("⚠️  NEEDS IMPROVEMENT: Several issues need attention")
    else:
        print("🚨 CRITICAL: Major file organization violations!")
    
    # Print recommendations
    if report['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save detailed report
    report_path = Path('system/utilities/reports/file_organization_compliance.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved: {report_path}")
    
    # Exit with appropriate code
    if report['compliance_score'] < 80:
        print("\n🚨 COMPLIANCE FAILURE: Score below 80. Please fix violations before proceeding.")
        sys.exit(1)
    else:
        print("\n✅ COMPLIANCE PASSED: File organization meets standards.")
        sys.exit(0)

if __name__ == "__main__":
    main()