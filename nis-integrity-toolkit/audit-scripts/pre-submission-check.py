#!/usr/bin/env python3
"""
⚡ NIS Pre-Submission Integrity Check
Fast 5-minute validation before any release or submission
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

class PreSubmissionChecker:
    """Fast integrity check for imminent releases"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.failures = []
        self.warnings = []
        self.passes = []
    
    def run_check(self) -> bool:
        """Run fast pre-submission check. Returns True if ready for submission."""
        
        print("⚡ NIS Pre-Submission Integrity Check")
        print("=" * 50)
        print(f"📁 Project: {self.project_path.absolute()}")
        print(f"⏰ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Critical checks (must pass)
        self._check_hardcoded_values()
        self._check_hype_language()
        self._check_unsubstantiated_claims()
        self._check_missing_evidence()
        self._check_basic_documentation()
        
        # Summary
        self._print_summary()
        
        # Pass/fail decision
        has_failures = len(self.failures) > 0
        
        if has_failures:
            print("\n❌ SUBMISSION BLOCKED - Fix issues above before release")
            return False
        else:
            print("\n✅ SUBMISSION APPROVED - Ready for release")
            return True
    
    def _check_hardcoded_values(self):
        """Check for hardcoded performance metrics"""
        print("🔍 Checking for hardcoded performance values...")
        
        py_files = list(self.project_path.rglob("*.py"))
        
        # Critical hardcoded patterns
        critical_patterns = [
            (r'consciousness_level\s*=\s*0\.\d+', 'consciousness_level'),
            (r'interpretability\s*=\s*0\.\d+', 'interpretability'),
            (r'physics_compliance\s*=\s*0\.\d+', 'physics_compliance'),
            (r'accuracy\s*=\s*0\.\d+', 'accuracy'),
            (r'confidence\s*=\s*0\.\d+', 'confidence'),
            (r'performance\s*=\s*0\.\d+', 'performance')
        ]
        
        found_hardcoded = False
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, name in critical_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        self.failures.append(f"Hardcoded {name} in {py_file.name}: {matches[0]}")
                        found_hardcoded = True
                        
            except Exception as e:
                self.warnings.append(f"Could not read {py_file.name}: {str(e)}")
        
        if not found_hardcoded:
            self.passes.append("No hardcoded performance values found")
    
    def _check_hype_language(self):
        """Check for unsupported hype language"""
        print("🎯 Checking for hype language...")
        
        # Critical hype terms that require evidence
        hype_terms = [
            'advanced multi-agent system',
            'KAN interpretability-driven',
            'innovative',
            'improvement',
            'low error rate',
            'optimized accuracy',
            'novel',
            'mathematically-inspired'
        ]
        
        readme_files = list(self.project_path.rglob("README*.md"))
        
        found_hype = False
        
        for readme in readme_files:
            try:
                content = readme.read_text(encoding='utf-8').lower()
                
                for term in hype_terms:
                    if term in content:
                        self.failures.append(f"Unsupported hype language in {readme.name}: '{term}'")
                        found_hype = True
                        
            except Exception as e:
                self.warnings.append(f"Could not read {readme.name}: {str(e)}")
        
        if not found_hype:
            self.passes.append("No unsupported hype language found")
    
    def _check_unsubstantiated_claims(self):
        """Check for specific technical claims without evidence"""
        print("📊 Checking for unsubstantiated claims...")
        
        # Specific claims that need evidence
        claim_patterns = [
            (r'(\d+\.?\d*)% (accuracy|interpretability|performance|compliance)', 'percentage claim'),
            (r'(\d+\.?\d*)(x|times) (faster|better|more accurate)', 'comparison claim'),
            (r'(zero|no) (hallucination|error|bias)', 'zero-error claim'),
            (r'(sub-second|millisecond) (processing|response)', 'speed claim')
        ]
        
        doc_files = list(self.project_path.rglob("*.md"))
        
        found_claims = False
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8')
                
                for pattern, claim_type in claim_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.failures.append(f"Unsubstantiated {claim_type} in {doc_file.name}: {matches[0]}")
                        found_claims = True
                        
            except Exception as e:
                self.warnings.append(f"Could not read {doc_file.name}: {str(e)}")
        
        if not found_claims:
            self.passes.append("No unsubstantiated technical claims found")
    
    def _check_missing_evidence(self):
        """Check for missing evidence for claims"""
        print("🧪 Checking for missing evidence...")
        
        # Look for test/benchmark files
        test_files = list(self.project_path.rglob("*test*.py"))
        benchmark_files = list(self.project_path.rglob("*benchmark*.py"))
        
        # Look for performance claims in documentation
        doc_files = list(self.project_path.rglob("*.md"))
        has_performance_claims = False
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8').lower()
                if any(term in content for term in ['performance', 'speed', 'accuracy', 'faster', 'better']):
                    has_performance_claims = True
                    break
            except:
                continue
        
        if has_performance_claims and not test_files and not benchmark_files:
            self.failures.append("Performance claims made but no test/benchmark files found")
        elif not has_performance_claims or (test_files or benchmark_files):
            self.passes.append("Evidence availability matches claims")
    
    def _check_basic_documentation(self):
        """Check for basic documentation requirements"""
        print("📝 Checking basic documentation...")
        
        # Check for README
        readme_files = list(self.project_path.rglob("README*.md"))
        if not readme_files:
            self.failures.append("No README.md file found")
        else:
            # Check README content
            readme = readme_files[0]
            try:
                content = readme.read_text(encoding='utf-8')
                
                # Check for basic sections
                required_sections = ['installation', 'usage', 'getting started']
                missing_sections = []
                
                for section in required_sections:
                    if section not in content.lower():
                        missing_sections.append(section)
                
                if missing_sections:
                    self.warnings.append(f"README missing basic sections: {', '.join(missing_sections)}")
                else:
                    self.passes.append("README contains basic required sections")
                    
            except Exception as e:
                self.warnings.append(f"Could not analyze README: {str(e)}")
        
        # Check for requirements.txt or similar
        dep_files = list(self.project_path.glob("requirements*.txt")) + list(self.project_path.glob("pyproject.toml"))
        if not dep_files:
            self.warnings.append("No dependency file found (requirements.txt, pyproject.toml)")
        else:
            self.passes.append("Dependency management file found")
    
    def _print_summary(self):
        """Print check summary"""
        print()
        print("=" * 50)
        print("📋 PRE-SUBMISSION CHECK SUMMARY")
        print("=" * 50)
        
        # Failures (blocking)
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}) - MUST FIX:")
            for i, failure in enumerate(self.failures, 1):
                print(f"   {i}. {failure}")
        
        # Warnings (should fix)
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}) - SHOULD FIX:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Passes (good!)
        if self.passes:
            print(f"\n✅ PASSES ({len(self.passes)}):")
            for i, pass_item in enumerate(self.passes, 1):
                print(f"   {i}. {pass_item}")
        
        print()
    
def main():
    """Main entry point"""
    
    # Check if we're in a git repository
    project_path = "."
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    
    checker = PreSubmissionChecker(project_path)
    success = checker.run_check()
    
    # Print quick fix suggestions for common issues
    if not success:
        print("\n🔧 QUICK FIXES:")
        print("1. Replace hardcoded values with actual calculations")
        print("2. Remove unsupported hype language from documentation")
        print("3. Provide evidence for all technical claims")
        print("4. Add benchmark/test files for performance claims")
        print("5. Ensure README has basic sections")
        print()
        print("💡 Use full audit for detailed analysis:")
        print("   python audit-scripts/full-audit.py --project-path . --output-report")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 