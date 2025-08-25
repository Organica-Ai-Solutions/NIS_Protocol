#!/usr/bin/env python3
"""
🔍 NIS Engineering Integrity Audit Tool
Comprehensive repository audit for technical accuracy and professional credibility
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import yaml

class NISIntegrityAuditor:
    """Comprehensive engineering integrity auditor for NIS protocol systems"""
    
    # Hype terms that require evidence
    HYPE_TERMS = {
        'consciousness': ['consciousness', 'conscious', 'self-aware', 'sentient'],
        'agi': ['agi', 'artificial general intelligence', 'general intelligence'],
        'revolutionary': ['revolutionary', 'breakthrough', 'unprecedented', 'world-first', 'first-ever'],
        'perfection': ['zero error', 'perfect', '100% accurate', 'flawless', 'bulletproof'],
        'quantum': ['quantum-inspired', 'quantum computing', 'quantum neural'],
        'interpretability': ['97.3%', '95%+', 'full interpretability', 'complete transparency']
    }
    
    # Technical claims that need code backing
    TECHNICAL_CLAIMS = [
        r'(\d+\.?\d*)% (accuracy|interpretability|performance|compliance)',
        r'(\d+\.?\d*)(x|times) (faster|better|more accurate)',
        r'(zero|no) (hallucination|error|bias)',
        r'real[- ]?time (processing|analysis|monitoring)',
        r'(sub[- ]?second|millisecond) (processing|response|analysis)'
    ]
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.results = {
            'audit_timestamp': datetime.now().isoformat(),
            'project_path': str(self.project_path),
            'issues': [],
            'metrics': {},
            'recommendations': [],
            'integrity_score': 0
        }
    
    def run_full_audit(self) -> Dict:
        """Run comprehensive integrity audit"""
        print("🔍 Starting NIS Engineering Integrity Audit...")
        print(f"📁 Project: {self.project_path}")
        print("=" * 60)
        
        # Core audit steps
        self._audit_documentation()
        self._audit_code_claims()
        self._audit_performance_claims()
        self._audit_architecture_alignment()
        self._audit_language_hype()
        self._calculate_integrity_score()
        self._generate_recommendations()
        
        print(f"\n🎯 Audit Complete! Integrity Score: {self.results['integrity_score']}/100")
        return self.results
    
    def _audit_documentation(self):
        """Audit all documentation for accuracy"""
        print("\n📝 Auditing Documentation...")
        
        docs = self._find_documentation_files()
        
        for doc_file in docs:
            try:
                content = doc_file.read_text(encoding='utf-8')
                
                # Check for unsubstantiated claims
                for category, terms in self.HYPE_TERMS.items():
                    for term in terms:
                        if re.search(term, content, re.IGNORECASE):
                            self._add_issue(
                                'documentation_hype',
                                f"Unsubstantiated {category} claim in {doc_file.name}",
                                {'file': str(doc_file), 'term': term},
                                'HIGH'
                            )
                
                # Check for specific technical claims
                for claim_pattern in self.TECHNICAL_CLAIMS:
                    matches = re.findall(claim_pattern, content, re.IGNORECASE)
                    for match in matches:
                        self._add_issue(
                            'unverified_claim',
                            f"Technical claim needs verification in {doc_file.name}: {match}",
                            {'file': str(doc_file), 'claim': match},
                            'MEDIUM'
                        )
                
            except Exception as e:
                self._add_issue(
                    'file_error',
                    f"Could not read {doc_file.name}: {str(e)}",
                    {'file': str(doc_file)},
                    'LOW'
                )
    
    def _audit_code_claims(self):
        """Verify that documented features have actual code implementation"""
        print("🔍 Auditing Code-Claim Alignment...")
        
        # Find all Python files
        py_files = list(self.project_path.rglob("*.py"))
        
        # Look for hardcoded "impressive" values
        hardcoded_patterns = [
            r'consciousness_level\s*=\s*0\.9\d+',
            r'interpretability\s*=\s*0\.9\d+',
            r'physics_compliance\s*=\s*0\.9\d+',
            r'accuracy\s*=\s*0\.9\d+',
            r'confidence\s*=\s*0\.9\d+'
        ]
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern in hardcoded_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        self._add_issue(
                            'hardcoded_metric',
                            f"Hardcoded performance metric in {py_file.name}: {match}",
                            {'file': str(py_file), 'pattern': match},
                            'HIGH'
                        )
                        
                # Look for TODO/FIXME indicating incomplete features
                todo_matches = re.findall(r'(TODO|FIXME|HACK).*', content, re.IGNORECASE)
                if todo_matches:
                    self._add_issue(
                        'incomplete_implementation',
                        f"Incomplete implementation markers in {py_file.name}",
                        {'file': str(py_file), 'count': len(todo_matches)},
                        'MEDIUM'
                    )
                    
            except Exception as e:
                self._add_issue(
                    'file_error',
                    f"Could not analyze {py_file.name}: {str(e)}",
                    {'file': str(py_file)},
                    'LOW'
                )
    
    def _audit_performance_claims(self):
        """Check if performance claims are backed by benchmarks"""
        print("📊 Auditing Performance Claims...")
        
        # Look for test/benchmark files
        test_files = list(self.project_path.rglob("*test*.py"))
        benchmark_files = list(self.project_path.rglob("*benchmark*.py"))
        
        if not test_files and not benchmark_files:
            self._add_issue(
                'missing_benchmarks',
                "No test or benchmark files found to validate performance claims",
                {'test_files': len(test_files), 'benchmark_files': len(benchmark_files)},
                'HIGH'
            )
        
        # Check requirements.txt for realistic dependencies
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            # Look for overly complex dependencies that might indicate unused sophistication
            complex_deps = ['transformers', 'torch', 'tensorflow', 'jax']
            found_complex = [dep for dep in complex_deps if dep in content.lower()]
            
            if found_complex:
                self._add_issue(
                    'complex_dependencies',
                    f"Complex ML dependencies found: {found_complex}. Are these actually used?",
                    {'dependencies': found_complex},
                    'MEDIUM'
                )
    
    def _audit_architecture_alignment(self):
        """Check if architecture documentation matches actual code structure"""
        print("🏗️ Auditing Architecture Alignment...")
        
        # Look for architecture diagrams or documentation
        arch_files = []
        for pattern in ['*architecture*', '*diagram*', '*design*']:
            arch_files.extend(self.project_path.rglob(pattern))
        
        # Check if claimed services/components actually exist
        claimed_services = []
        actual_services = []
        
        # Look for docker-compose or service definitions
        docker_files = list(self.project_path.rglob("docker-compose*.yml"))
        for docker_file in docker_files:
            try:
                content = yaml.safe_load(docker_file.read_text())
                if 'services' in content:
                    actual_services.extend(content['services'].keys())
            except:
                pass
        
        # Look for microservice directories
        service_dirs = [d for d in self.project_path.iterdir() 
                       if d.is_dir() and d.name not in ['.git', '__pycache__', '.pytest_cache']]
        
        self.results['metrics']['claimed_services'] = len(claimed_services)
        self.results['metrics']['actual_services'] = len(actual_services)
        self.results['metrics']['service_directories'] = len(service_dirs)
    
    def _audit_language_hype(self):
        """Audit for marketing language without technical backing"""
        print("🎯 Auditing Language for Hype...")
        
        # Check README files specifically
        readme_files = list(self.project_path.rglob("README*.md"))
        
        total_hype_instances = 0
        for readme in readme_files:
            try:
                content = readme.read_text(encoding='utf-8')
                
                # Count hype terms
                for category, terms in self.HYPE_TERMS.items():
                    for term in terms:
                        instances = len(re.findall(term, content, re.IGNORECASE))
                        total_hype_instances += instances
                        
                        if instances > 0:
                            self._add_issue(
                                'marketing_hype',
                                f"Marketing language in {readme.name}: '{term}' appears {instances} times",
                                {'file': str(readme), 'term': term, 'count': instances},
                                'MEDIUM' if instances > 3 else 'LOW'
                            )
                            
            except Exception as e:
                continue
        
        self.results['metrics']['total_hype_instances'] = total_hype_instances
    
    def _calculate_integrity_score(self):
        """Calculate overall integrity score (0-100)"""
        
        # Count issues by severity
        high_issues = len([i for i in self.results['issues'] if i['severity'] == 'HIGH'])
        medium_issues = len([i for i in self.results['issues'] if i['severity'] == 'MEDIUM'])
        low_issues = len([i for i in self.results['issues'] if i['severity'] == 'LOW'])
        
        # Calculate deductions
        score = 100
        score -= high_issues * 20    # High issues: -20 points each
        score -= medium_issues * 10  # Medium issues: -10 points each  
        score -= low_issues * 5      # Low issues: -5 points each
        
        # Bonus for good practices
        if self.results['metrics'].get('total_hype_instances', 0) == 0:
            score += 10  # No hype language bonus
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        
        self.results['integrity_score'] = score
        self.results['metrics']['high_issues'] = high_issues
        self.results['metrics']['medium_issues'] = medium_issues
        self.results['metrics']['low_issues'] = low_issues
    
    def _generate_recommendations(self):
        """Generate specific recommendations for improvement"""
        
        recommendations = []
        
        # High-priority recommendations
        high_issues = [i for i in self.results['issues'] if i['severity'] == 'HIGH']
        if high_issues:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Replace hardcoded performance metrics with actual calculations',
                'reason': f"Found {len(high_issues)} high-severity issues that damage credibility"
            })
        
        # Hype language recommendations
        hype_count = self.results['metrics'].get('total_hype_instances', 0)
        if hype_count > 5:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Rewrite marketing language with evidence-based descriptions',
                'reason': f"Found {hype_count} instances of unsupported hype language"
            })
        
        # Missing benchmarks
        if any(i['type'] == 'missing_benchmarks' for i in self.results['issues']):
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Create comprehensive benchmark suite to validate all performance claims',
                'reason': 'Performance claims lack validation'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'MEDIUM',
                'action': 'Implement pre-commit hooks with this audit tool',
                'reason': 'Prevent future integrity issues'
            },
            {
                'priority': 'LOW',
                'action': 'Add explicit limitations section to documentation',
                'reason': 'Transparent communication of constraints builds trust'
            }
        ])
        
        self.results['recommendations'] = recommendations
    
    def _find_documentation_files(self) -> List[Path]:
        """Find all documentation files"""
        doc_files = []
        patterns = ['*.md', '*.rst', '*.txt']
        
        for pattern in patterns:
            doc_files.extend(self.project_path.rglob(pattern))
        
        # Filter out obvious non-documentation
        excluded = ['.git', '__pycache__', 'node_modules', '.pytest_cache']
        doc_files = [f for f in doc_files if not any(ex in str(f) for ex in excluded)]
        
        return doc_files
    
    def _add_issue(self, issue_type: str, description: str, details: Dict, severity: str):
        """Add an issue to the audit results"""
        self.results['issues'].append({
            'type': issue_type,
            'description': description,
            'details': details,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        
        # Print issue immediately for feedback
        severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
        print(f"  {severity_emoji[severity]} {severity}: {description}")

def main():
    parser = argparse.ArgumentParser(description="NIS Engineering Integrity Audit Tool")
    parser.add_argument("--project-path", required=True, help="Path to project to audit")
    parser.add_argument("--output-report", action="store_true", help="Generate detailed report")
    parser.add_argument("--output-file", help="Output file for report (default: audit-report.json)")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = NISIntegrityAuditor(args.project_path)
    results = auditor.run_full_audit()
    
    # Generate report
    if args.output_report:
        output_file = args.output_file or "audit-report.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 AUDIT SUMMARY")
    print("=" * 60)
    print(f"Integrity Score: {results['integrity_score']}/100")
    print(f"Total Issues: {len(results['issues'])}")
    print(f"  🔴 High: {results['metrics'].get('high_issues', 0)}")
    print(f"  🟡 Medium: {results['metrics'].get('medium_issues', 0)}")
    print(f"  🟢 Low: {results['metrics'].get('low_issues', 0)}")
    
    if results['integrity_score'] >= 80:
        print("\n✅ PASS: Repository meets integrity standards")
    else:
        print("\n❌ FAIL: Repository needs improvement before release")
        print("\nTop Recommendations:")
        for rec in results['recommendations'][:3]:
            print(f"  • {rec['action']}")
    
    return 0 if results['integrity_score'] >= 80 else 1

if __name__ == "__main__":
    exit(main()) 