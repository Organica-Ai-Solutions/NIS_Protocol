#!/usr/bin/env python3
"""
🎭 NIS Hype vs Reality Auditor
Comprehensive audit to combat overblown claims, marketing language, and academic pretensions

This auditor specifically targets the three main issues:
1. Overblown Claims vs Reality (consciousness, physics, etc.)
2. Complexity vs Value (over-engineering assessment) 
3. Academic Pretensions (whitepaper quality, benchmarks)
"""

import os
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
import math


class HypeVsRealityAuditor:
    """Comprehensive auditor for marketing hype vs technical reality"""
    
    # Marketing hype terms that need evidence
    HYPE_BUZZWORDS = {
        'consciousness': {
            'terms': ['consciousness', 'conscious', 'self-aware', 'sentient', 'meta-cognitive', 'introspection'],
            'required_evidence': ['metacognitive_processor.py', 'consciousness_metrics', 'awareness_tracking']
        },
        'revolutionary': {
            'terms': ['revolutionary', 'breakthrough', 'unprecedented', 'world-first', 'first-ever', 'paradigm shift'],
            'required_evidence': ['peer_reviewed_paper', 'independent_benchmark', 'novel_algorithm']
        },
        'agi': {
            'terms': ['agi', 'artificial general intelligence', 'general intelligence', 'human-level ai'],
            'required_evidence': ['multi_domain_benchmarks', 'transfer_learning_proof', 'reasoning_validation']
        },
        'physics_informed': {
            'terms': ['physics-informed neural networks', 'PINN', 'physics validation', 'differential equations'],
            'required_evidence': ['torch.autograd', 'physics_residual', 'pde_solving', 'boundary_conditions']
        }
    }
    
    # Academic pretension indicators
    ACADEMIC_PRETENSION_PATTERNS = [
        r'whitepaper',
        r'research paper',
        r'technical paper',
        r'peer.?review',
        r'academic publication',
        r'arxiv',
        r'proceedings',
        r'conference paper'
    ]
    
    # Performance claims that need benchmarking
    PERFORMANCE_CLAIM_PATTERNS = [
        r'(\d+\.?\d*)\s*%\s*(accuracy|performance|compliance|efficiency)',
        r'(\d+\.?\d*)x\s*(faster|better|improvement)',
        r'(zero|no)\s*(error|hallucination|bias|failure)',
        r'(sub.?second|millisecond)\s*(response|processing)',
        r'(real.?time|instant)\s*(processing|analysis)'
    ]
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.results = {
            'audit_timestamp': datetime.now().isoformat(),
            'project_name': self.project_path.name,
            'hype_analysis': {},
            'complexity_analysis': {},
            'academic_analysis': {},
            'overall_integrity_score': 0,
            'critical_issues': [],
            'recommendations': []
        }
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete hype vs reality audit"""
        print("🎭 Starting Comprehensive Hype vs Reality Audit...")
        print(f"📁 Project: {self.project_path}")
        print("=" * 70)
        
        # Issue 1: Overblown Claims vs Reality
        self._audit_overblown_claims()
        
        # Issue 2: Complexity vs Value
        self._audit_complexity_vs_value()
        
        # Issue 3: Academic Pretensions
        self._audit_academic_pretensions()
        
        # Calculate overall scores and generate report
        self._calculate_overall_integrity_score()
        self._generate_comprehensive_recommendations()
        
        return self.results
    
    def _audit_overblown_claims(self):
        """Audit for overblown marketing claims vs actual implementation"""
        print("\n🎯 Auditing Overblown Claims vs Reality...")
        
        hype_analysis = {
            'buzzword_violations': [],
            'unverified_claims': [],
            'implementation_gaps': [],
            'marketing_to_tech_ratio': 0.0
        }
        
        # Scan all documentation for hype terms
        doc_files = list(self.project_path.glob("**/*.md")) + \
                   list(self.project_path.glob("**/*.rst")) + \
                   list(self.project_path.glob("**/*.txt"))
        
        marketing_terms = 0
        technical_terms = 0
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8', errors='ignore')
                
                # Count marketing vs technical language
                marketing_patterns = [
                    'revolutionary', 'breakthrough', 'unprecedented', 'consciousness', 
                    'agi', 'world-class', 'cutting-edge', 'state-of-the-art'
                ]
                technical_patterns = [
                    'algorithm', 'implementation', 'function', 'class', 'method',
                    'parameter', 'variable', 'return', 'exception'
                ]
                
                for pattern in marketing_patterns:
                    marketing_terms += len(re.findall(pattern, content, re.IGNORECASE))
                for pattern in technical_patterns:
                    technical_terms += len(re.findall(pattern, content, re.IGNORECASE))
                
                # Check for specific hype violations
                for category, hype_data in self.HYPE_BUZZWORDS.items():
                    for term in hype_data['terms']:
                        if re.search(term, content, re.IGNORECASE):
                            # Check if required evidence exists
                            evidence_found = self._check_implementation_evidence(hype_data['required_evidence'])
                            if not evidence_found:
                                hype_analysis['buzzword_violations'].append({
                                    'file': str(doc_file.relative_to(self.project_path)),
                                    'category': category,
                                    'term': term,
                                    'missing_evidence': [e for e in hype_data['required_evidence'] if e not in evidence_found],
                                    'severity': 'CRITICAL'
                                })
                
                # Check for specific performance claims
                for claim_pattern in self.PERFORMANCE_CLAIM_PATTERNS:
                    matches = re.findall(claim_pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Look for corresponding benchmark/test file
                        if not self._has_benchmark_evidence(match):
                            hype_analysis['unverified_claims'].append({
                                'file': str(doc_file.relative_to(self.project_path)),
                                'claim': match,
                                'evidence_status': 'MISSING',
                                'severity': 'HIGH'
                            })
                        
            except Exception as e:
                print(f"⚠️  Error processing {doc_file}: {e}")
        
        # Calculate marketing to technical ratio
        if technical_terms > 0:
            hype_analysis['marketing_to_tech_ratio'] = marketing_terms / technical_terms
        
        self.results['hype_analysis'] = hype_analysis
    
    def _check_implementation_evidence(self, required_evidence: List[str]) -> List[str]:
        """Check if implementation evidence exists for claimed capabilities"""
        evidence_found = []
        
        # Search for evidence in code files
        code_files = list(self.project_path.glob("**/*.py"))
        
        for evidence in required_evidence:
            found = False
            for code_file in code_files:
                try:
                    content = code_file.read_text(encoding='utf-8', errors='ignore')
                    if evidence in content or evidence in code_file.name:
                        evidence_found.append(evidence)
                        found = True
                        break
                except:
                    continue
            
            if not found:
                # Check for partial evidence (less strict)
                evidence_keywords = evidence.replace('_', ' ').replace('.py', '').split()
                for keyword in evidence_keywords:
                    if len(keyword) > 3:  # Avoid tiny words
                        for code_file in code_files[:10]:  # Sample check
                            try:
                                content = code_file.read_text(encoding='utf-8', errors='ignore')
                                if keyword in content.lower():
                                    found = True
                                    break
                            except:
                                continue
                        if found:
                            break
        
        return evidence_found
    
    def _has_benchmark_evidence(self, claim) -> bool:
        """Check if there's benchmark evidence for performance claims"""
        benchmark_patterns = ['test_', 'benchmark_', '_test.py', 'validation_', 'eval_']
        test_dirs = ['test', 'tests', 'benchmark', 'benchmarks', 'validation']
        
        # Check for test/benchmark files
        for pattern in benchmark_patterns:
            if list(self.project_path.glob(f"**/*{pattern}*")):
                return True
        
        for test_dir in test_dirs:
            test_path = self.project_path / test_dir
            if test_path.exists() and list(test_path.glob("**/*.py")):
                return True
        
        return False
    
    def _audit_complexity_vs_value(self):
        """Audit system complexity vs actual value delivered"""
        print("\n🔧 Auditing Complexity vs Value...")
        
        complexity_metrics = {
            'total_files': 0,
            'python_files': 0,
            'lines_of_code': 0,
            'cyclomatic_complexity': 0,
            'import_complexity': 0,
            'directory_depth': 0,
            'duplicate_functionality': [],
            'over_engineering_score': 0
        }
        
        # Count files and LOC
        all_files = list(self.project_path.rglob("*"))
        complexity_metrics['total_files'] = len([f for f in all_files if f.is_file()])
        
        python_files = list(self.project_path.glob("**/*.py"))
        complexity_metrics['python_files'] = len(python_files)
        
        # Analyze Python code complexity
        total_loc = 0
        total_imports = 0
        function_count = 0
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                total_loc += len(lines)
                
                # Count imports
                imports = len(re.findall(r'^(import|from)\s+', content, re.MULTILINE))
                total_imports += imports
                
                # Count functions
                functions = len(re.findall(r'^def\s+', content, re.MULTILINE))
                function_count += functions
                
                # Look for duplicate functionality
                if 'agent' in py_file.name.lower():
                    similar_files = [f for f in python_files if f != py_file and 'agent' in f.name.lower()]
                    if len(similar_files) > 5:  # Threshold for too many similar files
                        complexity_metrics['duplicate_functionality'].append({
                            'pattern': 'agent',
                            'files': len(similar_files) + 1,
                            'example_file': str(py_file.name)
                        })
                
            except Exception as e:
                continue
        
        complexity_metrics['lines_of_code'] = total_loc
        complexity_metrics['import_complexity'] = total_imports / max(complexity_metrics['python_files'], 1)
        
        # Calculate directory depth
        max_depth = 0
        for file_path in all_files:
            depth = len(file_path.relative_to(self.project_path).parts)
            max_depth = max(max_depth, depth)
        complexity_metrics['directory_depth'] = max_depth
        
        # Calculate over-engineering score
        # Based on files per functionality, LOC per file, etc.
        avg_loc_per_file = total_loc / max(complexity_metrics['python_files'], 1)
        files_per_1000_loc = complexity_metrics['python_files'] / max(total_loc / 1000, 1)
        
        over_engineering_indicators = 0
        if avg_loc_per_file < 50:  # Too many small files
            over_engineering_indicators += 1
        if files_per_1000_loc > 20:  # Too many files for LOC
            over_engineering_indicators += 1
        if complexity_metrics['directory_depth'] > 6:  # Too deep nesting
            over_engineering_indicators += 1
        if len(complexity_metrics['duplicate_functionality']) > 3:  # Too much duplication
            over_engineering_indicators += 1
        
        complexity_metrics['over_engineering_score'] = over_engineering_indicators / 4.0
        
        self.results['complexity_analysis'] = complexity_metrics
    
    def _audit_academic_pretensions(self):
        """Audit for academic pretensions vs actual research quality"""
        print("\n🎓 Auditing Academic Pretensions...")
        
        academic_analysis = {
            'whitepaper_count': 0,
            'whitepaper_quality_score': 0,
            'citation_issues': [],
            'methodology_gaps': [],
            'benchmark_credibility': 0,
            'peer_review_claims': []
        }
        
        # Find documents claiming to be academic papers
        doc_files = list(self.project_path.glob("**/*.md")) + \
                   list(self.project_path.glob("**/*.rst")) + \
                   list(self.project_path.glob("**/*.txt"))
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for academic pretension patterns
                for pattern in self.ACADEMIC_PRETENSION_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        academic_analysis['whitepaper_count'] += 1
                        
                        # Analyze quality of academic content
                        quality_score = self._assess_academic_quality(content, doc_file)
                        academic_analysis['whitepaper_quality_score'] = max(
                            academic_analysis['whitepaper_quality_score'], 
                            quality_score
                        )
                        
                        # Check for common academic issues
                        issues = self._check_academic_issues(content, doc_file)
                        academic_analysis['methodology_gaps'].extend(issues)
                        
                        break
                
            except Exception as e:
                continue
        
        self.results['academic_analysis'] = academic_analysis
    
    def _assess_academic_quality(self, content: str, doc_file: Path) -> float:
        """Assess the quality of academic content"""
        quality_indicators = {
            'has_abstract': bool(re.search(r'abstract', content, re.IGNORECASE)),
            'has_methodology': bool(re.search(r'methodology|methods?', content, re.IGNORECASE)),
            'has_results': bool(re.search(r'results?|findings?', content, re.IGNORECASE)),
            'has_references': bool(re.search(r'references?|bibliography', content, re.IGNORECASE)),
            'has_citations': len(re.findall(r'\[\d+\]|\(.*\d{4}.*\)', content)) > 0,
            'has_equations': len(re.findall(r'\$.*\$|\\[a-zA-Z]+', content)) > 0,
            'proper_structure': len(re.findall(r'^#+\s+', content, re.MULTILINE)) > 3
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        # Penalties for marketing language in academic docs
        marketing_terms = ['revolutionary', 'unprecedented', 'world-first', 'breakthrough']
        marketing_count = sum(len(re.findall(term, content, re.IGNORECASE)) for term in marketing_terms)
        
        if marketing_count > 5:
            quality_score *= 0.5  # Heavy penalty for marketing in academic content
        
        return quality_score
    
    def _check_academic_issues(self, content: str, doc_file: Path) -> List[Dict[str, Any]]:
        """Check for common academic integrity issues"""
        issues = []
        
        # Check for unsupported claims
        if 'breakthrough' in content.lower() and 'peer review' not in content.lower():
            issues.append({
                'file': str(doc_file.name),
                'type': 'UNSUPPORTED_BREAKTHROUGH_CLAIM',
                'severity': 'HIGH'
            })
        
        # Check for performance claims without methodology
        perf_claims = len(re.findall(r'\d+\.?\d*%', content))
        methodology_sections = len(re.findall(r'methodology|evaluation|experiment', content, re.IGNORECASE))
        
        if perf_claims > 3 and methodology_sections == 0:
            issues.append({
                'file': str(doc_file.name),
                'type': 'PERFORMANCE_CLAIMS_WITHOUT_METHODOLOGY',
                'severity': 'CRITICAL'
            })
        
        return issues
    
    def _calculate_overall_integrity_score(self):
        """Calculate overall integrity score across all dimensions"""
        print("\n📊 Calculating Overall Integrity Score...")
        
        hype_score = 100
        complexity_score = 100
        academic_score = 100
        
        # Hype analysis scoring
        hype_analysis = self.results['hype_analysis']
        hype_score -= len(hype_analysis['buzzword_violations']) * 15
        hype_score -= len(hype_analysis['unverified_claims']) * 10
        hype_score -= min(50, hype_analysis['marketing_to_tech_ratio'] * 100)
        
        # Complexity analysis scoring
        complexity_analysis = self.results['complexity_analysis']
        complexity_score -= complexity_analysis['over_engineering_score'] * 30
        complexity_score -= len(complexity_analysis['duplicate_functionality']) * 10
        
        # Academic analysis scoring
        academic_analysis = self.results['academic_analysis']
        if academic_analysis['whitepaper_count'] > 0:
            academic_score = academic_analysis['whitepaper_quality_score'] * 100
            academic_score -= len(academic_analysis['methodology_gaps']) * 20
        
        # Overall score (weighted average)
        overall_score = (hype_score * 0.5 + complexity_score * 0.3 + academic_score * 0.2)
        self.results['overall_integrity_score'] = max(0, min(100, overall_score))
        
        # Identify critical issues
        critical_issues = []
        if hype_score < 50:
            critical_issues.append("HIGH_MARKETING_HYPE")
        if complexity_score < 40:
            critical_issues.append("SEVERE_OVER_ENGINEERING") 
        if academic_score < 30:
            critical_issues.append("POOR_ACADEMIC_QUALITY")
        
        self.results['critical_issues'] = critical_issues
    
    def _generate_comprehensive_recommendations(self):
        """Generate comprehensive recommendations for improvement"""
        recommendations = []
        
        hype_analysis = self.results['hype_analysis']
        complexity_analysis = self.results['complexity_analysis']
        academic_analysis = self.results['academic_analysis']
        
        # Hype reduction recommendations
        if len(hype_analysis['buzzword_violations']) > 0:
            recommendations.append({
                'category': 'Marketing Claims',
                'priority': 'CRITICAL',
                'issue': f"Found {len(hype_analysis['buzzword_violations'])} unsubstantiated hype claims",
                'action': "Either implement genuine capabilities or remove exaggerated claims",
                'specific_actions': [
                    "Replace 'consciousness' claims with 'agent coordination'",
                    "Replace 'PINN' claims with actual differential equation solving",
                    "Add evidence for all performance metrics"
                ]
            })
        
        # Complexity reduction recommendations
        if complexity_analysis['over_engineering_score'] > 0.6:
            recommendations.append({
                'category': 'System Architecture',
                'priority': 'HIGH',
                'issue': f"Over-engineering score: {complexity_analysis['over_engineering_score']:.1%}",
                'action': "Simplify architecture and reduce unnecessary complexity",
                'specific_actions': [
                    f"Consolidate {len(complexity_analysis['duplicate_functionality'])} duplicate functionalities",
                    f"Reduce directory depth from {complexity_analysis['directory_depth']} levels",
                    "Merge similar agent modules"
                ]
            })
        
        # Academic quality recommendations
        if academic_analysis['whitepaper_quality_score'] < 0.5:
            recommendations.append({
                'category': 'Academic Content',
                'priority': 'MEDIUM',
                'issue': f"Academic quality score: {academic_analysis['whitepaper_quality_score']:.1%}",
                'action': "Improve academic rigor or remove academic pretensions",
                'specific_actions': [
                    "Add proper methodology sections",
                    "Include peer-reviewed references", 
                    "Remove marketing language from technical papers",
                    "Provide reproducible benchmarks"
                ]
            })
        
        self.results['recommendations'] = recommendations
    
    def generate_integrity_report(self, output_file: str = None) -> str:
        """Generate comprehensive integrity audit report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 🎭 NIS Protocol Integrity Audit Report

**Generated:** {timestamp}
**Project:** {self.results['project_name']}
**Overall Integrity Score:** {self.results['overall_integrity_score']:.1f}/100

## 🎯 Executive Summary

This audit examined three critical areas:
1. **Marketing Hype vs Reality** - Overblown claims without implementation
2. **Complexity vs Value** - Over-engineering and unnecessary complexity  
3. **Academic Pretensions** - Quality of research claims and whitepapers

### Critical Issues Identified: {len(self.results['critical_issues'])}
"""
        for issue in self.results['critical_issues']:
            report += f"- ❌ {issue.replace('_', ' ').title()}\n"
        
        report += f"""

## 📊 Detailed Analysis

### 1. Marketing Hype Analysis
- **Buzzword Violations:** {len(self.results['hype_analysis']['buzzword_violations'])}
- **Unverified Claims:** {len(self.results['hype_analysis']['unverified_claims'])}
- **Marketing-to-Technical Ratio:** {self.results['hype_analysis']['marketing_to_tech_ratio']:.2f}

#### Top Hype Issues:
"""
        
        for violation in self.results['hype_analysis']['buzzword_violations'][:5]:
            report += f"- **{violation['category'].title()}**: '{violation['term']}' in {violation['file']}\n"
        
        report += f"""

### 2. Complexity Analysis
- **Total Files:** {self.results['complexity_analysis']['total_files']}
- **Python Files:** {self.results['complexity_analysis']['python_files']}
- **Lines of Code:** {self.results['complexity_analysis']['lines_of_code']:,}
- **Over-engineering Score:** {self.results['complexity_analysis']['over_engineering_score']:.1%}
- **Directory Depth:** {self.results['complexity_analysis']['directory_depth']} levels

#### Complexity Issues:
"""
        for dup in self.results['complexity_analysis']['duplicate_functionality']:
            report += f"- **{dup['pattern'].title()} Duplication**: {dup['files']} similar files\n"
        
        report += f"""

### 3. Academic Quality Analysis
- **Whitepapers Found:** {self.results['academic_analysis']['whitepaper_count']}
- **Quality Score:** {self.results['academic_analysis']['whitepaper_quality_score']:.1%}
- **Methodology Gaps:** {len(self.results['academic_analysis']['methodology_gaps'])}

## 🚀 Recommendations for Improvement

"""
        
        for i, rec in enumerate(self.results['recommendations'], 1):
            report += f"""
### {i}. {rec['category']} ({rec['priority']} Priority)
**Issue:** {rec['issue']}
**Action:** {rec['action']}

**Specific Steps:**
"""
            for action in rec['specific_actions']:
                report += f"- {action}\n"
        
        report += f"""

## 🎯 Integrity Improvement Roadmap

### Phase 1: Critical Hype Reduction (Week 1)
- Remove or substantiate all consciousness/AGI claims
- Implement genuine physics validation or remove PINN claims  
- Add evidence for all performance metrics

### Phase 2: Complexity Reduction (Week 2-3)
- Consolidate duplicate agent functionality
- Simplify directory structure
- Reduce unnecessary abstraction layers

### Phase 3: Academic Quality (Week 4)
- Improve whitepaper methodology sections
- Add peer-reviewed references
- Remove marketing language from technical docs

### Success Metrics:
- Integrity Score > 80/100
- Zero unsubstantiated technical claims
- Marketing-to-technical ratio < 0.3
- Over-engineering score < 30%

---

**"Build systems so good that honest descriptions sound impressive"**
"""
        
        if output_file:
            Path(output_file).write_text(report)
            print(f"📄 Comprehensive report saved to {output_file}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive hype vs reality audit')
    parser.add_argument('--project-path', default='.', help='Path to project to audit')
    parser.add_argument('--output-report', help='Output report file path')
    
    args = parser.parse_args()
    
    print("🎭 NIS Protocol Hype vs Reality Auditor")
    print("Combating AI washing and ensuring honest representation")
    print("=" * 60)
    
    auditor = HypeVsRealityAuditor(args.project_path)
    results = auditor.run_comprehensive_audit()
    
    report = auditor.generate_integrity_report(args.output_report)
    
    # Summary
    score = results['overall_integrity_score']
    critical_count = len(results['critical_issues'])
    
    print(f"\n🎯 AUDIT COMPLETE")
    print(f"Integrity Score: {score:.1f}/100")
    print(f"Critical Issues: {critical_count}")
    
    if score < 60 or critical_count > 0:
        print("❌ INTEGRITY AUDIT FAILED - Immediate action required")
        exit(1)
    else:
        print("✅ INTEGRITY AUDIT PASSED - System maintains professional credibility")


if __name__ == "__main__":
    main()