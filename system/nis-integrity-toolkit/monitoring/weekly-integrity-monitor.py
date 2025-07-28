#!/usr/bin/env python3
"""
📊 NIS Weekly Integrity Monitor
Tracks engineering integrity metrics over time and generates trend reports
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

class IntegrityMonitor:
    """Long-term integrity tracking and trend analysis"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.history_file = self.project_path / "nis-integrity-toolkit" / "monitoring" / "integrity_history.json"
        self.reports_dir = self.project_path / "nis-integrity-toolkit" / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load historical data
        self.history = self._load_history()
        
        # Initialize metrics
        self.current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'integrity_score': 0,
            'hype_instances': 0,
            'hardcoded_values': 0,
            'unverified_claims': 0,
            'missing_evidence': 0,
            'documentation_alignment': 0,
            'test_coverage': 0,
            'benchmark_coverage': 0,
            'git_commits_since_last': 0,
            'code_quality_score': 0,
            'professional_language_score': 0
        }
    
    def run_weekly_check(self) -> Dict:
        """Run comprehensive weekly integrity check"""
        print("📊 NIS Weekly Integrity Monitor")
        print("=" * 50)
        print(f"📁 Project: {self.project_path}")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all monitoring checks
        self._check_hype_language()
        self._check_hardcoded_values()
        self._check_unverified_claims()
        self._check_documentation_alignment()
        self._check_test_coverage()
        self._check_benchmark_coverage()
        self._check_git_activity()
        self._check_code_quality()
        self._calculate_integrity_score()
        
        # Save to history
        self._save_to_history()
        
        # Generate reports
        self._generate_trend_report()
        self._generate_weekly_summary()
        
        print(f"\n🎯 Weekly Integrity Score: {self.current_metrics['integrity_score']}/100")
        
        return self.current_metrics
    
    def _check_hype_language(self):
        """Monitor hype language usage over time"""
        print("🎯 Checking hype language usage...")
        
        hype_terms = [
            'advanced', 'improvement', 'KAN interpretability-driven',
            'innovative', 'novel', 'optimized', 'low error rate',
            'mathematically-inspired', 'multi-agent system', 'advanced multi-agent system'
        ]
        
        total_instances = 0
        doc_files = list(self.project_path.rglob("*.md"))
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8').lower()
                for term in hype_terms:
                    instances = content.count(term)
                    total_instances += instances
            except:
                continue
        
        self.current_metrics['hype_instances'] = total_instances
        print(f"   📊 Hype language instances: {total_instances}")
    
    def _check_hardcoded_values(self):
        """Monitor hardcoded performance values"""
        print("🔍 Checking hardcoded values...")
        
        hardcoded_patterns = [
            r'consciousness_level\s*=\s*0\.\d+',
            r'interpretability\s*=\s*0\.\d+',
            r'physics_compliance\s*=\s*0\.\d+',
            r'accuracy\s*=\s*0\.\d+',
            r'confidence\s*=\s*0\.\d+'
        ]
        
        total_hardcoded = 0
        py_files = list(self.project_path.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in hardcoded_patterns:
                    import re
                    matches = re.findall(pattern, content)
                    total_hardcoded += len(matches)
            except:
                continue
        
        self.current_metrics['hardcoded_values'] = total_hardcoded
        print(f"   📊 Hardcoded values: {total_hardcoded}")
    
    def _check_unverified_claims(self):
        """Monitor unverified technical claims"""
        print("📊 Checking unverified claims...")
        
        claim_patterns = [
            r'(\d+\.?\d*)% (accuracy|interpretability|performance|compliance)',
            r'(\d+\.?\d*)(x|times) (faster|better|more accurate)',
            r'(zero|no) (hallucination|error|bias)',
            r'(sub-second|millisecond) (processing|response)'
        ]
        
        total_claims = 0
        doc_files = list(self.project_path.rglob("*.md"))
        
        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding='utf-8')
                for pattern in claim_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    total_claims += len(matches)
            except:
                continue
        
        self.current_metrics['unverified_claims'] = total_claims
        print(f"   📊 Technical claims: {total_claims}")
    
    def _check_documentation_alignment(self):
        """Check documentation-code alignment"""
        print("📝 Checking documentation alignment...")
        
        # Simple heuristic: compare README length to code complexity
        readme_files = list(self.project_path.rglob("README*.md"))
        py_files = list(self.project_path.rglob("*.py"))
        
        readme_lines = 0
        code_lines = 0
        
        for readme in readme_files:
            try:
                readme_lines += len(readme.read_text(encoding='utf-8').splitlines())
            except:
                continue
        
        for py_file in py_files:
            try:
                code_lines += len(py_file.read_text(encoding='utf-8').splitlines())
            except:
                continue
        
        # Calculate alignment score (1-100)
        if code_lines > 0:
            alignment_ratio = min(readme_lines / code_lines, 1.0)
            alignment_score = int(alignment_ratio * 100)
        else:
            alignment_score = 0
        
        self.current_metrics['documentation_alignment'] = alignment_score
        print(f"   📊 Documentation alignment: {alignment_score}/100")
    
    def _check_test_coverage(self):
        """Monitor test coverage"""
        print("🧪 Checking test coverage...")
        
        test_files = list(self.project_path.rglob("*test*.py"))
        py_files = list(self.project_path.rglob("*.py"))
        
        # Exclude test files from main code count
        main_py_files = [f for f in py_files if "test" not in f.name.lower()]
        
        if main_py_files:
            test_coverage = min(len(test_files) / len(main_py_files), 1.0) * 100
        else:
            test_coverage = 0
        
        self.current_metrics['test_coverage'] = int(test_coverage)
        print(f"   📊 Test coverage: {int(test_coverage)}%")
    
    def _check_benchmark_coverage(self):
        """Monitor benchmark coverage"""
        print("📈 Checking benchmark coverage...")
        
        benchmark_files = list(self.project_path.rglob("*benchmark*.py"))
        performance_claims = self.current_metrics['unverified_claims']
        
        if performance_claims > 0:
            benchmark_coverage = min(len(benchmark_files) / performance_claims, 1.0) * 100
        else:
            benchmark_coverage = 100 if len(benchmark_files) > 0 else 0
        
        self.current_metrics['benchmark_coverage'] = int(benchmark_coverage)
        print(f"   📊 Benchmark coverage: {int(benchmark_coverage)}%")
    
    def _check_git_activity(self):
        """Monitor git activity since last check"""
        print("🔄 Checking git activity...")
        
        try:
            # Get commits since last week
            last_week = datetime.now() - timedelta(days=7)
            last_week_str = last_week.strftime('%Y-%m-%d')
            
            import subprocess
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'--since={last_week_str}', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )
            
            if result.returncode == 0:
                commits = int(result.stdout.strip())
            else:
                commits = 0
                
        except:
            commits = 0
        
        self.current_metrics['git_commits_since_last'] = commits
        print(f"   📊 Commits this week: {commits}")
    
    def _check_code_quality(self):
        """Monitor code quality metrics"""
        print("🔧 Checking code quality...")
        
        py_files = list(self.project_path.rglob("*.py"))
        
        total_lines = 0
        total_comments = 0
        total_docstrings = 0
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                total_lines += len(lines)
                
                # Count comment lines
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        total_comments += 1
                    elif '"""' in stripped or "'''" in stripped:
                        total_docstrings += 1
                        
            except:
                continue
        
        if total_lines > 0:
            comment_ratio = (total_comments + total_docstrings) / total_lines
            quality_score = min(comment_ratio * 200, 100)  # Cap at 100
        else:
            quality_score = 0
        
        self.current_metrics['code_quality_score'] = int(quality_score)
        print(f"   📊 Code quality score: {int(quality_score)}/100")
    
    def _calculate_integrity_score(self):
        """Calculate overall integrity score"""
        
        # Base score
        score = 100
        
        # Deductions
        score -= self.current_metrics['hype_instances'] * 5
        score -= self.current_metrics['hardcoded_values'] * 10
        score -= self.current_metrics['unverified_claims'] * 8
        
        # Bonuses
        score += self.current_metrics['documentation_alignment'] * 0.1
        score += self.current_metrics['test_coverage'] * 0.2
        score += self.current_metrics['benchmark_coverage'] * 0.15
        score += self.current_metrics['code_quality_score'] * 0.1
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        
        self.current_metrics['integrity_score'] = int(score)
    
    def _load_history(self) -> List[Dict]:
        """Load historical integrity data"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        return []
    
    def _save_to_history(self):
        """Save current metrics to history"""
        self.history.append(self.current_metrics.copy())
        
        # Keep only last 52 weeks (1 year)
        if len(self.history) > 52:
            self.history = self.history[-52:]
        
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def _generate_trend_report(self):
        """Generate trend analysis report"""
        if len(self.history) < 2:
            return
        
        print("📈 Generating trend report...")
        
        # Create DataFrame from history
        df = pd.DataFrame(self.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NIS Engineering Integrity Trends', fontsize=16)
        
        # Integrity score trend
        axes[0, 0].plot(df['timestamp'], df['integrity_score'], marker='o')
        axes[0, 0].set_title('Integrity Score Over Time')
        axes[0, 0].set_ylabel('Score (0-100)')
        axes[0, 0].grid(True)
        
        # Hype language trend
        axes[0, 1].plot(df['timestamp'], df['hype_instances'], marker='o', color='red')
        axes[0, 1].set_title('Hype Language Usage')
        axes[0, 1].set_ylabel('Instances')
        axes[0, 1].grid(True)
        
        # Test coverage trend
        axes[1, 0].plot(df['timestamp'], df['test_coverage'], marker='o', color='green')
        axes[1, 0].set_title('Test Coverage')
        axes[1, 0].set_ylabel('Coverage %')
        axes[1, 0].grid(True)
        
        # Hardcoded values trend
        axes[1, 1].plot(df['timestamp'], df['hardcoded_values'], marker='o', color='orange')
        axes[1, 1].set_title('Hardcoded Values')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
        
        # Rotate x-axis labels
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.reports_dir / f"integrity_trends_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Trend report saved: {plot_file}")
    
    def _generate_weekly_summary(self):
        """Generate weekly summary report"""
        print("📋 Generating weekly summary...")
        
        # Calculate changes from last week
        changes = {}
        if len(self.history) >= 2:
            last_week = self.history[-2]
            for key in ['integrity_score', 'hype_instances', 'hardcoded_values', 'test_coverage']:
                current = self.current_metrics[key]
                previous = last_week.get(key, 0)
                changes[key] = current - previous
        
        # Generate summary
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'integrity_score': self.current_metrics['integrity_score'],
            'score_change': changes.get('integrity_score', 0),
            'hype_instances': self.current_metrics['hype_instances'],
            'hype_change': changes.get('hype_instances', 0),
            'hardcoded_values': self.current_metrics['hardcoded_values'],
            'hardcoded_change': changes.get('hardcoded_values', 0),
            'test_coverage': self.current_metrics['test_coverage'],
            'coverage_change': changes.get('test_coverage', 0),
            'commits_this_week': self.current_metrics['git_commits_since_last'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save summary
        summary_file = self.reports_dir / f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📋 Weekly summary saved: {summary_file}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 WEEKLY INTEGRITY SUMMARY")
        print("=" * 50)
        print(f"📅 Date: {summary['date']}")
        print(f"🎯 Integrity Score: {summary['integrity_score']}/100 ({summary['score_change']:+d})")
        print(f"🎭 Hype Language: {summary['hype_instances']} instances ({summary['hype_change']:+d})")
        print(f"🔍 Hardcoded Values: {summary['hardcoded_values']} ({summary['hardcoded_change']:+d})")
        print(f"🧪 Test Coverage: {summary['test_coverage']}% ({summary['coverage_change']:+d})")
        print(f"🔄 Commits This Week: {summary['commits_this_week']}")
        
        if summary['recommendations']:
            print(f"\n📝 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current metrics"""
        recommendations = []
        
        if self.current_metrics['integrity_score'] < 80:
            recommendations.append("Focus on improving overall integrity score")
        
        if self.current_metrics['hype_instances'] > 5:
            recommendations.append("Reduce hype language usage in documentation")
        
        if self.current_metrics['hardcoded_values'] > 3:
            recommendations.append("Replace hardcoded values with calculated metrics")
        
        if self.current_metrics['test_coverage'] < 50:
            recommendations.append("Increase test coverage for better validation")
        
        if self.current_metrics['benchmark_coverage'] < 70:
            recommendations.append("Add benchmarks for performance claims")
        
        if self.current_metrics['git_commits_since_last'] == 0:
            recommendations.append("Consider regular development activity")
        
        return recommendations

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NIS Weekly Integrity Monitor")
    parser.add_argument("--project-path", default=".", help="Path to project")
    parser.add_argument("--generate-plots", action="store_true", help="Generate trend plots")
    
    args = parser.parse_args()
    
    # Install required packages if not available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'matplotlib', 'pandas'])
        import matplotlib.pyplot as plt
        import pandas as pd
    
    monitor = IntegrityMonitor(args.project_path)
    results = monitor.run_weekly_check()
    
    # Exit with appropriate code
    if results['integrity_score'] >= 80:
        print("\n✅ Integrity monitoring: GOOD")
        return 0
    else:
        print("\n⚠️  Integrity monitoring: NEEDS ATTENTION")
        return 1

if __name__ == "__main__":
    exit(main()) 