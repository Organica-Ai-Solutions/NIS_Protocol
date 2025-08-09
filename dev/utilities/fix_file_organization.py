#!/usr/bin/env python3
"""
🔧 NIS Protocol File Organization Auto-Fix Utility

Automatically fixes common file organization violations by moving files
to their proper locations according to the organization rules.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class FileOrganizationFixer:
    def __init__(self, project_root: str = ".", dry_run: bool = False):
        self.project_root = Path(project_root).resolve()
        self.dry_run = dry_run
        self.fixes_applied = []
        self.errors = []
        
    def fix_organization(self) -> Dict:
        """Apply automatic fixes for file organization violations"""
        print("🔧 NIS Protocol File Organization Auto-Fix")
        print(f"📂 Project Root: {self.project_root}")
        print(f"🧪 Dry Run Mode: {'YES' if self.dry_run else 'NO'}")
        print("=" * 60)
        
        # Create necessary directories
        self._create_directories()
        
        # Fix root directory violations
        self._fix_root_violations()
        
        # Fix misplaced files
        self._fix_misplaced_files()
        
        # Clean up temporary files
        self._cleanup_temporary_files()
        
        return self._generate_fix_report()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            'dev/utilities',
            'dev/testing/root_cleanup',
            'dev/testing/benchmarks',
            'dev/testing/integration',
            'dev/documentation',
            'system/utilities/reports',
            'scripts/installation',
            'scripts/deployment',
            'scripts/maintenance',
            'scripts/utilities',
            'logs/archived',
            'logs/application'
        ]
        
        print("📁 Creating directory structure...")
        for dir_path in directories:
            full_path = self.project_root / dir_path
            if not self.dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path}")
    
    def _fix_root_violations(self):
        """Fix files that shouldn't be in root directory"""
        print("\n🚨 Fixing root directory violations...")
        
        allowed_files = {
            'main.py', 'requirements.txt', 'docker-compose.yml', 'Dockerfile',
            'README.md', 'LICENSE', '.gitignore', '.dockerignore', '.gitattributes',
            '.cursorrules', '.nojekyll', 'CNAME', 'start.sh', 'stop.sh', 'reset.sh',
            'NIS_Protocol_v3_COMPLETE_Postman_Collection.json'
        }
        
        for item in self.project_root.iterdir():
            if item.is_file() and item.name not in allowed_files:
                # Skip hidden files and system files
                if item.name.startswith('.') and item.name not in {'.env', '.env~'}:
                    continue
                    
                destination = self._determine_destination(item.name)
                if destination:
                    self._move_file(item, destination, "ROOT_VIOLATION")
    
    def _fix_misplaced_files(self):
        """Fix files that are in wrong subdirectories"""
        print("\n📁 Fixing misplaced files...")
        
        # Find and fix test files
        self._fix_test_files()
        
        # Find and fix utility files
        self._fix_utility_files()
        
        # Find and fix report files
        self._fix_report_files()
    
    def _fix_test_files(self):
        """Move test files to proper testing directories"""
        # Define directories to skip (external dependencies)
        skip_dirs = {'.venv', 'venv', 'env', 'models/bitnet/venv', '.git', '__pycache__', 'node_modules'}
        
        for file_path in self.project_root.rglob('test_*.py'):
            # Skip if already in proper location
            relative_path = file_path.relative_to(self.project_root)
            relative_str = str(relative_path).replace('\\', '/')
            
            # Skip virtual environments and external dependencies
            if any(skip_dir in relative_str for skip_dir in skip_dirs):
                continue
                
            if relative_str.startswith('dev/testing') or relative_str.startswith('.'):
                continue
                
            # Only move files in our project directories
            if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/', 'endpoint_fixes_backup/']):
                continue
                
            destination = self.project_root / 'dev/testing/root_cleanup' / file_path.name
            self._move_file(file_path, destination, "MISPLACED_TEST")
        
        # Also check for *_test.py files
        for file_path in self.project_root.rglob('*_test.py'):
            relative_path = file_path.relative_to(self.project_root)
            relative_str = str(relative_path).replace('\\', '/')
            
            # Skip virtual environments and external dependencies
            if any(skip_dir in relative_str for skip_dir in skip_dirs):
                continue
                
            if relative_str.startswith('dev/testing') or relative_str.startswith('.'):
                continue
                
            # Only move files in our project directories
            if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/', 'endpoint_fixes_backup/']):
                continue
                
            destination = self.project_root / 'dev/testing/root_cleanup' / file_path.name
            self._move_file(file_path, destination, "MISPLACED_TEST")
    
    def _fix_utility_files(self):
        """Move utility files to proper directories"""
        utility_patterns = ['*_utility.py', 'fix_*.py', 'generate_*.py']
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
                   relative_str.startswith('system/utilities') or \
                   relative_str.startswith('.'):
                    continue
                
                # Only move files in our project directories
                if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/']):
                    continue
                
                # Determine if it's a dev utility or system utility
                if any(keyword in file_path.name.lower() for keyword in ['nvidia', 'terminal', 'endpoint']):
                    destination = self.project_root / 'dev/utilities' / file_path.name
                else:
                    destination = self.project_root / 'system/utilities' / file_path.name
                
                self._move_file(file_path, destination, "MISPLACED_UTILITY")
    
    def _fix_report_files(self):
        """Move report files to proper directories"""
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
                if relative_str.startswith('system/utilities/reports') or \
                   relative_str.startswith('.'):
                    continue
                
                # Only move files in our project directories
                if not any(relative_str.startswith(proj_dir) for proj_dir in ['dev/', 'src/', 'system/', 'scripts/', 'endpoint_fixes_backup/']):
                    continue
                
                destination = self.project_root / 'system/utilities/reports' / file_path.name
                self._move_file(file_path, destination, "MISPLACED_REPORT")
    
    def _cleanup_temporary_files(self):
        """Clean up temporary files"""
        print("\n🧹 Cleaning up temporary files...")
        
        temp_patterns = ['*.tmp', '*~', 'temp_*']
        delete_files = ['commit_message.txt', 'test_curl.sh~', '.env~']
        
        # Delete specific temporary files
        for filename in delete_files:
            file_path = self.project_root / filename
            if file_path.exists():
                self._delete_file(file_path, "TEMPORARY_FILE")
        
        # Move other temporary files to cache
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                relative_path = file_path.relative_to(self.project_root)
                
                # Skip if already in proper location
                if str(relative_path).startswith('cache/') or \
                   str(relative_path).startswith('.git/'):
                    continue
                
                # Move to cache directory
                cache_dir = self.project_root / 'cache'
                if not self.dry_run:
                    cache_dir.mkdir(exist_ok=True)
                    
                destination = cache_dir / file_path.name
                self._move_file(file_path, destination, "TEMPORARY_FILE")
    
    def _determine_destination(self, filename: str) -> Path:
        """Determine where a file should be moved"""
        if filename.startswith('test_') and filename.endswith('.py'):
            return self.project_root / 'dev/testing/root_cleanup' / filename
        elif filename.endswith('_test.py'):
            return self.project_root / 'dev/testing/root_cleanup' / filename
        elif any(pattern in filename for pattern in ['utility', 'fix_', 'generate_']):
            return self.project_root / 'dev/utilities' / filename
        elif filename.endswith('_report.json') or filename.endswith('_summary.md'):
            return self.project_root / 'system/utilities/reports' / filename
        elif filename.endswith('.log'):
            return self.project_root / 'logs/archived' / filename
        elif filename.startswith('install_') and filename.endswith('.py'):
            return self.project_root / 'scripts/installation' / filename
        elif filename.endswith('.sh') and filename not in {'start.sh', 'stop.sh', 'reset.sh'}:
            return self.project_root / 'scripts/utilities' / filename
        elif filename.endswith('.md') and filename != 'README.md':
            if 'NVIDIA' in filename or 'NEMOTRON' in filename:
                return self.project_root / 'dev/documentation' / filename
            else:
                return self.project_root / 'system/docs' / filename
        elif filename in ['environment-template.txt']:
            return self.project_root / 'dev' / filename
        elif filename.endswith('.bat') or filename.endswith('.ps1'):
            return self.project_root / 'scripts/utilities' / filename
        else:
            return None  # Don't move unknown files automatically
    
    def _move_file(self, source: Path, destination: Path, fix_type: str):
        """Move a file from source to destination"""
        try:
            if not self.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            
            self.fixes_applied.append({
                'type': fix_type,
                'action': 'MOVE',
                'source': str(source.relative_to(self.project_root)),
                'destination': str(destination.relative_to(self.project_root)),
                'timestamp': datetime.now().isoformat()
            })
            
            action = "[DRY RUN] Would move" if self.dry_run else "Moved"
            print(f"  📦 {action}: {source.name} → {destination.relative_to(self.project_root)}")
            
        except Exception as e:
            self.errors.append({
                'type': fix_type,
                'action': 'MOVE',
                'source': str(source),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print(f"  ❌ Error moving {source.name}: {e}")
    
    def _delete_file(self, file_path: Path, fix_type: str):
        """Delete a temporary file"""
        try:
            if not self.dry_run:
                file_path.unlink()
            
            self.fixes_applied.append({
                'type': fix_type,
                'action': 'DELETE',
                'source': str(file_path.relative_to(self.project_root)),
                'destination': 'DELETED',
                'timestamp': datetime.now().isoformat()
            })
            
            action = "[DRY RUN] Would delete" if self.dry_run else "Deleted"
            print(f"  🗑️  {action}: {file_path.name}")
            
        except Exception as e:
            self.errors.append({
                'type': fix_type,
                'action': 'DELETE',
                'source': str(file_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print(f"  ❌ Error deleting {file_path.name}: {e}")
    
    def _generate_fix_report(self) -> Dict:
        """Generate report of fixes applied"""
        return {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'dry_run': self.dry_run,
            'summary': {
                'fixes_applied': len(self.fixes_applied),
                'errors_encountered': len(self.errors),
                'moves': len([f for f in self.fixes_applied if f['action'] == 'MOVE']),
                'deletions': len([f for f in self.fixes_applied if f['action'] == 'DELETE'])
            },
            'fixes': self.fixes_applied,
            'errors': self.errors
        }

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix NIS Protocol file organization')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making changes')
    parser.add_argument('--project-root', default='.', 
                       help='Path to project root directory')
    
    args = parser.parse_args()
    
    fixer = FileOrganizationFixer(args.project_root, args.dry_run)
    report = fixer.fix_organization()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🔧 FIX REPORT SUMMARY")
    print("=" * 60)
    print(f"📦 Files Moved: {report['summary']['moves']}")
    print(f"🗑️  Files Deleted: {report['summary']['deletions']}")
    print(f"❌ Errors: {report['summary']['errors_encountered']}")
    
    if report['summary']['fixes_applied'] == 0:
        print("🎉 No fixes needed - file organization is already compliant!")
    elif args.dry_run:
        print(f"\n🧪 DRY RUN: {report['summary']['fixes_applied']} fixes would be applied")
        print("Run without --dry-run to apply these fixes")
    else:
        print(f"\n✅ Applied {report['summary']['fixes_applied']} fixes successfully")
    
    if report['errors']:
        print("\n⚠️  Some errors occurred:")
        for error in report['errors']:
            print(f"  • {error['source']}: {error['error']}")
    
    # Save report
    if not args.dry_run:
        report_path = Path('system/utilities/reports/file_organization_fixes.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Fix report saved: {report_path}")

if __name__ == "__main__":
    main()