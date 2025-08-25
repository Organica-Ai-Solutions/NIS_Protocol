"""
UI Resource Generator for mcp-ui Integration

Generates interactive UI components for different types of data and workflows.
Creates mcp-ui compatible resources that render in the client.
"""

import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class UIResourceGenerator:
    """
    Generates mcp-ui compatible UI resources for different data types.
    
    Provides factory methods for common UI patterns like data grids,
    progress monitors, tabbed viewers, etc.
    """
    
    def __init__(self):
        self.supported_components = [
            "dataGrid", "tabs", "progressWithLogs", "timeline", 
            "diffView", "cards", "dashboard", "form"
        ]
        
    def get_supported_components(self) -> List[str]:
        """Get list of supported UI component types."""
        return self.supported_components.copy()
        
    def create_data_grid(self, items: List[Dict], title: str = "Data", 
                        searchable: bool = True, pagination: bool = True,
                        actions: List[Dict] = None) -> Dict[str, Any]:
        """Create a data grid UI resource in official mcp-ui format."""
        actions = actions or []
        
        # Generate columns from first item if available
        columns = []
        if items:
            for key in items[0].keys():
                columns.append({
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "sortable": True,
                    "type": self._infer_column_type(items[0][key])
                })
                
        # Official mcp-ui UIResource format
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://datagrid/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_data_grid_html(
                    title, columns, items, searchable, pagination, actions
                )
            }
        }
        
        return ui_resource
        
    def create_tabbed_viewer(self, tabs: Dict[str, Any], title: str = "Details") -> Dict[str, Any]:
        """Create a tabbed viewer UI resource."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://tabs/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_tabbed_viewer_html(title, tabs)
            }
        }
        
        return ui_resource
        
    def create_progress_monitor(self, run_id: str, status: str, progress: int,
                              logs: List[str] = None, cancelable: bool = True) -> Dict[str, Any]:
        """Create a progress monitor UI resource."""
        logs = logs or []
        
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://progress/{run_id}",
                "mimeType": "text/html",
                "text": self._generate_progress_monitor_html(
                    run_id, status, progress, logs, cancelable
                )
            }
        }
        
        return ui_resource
        
    def create_pipeline_status(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pipeline status dashboard."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://pipeline-status/{status_data.get('run_id', 'unknown')}",
                "mimeType": "text/html",
                "text": self._generate_pipeline_status_html(status_data)
            }
        }
        
        return ui_resource
        
    def create_research_plan_tree(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a research plan tree view."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://research-plan/{plan_data.get('plan_id', 'unknown')}",
                "mimeType": "text/html", 
                "text": self._generate_research_plan_html(plan_data)
            }
        }
        
        return ui_resource
        
    def create_research_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Create a research results viewer."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://research-results/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_research_results_html(results)
            }
        }
        
        return ui_resource
        
    def create_audit_timeline(self, timeline: List[Dict]) -> Dict[str, Any]:
        """Create an audit timeline view."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://audit-timeline/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_audit_timeline_html(timeline)
            }
        }
        
        return ui_resource
        
    def create_analysis_dashboard(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an analysis dashboard."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://analysis-dashboard/{analysis_data.get('analysis_id', 'unknown')}",
                "mimeType": "text/html",
                "text": self._generate_analysis_dashboard_html(analysis_data)
            }
        }
        
        return ui_resource
        
    def create_diff_viewer(self, diff_data: List[Dict]) -> Dict[str, Any]:
        """Create a code diff viewer."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://diff-viewer/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_diff_viewer_html(diff_data)
            }
        }
        
        return ui_resource
        
    def create_code_review_panel(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a code review panel."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://code-review/{review_data.get('review_id', 'unknown')}",
                "mimeType": "text/html",
                "text": self._generate_code_review_html(review_data)
            }
        }
        
        return ui_resource
        
    def create_data_viewer(self, data: Any, title: str = "Data") -> Dict[str, Any]:
        """Create a generic data viewer (fallback)."""
        ui_resource = {
            "type": "resource",
            "resource": {
                "uri": f"ui://data-viewer/{int(time.time() * 1000)}",
                "mimeType": "text/html",
                "text": self._generate_data_viewer_html(title, data)
            }
        }
        
        return ui_resource
        
    def _infer_column_type(self, value: Any) -> str:
        """Infer column type from value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, (list, dict)):
            return "object"
        else:
            return "string"
            
    def _generate_data_grid_html(self, title: str, columns: List[Dict], 
                                items: List[Dict], searchable: bool, 
                                pagination: bool, actions: List[Dict]) -> str:
        """Generate HTML for data grid component."""
        # Basic HTML structure with inline CSS and JavaScript
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        .data-grid {{ background: white; border-radius: 8px; border: 1px solid #e1e5e9; }}
        .grid-header {{ padding: 16px; border-bottom: 1px solid #e1e5e9; display: flex; justify-content: space-between; align-items: center; }}
        .grid-title {{ font-size: 18px; font-weight: 600; }}
        .search-box {{ padding: 8px 12px; border: 1px solid #d0d7de; border-radius: 6px; }}
        .grid-table {{ width: 100%; border-collapse: collapse; }}
        .grid-table th {{ background: #f6f8fa; padding: 12px; text-align: left; border-bottom: 1px solid #d0d7de; font-weight: 600; }}
        .grid-table td {{ padding: 12px; border-bottom: 1px solid #e1e5e9; }}
        .grid-table tr:hover {{ background: #f6f8fa; }}
        .action-btn {{ background: #0969da; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; margin-right: 8px; }}
        .action-btn:hover {{ background: #0550ae; }}
        .pagination {{ padding: 16px; display: flex; justify-content: center; gap: 8px; }}
        .page-btn {{ padding: 8px 12px; border: 1px solid #d0d7de; background: white; cursor: pointer; border-radius: 6px; }}
        .page-btn.active {{ background: #0969da; color: white; }}
    </style>
</head>
<body>
    <div class="data-grid">
        <div class="grid-header">
            <h2 class="grid-title">{title}</h2>
            {f'<input type="text" class="search-box" placeholder="Search..." oninput="filterTable(this.value)">' if searchable else ''}
        </div>
        <table class="grid-table" id="dataTable">
            <thead>
                <tr>
                    {self._generate_table_headers(columns, actions)}
                </tr>
            </thead>
            <tbody>
                {self._generate_table_rows(columns, items, actions)}
            </tbody>
        </table>
        {f'<div class="pagination" id="pagination"></div>' if pagination else ''}
    </div>

    <script>
        function filterTable(query) {{
            const table = document.getElementById('dataTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {{
                const row = rows[i];
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query.toLowerCase()) ? '' : 'none';
            }}
        }}
        
        function executeAction(action, rowData) {{
            window.parent.postMessage({{
                type: 'tool',
                payload: {{
                    toolName: action,
                    params: rowData
                }}
            }}, '*');
        }}
        
        function previewItem(id) {{
            window.parent.postMessage({{
                type: 'intent',
                payload: {{
                    intent: 'open_preview',
                    params: {{ id: id }}
                }}
            }}, '*');
        }}
    </script>
</body>
</html>"""
        return html
        
    def _generate_table_headers(self, columns: List[Dict], actions: List[Dict]) -> str:
        """Generate table headers."""
        headers = []
        for col in columns:
            headers.append(f'<th>{col["label"]}</th>')
        if actions:
            headers.append('<th>Actions</th>')
        return "".join(headers)
        
    def _generate_table_rows(self, columns: List[Dict], items: List[Dict], actions: List[Dict]) -> str:
        """Generate table rows."""
        rows = []
        for item in items:
            row_cells = []
            for col in columns:
                value = item.get(col["key"], "")
                if col["key"] == "id" or col["key"] == "name":
                    # Make ID/name clickable for preview
                    row_cells.append(f'<td><a href="#" onclick="previewItem(\'{value}\')">{value}</a></td>')
                else:
                    row_cells.append(f'<td>{self._format_cell_value(value)}</td>')
            
            if actions:
                action_btns = []
                for action in actions:
                    action_btns.append(f'<button class="action-btn" onclick="executeAction(\'{action["name"]}\', {json.dumps(item)})">{action["label"]}</button>')
                row_cells.append(f'<td>{"".join(action_btns)}</td>')
                
            rows.append(f'<tr>{"".join(row_cells)}</tr>')
        return "".join(rows)
        
    def _format_cell_value(self, value: Any) -> str:
        """Format cell value for display."""
        if isinstance(value, dict):
            return f'<details><summary>Object</summary><pre>{json.dumps(value, indent=2)}</pre></details>'
        elif isinstance(value, list):
            return f'<details><summary>Array ({len(value)})</summary><pre>{json.dumps(value, indent=2)}</pre></details>'
        elif isinstance(value, bool):
            return "✓" if value else "✗"
        else:
            return str(value)
            
    def _generate_tabbed_viewer_html(self, title: str, tabs: Dict[str, Any]) -> str:
        """Generate HTML for tabbed viewer component."""
        tab_headers = []
        tab_contents = []
        
        for i, (tab_name, tab_data) in enumerate(tabs.items()):
            active_class = "active" if i == 0 else ""
            tab_headers.append(f'<button class="tab-btn {active_class}" onclick="showTab(\'{tab_name}\')">{tab_name}</button>')
            
            display_style = "block" if i == 0 else "none"
            content = self._format_tab_content(tab_data)
            tab_contents.append(f'<div id="{tab_name}" class="tab-content" style="display: {display_style}">{content}</div>')
            
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        .tab-container {{ background: white; border-radius: 8px; border: 1px solid #e1e5e9; }}
        .tab-header {{ display: flex; border-bottom: 1px solid #e1e5e9; }}
        .tab-btn {{ background: none; border: none; padding: 12px 20px; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab-btn.active {{ border-bottom-color: #0969da; color: #0969da; background: #f6f8fa; }}
        .tab-content {{ padding: 20px; }}
        .json-viewer {{ background: #f6f8fa; padding: 16px; border-radius: 6px; font-family: Monaco, 'Courier New', monospace; white-space: pre-wrap; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e1e5e9; }}
        th {{ background: #f6f8fa; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="tab-container">
        <div class="tab-header">
            {" ".join(tab_headers)}
        </div>
        {" ".join(tab_contents)}
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.style.display = 'none');
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).style.display = 'block';
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>"""
        return html
        
    def _format_tab_content(self, data: Any) -> str:
        """Format content for a tab."""
        if isinstance(data, dict):
            if self._looks_like_table_data(data):
                return self._dict_to_table(data)
            else:
                return f'<div class="json-viewer">{json.dumps(data, indent=2)}</div>'
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return self._list_to_table(data)
            else:
                return f'<div class="json-viewer">{json.dumps(data, indent=2)}</div>'
        else:
            return f'<div class="json-viewer">{str(data)}</div>'
            
    def _looks_like_table_data(self, data: dict) -> bool:
        """Check if dict looks like table data."""
        if not data:
            return False
        
        # Check if all values are simple types (not nested objects/arrays)
        for value in data.values():
            if isinstance(value, (dict, list)):
                return False
        return True
        
    def _dict_to_table(self, data: dict) -> str:
        """Convert dict to HTML table."""
        rows = []
        for key, value in data.items():
            rows.append(f'<tr><th>{key}</th><td>{self._format_cell_value(value)}</td></tr>')
        return f'<table>{"".join(rows)}</table>'
        
    def _list_to_table(self, data: list) -> str:
        """Convert list of dicts to HTML table."""
        if not data:
            return '<p>No data</p>'
            
        # Get headers from first item
        headers = list(data[0].keys())
        header_row = "".join(f'<th>{h}</th>' for h in headers)
        
        # Generate rows
        rows = []
        for item in data:
            row_cells = []
            for header in headers:
                value = item.get(header, "")
                row_cells.append(f'<td>{self._format_cell_value(value)}</td>')
            rows.append(f'<tr>{"".join(row_cells)}</tr>')
            
        return f'<table><thead><tr>{header_row}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        
    def _generate_progress_monitor_html(self, run_id: str, status: str, 
                                      progress: int, logs: List[str], 
                                      cancelable: bool) -> str:
        """Generate HTML for progress monitor."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pipeline Progress</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        .progress-container {{ background: white; border-radius: 8px; border: 1px solid #e1e5e9; padding: 20px; }}
        .progress-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .run-id {{ font-family: Monaco, 'Courier New', monospace; background: #f6f8fa; padding: 4px 8px; border-radius: 4px; }}
        .status {{ padding: 4px 12px; border-radius: 12px; font-size: 14px; font-weight: 500; }}
        .status.running {{ background: #dbeafe; color: #1e40af; }}
        .status.completed {{ background: #dcfce7; color: #15803d; }}
        .status.failed {{ background: #fee2e2; color: #dc2626; }}
        .progress-bar {{ width: 100%; height: 8px; background: #f3f4f6; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
        .progress-fill {{ height: 100%; background: #0969da; transition: width 0.3s ease; }}
        .logs-container {{ background: #1f2937; color: #f9fafb; padding: 16px; border-radius: 6px; font-family: Monaco, 'Courier New', monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }}
        .cancel-btn {{ background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; }}
        .cancel-btn:hover {{ background: #b91c1c; }}
        .refresh-btn {{ background: #0969da; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin-right: 8px; }}
        .refresh-btn:hover {{ background: #0550ae; }}
    </style>
</head>
<body>
    <div class="progress-container">
        <div class="progress-header">
            <div>
                <h3>Pipeline Execution</h3>
                <span class="run-id">Run ID: {run_id}</span>
            </div>
            <div>
                <span class="status {status.lower()}">{status.upper()}</span>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress}%"></div>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <strong>{progress}% Complete</strong>
        </div>
        
        <div style="margin-bottom: 20px;">
            <button class="refresh-btn" onclick="refreshStatus()">Refresh Status</button>
            {f'<button class="cancel-btn" onclick="cancelRun()">Cancel Run</button>' if cancelable and status.lower() == 'running' else ''}
        </div>
        
        <h4>Execution Logs</h4>
        <div class="logs-container" id="logs">
            {self._format_logs(logs)}
        </div>
    </div>

    <script>
        function refreshStatus() {{
            window.parent.postMessage({{
                type: 'tool',
                payload: {{
                    toolName: 'pipeline.status',
                    params: {{ run_id: '{run_id}' }}
                }}
            }}, '*');
        }}
        
        function cancelRun() {{
            if (confirm('Are you sure you want to cancel this pipeline run?')) {{
                window.parent.postMessage({{
                    type: 'tool',
                    payload: {{
                        toolName: 'pipeline.cancel',
                        params: {{ run_id: '{run_id}' }}
                    }}
                }}, '*');
            }}
        }}
        
        // Auto-refresh every 5 seconds for running pipelines
        {f"setInterval(refreshStatus, 5000);" if status.lower() == 'running' else ''}
    </script>
</body>
</html>"""
        return html
        
    def _format_logs(self, logs: List[str]) -> str:
        """Format logs for display."""
        if not logs:
            return '<div style="color: #9ca3af;">No logs available</div>'
        
        formatted_logs = []
        for log in logs:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_logs.append(f'<div>[{timestamp}] {log}</div>')
        return "".join(formatted_logs)
        
    def _generate_pipeline_status_html(self, status_data: Dict[str, Any]) -> str:
        """Generate HTML for pipeline status dashboard."""
        # This would contain a more detailed pipeline status view
        # For brevity, using the progress monitor as base
        return self._generate_progress_monitor_html(
            status_data.get("run_id", "unknown"),
            status_data.get("status", "unknown"),
            status_data.get("progress", 0),
            status_data.get("logs", []),
            True
        )
        
    def _generate_research_plan_html(self, plan_data: Dict[str, Any]) -> str:
        """Generate HTML for research plan tree."""
        # Simplified version - would create an interactive tree view
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Plan</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        .plan-container {{ background: white; border-radius: 8px; border: 1px solid #e1e5e9; padding: 20px; }}
        .plan-section {{ margin-bottom: 24px; }}
        .plan-section h3 {{ color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
        .objectives-list {{ list-style-type: none; padding: 0; }}
        .objectives-list li {{ background: #f3f4f6; margin: 8px 0; padding: 12px; border-radius: 6px; border-left: 4px solid #0969da; }}
    </style>
</head>
<body>
    <div class="plan-container">
        <h2>{plan_data.get('goal', 'Research Plan')}</h2>
        
        <div class="plan-section">
            <h3>Objectives</h3>
            <ul class="objectives-list">
                {self._format_list_items(plan_data.get('objectives', []))}
            </ul>
        </div>
        
        <div class="plan-section">
            <h3>Methodology</h3>
            <div class="json-viewer">{json.dumps(plan_data.get('methodology', {}), indent=2)}</div>
        </div>
    </div>
</body>
</html>"""
        return html
        
    def _format_list_items(self, items: List[str]) -> str:
        """Format list items for HTML."""
        return "".join(f'<li>{item}</li>' for item in items)
        
    def _generate_research_results_html(self, results: List[Dict]) -> str:
        """Generate HTML for research results."""
        # Use data grid for research results
        return self._generate_data_grid_html(
            "Research Results", 
            [
                {"key": "title", "label": "Title", "type": "string"},
                {"key": "authors", "label": "Authors", "type": "array"},
                {"key": "year", "label": "Year", "type": "number"},
                {"key": "relevance_score", "label": "Relevance", "type": "number"}
            ],
            results, True, True, []
        )
        
    def _generate_audit_timeline_html(self, timeline: List[Dict]) -> str:
        """Generate HTML for audit timeline."""
        # Simplified timeline view
        return self._generate_data_grid_html(
            "Audit Timeline",
            [
                {"key": "timestamp", "label": "Time", "type": "string"},
                {"key": "event", "label": "Event", "type": "string"},
                {"key": "component", "label": "Component", "type": "string"},
                {"key": "details", "label": "Details", "type": "string"}
            ],
            timeline, True, True, []
        )
        
    def _generate_analysis_dashboard_html(self, analysis_data: Dict[str, Any]) -> str:
        """Generate HTML for analysis dashboard."""
        return self._generate_tabbed_viewer_html(
            "Analysis Dashboard",
            {
                "Performance": analysis_data.get("performance", {}),
                "Patterns": analysis_data.get("patterns", []),
                "Anomalies": analysis_data.get("anomalies", []),
                "Recommendations": analysis_data.get("recommendations", [])
            }
        )
        
    def _generate_diff_viewer_html(self, diff_data: List[Dict]) -> str:
        """Generate HTML for diff viewer."""
        return self._generate_data_grid_html(
            "Code Changes",
            [
                {"key": "type", "label": "Type", "type": "string"},
                {"key": "line_start", "label": "Line", "type": "number"},
                {"key": "old_content", "label": "Old", "type": "string"},
                {"key": "new_content", "label": "New", "type": "string"}
            ],
            diff_data, False, False, []
        )
        
    def _generate_code_review_html(self, review_data: Dict[str, Any]) -> str:
        """Generate HTML for code review panel."""
        return self._generate_tabbed_viewer_html(
            f"Code Review (Score: {review_data.get('overall_score', 'N/A')})",
            {
                "Summary": {
                    "score": review_data.get("overall_score"),
                    "grade": review_data.get("overall_grade"),
                    "summary": review_data.get("summary")
                },
                "Issues": review_data.get("issues", []),
                "Recommendations": review_data.get("recommendations", []),
                "Categories": review_data.get("categories", {})
            }
        )
        
    def _generate_data_viewer_html(self, title: str, data: Any) -> str:
        """Generate HTML for generic data viewer."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        .data-container {{ background: white; border-radius: 8px; border: 1px solid #e1e5e9; padding: 20px; }}
        .json-viewer {{ background: #f6f8fa; padding: 16px; border-radius: 6px; font-family: Monaco, 'Courier New', monospace; white-space: pre-wrap; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="data-container">
        <h2>{title}</h2>
        <div class="json-viewer">{json.dumps(data, indent=2, default=str)}</div>
    </div>
</body>
</html>"""
        return html
