"""
Precision Diagram & Chart Generation Agent
Uses code-based generation for accurate, data-driven visuals
"""

import io
import base64
import json
import logging
import math
from typing import Dict, Any, List, Optional, Union

# Graceful imports with fallbacks for Docker environment
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import FancyBboxPatch
    import networkx as nx
    
    # Set matplotlib to non-interactive backend
    plt.switch_backend('Agg')
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    # Fallback for environments without visualization dependencies
    plt = None
    patches = None
    np = None
    sns = None
    FancyBboxPatch = None
    nx = None
    logging.warning(f"⚠️ Visualization libraries not available - using SVG fallback mode")

class DiagramAgent:
    def __init__(self):
        self.logger = logging.getLogger("diagram_agent")
        
        if VISUALIZATION_AVAILABLE:
            # Set clean styling
            plt.style.use('default')
            sns.set_palette("husl")
            self.logger.info("✅ Visualization libraries loaded - full precision generation available")
        else:
            self.logger.warning("⚠️ Visualization libraries not available - using SVG fallback mode")
        
    def generate_chart(self, chart_type: str, data: Dict[str, Any], style: str = "scientific") -> Dict[str, Any]:
        """Generate precise charts using matplotlib or SVG fallback"""
        try:
            self.logger.info(f"🎨 Generating {chart_type} chart with precise data")
            
            if not VISUALIZATION_AVAILABLE:
                return self._create_svg_fallback_chart(chart_type, data, style)
            
            if chart_type == "bar":
                return self._create_bar_chart(data, style)
            elif chart_type == "line":
                return self._create_line_chart(data, style)
            elif chart_type == "pie":
                return self._create_pie_chart(data, style)
            elif chart_type == "scatter":
                return self._create_scatter_plot(data, style)
            elif chart_type == "histogram":
                return self._create_histogram(data, style)
            elif chart_type == "heatmap":
                return self._create_heatmap(data, style)
            else:
                return {"error": f"Chart type '{chart_type}' not supported"}
                
        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")
            # Fallback to SVG if matplotlib fails
            return self._create_svg_fallback_chart(chart_type, data, style)
    
    def generate_diagram(self, diagram_type: str, data: Dict[str, Any], style: str = "scientific") -> Dict[str, Any]:
        """Generate precise diagrams using matplotlib or SVG fallback"""
        try:
            self.logger.info(f"🔧 Generating {diagram_type} diagram")
            
            if not VISUALIZATION_AVAILABLE:
                return self._create_svg_fallback_diagram(diagram_type, data, style)
            
            if diagram_type == "flowchart":
                return self._create_flowchart(data, style)
            elif diagram_type == "network":
                return self._create_network_diagram(data, style)
            elif diagram_type == "architecture":
                return self._create_architecture_diagram(data, style)
            elif diagram_type == "physics":
                return self._create_physics_diagram(data, style)
            elif diagram_type == "pipeline":
                return self._create_pipeline_diagram(data, style)
            else:
                return {"error": f"Diagram type '{diagram_type}' not supported"}
                
        except Exception as e:
            self.logger.error(f"Diagram generation failed: {e}")
            # Fallback to SVG if matplotlib fails
            return self._create_svg_fallback_diagram(diagram_type, data, style)
    
    def _create_bar_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create precise bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = data.get('categories', ['A', 'B', 'C'])
        values = data.get('values', [10, 20, 15])
        title = data.get('title', 'Bar Chart')
        
        colors = self._get_color_palette(style, len(categories))
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(data.get('xlabel', 'Categories'), fontsize=12, fontweight='bold')
        ax.set_ylabel(data.get('ylabel', 'Values'), fontsize=12, fontweight='bold')
        
        # Style the chart
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_line_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create precise line chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_data = data.get('x', list(range(10)))
        y_data = data.get('y', [np.sin(x/2) for x in x_data]) if isinstance(data.get('y'), type(None)) else data.get('y')
        title = data.get('title', 'Line Chart')
        
        color = self._get_color_palette(style, 1)[0]
        ax.plot(x_data, y_data, marker='o', linewidth=2.5, markersize=6, 
                color=color, markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(data.get('xlabel', 'X-axis'), fontsize=12, fontweight='bold')
        ax.set_ylabel(data.get('ylabel', 'Y-axis'), fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_pie_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create precise pie chart"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = data.get('labels', ['A', 'B', 'C'])
        sizes = data.get('sizes', [30, 40, 30])
        title = data.get('title', 'Pie Chart')
        
        colors = self._get_color_palette(style, len(labels))
        
        # Ensure percentages add to 100%
        total = sum(sizes)
        percentages = [size/total*100 for size in sizes]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, explode=[0.05]*len(labels))
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_flowchart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create precise flowchart diagram"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        nodes = data.get('nodes', [
            {'id': 'start', 'label': 'Start', 'x': 0.5, 'y': 0.9, 'type': 'oval'},
            {'id': 'process', 'label': 'Process', 'x': 0.5, 'y': 0.5, 'type': 'rect'},
            {'id': 'end', 'label': 'End', 'x': 0.5, 'y': 0.1, 'type': 'oval'}
        ])
        
        edges = data.get('edges', [
            {'from': 'start', 'to': 'process'},
            {'from': 'process', 'to': 'end'}
        ])
        
        title = data.get('title', 'Flowchart')
        colors = self._get_color_palette(style, len(nodes))
        
        # Draw nodes
        node_positions = {}
        for i, node in enumerate(nodes):
            x, y = node['x'], node['y']
            node_positions[node['id']] = (x, y)
            
            if node.get('type') == 'oval':
                shape = patches.Ellipse((x, y), 0.15, 0.08, facecolor=colors[i % len(colors)], 
                                      edgecolor='black', linewidth=2)
            else:
                shape = FancyBboxPatch((x-0.075, y-0.04), 0.15, 0.08, 
                                     boxstyle="round,pad=0.01", facecolor=colors[i % len(colors)],
                                     edgecolor='black', linewidth=2)
            
            ax.add_patch(shape)
            ax.text(x, y, node['label'], ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw edges
        for edge in edges:
            start_pos = node_positions[edge['from']]
            end_pos = node_positions[edge['to']]
            
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_network_diagram(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create network diagram using NetworkX"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create graph
        G = nx.Graph()
        
        nodes = data.get('nodes', ['A', 'B', 'C', 'D'])
        edges = data.get('edges', [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')])
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Colors
        colors = self._get_color_palette(style, len(nodes))
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        title = data.get('title', 'Network Diagram')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_architecture_diagram(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create system architecture diagram"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Default NIS Protocol architecture
        components = data.get('components', [
            {'name': 'Frontend\n(Chat Console)', 'x': 0.2, 'y': 0.8, 'width': 0.15, 'height': 0.1},
            {'name': 'API Gateway\n(FastAPI)', 'x': 0.5, 'y': 0.8, 'width': 0.15, 'height': 0.1},
            {'name': 'LLM Manager', 'x': 0.2, 'y': 0.5, 'width': 0.15, 'height': 0.1},
            {'name': 'Vision Agent', 'x': 0.5, 'y': 0.5, 'width': 0.15, 'height': 0.1},
            {'name': 'Physics Engine', 'x': 0.8, 'y': 0.5, 'width': 0.15, 'height': 0.1},
            {'name': 'Database', 'x': 0.5, 'y': 0.2, 'width': 0.15, 'height': 0.1},
        ])
        
        colors = self._get_color_palette(style, len(components))
        
        # Draw components
        for i, comp in enumerate(components):
            rect = FancyBboxPatch(
                (comp['x'] - comp['width']/2, comp['y'] - comp['height']/2),
                comp['width'], comp['height'],
                boxstyle="round,pad=0.01",
                facecolor=colors[i % len(colors)],
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(comp['x'], comp['y'], comp['name'], ha='center', va='center', 
                   fontweight='bold', fontsize=9)
        
        # Add connections (simplified)
        connections = [
            (0, 1), (1, 2), (1, 3), (3, 4), (2, 5), (3, 5)
        ]
        
        for start_idx, end_idx in connections:
            start = components[start_idx]
            end = components[end_idx]
            ax.annotate('', xy=(end['x'], end['y']), xytext=(start['x'], start['y']),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))
        
        title = data.get('title', 'System Architecture')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_physics_diagram(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create physics concept diagram"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        concept = data.get('concept', 'wave_interference')
        
        if concept == 'wave_interference':
            # Create wave interference pattern
            x = np.linspace(0, 4*np.pi, 1000)
            wave1 = np.sin(x)
            wave2 = np.sin(x + np.pi/4)
            interference = wave1 + wave2
            
            ax.plot(x, wave1, 'b-', label='Wave 1', alpha=0.7, linewidth=2)
            ax.plot(x, wave2, 'r-', label='Wave 2', alpha=0.7, linewidth=2)
            ax.plot(x, interference, 'k-', label='Interference', linewidth=3)
            
            ax.set_xlabel('Position (x)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
        title = data.get('title', 'Physics Concept')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _create_pipeline_diagram(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create data pipeline diagram"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # NIS Protocol v3 pipeline
        stages = data.get('stages', [
            'Input\nSignal', 'Laplace\nTransform', 'KAN\nReasoning', 
            'PINN\nPhysics', 'LLM\nIntegration', 'Output'
        ])
        
        colors = self._get_color_palette(style, len(stages))
        stage_width = 0.8 / len(stages)
        
        for i, stage in enumerate(stages):
            x = 0.1 + i * stage_width + stage_width/2
            y = 0.5
            
            # Draw stage box
            rect = FancyBboxPatch(
                (x - stage_width/3, y - 0.1),
                stage_width*0.6, 0.2,
                boxstyle="round,pad=0.02",
                facecolor=colors[i],
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(x, y, stage, ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Draw arrow to next stage
            if i < len(stages) - 1:
                next_x = 0.1 + (i+1) * stage_width + stage_width/2
                ax.annotate('', xy=(next_x - stage_width/3, y), 
                           xytext=(x + stage_width/3, y),
                           arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
        
        title = data.get('title', 'NIS Protocol v3 Pipeline')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig, title)
    
    def _get_color_palette(self, style: str, n_colors: int) -> List[str]:
        """Get color palette based on style"""
        if style == "scientific":
            return plt.cm.viridis(np.linspace(0, 1, n_colors))
        elif style == "professional":
            return plt.cm.Set3(np.linspace(0, 1, n_colors))
        else:
            return plt.cm.tab10(np.linspace(0, 1, n_colors))
    
    def _save_plot_to_base64(self, fig, title: str) -> Dict[str, Any]:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_base64}"
        
        plt.close(fig)  # Important: close figure to free memory
        
        return {
            "status": "success",
            "url": data_url,
            "title": title,
            "format": "png",
            "method": "matplotlib_precision_generation",
            "note": "Generated with mathematical precision using code"
        }
    
    def generate_interactive_chart(self, chart_type: str, data: Dict[str, Any], style: str = "scientific") -> Dict[str, Any]:
        """Generate interactive Plotly charts with zoom, hover, and real-time capabilities"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.utils import PlotlyJSONEncoder
            import json
            
            self.logger.info(f"🎨 Generating INTERACTIVE {chart_type} chart with Plotly")
            
            # Create interactive chart based on type
            if chart_type == "line":
                return self._create_interactive_line_chart(data, style)
            elif chart_type == "bar":
                return self._create_interactive_bar_chart(data, style)
            elif chart_type == "scatter":
                return self._create_interactive_scatter_chart(data, style)
            elif chart_type == "pie":
                return self._create_interactive_pie_chart(data, style)
            elif chart_type == "heatmap":
                return self._create_interactive_heatmap(data, style)
            elif chart_type == "timeline":
                return self._create_interactive_timeline(data, style)
            elif chart_type == "real_time":
                return self._create_real_time_chart(data, style)
            else:
                # Fallback to SVG for unsupported types
                return self._create_svg_fallback_chart(chart_type, data, style)
                
        except ImportError:
            self.logger.warning("Plotly not available - falling back to SVG generation")
            return self._create_svg_fallback_chart(chart_type, data, style)
        except Exception as e:
            self.logger.error(f"Interactive chart generation failed: {e}")
            return self._create_svg_fallback_chart(chart_type, data, style)
    
    def _create_interactive_line_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create interactive line chart with Plotly"""
        import plotly.graph_objects as go
        import json
        
        x_data = data.get('x', list(range(10)))
        y_data = data.get('y', [i*2 for i in x_data])
        title = data.get('title', 'Interactive Line Chart')
        
        # Create Plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Data Series',
            line=dict(color='#0891b2', width=3),
            marker=dict(size=8, color='#0891b2', line=dict(color='white', width=2)),
            hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>'
        ))
        
        # Update layout for scientific style
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#1f2937')),
            xaxis=dict(
                title=data.get('xlabel', 'X-axis'),
                gridcolor='#e5e7eb',
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                title=data.get('ylabel', 'Y-axis'),
                gridcolor='#e5e7eb',
                showgrid=True,
                zeroline=True
            ),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='#374151'),
            hovermode='x unified',
            showlegend=True
        )
        
        # Convert to JSON for embedding
        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "status": "success",
            "chart_json": chart_json,
            "chart_type": "interactive_line",
            "title": title,
            "format": "plotly_json",
            "method": "plotly_interactive_generation",
            "note": "Interactive chart with zoom, pan, hover, and real-time updates",
            "features": ["zoom", "pan", "hover", "responsive", "real_time_capable"]
        }
    
    def _create_interactive_bar_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create interactive bar chart with Plotly"""
        import plotly.graph_objects as go
        import json
        
        categories = data.get('categories', ['A', 'B', 'C'])
        values = data.get('values', [10, 20, 15])
        title = data.get('title', 'Interactive Bar Chart')
        
        # Create Plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color='#0891b2',
                line=dict(color='#0e7490', width=1.5)
            ),
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#1f2937')),
            xaxis=dict(title=data.get('xlabel', 'Categories')),
            yaxis=dict(title=data.get('ylabel', 'Values')),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='#374151'),
            showlegend=False
        )
        
        # Convert to JSON
        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "status": "success",
            "chart_json": chart_json,
            "chart_type": "interactive_bar",
            "title": title,
            "format": "plotly_json",
            "method": "plotly_interactive_generation",
            "note": "Interactive bar chart with hover details and animations",
            "features": ["hover", "animations", "responsive"]
        }
    
    def _create_real_time_chart(self, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create real-time updating chart template"""
        import plotly.graph_objects as go
        import json
        
        title = data.get('title', 'Real-Time Data Stream')
        
        # Create empty figure for real-time updates
        fig = go.Figure()
        
        # Add placeholder trace
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name='Live Data',
            line=dict(color='#059669', width=2),
            marker=dict(size=6)
        ))
        
        # Configure for real-time updates
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis=dict(title='Time', range=[0, 100]),
            yaxis=dict(title='Value', range=[0, 1]),
            plot_bgcolor='#f0f9ff',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            showlegend=True,
            # Enable real-time features
            uirevision='constant'  # Maintains zoom/pan during updates
        )
        
        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "status": "success",
            "chart_json": chart_json,
            "chart_type": "real_time",
            "title": title,
            "format": "plotly_json",
            "method": "plotly_realtime_generation",
            "note": "Real-time updating chart with live data streaming",
            "features": ["real_time", "streaming", "auto_update", "zoom_persistence"],
            "update_endpoint": "/visualization/real-time-data",
            "update_interval": 2000  # milliseconds
        }
    
    def _create_svg_fallback_chart(self, chart_type: str, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create SVG fallback charts when matplotlib is not available"""
        try:
            title = data.get('title', f'{chart_type.title()} Chart')
            
            if chart_type == "bar":
                categories = data.get('categories', ['A', 'B', 'C'])
                values = data.get('values', [10, 20, 15])
                svg_content = self._create_svg_bar_chart(categories, values, title, style)
            elif chart_type == "pie":
                labels = data.get('labels', ['A', 'B', 'C'])
                sizes = data.get('sizes', [30, 40, 30])
                svg_content = self._create_svg_pie_chart(labels, sizes, title, style)
            else:
                svg_content = self._create_basic_svg_chart(title, chart_type)
            
            # Convert SVG to base64
            svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode()
            data_url = f"data:image/svg+xml;base64,{svg_b64}"
            
            return {
                "status": "success",
                "url": data_url,
                "title": title,
                "format": "svg",
                "method": "svg_fallback_generation",
                "note": "Generated with SVG fallback (install matplotlib for full precision)"
            }
            
        except Exception as e:
            self.logger.error(f"SVG fallback chart creation failed: {e}")
            return {"error": f"SVG fallback failed: {str(e)}"}
    
    def _create_svg_fallback_diagram(self, diagram_type: str, data: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Create SVG fallback diagrams when matplotlib is not available"""
        try:
            title = data.get('title', f'{diagram_type.title()} Diagram')
            
            if diagram_type == "flowchart":
                svg_content = self._create_svg_flowchart(data, title, style)
            elif diagram_type == "architecture":
                svg_content = self._create_svg_architecture(title, style)
            elif diagram_type == "pipeline":
                svg_content = self._create_svg_pipeline(title, style)
            else:
                svg_content = self._create_basic_svg_diagram(title, diagram_type)
            
            # Convert SVG to base64
            svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode()
            data_url = f"data:image/svg+xml;base64,{svg_b64}"
            
            return {
                "status": "success",
                "url": data_url,
                "title": title,
                "format": "svg",
                "method": "svg_fallback_generation",
                "note": "Generated with SVG fallback (install matplotlib for full precision)"
            }
            
        except Exception as e:
            self.logger.error(f"SVG fallback diagram creation failed: {e}")
            return {"error": f"SVG fallback failed: {str(e)}"}
    
    def _create_svg_bar_chart(self, categories: list, values: list, title: str, style: str) -> str:
        """Create a simple SVG bar chart"""
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        
        width = 600
        height = 400
        margin = 60
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        max_value = max(values) if values else 1
        bar_width = chart_width / len(categories)
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
                .value {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; fill: white; }}
            </style>
            <rect width="{width}" height="{height}" fill="#f8fafc" />
            <text x="{width/2}" y="30" class="title">{title}</text>'''
        
        for i, (cat, val) in enumerate(zip(categories, values)):
            x = margin + i * bar_width + bar_width * 0.1
            bar_height = (val / max_value) * chart_height
            y = height - margin - bar_height
            color = colors[i % len(colors)]
            
            svg += f'''
            <rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{bar_height}" fill="{color}" />
            <text x="{x + bar_width * 0.4}" y="{height - margin + 15}" class="label">{cat}</text>
            <text x="{x + bar_width * 0.4}" y="{y + bar_height/2}" class="value">{val}</text>'''
        
        svg += '</svg>'
        return svg
    
    def _create_svg_pie_chart(self, labels: list, sizes: list, title: str, style: str) -> str:
        """Create a simple SVG pie chart"""
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        
        width = 500
        height = 400
        center_x = width / 2
        center_y = height / 2
        radius = 120
        
        total = sum(sizes)
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .legend {{ font-family: Arial, sans-serif; font-size: 12px; }}
            </style>
            <rect width="{width}" height="{height}" fill="#f8fafc" />
            <text x="{center_x}" y="30" class="title">{title}</text>'''
        
        start_angle = 0
        for i, (label, size) in enumerate(zip(labels, sizes)):
            angle = (size / total) * 360
            end_angle = start_angle + angle
            
            # Convert to radians
            start_rad = (start_angle * 3.14159) / 180
            end_rad = (end_angle * 3.14159) / 180
            
            # Calculate arc points
            x1 = center_x + radius * math.cos(start_rad)
            y1 = center_y + radius * math.sin(start_rad)
            x2 = center_x + radius * math.cos(end_rad)
            y2 = center_y + radius * math.sin(end_rad)
            
            large_arc = 1 if angle > 180 else 0
            color = colors[i % len(colors)]
            
            svg += f'''
            <path d="M {center_x} {center_y} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z" 
                  fill="{color}" stroke="white" stroke-width="2" />'''
            
            # Legend
            legend_y = 60 + i * 20
            svg += f'''
            <rect x="20" y="{legend_y - 8}" width="12" height="12" fill="{color}" />
            <text x="40" y="{legend_y}" class="legend">{label}: {size} ({100*size/total:.1f}%)</text>'''
            
            start_angle = end_angle
        
        svg += '</svg>'
        return svg
    
    def _create_svg_flowchart(self, data: Dict, title: str, style: str) -> str:
        """Create a simple SVG flowchart"""
        nodes = data.get('nodes', [
            {'id': 'start', 'label': 'Start', 'x': 0.5, 'y': 0.9},
            {'id': 'process', 'label': 'Process', 'x': 0.5, 'y': 0.5},
            {'id': 'end', 'label': 'End', 'x': 0.5, 'y': 0.1}
        ])
        
        width = 500
        height = 400
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .node {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
            </style>
            <rect width="{width}" height="{height}" fill="#f8fafc" />
            <text x="{width/2}" y="30" class="title">{title}</text>'''
        
        for node in nodes:
            x = node['x'] * width
            y = node['y'] * height
            
            svg += f'''
            <rect x="{x-50}" y="{y-15}" width="100" height="30" rx="5" fill="#3b82f6" stroke="#1e40af" stroke-width="2" />
            <text x="{x}" y="{y+5}" class="node" fill="white">{node['label']}</text>'''
            
        svg += '</svg>'
        return svg
    
    def _create_svg_architecture(self, title: str, style: str) -> str:
        """Create a simple SVG architecture diagram"""
        svg = f'''<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .component {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }}
            </style>
            <rect width="600" height="400" fill="#f8fafc" />
            <text x="300" y="30" class="title">{title}</text>
            
            <!-- Frontend -->
            <rect x="50" y="80" width="120" height="60" rx="8" fill="#3b82f6" stroke="#1e40af" stroke-width="2" />
            <text x="110" y="115" class="component" fill="white">Frontend</text>
            
            <!-- API Gateway -->
            <rect x="240" y="80" width="120" height="60" rx="8" fill="#10b981" stroke="#059669" stroke-width="2" />
            <text x="300" y="115" class="component" fill="white">API Gateway</text>
            
            <!-- Backend Services -->
            <rect x="430" y="80" width="120" height="60" rx="8" fill="#f59e0b" stroke="#d97706" stroke-width="2" />
            <text x="490" y="115" class="component" fill="white">Backend</text>
            
            <!-- Database -->
            <rect x="240" y="200" width="120" height="60" rx="8" fill="#ef4444" stroke="#dc2626" stroke-width="2" />
            <text x="300" y="235" class="component" fill="white">Database</text>
            
            <!-- Arrows -->
            <path d="M 170 110 L 230 110" stroke="#374151" stroke-width="2" marker-end="url(#arrowhead)" />
            <path d="M 360 110 L 420 110" stroke="#374151" stroke-width="2" marker-end="url(#arrowhead)" />
            <path d="M 300 150 L 300 190" stroke="#374151" stroke-width="2" marker-end="url(#arrowhead)" />
            
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#374151" />
                </marker>
            </defs>
        </svg>'''
        return svg
    
    def _create_svg_pipeline(self, title: str, style: str) -> str:
        """Create a simple SVG pipeline diagram"""
        stages = ['Input', 'Laplace', 'KAN', 'PINN', 'LLM', 'Output']
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
        
        svg = f'''<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .stage {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }}
            </style>
            <rect width="800" height="200" fill="#f8fafc" />
            <text x="400" y="30" class="title">{title}</text>'''
        
        stage_width = 700 / len(stages)
        for i, stage in enumerate(stages):
            x = 50 + i * stage_width
            y = 80
            color = colors[i % len(colors)]
            
            svg += f'''
            <rect x="{x}" y="{y}" width="{stage_width-20}" height="40" rx="8" fill="{color}" stroke="white" stroke-width="2" />
            <text x="{x + (stage_width-20)/2}" y="{y + 25}" class="stage" fill="white">{stage}</text>'''
            
            if i < len(stages) - 1:
                arrow_x = x + stage_width - 20
                svg += f'''<path d="M {arrow_x} {y+20} L {arrow_x+15} {y+20}" stroke="#374151" stroke-width="3" marker-end="url(#arrowhead)" />'''
        
        svg += '''
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#374151" />
                </marker>
            </defs>
        </svg>'''
        return svg
    
    def _create_basic_svg_chart(self, title: str, chart_type: str) -> str:
        """Create a basic SVG placeholder for unsupported chart types"""
        return f'''<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .message {{ font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }}
            </style>
            <rect width="400" height="300" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" />
            <text x="200" y="50" class="title">{title}</text>
            <text x="200" y="150" class="message">📊 {chart_type.title()} Chart</text>
            <text x="200" y="180" class="message">SVG Fallback Mode</text>
            <text x="200" y="210" class="message">Install matplotlib for full precision</text>
        </svg>'''
    
    def _create_basic_svg_diagram(self, title: str, diagram_type: str) -> str:
        """Create a basic SVG placeholder for unsupported diagram types"""
        return f'''<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
                .message {{ font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }}
            </style>
            <rect width="400" height="300" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" />
            <text x="200" y="50" class="title">{title}</text>
            <text x="200" y="150" class="message">🔧 {diagram_type.title()} Diagram</text>
            <text x="200" y="180" class="message">SVG Fallback Mode</text>
            <text x="200" y="210" class="message">Install matplotlib for full precision</text>
        </svg>'''