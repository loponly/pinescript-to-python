"""Simple web dashboard for viewing optimization results."""

from typing import Dict, List, Optional, Any
import pandas as pd
import json
import os
from datetime import datetime
import logging

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .database_manager import DatabaseManager
from ..optimization.optimization_results import OptimizationResults


class Dashboard:
    """
    Simple web dashboard for viewing optimization results.
    
    Features:
    - Interactive charts and tables
    - Best strategy filtering
    - Performance comparisons
    - Export capabilities
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize dashboard.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        if not STREAMLIT_AVAILABLE:
            self.logger.warning(
                "Streamlit not available. Install with: pip install streamlit plotly"
            )
    
    def create_html_dashboard(
        self,
        results: OptimizationResults,
        output_file: str = "dashboard.html"
    ) -> str:
        """
        Create a simple HTML dashboard.
        
        Args:
            results: Optimization results
            output_file: Output HTML file
            
        Returns:
            Path to created HTML file
        """
        # Generate summaries if not already done
        if not results.summaries:
            results.generate_summaries()
        
        # Get performance statistics
        stats = results.get_performance_statistics()
        best_per_stock = results.get_best_strategies_per_stock()
        best_per_timeframe = results.get_best_strategies_per_timeframe()
        
        # Create HTML content
        html_content = self._generate_html_content(
            results.summaries,
            stats,
            best_per_stock,
            best_per_timeframe
        )
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML dashboard created: {output_file}")
        return output_file
    
    def run_streamlit_dashboard(self, port: int = 8501) -> None:
        """
        Run Streamlit dashboard.
        
        Args:
            port: Port to run dashboard on
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Install with: pip install streamlit plotly")
        
        # Create streamlit app file
        app_content = self._create_streamlit_app()
        
        with open("streamlit_dashboard.py", "w") as f:
            f.write(app_content)
        
        # Run streamlit
        os.system(f"streamlit run streamlit_dashboard.py --server.port {port}")
    
    def _generate_html_content(
        self,
        summaries: List,
        stats: Dict[str, Any],
        best_per_stock: Dict[str, Any],
        best_per_timeframe: Dict[str, Any]
    ) -> str:
        """Generate HTML dashboard content."""
        
        # Convert summaries to HTML table
        summaries_data = [s.to_dict() for s in summaries]
        df_summaries = pd.DataFrame(summaries_data)
        
        # Sort by profit factor
        df_summaries = df_summaries.sort_values('profit_factor', ascending=False)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Optimization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e3f2fd; border-radius: 8px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Optimization Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="metric-box">
                    <div class="metric-value">{stats.get('total_combinations_tested', 0)}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{stats.get('unique_stock_timeframe_pairs', 0)}</div>
                    <div class="metric-label">Stock/Timeframe Pairs</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{stats.get('profitable_strategies', 0)}</div>
                    <div class="metric-label">Profitable Strategies</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{stats.get('avg_profit_factor', 0):.2f}</div>
                    <div class="metric-label">Avg Profit Factor</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{stats.get('avg_win_rate', 0):.1%}</div>
                    <div class="metric-label">Avg Win Rate</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top 20 Strategies</h2>
                {self._dataframe_to_html_table(df_summaries.head(20))}
            </div>
            
            <div class="section">
                <h2>Best Strategy Per Stock</h2>
                {self._dict_to_html_table(best_per_stock, "Best Per Stock")}
            </div>
            
            <div class="section">
                <h2>Best Strategy Per Timeframe</h2>
                {self._dict_to_html_table(best_per_timeframe, "Best Per Timeframe")}
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def _dataframe_to_html_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to HTML table with formatting."""
        if df.empty:
            return "<p>No data available</p>"
        
        html = "<table>"
        
        # Header
        html += "<tr>"
        for col in df.columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += "</tr>"
        
        # Rows
        for _, row in df.iterrows():
            html += "<tr>"
            for col in df.columns:
                value = row[col]
                
                # Format numeric values
                if isinstance(value, float):
                    if col in ['profit_factor']:
                        formatted_value = f"{value:.2f}"
                        css_class = "positive" if value > 1.0 else "negative"
                    elif col in ['win_rate']:
                        formatted_value = f"{value:.1%}"
                        css_class = "positive" if value > 0.5 else "negative"
                    elif col in ['max_drawdown']:
                        formatted_value = f"{value:.1%}"
                        css_class = "negative" if value < 0 else "positive"
                    elif col in ['total_return']:
                        formatted_value = f"{value:.1%}"
                        css_class = "positive" if value > 0 else "negative"
                    else:
                        formatted_value = f"{value:.2f}"
                        css_class = ""
                    
                    html += f'<td class="{css_class}">{formatted_value}</td>'
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
    def _dict_to_html_table(self, data_dict: Dict[str, Any], title: str) -> str:
        """Convert dictionary to HTML table."""
        if not data_dict:
            return "<p>No data available</p>"
        
        html = f"<table><caption>{title}</caption>"
        
        # Header
        sample_item = next(iter(data_dict.values()))
        if hasattr(sample_item, 'to_dict'):
            sample_dict = sample_item.to_dict()
        else:
            sample_dict = sample_item
        
        html += "<tr><th>Key</th>"
        for key in sample_dict.keys():
            html += f"<th>{key.replace('_', ' ').title()}</th>"
        html += "</tr>"
        
        # Rows
        for key, item in data_dict.items():
            if hasattr(item, 'to_dict'):
                item_dict = item.to_dict()
            else:
                item_dict = item
            
            html += f"<tr><td><strong>{key}</strong></td>"
            for value in item_dict.values():
                if isinstance(value, float):
                    if 'profit' in str(value).lower():
                        html += f"<td>{value:.2f}</td>"
                    else:
                        html += f"<td>{value:.3f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
    def _create_streamlit_app(self) -> str:
        """Create Streamlit app content."""
        return '''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis.database_manager import DatabaseManager

# Page config
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize database manager
@st.cache_resource
def get_db_manager():
    return DatabaseManager()

db = get_db_manager()

# Title
st.title("📈 Trading Strategy Optimization Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")

# Get available options
@st.cache_data
def get_filter_options():
    df = db.query_results(limit=10000)
    return {
        'symbols': sorted(df['symbol'].unique().tolist()),
        'timeframes': sorted(df['timeframe'].unique().tolist())
    }

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

options = get_filter_options()

selected_symbols = st.sidebar.multiselect(
    "Select Symbols",
    options['symbols'],
    default=options['symbols'][:10] if len(options['symbols']) > 10 else options['symbols']
)

selected_timeframes = st.sidebar.multiselect(
    "Select Timeframes", 
    options['timeframes'],
    default=options['timeframes']
)

min_profit_factor = st.sidebar.slider(
    "Min Profit Factor",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1
)

max_drawdown = st.sidebar.slider(
    "Max Drawdown %",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=1.0
) / 100

# Load data
@st.cache_data
def load_data(symbols, timeframes, min_pf, max_dd):
    return db.query_results(
        symbols=symbols,
        timeframes=timeframes,
        min_profit_factor=min_pf,
        max_drawdown=max_dd,
        limit=1000
    )

df = load_data(selected_symbols, selected_timeframes, min_profit_factor, max_drawdown)

if df.empty:
    st.warning("No data found with current filters")
    st.stop()

# Overview metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Strategies", len(df))

with col2:
    avg_pf = df['profit_factor'].mean()
    st.metric("Avg Profit Factor", f"{avg_pf:.2f}")

with col3:
    avg_wr = df['win_rate'].mean()
    st.metric("Avg Win Rate", f"{avg_wr:.1%}")

with col4:
    profitable = (df['profit_factor'] > 1.0).sum()
    st.metric("Profitable Strategies", profitable)

with col5:
    best_pf = df['profit_factor'].max()
    st.metric("Best Profit Factor", f"{best_pf:.2f}")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profit Factor Distribution")
    fig_hist = px.histogram(
        df, 
        x='profit_factor', 
        bins=30,
        title="Distribution of Profit Factors"
    )
    fig_hist.add_vline(x=1.0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Win Rate vs Profit Factor")
    fig_scatter = px.scatter(
        df,
        x='win_rate',
        y='profit_factor',
        color='timeframe',
        hover_data=['symbol', 'max_drawdown'],
        title="Win Rate vs Profit Factor by Timeframe"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Best strategies
st.subheader("Best Strategies by Stock")
best_per_stock = db.get_best_strategies_per_stock(limit=20)
st.dataframe(best_per_stock, use_container_width=True)

st.subheader("Best Strategies by Timeframe")
best_per_timeframe = db.get_best_strategies_per_timeframe()
st.dataframe(best_per_timeframe, use_container_width=True)

# Detailed results
st.subheader("All Results")
st.dataframe(df, use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name=f"optimization_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
'''
    
    def create_performance_report(
        self,
        results: OptimizationResults,
        output_file: str = "performance_report.html"
    ) -> str:
        """
        Create a detailed performance report.
        
        Args:
            results: Optimization results
            output_file: Output HTML file
            
        Returns:
            Path to created report
        """
        # This would create a more detailed report with charts
        # For now, use the basic HTML dashboard
        return self.create_html_dashboard(results, output_file)
