import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
import numpy as np
from config import Config
from io import BytesIO
import base64

def create_consumption_rainfall_chart(merged_data):
    """
    Create a chart showing consumption vs rainfall over time
    """
    if merged_data.empty:
        return json.dumps({})
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add consumption trace
    fig.add_trace(
        go.Bar(
            x=merged_data['dekad_period'],
            y=merged_data['total_consumption'],
            name="Consumption (m³)",
            marker_color=Config.CHART_COLORS['primary']
        ),
        secondary_y=False,
    )
    
    # Add rainfall trace
    fig.add_trace(
        go.Scatter(
            x=merged_data['dekad_period'],
            y=merged_data['total_rainfall'],
            name="Rainfall (mm)",
            marker_color=Config.CHART_COLORS['secondary'],
            line=dict(width=3)
        ),
        secondary_y=True,
    )
    
    # Add anomaly markers if available
    if 'is_anomaly' in merged_data.columns:
        anomalies = merged_data[merged_data['is_anomaly'] == True]
        
        fig.add_trace(
            go.Scatter(
                x=anomalies['dekad_period'],
                y=anomalies['total_consumption'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=Config.CHART_COLORS['danger'],
                    size=12,
                    symbol='x'
                )
            ),
            secondary_y=False
        )
    
    # Set titles
    fig.update_layout(
        title_text="Water Consumption vs Rainfall",
        xaxis_title="Time Period",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Consumption (m³)", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    
    return fig.to_json()

def create_consumption_range_chart(consumption_data, predictions=None):
    """
    Create a chart showing expected consumption range vs actual
    """
    if consumption_data.empty:
        return json.dumps({})
    
    fig = go.Figure()
    
    # Add actual consumption
    fig.add_trace(
        go.Scatter(
            x=consumption_data['dekad_period'],
            y=consumption_data['avg_consumption'],
            name="Actual Consumption",
            line=dict(color=Config.CHART_COLORS['primary'], width=3)
        )
    )
    
    # Add expected range if predictions are available
    if predictions is not None and not predictions.empty:
        fig.add_trace(
            go.Scatter(
                x=predictions['dekad_period'],
                y=predictions['lower_bound'],
                line=dict(width=0),
                name="Lower Bound"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions['dekad_period'],
                y=predictions['upper_bound'],
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(width=0),
                name="Upper Bound"
            )
        )
        
        # Add predicted consumption
        fig.add_trace(
            go.Scatter(
                x=predictions['dekad_period'],
                y=predictions['predicted_consumption'],
                name="Predicted Consumption",
                line=dict(color=Config.CHART_COLORS['secondary'], width=2, dash='dash')
            )
        )
    
    # Set titles and layout
    fig.update_layout(
        title_text="Water Consumption: Actual vs Expected Range",
        xaxis_title="Time Period",
        yaxis_title="Average Consumption (m³)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig.to_json()

def create_correlation_heatmap(merged_data):
    """
    Create a correlation heatmap between consumption and rainfall
    """
    if merged_data.empty:
        return json.dumps({})
    
    # Select numerical columns
    numerical_cols = ['total_consumption', 'avg_consumption', 'total_rainfall', 'avg_rainfall']
    
    # Create correlation matrix
    corr_data = merged_data[numerical_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_data,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Between Consumption and Rainfall"
    )
    
    return fig.to_json()

def plt_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str