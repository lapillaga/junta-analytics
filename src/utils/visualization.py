from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualizationHelper:
    """Helper class for creating interactive visualizations"""

    def __init__(self):
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.theme = {
            'background_color': '#f8f9fa',
            'paper_bgcolor': 'white',
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'title_font_size': 16
        }

    def create_rainfall_consumption_timeline(self, data: pd.DataFrame) -> str:
        """Create dual-axis timeline chart for rainfall vs consumption"""

        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Monthly Rainfall vs Water Consumption"]
        )

        # Add consumption line
        fig.add_trace(
            go.Scatter(
                x=data['period_dt'],
                y=data['avg_consumption'],
                name='Avg Consumption (m³)',
                line=dict(color='#1f77b4', width=3),
                mode='lines+markers',
                marker=dict(size=6)
            ),
            secondary_y=False,
        )

        # Add rainfall line
        fig.add_trace(
            go.Scatter(
                x=data['period_dt'],
                y=data['avg_rainfall'],
                name='Avg Rainfall (mm)',
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers',
                marker=dict(size=6)
            ),
            secondary_y=True,
        )

        # Set y-axes titles
        fig.update_yaxes(
            title_text="Water Consumption (m³)",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Rainfall (mm)",
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Temporal Correlation: Rainfall vs Water Consumption',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Period',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['background_color'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_correlation_scatter(self, data: pd.DataFrame) -> str:
        """Create scatter plot showing rainfall-consumption correlation"""

        # Calculate correlation coefficient
        correlation = data['avg_rainfall'].corr(data['avg_consumption'])

        fig = go.Figure()

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=data['avg_rainfall'],
                y=data['avg_consumption'],
                mode='markers',
                name='Monthly Data',
                marker=dict(
                    size=8,
                    color=data.index,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time Period")
                ),
                text=data['period_str'],
                hovertemplate=(
                    '<b>Period:</b> %{text}<br>'
                    '<b>Rainfall:</b> %{x:.1f} mm<br>'
                    '<b>Consumption:</b> %{y:.1f} m³<br>'
                    '<extra></extra>'
                )
            )
        )

        # Add trend line
        z = np.polyfit(data['avg_rainfall'], data['avg_consumption'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data['avg_rainfall'].min(),
                              data['avg_rainfall'].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name=f'Trend Line (r={correlation:.3f})',
                line=dict(color='red', dash='dash', width=2)
            )
        )

        fig.update_layout(
            title={
                'text': f'Rainfall vs Consumption Correlation (r = {correlation:.3f})',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Average Rainfall (mm)',
            yaxis_title='Average Consumption (m³)',
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['background_color'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_seasonal_heatmap(self, data: pd.DataFrame) -> str:
        """Create heatmap showing consumption patterns by month and rainfall intensity"""

        # Create month and rainfall intensity columns if not exist
        if 'month' not in data.columns:
            data['month'] = pd.to_datetime(data['period_str']).dt.month

        # Create rainfall intensity bins
        data['rainfall_bin'] = pd.cut(
            data['avg_rainfall'],
            bins=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )

        # Create pivot table
        heatmap_data = data.pivot_table(
            values='avg_consumption',
            index='rainfall_bin',
            columns='month',
            aggfunc='mean'
        ).round(1)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=[f'Month {i}' for i in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale='RdYlBu_r',
                hoverongaps=False,
                hovertemplate=(
                    '<b>Month:</b> %{x}<br>'
                    '<b>Rainfall Level:</b> %{y}<br>'
                    '<b>Avg Consumption:</b> %{z:.1f} m³<br>'
                    '<extra></extra>'
                )
            )
        )

        fig.update_layout(
            title={
                'text': 'Consumption Patterns by Month and Rainfall Intensity',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Month',
            yaxis_title='Rainfall Intensity',
            paper_bgcolor=self.theme['paper_bgcolor'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_neighborhood_consumption(self,
                                        neighborhood_data: pd.DataFrame) -> str:
        """Create bar chart of consumption by neighborhood"""

        # Aggregate data by neighborhood
        neighborhood_stats = neighborhood_data.groupby(
            'neighborhood_name').agg({
            'total_consumption': 'sum',
            'avg_consumption': 'mean',
            'active_meters': 'sum'
        }).reset_index()

        # Calculate consumption per meter
        neighborhood_stats['consumption_per_meter'] = (
            neighborhood_stats['total_consumption'] / neighborhood_stats[
            'active_meters']
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=neighborhood_stats['neighborhood_name'],
                y=neighborhood_stats['consumption_per_meter'],
                name='Consumption per Meter',
                marker_color='lightblue',
                hovertemplate=(
                    '<b>Neighborhood:</b> %{x}<br>'
                    '<b>Consumption per Meter:</b> %{y:.1f} m³<br>'
                    '<b>Total Meters:</b> %{customdata[0]}<br>'
                    '<b>Total Consumption:</b> %{customdata[1]:.1f} m³<br>'
                    '<extra></extra>'
                ),
                customdata=neighborhood_stats[
                    ['active_meters', 'total_consumption']].values
            )
        )

        fig.update_layout(
            title={
                'text': 'Average Consumption per Meter by Neighborhood',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Neighborhood',
            yaxis_title='Consumption per Meter (m³)',
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['background_color'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_anomaly_detection_chart(self, anomaly_data: List[Dict]) -> str:
        """Create chart showing detected anomalies"""

        if not anomaly_data:
            # Return empty chart if no data
            fig = go.Figure()
            fig.update_layout(
                title='No Anomalies Detected',
                annotations=[dict(
                    text="No anomalous readings found",
                    showarrow=False,
                    x=0.5, y=0.5
                )]
            )
            return fig.to_json()

        df = pd.DataFrame(anomaly_data)

        # Separate normal and anomalous readings
        normal_readings = df[~df['is_anomaly']]
        anomalous_readings = df[df['is_anomaly']]

        fig = go.Figure()

        # Add normal readings
        if len(normal_readings) > 0:
            fig.add_trace(
                go.Scatter(
                    x=normal_readings.index,
                    y=normal_readings['consumption'],
                    mode='markers',
                    name='Normal Readings',
                    marker=dict(color='green', size=6),
                    hovertemplate=(
                        '<b>Reading #:</b> %{x}<br>'
                        '<b>Consumption:</b> %{y} m³<br>'
                        '<b>Status:</b> Normal<br>'
                        '<extra></extra>'
                    )
                )
            )

        # Add anomalous readings
        if len(anomalous_readings) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalous_readings.index,
                    y=anomalous_readings['consumption'],
                    mode='markers',
                    name='Anomalous Readings',
                    marker=dict(
                        color='red',
                        size=anomalous_readings['confidence'] * 15 + 5,
                        symbol='x'
                    ),
                    hovertemplate=(
                        '<b>Reading #:</b> %{x}<br>'
                        '<b>Consumption:</b> %{y} m³<br>'
                        '<b>Confidence:</b> %{customdata[0]:.1%}<br>'
                        '<b>Reason:</b> %{customdata[1]}<br>'
                        '<extra></extra>'
                    ),
                    customdata=anomalous_readings[
                        ['confidence', 'reason']].values
                )
            )

        fig.update_layout(
            title={
                'text': 'Anomaly Detection Results',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Reading Index',
            yaxis_title='Consumption (m³)',
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['background_color'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_consumption_prediction_chart(self,
                                            historical_data: pd.DataFrame,
                                            predictions: pd.DataFrame) -> str:
        """Create chart showing consumption predictions"""

        fig = go.Figure()

        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data['period_dt'],
                y=historical_data['avg_consumption'],
                mode='lines+markers',
                name='Historical Consumption',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            )
        )

        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=predictions['period_dt'],
                y=predictions['predicted_consumption'],
                mode='lines+markers',
                name='Predicted Consumption',
                line=dict(color='red', dash='dash', width=2),
                marker=dict(size=6, symbol='diamond')
            )
        )

        # Add confidence intervals if available
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions['period_dt'],
                    y=predictions['upper_bound'],
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=predictions['period_dt'],
                    y=predictions['lower_bound'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='Confidence Interval',
                    hoverinfo='skip'
                )
            )

        fig.update_layout(
            title={
                'text': 'Water Consumption Prediction',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            xaxis_title='Period',
            yaxis_title='Consumption (m³)',
            hovermode='x unified',
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['background_color'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_kpi_cards(self, stats: Dict) -> Dict[str, str]:
        """Create KPI cards data for dashboard"""

        kpi_data = {
            'total_periods': {
                'value': stats.get('total_periods', 0),
                'title': 'Analysis Periods',
                'icon': 'calendar',
                'color': 'primary'
            },
            'avg_consumption': {
                'value': f"{stats.get('consumption_stats', {}).get('mean', 0):.1f}",
                'title': 'Avg Monthly Consumption (m³)',
                'icon': 'droplet',
                'color': 'info'
            },
            'avg_rainfall': {
                'value': f"{stats.get('rainfall_stats', {}).get('mean', 0):.1f}",
                'title': 'Avg Monthly Rainfall (mm)',
                'icon': 'cloud-rain',
                'color': 'success'
            },
            'correlation': {
                'value': f"{stats.get('correlation', {}).get('rainfall_consumption', 0):.3f}",
                'title': 'Rainfall-Consumption Correlation',
                'icon': 'trending-up',
                'color': 'warning' if abs(
                    stats.get('correlation', {}).get('rainfall_consumption',
                                                     0)) < 0.3 else 'success'
            }
        }

        return kpi_data

    def create_risk_classification_chart(self, risk_data: pd.DataFrame) -> str:
        """Create chart showing risk period classification"""

        # Count periods by risk level
        risk_counts = risk_data['risk_level'].value_counts()

        colors = {
            'Low Risk': '#28a745',
            'Normal': '#ffc107',
            'High Risk': '#dc3545'
        }

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(
                        colors=[colors.get(label, '#6c757d') for label in
                                risk_counts.index]
                    ),
                    hovertemplate=(
                        '<b>Risk Level:</b> %{label}<br>'
                        '<b>Periods:</b> %{value}<br>'
                        '<b>Percentage:</b> %{percent}<br>'
                        '<extra></extra>'
                    )
                )
            ]
        )

        fig.update_layout(
            title={
                'text': 'Water Risk Period Classification',
                'x': 0.5,
                'font': {'size': self.theme['title_font_size']}
            },
            paper_bgcolor=self.theme['paper_bgcolor'],
            font={'family': self.theme['font_family'],
                  'size': self.theme['font_size']}
        )

        return fig.to_json()

    def create_dashboard_layout(self) -> str:
        """Create the main dashboard layout HTML"""

        dashboard_html = """
        <div class="container-fluid">
            <!-- KPI Cards Row -->
            <div class="row mb-4" id="kpi-cards">
                <!-- KPI cards will be populated by JavaScript -->
            </div>

            <!-- Main Charts Row -->
            <div class="row mb-4">
                <div class="col-lg-8">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Rainfall vs Consumption Timeline</h5>
                        </div>
                        <div class="card-body">
                            <div id="timeline-chart" style="height: 400px;"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Correlation Analysis</h5>
                        </div>
                        <div class="card-body">
                            <div id="correlation-chart" style="height: 400px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Secondary Charts Row -->
            <div class="row mb-4">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Seasonal Patterns</h5>
                        </div>
                        <div class="card-body">
                            <div id="seasonal-heatmap" style="height: 350px;"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Neighborhood Analysis</h5>
                        </div>
                        <div class="card-body">
                            <div id="neighborhood-chart" style="height: 350px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Predictions and Anomalies Row -->
            <div class="row mb-4">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Consumption Predictions</h5>
                        </div>
                        <div class="card-body">
                            <div id="prediction-chart" style="height: 350px;"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Risk Classification</h5>
                        </div>
                        <div class="card-body">
                            <div id="risk-chart" style="height: 350px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Anomaly Detection Section -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">Anomaly Detection</h5>
                            <button class="btn btn-primary btn-sm" onclick="testAnomaly()">
                                Test New Reading
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-lg-8">
                                    <div id="anomaly-chart" style="height: 300px;"></div>
                                </div>
                                <div class="col-lg-4">
                                    <div id="anomaly-test-form">
                                        <h6>Test Anomaly Detection</h6>
                                        <form id="anomaly-form">
                                            <div class="mb-2">
                                                <label class="form-label">Water Meter ID</label>
                                                <input type="number" class="form-control" id="meter-id" required>
                                            </div>
                                            <div class="mb-2">
                                                <label class="form-label">Previous Reading</label>
                                                <input type="number" class="form-control" id="previous-reading" required>
                                            </div>
                                            <div class="mb-2">
                                                <label class="form-label">Current Reading</label>
                                                <input type="number" class="form-control" id="current-reading" required>
                                            </div>
                                            <div class="mb-2">
                                                <label class="form-label">Days Billed</label>
                                                <input type="number" class="form-control" id="days-billed" value="30" required>
                                            </div>
                                            <button type="submit" class="btn btn-success btn-sm">Check Anomaly</button>
                                        </form>
                                        <div id="anomaly-result" class="mt-3"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

        return dashboard_html
