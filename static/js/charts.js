// Charts utility functions for Junta Analytics

// Chart color palette
const CHART_COLORS = {
  primary: '#3B82F6',    // Blue
  secondary: '#10B981',  // Green
  warning: '#F59E0B',    // Amber
  danger: '#EF4444',     // Red
  info: '#06B6D4',       // Cyan
  purple: '#8B5CF6',     // Purple
  pink: '#EC4899',       // Pink
  gray: '#6B7280',       // Gray
};

// Chart templates for common visualizations
const ChartTemplates = {
  // Consumption vs Rainfall chart template
  consumptionVsRainfall: function(elementId, data) {
    if (!data || data.length === 0) {
      document.getElementById(elementId).innerHTML = `
        <div class="flex items-center justify-center h-full text-gray-500">
          <p>No data available to display chart</p>
        </div>
      `;
      return;
    }
    
    // Prepare data
    const dates = data.map(item => item.dekad_period || item.date);
    const consumption = data.map(item => item.total_consumption || item.consumption);
    const rainfall = data.map(item => item.total_rainfall || item.rainfall);
    
    // Create anomaly data if available
    let anomaliesX = [];
    let anomaliesY = [];
    
    if (data[0].hasOwnProperty('is_anomaly')) {
      data.forEach((item, index) => {
        if (item.is_anomaly) {
          anomaliesX.push(dates[index]);
          anomaliesY.push(consumption[index]);
        }
      });
    }
    
    // Create figure with secondary y-axis
    const layout = {
      title: 'Water Consumption vs Rainfall',
      xaxis: {
        title: 'Time Period',
      },
      yaxis: {
        title: 'Consumption (m³)',
        titlefont: {color: CHART_COLORS.primary},
        tickfont: {color: CHART_COLORS.primary}
      },
      yaxis2: {
        title: 'Rainfall (mm)',
        titlefont: {color: CHART_COLORS.secondary},
        tickfont: {color: CHART_COLORS.secondary},
        overlaying: 'y',
        side: 'right'
      },
      legend: {
        orientation: 'h',
        y: 1.1
      }
    };
    
    const traces = [
      {
        type: 'bar',
        x: dates,
        y: consumption,
        name: 'Consumption (m³)',
        marker: {
          color: CHART_COLORS.primary
        },
        yaxis: 'y'
      },
      {
        type: 'scatter',
        x: dates,
        y: rainfall,
        name: 'Rainfall (mm)',
        marker: {
          color: CHART_COLORS.secondary
        },
        line: {
          width: 3
        },
        yaxis: 'y2'
      }
    ];
    
    // Add anomaly markers if data is available
    if (anomaliesX.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: anomaliesX,
        y: anomaliesY,
        name: 'Anomalies',
        marker: {
          color: CHART_COLORS.danger,
          size: 12,
          symbol: 'x'
        },
        yaxis: 'y'
      });
    }
    
    Plotly.newPlot(elementId, traces, layout);
  },
  
  // Consumption range chart template
  consumptionRange: function(elementId, actualData, forecastData) {
    if (!actualData || actualData.length === 0) {
      document.getElementById(elementId).innerHTML = `
        <div class="flex items-center justify-center h-full text-gray-500">
          <p>No data available to display chart</p>
        </div>
      `;
      return;
    }
    
    // Prepare actual data
    const dates = actualData.map(item => item.dekad_period || item.date);
    const consumption = actualData.map(item => item.avg_consumption || item.consumption);
    
    const traces = [
      {
        type: 'scatter',
        x: dates,
        y: consumption,
        name: 'Actual Consumption',
        line: {
          color: CHART_COLORS.primary,
          width: 3
        }
      }
    ];
    
    // Add forecast data if available
    if (forecastData && forecastData.length > 0) {
      const forecastDates = forecastData.map(item => item.dekad_period || item.date);
      const forecastValues = forecastData.map(item => item.predicted_consumption || item.forecast);
      const lowerBounds = forecastData.map(item => item.lower_bound);
      const upperBounds = forecastData.map(item => item.upper_bound);
      
      // Add lower bound
      traces.push({
        type: 'scatter',
        x: forecastDates,
        y: lowerBounds,
        name: 'Lower Bound',
        line: {
          width: 0
        },
        showlegend: false
      });
      
      // Add upper bound with fill
      traces.push({
        type: 'scatter',
        x: forecastDates,
        y: upperBounds,
        name: 'Expected Range',
        fill: 'tonexty',
        fillcolor: 'rgba(59, 130, 246, 0.2)',
        line: {
          width: 0
        }
      });
      
      // Add forecast line
      traces.push({
        type: 'scatter',
        x: forecastDates,
        y: forecastValues,
        name: 'Predicted Consumption',
        line: {
          color: CHART_COLORS.secondary,
          width: 2,
          dash: 'dash'
        }
      });
    }
    
    const layout = {
      title: 'Water Consumption: Actual vs Expected Range',
      xaxis: {
        title: 'Time Period'
      },
      yaxis: {
        title: 'Average Consumption (m³)'
      },
      legend: {
        orientation: 'h',
        y: 1.1
      }
    };
    
    Plotly.newPlot(elementId, traces, layout);
  },
  
  // Correlation heatmap template
  correlationHeatmap: function(elementId, correlationData) {
    if (!correlationData) {
      document.getElementById(elementId).innerHTML = `
        <div class="flex items-center justify-center h-full text-gray-500">
          <p>No data available to display chart</p>
        </div>
      `;
      return;
    }
    
    // Get labels and values from correlation data
    const labels = Object.keys(correlationData);
    const values = labels.map(row => {
      return labels.map(col => correlationData[row][col]);
    });
    
    const data = [{
      z: values,
      x: labels,
      y: labels,
      type: 'heatmap',
      colorscale: 'RdBu',
      zmin: -1,
      zmax: 1,
      text: values.map(row => row.map(value => value.toFixed(2))),
      hoverinfo: 'text',
      showscale: true
    }];
    
    const layout = {
      title: 'Correlation Between Consumption and Rainfall',
      xaxis: {
        ticks: '',
        side: 'top'
      },
      yaxis: {
        ticks: '',
        automargin: true
      },
      annotations: []
    };
    
    // Add correlation values as text annotations
    labels.forEach((rowLabel, i) => {
      labels.forEach((colLabel, j) => {
        layout.annotations.push({
          xref: 'x',
          yref: 'y',
          x: colLabel,
          y: rowLabel,
          text: values[i][j].toFixed(2),
          font: {
            color: Math.abs(values[i][j]) > 0.5 ? 'white' : 'black'
          },
          showarrow: false
        });
      });
    });
    
    Plotly.newPlot(elementId, data, layout);
  }
};

// Export chart utilities
window.JuntaAnalytics = window.JuntaAnalytics || {};
window.JuntaAnalytics.Charts = {
  colors: CHART_COLORS,
  templates: ChartTemplates
};