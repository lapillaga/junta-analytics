// Global variables
let dashboardData = null;

// API endpoints
const API_ENDPOINTS = {
    dashboard: '/api/dashboard-data',
    detectAnomaly: '/api/detect-anomaly',
    predictConsumption: '/api/predict-consumption',
    waterMeters: '/api/water-meters',
    neighborhoods: '/api/neighborhoods',
    recentAnomalies: '/api/recent-anomalies',
    modelInfo: '/api/model-info',
    health: '/health'
};

// Utility functions
const utils = {
    formatNumber: (num, decimals = 1) => {
        if (typeof num !== 'number') return 'N/A';
        return num.toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    },

    formatDate: (dateString) => {
        if (!dateString) return 'Never';
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },

    showLoading: (elementId) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="chart-loading">
                    <div class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2 text-muted">Loading chart...</p>
                    </div>
                </div>
            `;
        }
    },

    showError: (elementId, message) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-warning text-center">
                    <i class="fas fa-exclamation-triangle mb-2" style="font-size: 2rem;"></i>
                    <p class="mb-0">${message}</p>
                </div>
            `;
        }
    },

    showSuccess: (message) => {
        const toast = document.createElement('div');
        toast.className = 'position-fixed top-0 end-0 p-3';
        toast.style.zIndex = '9999';
        toast.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-header bg-success text-white">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong class="me-auto">Success</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>
        `;
        document.body.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 5000);
    },

    showErrorToast: (message) => {
        const toast = document.createElement('div');
        toast.className = 'position-fixed top-0 end-0 p-3';
        toast.style.zIndex = '9999';
        toast.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-header bg-danger text-white">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong class="me-auto">Error</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>
        `;
        document.body.appendChild(toast);

        // Auto remove after 8 seconds
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 8000);
    }
};

// API functions
const api = {
    get: async (endpoint) => {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API GET error for ${endpoint}:`, error);
            throw error;
        }
    },

    post: async (endpoint, data) => {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API POST error for ${endpoint}:`, error);
            throw error;
        }
    }
};

// Dashboard functions
const dashboard = {
    load: async () => {
        try {
            // Show loading state
            document.getElementById('loading-state').style.display = 'block';
            document.getElementById('error-state').style.display = 'none';
            document.getElementById('dashboard-content').style.display = 'none';

            // Fetch dashboard data
            dashboardData = await api.get(API_ENDPOINTS.dashboard);

            // Render dashboard components
            dashboard.renderKPICards(dashboardData.kpi_cards);
            dashboard.renderCharts(dashboardData.charts);
            dashboard.updateLastUpdated(dashboardData.last_updated);

            // Load recent anomalies
            await dashboard.loadRecentAnomalies();

            // Show dashboard content
            document.getElementById('loading-state').style.display = 'none';
            document.getElementById('dashboard-content').style.display = 'block';
            document.getElementById('dashboard-content').classList.add('fade-in');

        } catch (error) {
            console.error('Error loading dashboard:', error);
            document.getElementById('loading-state').style.display = 'none';
            document.getElementById('error-state').style.display = 'block';
            document.getElementById('error-message').textContent = error.message;
        }
    },

    renderKPICards: (kpiData) => {
        const container = document.getElementById('kpi-cards-container');
        if (!container || !kpiData) return;

        const kpiHTML = `
            <div class="row mb-4">
                ${Object.entries(kpiData).map(([key, kpi]) => `
                    <div class="col-lg-3 col-md-6 mb-3">
                        <div class="kpi-card ${kpi.color}">
                            <div class="kpi-icon">
                                <i class="fas fa-${kpi.icon}"></i>
                            </div>
                            <div class="kpi-value">${kpi.value}</div>
                            <div class="kpi-title">${kpi.title}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        container.innerHTML = kpiHTML;
    },

    renderCharts: (chartsData) => {
        if (!chartsData) return;

        // Render each chart
        Object.entries(chartsData).forEach(([chartName, chartData]) => {
            if (chartData) {
                dashboard.renderChart(chartName, chartData);
            } else {
                utils.showError(`${chartName}-chart`, 'Chart data not available');
            }
        });
    },

    renderChart: (chartName, chartData) => {
        const elementId = `${chartName.replace('_', '-')}-chart`;
        const element = document.getElementById(elementId);

        if (!element) {
            console.warn(`Chart element ${elementId} not found`);
            return;
        }

        try {
            // Parse chart data if it's a string
            const plotData = typeof chartData === 'string' ? JSON.parse(chartData) : chartData;

            // Configure responsive layout
            const layout = {
                ...plotData.layout,
                responsive: true,
                autosize: true,
                margin: { l: 50, r: 50, t: 50, b: 50 }
            };

            const config = {
                responsive: true,
                displayModeBar: false,
                displaylogo: false
            };

            // Render the chart
            Plotly.newPlot(element, plotData.data, layout, config);

        } catch (error) {
            console.error(`Error rendering ${chartName} chart:`, error);
            utils.showError(elementId, 'Error loading chart');
        }
    },

    loadRecentAnomalies: async () => {
        try {
            const anomaliesData = await api.get(API_ENDPOINTS.recentAnomalies);
            dashboard.renderAnomaliesList(anomaliesData.anomalies);

            if (anomaliesData.chart) {
                dashboard.renderChart('anomaly', anomaliesData.chart);
            }
        } catch (error) {
            console.error('Error loading recent anomalies:', error);
            utils.showError('anomaly-list', 'Unable to load recent anomalies');
        }
    },

    renderAnomaliesList: (anomalies) => {
        const container = document.getElementById('anomaly-list');
        if (!container) return;

        if (!anomalies || anomalies.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="fas fa-check-circle fa-2x mb-2"></i>
                    <p>No recent anomalies detected</p>
                </div>
            `;
            return;
        }

        const anomaliesHTML = anomalies.map(anomaly => {
            const badgeClass = anomaly.confidence > 0.8 ? 'high' :
                             anomaly.confidence > 0.5 ? 'medium' : 'low';

            return `
                <div class="anomaly-item ${anomaly.is_anomaly ? 'warning' : 'normal'}">
                    <div class="anomaly-item-header">
                        <strong>Meter #${anomaly.water_meter_id || 'Unknown'}</strong>
                        <span class="anomaly-badge ${badgeClass}">
                            ${Math.round(anomaly.confidence * 100)}%
                        </span>
                    </div>
                    <div class="anomaly-details">
                        <small class="text-muted">
                            Consumption: ${utils.formatNumber(anomaly.consumption)} m³<br>
                            ${anomaly.reason}<br>
                            <em>${utils.formatDate(anomaly.detected_at)}</em>
                        </small>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = anomaliesHTML;
    },

    updateLastUpdated: (timestamp) => {
        const element = document.getElementById('last-updated');
        if (element && timestamp) {
            element.textContent = utils.formatDate(timestamp);
        }
    },

    refresh: async () => {
        try {
            // Show loading in refresh button
            const refreshBtn = document.querySelector('[onclick="refreshDashboard()"]');
            if (refreshBtn) {
                const originalHTML = refreshBtn.innerHTML;
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
                refreshBtn.disabled = true;

                // Reload dashboard
                await dashboard.load();

                // Restore button
                refreshBtn.innerHTML = originalHTML;
                refreshBtn.disabled = false;

                utils.showSuccess('Dashboard refreshed successfully');
            }
        } catch (error) {
            console.error('Error refreshing dashboard:', error);
            utils.showErrorToast('Failed to refresh dashboard');
        }
    }
};

// Anomaly detection functions
const anomalyDetection = {
    test: async () => {
        try {
            // Get form data
            const formData = {
                water_meter_id: parseInt(document.getElementById('meter-id').value),
                previous_reading: parseInt(document.getElementById('previous-reading').value),
                current_reading: parseInt(document.getElementById('current-reading').value),
                days_billed: parseInt(document.getElementById('days-billed').value)
            };

            // Validate form data
            if (!formData.water_meter_id || !formData.previous_reading ||
                !formData.current_reading || !formData.days_billed) {
                utils.showErrorToast('Please fill in all required fields');
                return;
            }

            if (formData.current_reading < formData.previous_reading) {
                utils.showErrorToast('Current reading cannot be less than previous reading');
                return;
            }

            // Show loading state
            const resultContainer = document.getElementById('anomaly-result');
            const contentContainer = document.getElementById('anomaly-result-content');

            resultContainer.style.display = 'block';
            contentContainer.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                        <span class="visually-hidden">Analyzing...</span>
                    </div>
                    <span class="ms-2">Analyzing reading...</span>
                </div>
            `;

            // Make API call
            const result = await api.post(API_ENDPOINTS.detectAnomaly, formData);

            // Display result
            anomalyDetection.displayResult(result);

        } catch (error) {
            console.error('Error testing anomaly detection:', error);
            utils.showErrorToast('Failed to analyze reading: ' + error.message);

            // Hide result container on error
            document.getElementById('anomaly-result').style.display = 'none';
        }
    },

    displayResult: (result) => {
        const contentContainer = document.getElementById('anomaly-result-content');
        if (!contentContainer) return;

        const consumption = result.consumption || 0;
        const dailyConsumption = result.consumption_per_day || 0;

        const alertClass = result.is_anomaly ? 'alert-warning' : 'alert-success';
        const iconClass = result.is_anomaly ? 'fa-exclamation-triangle' : 'fa-check-circle';
        const statusText = result.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL READING';

        const resultHTML = `
            <div class="alert ${alertClass} mb-3">
                <div class="d-flex align-items-center">
                    <i class="fas ${iconClass} fa-2x me-3"></i>
                    <div>
                        <h6 class="mb-1">${statusText}</h6>
                        <p class="mb-0">Confidence: ${Math.round(result.confidence * 100)}%</p>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h6>Consumption Details</h6>
                    <ul class="list-unstyled">
                        <li><strong>Total:</strong> ${utils.formatNumber(consumption)} m³</li>
                        <li><strong>Daily Average:</strong> ${utils.formatNumber(dailyConsumption)} m³/day</li>
                        <li><strong>Anomaly Score:</strong> ${utils.formatNumber(result.anomaly_score || 0, 3)}</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6>Analysis</h6>
                    <p class="mb-2"><strong>Reason:</strong> ${result.reason}</p>
                    <p class="mb-0"><strong>Recommendation:</strong> ${result.recommendation}</p>
                </div>
            </div>
        `;

        contentContainer.innerHTML = resultHTML;
    },

    clearForm: () => {
        document.getElementById('anomaly-test-form').reset();
        document.getElementById('anomaly-result').style.display = 'none';
    }
};

// Export functions
const exportFunctions = {
    data: () => {
        if (!dashboardData) {
            utils.showErrorToast('No data available to export');
            return;
        }

        try {
            // Create CSV content
            const csvContent = exportFunctions.createCSV();

            // Create download link
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);

            link.setAttribute('href', url);
            link.setAttribute('download', `water_management_data_${new Date().toISOString().split('T')[0]}.csv`);
            link.style.visibility = 'hidden';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            utils.showSuccess('Data exported successfully');

        } catch (error) {
            console.error('Error exporting data:', error);
            utils.showErrorToast('Failed to export data');
        }
    },

    createCSV: () => {
        if (!dashboardData || !dashboardData.stats) {
            return 'No data available';
        }

        const stats = dashboardData.stats;
        const csvRows = [
            'Metric,Value',
            `Total Periods,${stats.total_periods}`,
            `Date Range Start,${stats.date_range?.start || 'N/A'}`,
            `Date Range End,${stats.date_range?.end || 'N/A'}`,
            `Mean Consumption,${stats.consumption_stats?.mean || 'N/A'}`,
            `Std Consumption,${stats.consumption_stats?.std || 'N/A'}`,
            `Min Consumption,${stats.consumption_stats?.min || 'N/A'}`,
            `Max Consumption,${stats.consumption_stats?.max || 'N/A'}`,
            `Mean Rainfall,${stats.rainfall_stats?.mean || 'N/A'}`,
            `Std Rainfall,${stats.rainfall_stats?.std || 'N/A'}`,
            `Min Rainfall,${stats.rainfall_stats?.min || 'N/A'}`,
            `Max Rainfall,${stats.rainfall_stats?.max || 'N/A'}`,
            `Correlation,${stats.correlation?.rainfall_consumption || 'N/A'}`
        ];

        return csvRows.join('\n');
    }
};

// Global functions (called from HTML)
window.loadDashboard = dashboard.load;
window.refreshDashboard = dashboard.refresh;
window.testAnomaly = anomalyDetection.test;
window.exportData = exportFunctions.data;

// Modal event handlers
document.addEventListener('DOMContentLoaded', function() {
    // Clear form when modal is hidden
    const anomalyModal = document.getElementById('anomalyTestModal');
    if (anomalyModal) {
        anomalyModal.addEventListener('hidden.bs.modal', anomalyDetection.clearForm);
    }

    // Form submission handler
    const anomalyForm = document.getElementById('anomaly-test-form');
    if (anomalyForm) {
        anomalyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            anomalyDetection.test();
        });
    }
});

// Auto-refresh functionality
const autoRefresh = {
    interval: null,

    start: (intervalMinutes = 5) => {
        autoRefresh.stop(); // Clear any existing interval
        autoRefresh.interval = setInterval(() => {
            dashboard.refresh();
        }, intervalMinutes * 60 * 1000);

        console.log(`Auto-refresh started: every ${intervalMinutes} minutes`);
    },

    stop: () => {
        if (autoRefresh.interval) {
            clearInterval(autoRefresh.interval);
            autoRefresh.interval = null;
            console.log('Auto-refresh stopped');
        }
    }
};

// Initialize auto-refresh when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Start auto-refresh after 1 minute delay
    setTimeout(() => {
        autoRefresh.start(5); // Refresh every 5 minutes
    }, 60000);
});

// Handle visibility change to pause/resume auto-refresh
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        autoRefresh.stop();
    } else {
        autoRefresh.start(5);
    }
});

// Health check function
const healthCheck = {
    check: async () => {
        try {
            const health = await api.get(API_ENDPOINTS.health);
            console.log('Health check:', health);
            return health.status === 'healthy';
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    },

    monitor: () => {
        setInterval(async () => {
            const isHealthy = await healthCheck.check();
            if (!isHealthy) {
                utils.showErrorToast('System health check failed. Some features may not work properly.');
            }
        }, 300000); // Check every 5 minutes
    }
};

// Start health monitoring
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        healthCheck.monitor();
    }, 30000); // Start after 30 seconds
});