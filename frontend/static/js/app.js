// Chart configurations
Chart.defaults.color = '#a0a5b5';
Chart.defaults.font.family = 'Inter';

let forecastChart = null;
let historyChart = null;
let myGlobe = null;
let markerNodes = {};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Extract marker template and make it the real element
    const templateContainer = document.getElementById('marker-template-container');
    const markerEl = templateContainer.querySelector('.marker-container');
    
    // Cache references to marker nodes before detachment
    markerNodes.aqi = markerEl.querySelector('#marker-aqi');
    markerNodes.ring = markerEl.querySelector('#marker-color-ring');
    markerNodes.popupAqi = markerEl.querySelector('#popup-aqi');
    markerNodes.popupBox = markerEl.querySelector('#popup-aqi-box');
    markerNodes.popupLevel = markerEl.querySelector('#popup-level');
    markerNodes.popupPm25 = markerEl.querySelector('#popup-pm25');
    markerNodes.popupPred = markerEl.querySelector('#popup-pred');
    markerNodes.popupIcon = markerEl.querySelector('.popup-icon');

    markerEl.style.transform = 'translate(-50%, -50%)';
    markerEl.style.pointerEvents = 'auto'; // ensure interactions work
    templateContainer.innerHTML = ''; // clear template to maintain unique IDs
    
    // Initialize 3D Globe
    myGlobe = Globe()
      (document.getElementById('globeViz'))
      .globeImageUrl('//unpkg.com/three-globe/example/img/earth-night.jpg')
      .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
      .backgroundColor('rgba(0,0,0,0)')
      .showAtmosphere(true)
      .atmosphereColor('#3b82f6')
      .atmosphereAltitude(0.25)
      .htmlElementsData([{ lat: 10.7626, lng: 106.6602 }])
      .htmlElement(() => markerEl);

    // Initial camera position pointing at HCMC
    myGlobe.pointOfView({ lat: 10.7626, lng: 106.6602, altitude: 2.0 });
    myGlobe.controls().autoRotate = true;
    myGlobe.controls().autoRotateSpeed = 0.5;
    
    fetchModelInfo();
    updateDashboard();
    
    // Auto refresh every 30 seconds
    setInterval(updateDashboard, 30000);
    
    // Setup advance time button for simulation
    document.getElementById('advance-time-btn').addEventListener('click', () => {
        fetch('/api/advance')
            .then(res => res.json())
            .then(data => {
                console.log("Time advanced, new index:", data.new_idx);
                updateDashboard();
            });
    });
});

async function updateDashboard() {
    try {
        await Promise.all([
            fetchCurrentData(),
            fetchForecastData(),
            fetchHistoryData()
        ]);
        
        // Update globe marker colors based on current AQI
        const currentAqiBox = document.getElementById('marker-color-ring');
        const markerAqi = document.getElementById('marker-aqi');
        const popupBox = document.getElementById('popup-aqi-box');
        
        // The color is set in fetchCurrentData and applied to elements there
    } catch (error) {
        console.error("Error updating dashboard:", error);
    }
}

async function fetchCurrentData() {
    const res = await fetch('/api/current');
    const data = await res.json();
    
    // Update Header
    document.getElementById('current-time').textContent = data.timestamp;
    
    // Update Main Stats
    const aqiEl = document.getElementById('current-aqi');
    aqiEl.textContent = data.aqi;
    aqiEl.style.color = data.color;
    aqiEl.style.textShadow = `0 0 20px ${data.color}80`; // Add glow
    
    // Update Risk Card
    document.getElementById('risk-percent').textContent = `${data.risk.percent}%`;
    document.getElementById('risk-desc').textContent = data.risk.description;
    const riskCard = document.querySelector('.risk-card');
    
    // Dynamic risk background
    if(data.risk.percent > 70) {
        riskCard.style.background = 'linear-gradient(135deg, rgba(80, 20, 20, 0.8), rgba(100, 30, 30, 0.6))';
    } else if(data.risk.percent > 40) {
        riskCard.style.background = 'linear-gradient(135deg, rgba(80, 60, 20, 0.8), rgba(100, 80, 30, 0.6))';
    } else {
        riskCard.style.background = 'linear-gradient(135deg, rgba(20, 30, 80, 0.8), rgba(30, 40, 100, 0.6))';
    }
    
    // Update Weather
    document.getElementById('w-temp').textContent = `${data.weather.temperature}°C`;
    document.getElementById('w-hum').textContent = `${data.weather.humidity}%`;
    document.getElementById('w-wind').textContent = `${data.weather.wind_speed} km/h`;
    document.getElementById('w-press').textContent = `${data.weather.pressure} hPa`;
    
    // Update Globe markers via cached nodes
    markerNodes.aqi.textContent = data.aqi;
    markerNodes.aqi.style.background = data.color;
    markerNodes.ring.style.borderColor = data.color;
    
    // Update Popup
    markerNodes.popupAqi.textContent = data.aqi;
    markerNodes.popupBox.style.background = data.color;
    markerNodes.popupLevel.textContent = data.level;
    markerNodes.popupPm25.textContent = `PM2.5: ${data.pm25} µg/m³`;
    markerNodes.popupPred.textContent = `${data.forecast_24h} µg/m³`;
    markerNodes.popupIcon.style.color = data.color;
}

async function fetchForecastData() {
    const res = await fetch('/api/forecast');
    const data = await res.json();
    
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(245, 158, 11, 0.5)'); // gold
    gradient.addColorStop(1, 'rgba(245, 158, 11, 0.0)');
    
    if (forecastChart) {
        forecastChart.data.labels = data.labels;
        forecastChart.data.datasets[0].data = data.values;
        forecastChart.update();
    } else {
        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Predicted PM2.5',
                    data: data.values,
                    borderColor: '#f59e0b',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#191e32',
                    pointBorderColor: '#f59e0b',
                    pointRadius: 3,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(25, 30, 50, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { maxTicksLimit: 8 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

async function fetchHistoryData() {
    const res = await fetch('/api/history');
    const data = await res.json();
    
    const ctx = document.getElementById('historyChart').getContext('2d');
    
    if (historyChart) {
        historyChart.data.labels = data.labels;
        historyChart.data.datasets[0].data = data.actual;
        historyChart.data.datasets[1].data = data.predicted;
        historyChart.update();
    } else {
        historyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Actual',
                        data: data.actual,
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 0
                    },
                    {
                        label: 'Predicted',
                        data: data.predicted,
                        borderColor: '#3b82f6', // blue
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: { boxWidth: 12, usePointStyle: true }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { maxTicksLimit: 6 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                }
            }
        });
    }
}

async function fetchModelInfo() {
    const res = await fetch('/api/model-info');
    const data = await res.json();
    
    document.getElementById('model-name').textContent = data.model_name;
    document.getElementById('m-rmse').textContent = data.metrics.rmse.toFixed(2);
    document.getElementById('m-mae').textContent = data.metrics.mae.toFixed(2);
    document.getElementById('m-mape').textContent = data.metrics.mape.toFixed(2);
}

// Interactivity 
window.zoomGlobe = function(dir) {
    if(!myGlobe) return;
    const currentR = myGlobe.pointOfView().altitude;
    myGlobe.pointOfView({ altitude: currentR - (dir * 0.5) }, 500);
};

window.toggleGlobeStyle = function(style) {
    if(!myGlobe) return;
    if (style === 'map') {
        myGlobe.globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg');
        document.getElementById('btn-iq-map').classList.replace('btn-secondary', 'btn-primary');
        document.getElementById('btn-3d-globe').classList.replace('btn-primary', 'btn-secondary');
    } else {
        myGlobe.globeImageUrl('//unpkg.com/three-globe/example/img/earth-night.jpg');
        document.getElementById('btn-3d-globe').classList.replace('btn-secondary', 'btn-primary');
        document.getElementById('btn-iq-map').classList.replace('btn-primary', 'btn-secondary');
    }
};

window.showToast = function(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
};

