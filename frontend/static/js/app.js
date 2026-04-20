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
    
    // Initialize 3D Globe with high-res textures
    const globeContainer = document.getElementById('globeViz');
    myGlobe = Globe()
      (globeContainer)
      .globeImageUrl('//unpkg.com/three-globe@2.41.12/example/img/earth-blue-marble.jpg')
      .bumpImageUrl('//unpkg.com/three-globe@2.41.12/example/img/earth-topology.png')
      .backgroundColor('rgba(0,0,0,0)')
      .showAtmosphere(true)
      .atmosphereColor('#3b82f6')
      .atmosphereAltitude(0.2)
      .width(globeContainer.clientWidth)
      .height(globeContainer.clientHeight)
      .htmlElementsData([{ lat: 10.7626, lng: 106.6602 }])
      .htmlElement(() => markerEl)
      .ringsData([{ lat: 10.7626, lng: 106.6602, color: 'rgba(255, 126, 0, 0.6)' }])
      .ringColor(d => d.color)
      .ringMaxRadius(3)
      .ringPropagationSpeed(2)
      .ringRepeatPeriod(600)
      // HCMC boundary polygon - initially hidden
      .polygonsData([])
      .polygonCapColor(() => 'rgba(255, 126, 0, 0.15)')
      .polygonSideColor(() => 'rgba(255, 126, 0, 0.6)')
      .polygonStrokeColor(() => 'rgba(255, 200, 50, 1)')
      .polygonAltitude(0.025)
      .polygonsTransitionDuration(1000);

    // Dynamic resize handler
    window.addEventListener('resize', () => {
        myGlobe.width(globeContainer.clientWidth);
        myGlobe.height(globeContainer.clientHeight);
    });

    // Premium Camera Position
    myGlobe.pointOfView({ lat: 10.7626, lng: 106.6602, altitude: 2.2 }, 0);
    myGlobe.controls().autoRotate = true;
    myGlobe.controls().autoRotateSpeed = 0.3;
    
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
    } catch (error) {
        console.error("Error updating dashboard:", error);
    }
}

async function fetchCurrentData() {
    const res = await fetch('/api/current');
    const data = await res.json();
    
    // Update Header — show HH:MM in large Orbitron clock
    // data.timestamp format: "DD/MM/YYYY HH:MM"
    const timeEl = document.getElementById('current-time');
    if (timeEl && data.timestamp) {
        const parts = data.timestamp.split(' ');
        timeEl.textContent = parts.length >= 2 ? parts[1] : data.timestamp;
    }
    
    // Update Main Stats
    const aqiEl = document.getElementById('current-aqi');
    aqiEl.textContent = data.aqi;
    aqiEl.style.color = data.color;

    // PM2.5 + level in detail rows
    const pm25El = document.getElementById('current-pm25');
    if (pm25El) {
        pm25El.innerHTML = `${data.pm25} <small>µg/m³</small>`;
        pm25El.style.color = data.color;
    }
    const levelEl = document.getElementById('current-level');
    if (levelEl) {
        levelEl.textContent = data.level;
        levelEl.style.color = data.color;
    }

    
    // Update 24h Forecast Risk Card
    const forecast24h = data.forecast_24h;
    document.getElementById('forecast-24h-val').textContent = forecast24h;
    document.getElementById('forecast-24h-val').style.color = data.color;
    document.getElementById('risk-percent').textContent = `${data.risk.percent}%`;
    document.getElementById('risk-percent').style.color = data.color;
    document.getElementById('risk-desc').textContent = data.risk.description;
    // Move dot indicator on the gradient bar (clamp to 2%–96% for visual padding)
    const barPct = Math.min(96, Math.max(2, data.risk.percent));
    document.getElementById('risk-bar-fill').style.left = `calc(${barPct}% - 7px)`;
    document.getElementById('risk-bar-fill').style.boxShadow = `0 0 10px 2px ${data.color}`;

    
    // Dynamically update the 3D surface ring glow covering the city
    myGlobe.ringsData([{ lat: 10.7626, lng: 106.6602, color: data.color }]);
    
    // Update polygon boundary color if zoomed in
    if (isZoomedIn) {
        myGlobe
            .polygonCapColor(() => 'rgba(255, 126, 0, 0.15)')
            .polygonSideColor(() => 'rgba(255, 126, 0, 0.6)')
            .polygonStrokeColor(() => 'rgba(255, 200, 50, 1)')
            .polygonsData([HCMC_BOUNDARY]);
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
    
    // Update CSS custom property for pulse/sparkle glow color
    markerNodes.ring.style.boxShadow = `inset 0 0 20px ${data.color}`;
    markerNodes.aqi.style.boxShadow = `0 0 30px 10px ${data.color}60, inset 0 0 15px rgba(255,255,255,0.8)`;
    
    // Update Popup
    markerNodes.popupAqi.textContent = data.aqi;
    markerNodes.popupBox.style.background = data.color;
    markerNodes.popupLevel.textContent = data.level;
    markerNodes.popupIcon.style.color = data.color;
}

async function fetchForecastData() {
    const res = await fetch('/api/forecast');
    const data = await res.json();
    
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)'); // blue gradient
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');
    
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

    // Model badge (shorten display name)
    document.getElementById('model-name').textContent = 'XGBoost Tuned';
    // Hidden values kept for reference
    document.getElementById('m-rmse').textContent = data.metrics.rmse.toFixed(2);

    // MAE in plain sentence: "within X µg/m³"
    const mae = data.metrics.mae;
    document.getElementById('m-mae').textContent = mae.toFixed(1);

    // Compute intuitive accuracy % : score = (1 - MAE / 50) * 100
    // 50 µg/m³ is used as "completely inaccurate" baseline for HCMC PM2.5 range
    const accPct = Math.round(Math.max(0, Math.min(100, (1 - mae / 50) * 100)));
    document.getElementById('intel-acc-pct').textContent = `${accPct}%`;

    // Animate accuracy bar fill
    const barEl = document.getElementById('intel-bar-fill');
    if (barEl) {
        // colour: green > 75%, yellow 50–75%, orange < 50%
        const color = accPct >= 75 ? '#4ade80' : accPct >= 50 ? '#facc15' : '#fb923c';
        barEl.style.width = `${accPct}%`;
        barEl.style.background = `linear-gradient(to right, ${color}88, ${color})`;
    }
}

// Simplified administrative boundary of Ho Chi Minh City (GeoJSON)
const HCMC_BOUNDARY = {
    type: "Feature",
    properties: { name: "Hồ Chí Minh City", area: "Monitoring Zone" },
    geometry: {
        type: "Polygon",
        coordinates: [[
            [106.3637, 10.7538],
            [106.3670, 10.8150],
            [106.3890, 10.8680],
            [106.4250, 10.9200],
            [106.4800, 10.9620],
            [106.5400, 10.9900],
            [106.6050, 11.0050],
            [106.6700, 11.0000],
            [106.7300, 10.9750],
            [106.7850, 10.9350],
            [106.8250, 10.8800],
            [106.8550, 10.8150],
            [106.8750, 10.7500],
            [106.8850, 10.6800],
            [106.8800, 10.6100],
            [106.8600, 10.5500],
            [106.8250, 10.4950],
            [106.7750, 10.4500],
            [106.7150, 10.4150],
            [106.6500, 10.3900],
            [106.5800, 10.3800],
            [106.5100, 10.3850],
            [106.4450, 10.4050],
            [106.3950, 10.4400],
            [106.3600, 10.4900],
            [106.3400, 10.5500],
            [106.3350, 10.6200],
            [106.3400, 10.6850],
            [106.3637, 10.7538]
        ]]
    }
};

let isZoomedIn = false;

// Interactivity 
window.zoomToHCMC = function() {
    if(!myGlobe) return;
    
    if (isZoomedIn) {
        // Zoom back out and hide boundary
        myGlobe.pointOfView({ lat: 10.7626, lng: 106.6602, altitude: 2.2 }, 1800);
        myGlobe.polygonsData([]);
        myGlobe.controls().autoRotate = true;
        isZoomedIn = false;
    } else {
        // Set bright, clearly visible polygon colors
        myGlobe
            .polygonCapColor(() => 'rgba(255, 126, 0, 0.15)')
            .polygonSideColor(() => 'rgba(255, 126, 0, 0.7)')
            .polygonStrokeColor(() => 'rgba(255, 200, 50, 1)')
            .polygonAltitude(0.025);
        
        // Zoom in and show HCMC boundary
        myGlobe.pointOfView({ lat: 10.7626, lng: 106.6602, altitude: 0.4 }, 2000);
        
        // Show the boundary polygon after zoom has started
        setTimeout(() => {
            myGlobe.polygonsData([HCMC_BOUNDARY]);
        }, 600);
        
        myGlobe.controls().autoRotate = false;
        isZoomedIn = true;
    }
};

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

