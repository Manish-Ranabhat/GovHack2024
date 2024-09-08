document.addEventListener("DOMContentLoaded", function () {
    // Initialize the map centered on Melbourne
    var map = L.map('map').setView([-37.8136, 144.9631], 8); // Default center at Melbourne

    // Set up the base map layer using OpenStreetMap
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    let accidentHeatmapLayer = null;
    let nodeData = [];

    // Function to generate the accident heatmap
    function generateAccidentHeatmap(data) {
        const heatPoints = data
            .filter(d => d.latitude && d.longitude && !isNaN(d.latitude) && !isNaN(d.longitude))
            .map(d => [parseFloat(d.latitude), parseFloat(d.longitude), 1]);  // 1 is the weight

        // Clear any existing layers before adding a new heatmap
        if (accidentHeatmapLayer) {
            map.removeLayer(accidentHeatmapLayer);
        }

        if (heatPoints.length > 0) {
            accidentHeatmapLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15 }).addTo(map);  // Add heatmap
        } else {
            console.log("No valid data points for the accident heatmap.");
        }
    }

    // Fetch MongoDB accident data and merge with CSV
    function fetchAccidentReportsAndCSV() {
        // Fetch CSV data
        Papa.parse(csvUrl, {
            download: true,
            header: true,
            complete: function (results) {
                nodeData = results.data;

                // Fetch MongoDB data after CSV data is loaded
                fetch('/get-reports')
                    .then(response => response.json())
                    .then(mongoData => {
                        const mongoReports = mongoData.reports;

                        // Combine MongoDB and CSV data
                        const combinedData = [...nodeData.map(d => ({
                            latitude: d.LATITUDE,
                            longitude: d.LONGITUDE,
                            address: d.LGA_NAME
                        })), ...mongoReports];

                        generateAccidentHeatmap(combinedData);
                    })
                    .catch(error => console.error('Error fetching MongoDB data:', error));
            }
        });
    }

    // Call this function to load data when the page loads
    fetchAccidentReportsAndCSV();

    // Function to filter accidents by location
    function filterAccidents(searchLocation) {
        const filteredData = nodeData.filter(row => {
            return row.LGA_NAME && row.LGA_NAME.toLowerCase().includes(searchLocation);
        });

        // Fetch MongoDB data as well
        fetch('/get-reports')
            .then(response => response.json())
            .then(mongoData => {
                const mongoReports = mongoData.reports.filter(report =>
                    report.address.toLowerCase().includes(searchLocation)
                );

                const filteredDataCombined = [
                    ...filteredData.map(row => ({
                        latitude: row.LATITUDE,
                        longitude: row.LONGITUDE,
                    })),
                    ...mongoReports
                ];

                const heatPoints = filteredDataCombined.map(row => [
                    parseFloat(row.latitude),
                    parseFloat(row.longitude),
                    1 // Weight for each point
                ]);

                if (accidentHeatmapLayer) {
                    map.removeLayer(accidentHeatmapLayer);
                }

                accidentHeatmapLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15 }).addTo(map);

                const bounds = L.latLngBounds(heatPoints.map(p => [p[0], p[1]]));
                map.fitBounds(bounds, { padding: [20, 20] });
            });
    }

    // Filter button functionality
    document.getElementById("searchButton").addEventListener("click", function () {
        const searchLocation = document.getElementById("locationSearch").value.toLowerCase().trim();

        if (!searchLocation) {
            alert('Please enter a location to search.');
            return;
        }

        console.log(`Searching for location: ${searchLocation}`);
        filterAccidents(searchLocation);
    });

    // Function to get the accident prediction for a specific location
    function getAccidentPrediction(locationName) {
        fetch('/predict-accident', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location_name: locationName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                const tableBody = document.getElementById('results-table-body');
                tableBody.innerHTML = ''; // Clear any previous results
                
                // Iterate over predictions
                data.predictions.forEach(prediction => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${prediction.latitude}</td>
                        <td>${prediction.longitude}</td>
                        <td>${prediction.prediction} (Probability: ${prediction.accident_probability})</td>
                    `;
                    tableBody.appendChild(row);

                    // Add markers to the map
                    L.marker([prediction.latitude, prediction.longitude]).addTo(map)
                        .bindPopup(`<b>${locationName}</b><br>Predicted Crashes: ${prediction.prediction}<br> Probability: ${prediction.accident_probability}`)
                        .openPopup();
                });

                if (data.predictions.length > 0) {
                    const firstPrediction = data.predictions[0];
                    map.setView([firstPrediction.latitude, firstPrediction.longitude], 14);
                }
            }
        })
        .catch(error => console.error('Error fetching prediction:', error));
    }

    // Prediction form submission
    document.getElementById('prediction-form').addEventListener('submit', function(event) {
        event.preventDefault();

        const locationName = document.getElementById('location_name').value;
        console.log(`Submitting location: ${locationName}`);  // Debugging log

        // Get the accident prediction for the entered location name
        getAccidentPrediction(locationName);
    });

    // Search button functionality for accident prediction
    document.getElementById("searchPredictionButton").addEventListener("click", function () {
        const locationName = document.getElementById("locationSearch").value;

        if (!locationName) {
            alert('Please enter a location name.');
            return;
        }

        // Get the accident prediction for the entered location name
        getAccidentPrediction(locationName);
    });

    // Automatically load accident heatmap when the map loads
    generateAccidentHeatmap(nodeData);
});
