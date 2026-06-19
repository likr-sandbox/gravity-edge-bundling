export function parseCSV(text) {
  const lines = text.split(/\r?\n/);
  if (lines.length === 0) return [];

  const headers = lines[0].split(',');
  const result = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const row = [];
    let insideQuote = false;
    let entry = '';

    for (let j = 0; j < line.length; j++) {
      const char = line[j];
      if (char === '"') {
        insideQuote = !insideQuote;
      } else if (char === ',' && !insideQuote) {
        row.push(entry.trim());
        entry = '';
      } else {
        entry += char;
      }
    }
    row.push(entry.trim());

    if (row.length === headers.length) {
      const obj = {};
      for (let h = 0; h < headers.length; h++) {
        obj[headers[h]] = row[h];
      }
      result.push(obj);
    }
  }
  return result;
}

export async function loadDatasets() {
  const airportsRes = await fetch('/airports.csv');
  const airportsText = await airportsRes.text();
  const airportsRaw = parseCSV(airportsText);

  const flightsRes = await fetch('/flights.csv');
  const flightsText = await flightsRes.text();
  const flightsRaw = parseCSV(flightsText);

  // Create airport map
  const airportMap = new Map();
  airportsRaw.forEach(ap => {
    const lat = parseFloat(ap.latitude);
    const lng = parseFloat(ap.longitude);
    if (!isNaN(lat) && !isNaN(lng)) {
      airportMap.set(ap.iata, {
        iata: ap.iata,
        name: ap.name,
        city: ap.city,
        state: ap.state,
        country: ap.country,
        lat,
        lng,
        mass: 1.0,
        degree: 0,
        flightCount: 0,
      });
    }
  });

  // Filter flights and count degrees
  const validFlights = [];
  flightsRaw.forEach(fl => {
    const origin = airportMap.get(fl.origin);
    const dest = airportMap.get(fl.destination);
    const count = parseInt(fl.count) || 1;

    if (origin && dest) {
      origin.degree += 1;
      dest.degree += 1;
      origin.flightCount += count;
      dest.flightCount += count;

      validFlights.push({
        origin: fl.origin,
        destination: fl.destination,
        count,
      });
    }
  });

  // Keep only airports that have at least one flight (active airports)
  const activeAirports = Array.from(airportMap.values()).filter(ap => ap.degree > 0);

  return {
    airports: activeAirports,
    flights: validFlights,
  };
}

// Function to scale coordinates to fit target grid (e.g. width x height)
// Standard Mercator projection or simple bounding box mapping
export function scaleCoordinates(airports, width, height, padding = 40) {
  if (airports.length === 0) return [];

  // Find longitude and latitude boundaries
  // Limit to continental US coordinates for clean layout since this dataset is US airports
  const continental = airports.filter(
    ap => ap.lng >= -125 && ap.lng <= -66 && ap.lat >= 24 && ap.lat <= 50
  );

  const targetList = continental.length > 0 ? continental : airports;

  let minLng = Infinity, maxLng = -Infinity;
  let minLat = Infinity, maxLat = -Infinity;

  targetList.forEach(ap => {
    if (ap.lng < minLng) minLng = ap.lng;
    if (ap.lng > maxLng) maxLng = ap.lng;
    if (ap.lat < minLat) minLat = ap.lat;
    if (ap.lat > maxLat) maxLat = ap.lat;
  });

  const lngRange = maxLng - minLng;
  const latRange = maxLat - minLat;

  // Simple Mercator-like stretching in Y to avoid squishing
  const mapWidth = width - 2 * padding;
  const mapHeight = height - 2 * padding;

  const scale = Math.min(mapWidth / lngRange, mapHeight / latRange);

  const scaledAirports = targetList.map(ap => {
    // Map longitude: min is left (0), max is right (width)
    const x = padding + (ap.lng - minLng) * (mapWidth / lngRange);
    // Map latitude: max is top (0), min is bottom (height) (canvas Y points down)
    const y = padding + (maxLat - ap.lat) * (mapHeight / latRange);

    return {
      ...ap,
      x: Math.max(0.0, Math.min(width - 1, x)),
      y: Math.max(0.0, Math.min(height - 1, y)),
    };
  });

  return scaledAirports;
}
