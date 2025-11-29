import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { EmergencyReport, GeoLocation } from '../types';
import L from 'leaflet';

// Fix for default Leaflet marker icons in webpack/react environments
const iconUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png";
const iconRetinaUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png";
const shadowUrl = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png";

const DefaultIcon = L.icon({
    iconUrl,
    iconRetinaUrl,
    shadowUrl,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

// Custom red icon for emergencies
const EmergencyIcon = L.icon({
    iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

interface MapProps {
  userLocation: GeoLocation | null;
  reports: EmergencyReport[];
}

export const Map: React.FC<MapProps> = ({ userLocation, reports }) => {
  const center = userLocation || { lat: 34.0522, lng: -118.2437 }; // Default LA

  return (
    <MapContainer center={[center.lat, center.lng]} zoom={13} scrollWheelZoom={true} className="w-full h-full z-0">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* User Location */}
      {userLocation && (
        <Marker position={[userLocation.lat, userLocation.lng]}>
          <Popup>You are here</Popup>
        </Marker>
      )}

      {/* Reports */}
      {reports.map((report) => (
        <Marker 
          key={report.id} 
          position={[report.location.lat, report.location.lng]}
          icon={EmergencyIcon}
        >
          <Popup>
            <div className="text-gray-900">
              <strong className="block text-red-600 uppercase text-xs">{report.emergencyType}</strong>
              <p className="m-0 text-sm font-semibold">{report.description}</p>
              <div className="mt-1 text-xs text-gray-500">
                <p>Needs: {report.criticalNeeds || 'None specified'}</p>
                <p>People: {report.peopleCount || 'Unknown'}</p>
                <p>{new Date(report.timestamp).toLocaleTimeString()}</p>
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};
