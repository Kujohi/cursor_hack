export interface GeoLocation {
  lat: number;
  lng: number;
}

export interface EmergencyReport {
  id: string;
  timestamp: number;
  location: GeoLocation;
  reporterName?: string;
  emergencyType: string;
  description: string;
  peopleCount?: number;
  criticalNeeds?: string;
  status: 'pending' | 'dispatched' | 'resolved';
}

export interface AudioVisualizerState {
  isRecording: boolean;
  volume: number;
}
