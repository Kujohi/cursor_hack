import React, { useState, useEffect, useRef } from 'react';
import { GeminiLiveService } from './services/geminiService';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Map } from './components/Map';
import { EmergencyReport, GeoLocation } from './types';
import { 
  PhoneIcon, 
  ShieldCheckIcon, 
  ExclamationTriangleIcon,
  XMarkIcon,
  MapPinIcon
} from '@heroicons/react/24/solid';

const App: React.FC = () => {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<string>('Ready');
  const [audioLevel, setAudioLevel] = useState(0);
  const [userLocation, setUserLocation] = useState<GeoLocation | null>(null);
  const [reports, setReports] = useState<EmergencyReport[]>([]);
  const [showReports, setShowReports] = useState(true);
  
  // Refs
  const geminiRef = useRef<GeminiLiveService | null>(null);

  // Initialize Location
  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.watchPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
        },
        (error) => console.error('Geo error', error),
        { enableHighAccuracy: true }
      );
    }
  }, []);

  // Initialize Gemini Service
  const toggleConnection = async () => {
    if (isConnected) {
      geminiRef.current?.stop();
      setIsConnected(false);
      setStatus('Ready');
      return;
    }

    if (!userLocation) {
      alert("We need your location to connect you to help. Please enable GPS.");
      return;
    }

    try {
      const service = new GeminiLiveService();
      geminiRef.current = service;

      service.onStatusChange = (s) => setStatus(s);
      service.onAudioLevel = (l) => setAudioLevel(l);
      
      service.onReportSubmitted = (data: any) => {
        const newReport: EmergencyReport = {
          id: Date.now().toString(),
          timestamp: Date.now(),
          location: userLocation,
          emergencyType: data.emergencyType,
          description: data.description,
          peopleCount: data.peopleCount,
          criticalNeeds: data.criticalNeeds,
          status: 'pending'
        };
        setReports(prev => [newReport, ...prev]);
      };

      await service.connect();
      setIsConnected(true);
    } catch (e) {
      console.error(e);
      setStatus('Connection Failed');
      setIsConnected(false);
    }
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-gray-900 text-white">
      
      {/* 1. Full Screen Map Layer (Z-0) */}
      <div className="absolute inset-0 z-0">
        <Map userLocation={userLocation} reports={reports} />
      </div>

      {/* 2. Dark Overlay when connected (Z-10) - Focuses attention on voice interaction */}
      <div 
        className={`absolute inset-0 z-10 bg-gray-900/80 backdrop-blur-sm transition-opacity duration-500 pointer-events-none ${isConnected ? 'opacity-100' : 'opacity-0'}`}
      />

      {/* 3. Header (Z-20) */}
      <header className="absolute top-0 left-0 right-0 z-20 h-20 bg-gradient-to-b from-gray-900/90 to-transparent flex items-center justify-between px-6 pt-2 pointer-events-none">
        <div className="flex items-center gap-2 text-red-500 pointer-events-auto bg-gray-900/50 backdrop-blur-md px-4 py-2 rounded-full border border-gray-700/50 shadow-lg">
          <ExclamationTriangleIcon className="w-6 h-6" />
          <span className="font-bold text-lg tracking-tight text-white">Disaster<span className="text-red-500">Connect</span></span>
        </div>
      </header>

      {/* 4. Incident List Floating Card (Z-20) */}
      <div className={`absolute top-24 left-4 z-20 w-80 transition-transform duration-300 ${showReports ? 'translate-x-0' : '-translate-x-[120%]'}`}>
        <div className="bg-gray-800/90 backdrop-blur-md rounded-xl border border-gray-700 shadow-2xl overflow-hidden flex flex-col max-h-[50vh]">
          <div className="p-4 bg-gray-800/50 border-b border-gray-700 flex justify-between items-center">
            <h3 className="font-bold flex items-center gap-2 text-sm">
              <ShieldCheckIcon className="w-4 h-4 text-blue-400" />
              Real-time Incidents
            </h3>
            <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full border border-red-500/20">
              {reports.length} Active
            </span>
          </div>
          <div className="overflow-y-auto p-2 space-y-2">
            {reports.length === 0 ? (
              <div className="p-4 text-center text-gray-500 text-xs">
                No incidents reported nearby.
                <br/>Stay safe.
              </div>
            ) : (
              reports.map(r => (
                <div key={r.id} className="p-3 bg-gray-700/40 rounded-lg border-l-2 border-red-500 hover:bg-gray-700/60 transition-colors">
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-bold text-red-400 text-xs uppercase tracking-wider">{r.emergencyType}</span>
                    <span className="text-[10px] text-gray-400 font-mono">{new Date(r.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                  </div>
                  <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">{r.description}</p>
                  {r.criticalNeeds && (
                     <div className="mt-2 text-[10px] flex gap-1 flex-wrap">
                        <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-300 rounded border border-blue-500/20">Needed: {r.criticalNeeds}</span>
                     </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Toggle Button for Incident List (Mobile friendly) */}
      <button 
        onClick={() => setShowReports(!showReports)}
        className="absolute top-24 left-4 z-10 p-2 bg-gray-800 text-white rounded-full shadow-lg border border-gray-700 hover:bg-gray-700 transition-opacity"
        style={{ opacity: showReports ? 0 : 1, pointerEvents: showReports ? 'none' : 'auto' }}
      >
        <MapPinIcon className="w-6 h-6" />
      </button>

      {/* 5. Main Action Area (Z-30) */}
      <div className="absolute bottom-0 left-0 right-0 z-30 flex flex-col items-center justify-end pb-8 pointer-events-none">
        
        {/* Audio Visualizer (Attached to bottom area when active) */}
        <div className={`transition-all duration-500 ease-out flex flex-col items-center justify-center mb-4 ${isConnected ? 'scale-100 opacity-100 translate-y-0' : 'scale-50 opacity-0 translate-y-20 h-0'}`}>
          <div className="w-64 h-64 relative flex items-center justify-center">
             <div className="absolute inset-0 flex items-center justify-center">
                <AudioVisualizer isActive={isConnected} level={audioLevel} />
             </div>
          </div>
          <p className="text-gray-400 font-mono text-sm mt-4 animate-pulse pointer-events-auto bg-gray-900/80 px-4 py-1 rounded-full">
            {status}
          </p>
        </div>

        {/* Call Button */}
        <div className="pointer-events-auto relative group">
           {isConnected && <div className="pulse-ring absolute inset-0 rounded-full border-red-500/50"></div>}
           <button 
            onClick={toggleConnection}
            className={`
              relative flex items-center justify-center rounded-full shadow-2xl transition-all duration-300 
              ${isConnected 
                ? 'w-20 h-20 bg-gray-800 text-white border-2 border-gray-600 hover:bg-gray-700 hover:border-red-500' 
                : 'w-24 h-24 bg-red-600 text-white hover:bg-red-700 hover:scale-105 shadow-red-900/50'
              }
            `}
          >
            {isConnected ? (
              <XMarkIcon className="w-8 h-8" />
            ) : (
              <PhoneIcon className="w-10 h-10" />
            )}
          </button>
        </div>

        {/* Helper Text (Only when not connected) */}
        <div className={`mt-4 bg-gray-900/80 backdrop-blur px-6 py-2 rounded-full border border-gray-700 pointer-events-auto transition-opacity duration-300 ${isConnected ? 'opacity-0' : 'opacity-100'}`}>
           <p className="text-sm font-medium text-gray-300">
             Tap emergency button to report
           </p>
        </div>

      </div>
      
      {/* Background Gradient for bottom area readability */}
      <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-gray-900 via-gray-900/80 to-transparent z-10 pointer-events-none" />

    </div>
  );
};

export default App;
