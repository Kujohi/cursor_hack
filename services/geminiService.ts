import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { arrayBufferToBase64, decodeAudioData, float32ToInt16PCM, base64ToUint8Array } from "./audioUtils";

const API_KEY = process.env.API_KEY || '';
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

// Define the function tool
const reportEmergencyTool: FunctionDeclaration = {
  name: 'reportEmergency',
  description: 'Submit an emergency report to the dispatch center when the user provides sufficient information about their situation.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      emergencyType: {
        type: Type.STRING,
        description: 'Type of disaster (e.g., Flood, Fire, Injury, Trapped, Storm).'
      },
      description: {
        type: Type.STRING,
        description: 'A concise summary of the situation and what happened.'
      },
      peopleCount: {
        type: Type.NUMBER,
        description: 'Estimated number of people involved or in danger.'
      },
      criticalNeeds: {
        type: Type.STRING,
        description: 'Specific assistance needed (e.g., Boat, Medical, Water, Evacuation).'
      },
    },
    required: ['emergencyType', 'description']
  }
};

export class GeminiLiveService {
  private ai: GoogleGenAI;
  private inputAudioContext: AudioContext | null = null;
  private outputAudioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private nextStartTime = 0;
  private sessionPromise: Promise<any> | null = null;
  private currentSession: any = null; // Stored reference to session for closing
  
  public onReportSubmitted: ((report: any) => void) | null = null;
  public onAudioLevel: ((level: number) => void) | null = null;
  public onStatusChange: ((status: string) => void) | null = null;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: API_KEY });
  }

  async connect() {
    this.onStatusChange?.('Initializing Audio...');
    
    this.inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    this.outputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    
    // Resume audio contexts if suspended (browser policy)
    if (this.inputAudioContext.state === 'suspended') await this.inputAudioContext.resume();
    if (this.outputAudioContext.state === 'suspended') await this.outputAudioContext.resume();

    this.onStatusChange?.('Requesting Mic...');
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    this.onStatusChange?.('Connecting to HQ...');
    
    this.sessionPromise = this.ai.live.connect({
      model: MODEL_NAME,
      callbacks: {
        onopen: this.handleOpen.bind(this),
        onmessage: this.handleMessage.bind(this),
        onclose: () => {
             this.onStatusChange?.('Disconnected');
             this.stop();
        },
        onerror: (err) => {
            console.error('Live API Error:', err);
            this.onStatusChange?.('Connection Error');
        },
      },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
        },
        systemInstruction: `You are an emergency response dispatcher for "Disaster Connect". 
        Your goal is to calmly collect information from users in distress. 
        Keep responses concise and reassuring. 
        You MUST ask for: 
        1. The type of emergency.
        2. Immediate needs.
        3. Number of people.
        Once you have this info, IMMEDIATELY call the 'reportEmergency' function to log the ticket.
        Do not ask for latitude/longitude, the app handles that automatically.
        After reporting, tell the user help is being coordinated.`,
        tools: [{ functionDeclarations: [reportEmergencyTool] }]
      }
    });

    // We wait for the session to be established before allowing input streaming
    this.currentSession = await this.sessionPromise;
  }

  private handleOpen() {
    this.onStatusChange?.('Live');
    if (!this.inputAudioContext || !this.stream) return;

    this.source = this.inputAudioContext.createMediaStreamSource(this.stream);
    this.processor = this.inputAudioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Calculate volume for visualizer
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      this.onAudioLevel?.(rms);

      // Convert to PCM and send
      const pcmData = float32ToInt16PCM(inputData);
      const uint8Params = new Uint8Array(pcmData.buffer);
      const base64Data = arrayBufferToBase64(uint8Params.buffer);

      if (this.currentSession) {
        this.currentSession.sendRealtimeInput({
          media: {
            mimeType: 'audio/pcm;rate=16000',
            data: base64Data
          }
        });
      }
    };

    this.source.connect(this.processor);
    this.processor.connect(this.inputAudioContext.destination);
  }

  private async handleMessage(message: LiveServerMessage) {
    // Handle Audio Output
    const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
    if (base64Audio && this.outputAudioContext) {
      const audioData = base64ToUint8Array(base64Audio);
      this.nextStartTime = Math.max(this.nextStartTime, this.outputAudioContext.currentTime);
      
      try {
        const buffer = await decodeAudioData(audioData, this.outputAudioContext);
        const source = this.outputAudioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.outputAudioContext.destination);
        source.start(this.nextStartTime);
        this.nextStartTime += buffer.duration;
      } catch (e) {
        console.error("Error decoding audio", e);
      }
    }

    // Handle Tool Calls
    if (message.toolCall) {
      const responses = [];
      for (const fc of message.toolCall.functionCalls) {
        if (fc.name === 'reportEmergency') {
          // Trigger the frontend callback
          this.onReportSubmitted?.(fc.args);
          responses.push({
            id: fc.id,
            name: fc.name,
            response: { result: { status: 'success', ticketId: Date.now().toString() } }
          });
        }
      }
      
      if (responses.length > 0 && this.currentSession) {
         this.currentSession.sendToolResponse({
           functionResponses: responses
         });
      }
    }
  }

  stop() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    if (this.source) {
      this.source.disconnect();
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    if (this.inputAudioContext) {
      this.inputAudioContext.close();
    }
    if (this.outputAudioContext) {
      this.outputAudioContext.close();
    }
    
    this.processor = null;
    this.source = null;
    this.stream = null;
    this.inputAudioContext = null;
    this.outputAudioContext = null;
    this.currentSession = null;
    this.onStatusChange?.('Disconnected');
  }
}