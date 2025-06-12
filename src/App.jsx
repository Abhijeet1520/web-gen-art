import React, { useState, useRef, useEffect, useCallback } from 'react';
import FeedContainer from './components/FeedContainer';
import ControlPanel from './components/ControlPanel';
import ToastContainer from './components/ToastContainer';

// NGROK header for bypassing browser warnings
const NGROK_HEADER = { 'ngrok-skip-browser-warning': '69420' };

// Texture prompts for suggestions (exactly matching HTML)
const TEXTURE_PROMPTS = [
  "transform this space into a medieval stone castle with torch-lit walls, wooden beams, and weathered stone textures, highly detailed",
  "convert this scene to a tropical jungle environment with lush vegetation overtaking existing structures, vines climbing walls, moss on surfaces",
  "change current environment to futuristic cyberpunk with neon-lit edges, holographic displays replacing existing frames, metallic surfaces",
  "transform this into a winter wonderland with snow-covered surfaces, icicles hanging from edges, frosted windows, soft blue lighting",
  "convert this area to an underwater scene with coral reefs growing on structures, seaweed replacing vertical elements, bubbles floating upward",
  "change this setting to an ancient temple with moss-covered stone walls, hieroglyphics carved into surfaces, golden artifacts scattered about",
  "transform current scene into a steampunk workshop with brass gears on walls, copper pipes replacing fixtures, vintage machinery integrated",
  "convert this space to a fantasy crystal cave with glowing gems embedded in walls, crystalline formations replacing furniture, ethereal lighting",
  "transform this into a post-apocalyptic abandoned version with plant overgrowth, peeling paint, structural decay, dust particles in light beams",
  "change this environment to a luxury gold and marble palace with ornate decorations, gilded surfaces, polished marble floors, classical columns"
];

function App() {
  // App state - matching HTML exactly
  const [isRunning, setIsRunning] = useState(false);
  const [isOfflineMode, setIsOfflineMode] = useState(true);
  const [apiAvailable, setApiAvailable] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [generatedImage, setGeneratedImage] = useState('');
  const [models, setModels] = useState(['offline_mode']);
  const [selectedModel, setSelectedModel] = useState('offline_mode');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  // Updated default values to match HTML exactly
  const [prompt, setPrompt] = useState('transform this space into a medieval stone castle with torch-lit walls, wooden beams, and weathered stone textures, highly detailed');
  const [negativePrompt, setNegativePrompt] = useState('blurry, watermark, text, low quality');
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(15); // Matching HTML
  const [strength, setStrength] = useState(0.4); // Matching HTML
  const [requestInterval, setRequestInterval] = useState(1500);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progressInfo, setProgressInfo] = useState(null);
  const [serverInfo, setServerInfo] = useState('Server information will appear here when connected...');
  const [promptSuggestions, setPromptSuggestions] = useState([]);
  const [toasts, setToasts] = useState([]);
  
  // References
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement('canvas'));
  const cameraStreamRef = useRef(null);
  const processingIntervalRef = useRef(null);
  const apiCheckIntervalRef = useRef(null);
  const websocketRef = useRef(null);
  const websocketReconnectTimerRef = useRef(null);
  const lastRequestTimeRef = useRef(0);
  const lastApiErrorTimeRef = useRef(0);
  
  // Toast notification system
  const showToast = useCallback((type, title, message) => {
    const id = Date.now();
    const newToast = { id, type, title, message };
    
    setToasts(prevToasts => {
      const updatedToasts = [newToast, ...prevToasts];
      // Limit maximum number of toasts to prevent overflow
      return updatedToasts.slice(0, 5);
    });
    
    // Auto remove after 5 seconds
    setTimeout(() => {
      setToasts(prevToasts => prevToasts.filter(toast => toast.id !== id));
    }, 5000);
    
    return id;
  }, []);
  
  const removeToast = useCallback((id) => {
    setToasts(prevToasts => prevToasts.filter(toast => toast.id !== id));
  }, []);
  
  // Update API status and ensure UI controls are properly enabled/disabled
  const updateApiStatus = useCallback((status, suppressToast = false) => {
    setApiAvailable(status);
    
    // Show toast notifications for connection status changes
    if (!suppressToast && !isOfflineMode) {
      if (status) {
        showToast('success', 'API Connection', 'Successfully connected to the server API');
      } else {
        // Only show error if we haven't shown one recently (avoid spam)
        const now = Date.now();
        if (now - lastApiErrorTimeRef.current > 30000) { // 30 seconds cooldown
          showToast('error', 'API Connection', 'Failed to connect to the server API. Using offline mode.');
          lastApiErrorTimeRef.current = now;
        }
      }
    }
  }, [isOfflineMode, showToast]);
  
  // Setup WebSocket connection with better error handling
  const setupWebSocket = useCallback(() => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    
    const wsUrl = apiUrl.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
    
    try {
      setConnectionStatus('connecting');
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        console.log('WebSocket connected');
        clearTimeout(websocketReconnectTimerRef.current);
        setConnectionStatus('connected');
        showToast('success', 'WebSocket', 'Connected to server websocket');
      };
      
      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'progress') {
            setProgressInfo(data);
            if (data.status === 'starting') {
              setIsGenerating(true);
            } else if (data.status === 'completed') {
              setIsGenerating(false);
              // Hide progress after 2 seconds
              setTimeout(() => setProgressInfo(null), 2000);
            } else {
              setIsGenerating(true);
            }
          } else if (data.type === 'status') {
            console.log('Server status:', data);
            
            // Update server info
            let infoText = `Server Status: Connected`;
            
            if (data.models && data.models.length) {
              infoText += ` • Models: ${data.models.length} available`;
              // Update models state with available models + offline mode
              const allModels = [...data.models];
              if (!allModels.includes('offline_mode')) {
                allModels.push('offline_mode');
              }
              setModels(allModels);
            }
            
            if (data.cuda_available) {
              infoText += ` • GPU: ${data.cuda_device || 'Available'}`;
              if (data.cuda_memory_gb) {
                infoText += ` (${data.cuda_memory_gb} GB)`;
              }
            } else {
              infoText += ` • GPU: Not available - using CPU`;
            }
            
            setServerInfo(infoText);
          } else if (data.type === 'error') {
            console.error('Server error:', data.message);
            showToast('error', 'Server Error', data.message);
          } else if (data.type === 'warning') {
            console.warn('Server warning:', data.message);
            showToast('warning', 'Server Warning', data.message);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        }
      };
      
      websocketRef.current.onclose = () => {
        console.log('WebSocket closed');
        setConnectionStatus('disconnected');
        
        // Try to reconnect after a delay only if not in offline mode
        clearTimeout(websocketReconnectTimerRef.current);
        if (!isOfflineMode) {
          websocketReconnectTimerRef.current = setTimeout(() => {
            setConnectionStatus('connecting');
            setupWebSocket();
          }, 5000);
        }
      };
      
      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
        // Don't show toast for WebSocket errors when server is likely down
        // Only log to console
      };
      
      return true;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('disconnected');
      return false;
    }
  }, [apiUrl, isOfflineMode, showToast]);
  
  // Check API availability with better error handling
  const checkApiAvailability = useCallback(async (suppressToast = false) => {
    // First try the websocket connection
    setupWebSocket();
    
    // Then check the REST API
    try {
      const response = await fetch(`${apiUrl}/status`, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json',
          ...NGROK_HEADER
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        updateApiStatus(true, suppressToast);
        
        // Update server info
        let infoText = `Server Status: Connected`;
        
        if (data.models && data.models.length) {
          infoText += ` • Models: ${data.models.length} available`;
          // IMPORTANT: Update models state when API connection is successful
          const availableModels = Array.isArray(data.models) ? data.models : [];
          const allModels = [...availableModels];
          if (!allModels.includes('offline_mode')) {
            allModels.push('offline_mode');
          }
          setModels(allModels);
          console.log('Models loaded:', allModels);
        }
        
        if (data.cuda_available) {
          infoText += ` • GPU: ${data.cuda_device || 'Available'}`;
          if (data.cuda_memory_gb) {
            infoText += ` (${data.cuda_memory_gb.toFixed(1)} GB)`;
          }
        } else {
          infoText += ` • GPU: Not available - using CPU`;
        }
        
        setServerInfo(infoText);
        
        return true;
      } else {
        throw new Error('API returned error status');
      }
    } catch (error) {
      console.error('API check failed:', error);
      updateApiStatus(false, suppressToast);
      
      // Load default offline model option when API is not available
      setModels(['offline_mode']);
      
      // Update server info
      setServerInfo('Server Status: Disconnected - Using offline mode');
      
      return false;
    }
  }, [apiUrl, setupWebSocket, updateApiStatus]);
  
  // Toggle offline mode with proper control state management
  const toggleOfflineMode = useCallback(() => {
    const newOfflineMode = !isOfflineMode;
    setIsOfflineMode(newOfflineMode);
    
    if (newOfflineMode) {
      showToast('info', 'Mode Changed', 'Switched to offline mode. Camera feed will be displayed without processing.');
      setSelectedModel('offline_mode');
      // Close WebSocket when going offline
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
      setConnectionStatus('disconnected');
      // Reset to offline mode only
      setModels(['offline_mode']);
    } else {
      checkApiAvailability();
      showToast('info', 'Mode Changed', 'Trying to connect to API server...');
    }
    
    // Reset progress UI
    setProgressInfo(null);
  }, [isOfflineMode, checkApiAvailability, showToast]);
  
  // Setup camera devices
  const setupCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      setCameraDevices(videoDevices);
      
      if (videoDevices.length > 0 && !selectedCamera) {
        setSelectedCamera(videoDevices[0].deviceId);
        startCamera(videoDevices[0].deviceId);
      } else if (videoDevices.length === 0) {
        showToast('error', 'Camera Error', 'No cameras found on your device');
      }
    } catch (error) {
      console.error('Error accessing cameras:', error);
      showToast('error', 'Camera Error', 'Error accessing cameras. Please ensure you have granted camera permissions.');
    }
  }, [selectedCamera, showToast]);
  
  // Start camera with selected device
  const startCamera = useCallback(async (deviceId) => {
    // Stop any existing stream
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    try {
      const constraints = {
        video: { deviceId: deviceId ? { exact: deviceId } : undefined }
      };
      
      cameraStreamRef.current = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = cameraStreamRef.current;
      }
      showToast('success', 'Camera', 'Camera started successfully');
    } catch (error) {
      console.error('Error starting camera:', error);
      showToast('error', 'Camera Error', `Error starting camera: ${error.message}`);
    }
  }, [showToast]);
  
  // Capture image from video
  const captureImage = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth) return null;
    
    const canvas = canvasRef.current;
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);
  
  // Process a single frame
  const processFrame = useCallback(async () => {
    if (!isRunning) return;
    
    const imageDataUrl = captureImage();
    if (!imageDataUrl) {
      showToast('error', 'Camera Error', 'Failed to capture image from camera');
      return;
    }
    
    // If we're already generating or it's too soon since the last request, skip this frame
    const currentTime = Date.now();
    if (isGenerating || (currentTime - lastRequestTimeRef.current < requestInterval)) {
      return;
    }
    
    // In offline mode, just display the camera feed
    if (isOfflineMode || !apiAvailable || selectedModel === 'offline_mode') {
      setGeneratedImage(imageDataUrl);
      return;
    }
    
    try {
      lastRequestTimeRef.current = currentTime;
      
      const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...NGROK_HEADER
        },
        body: JSON.stringify({
          model: selectedModel,
          prompt: prompt,
          negative_prompt: negativePrompt,
          steps: parseInt(steps),
          guidance_scale: parseFloat(guidanceScale),
          strength: parseFloat(strength),
          width: 512,
          height: 512,
          image_b64: imageDataUrl
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      setGeneratedImage(data.image);
      
      // If we got here, API is available
      if (!apiAvailable) {
        updateApiStatus(true, true); // Suppress toast to avoid spam
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setGeneratedImage(imageDataUrl); // Fallback to showing camera feed
      // Only show error toast if this is the first error in a while
      const now = Date.now();
      if (now - lastApiErrorTimeRef.current > 30000) {
        showToast('error', 'Generation Error', 'Failed to process image');
        lastApiErrorTimeRef.current = now;
      }
      updateApiStatus(false, true); // Suppress toast to avoid spam
    }
  }, [
    isRunning, captureImage, isGenerating, requestInterval, 
    isOfflineMode, apiAvailable, selectedModel, apiUrl,
    prompt, negativePrompt, steps, guidanceScale, strength, 
    showToast, updateApiStatus
  ]);
  
  // Process a single frame once
  const processSingleFrame = useCallback(async () => {
    // Don't do anything if continuous processing is already running
    if (isRunning) {
      showToast('warning', 'Already Running', 'Please stop continuous processing first before capturing a single frame');
      return;
    }
    
    const imageDataUrl = captureImage();
    if (!imageDataUrl) {
      showToast('error', 'Camera Error', 'Failed to capture image from camera');
      return;
    }
    
    // In offline mode, just display the camera feed
    if (isOfflineMode || !apiAvailable || selectedModel === 'offline_mode') {
      setGeneratedImage(imageDataUrl);
      showToast('info', 'Offline Mode', 'Image captured and displayed (offline mode)');
      return;
    }
    
    try {
      // Show processing indicator
      setProgressInfo({
        step: 0,
        total: parseInt(steps),
        percentage: 0,
        status: "starting",
        message: "Starting single frame generation"
      });
      
      // Temporarily set isGenerating to prevent other requests
      setIsGenerating(true);
      
      const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...NGROK_HEADER
        },
        body: JSON.stringify({
          model: selectedModel,
          prompt: prompt,
          negative_prompt: negativePrompt,
          steps: parseInt(steps),
          guidance_scale: parseFloat(guidanceScale),
          strength: parseFloat(strength),
          width: 512,
          height: 512,
          image_b64: imageDataUrl
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      setGeneratedImage(data.image);
      showToast('success', 'Generation Complete', 'Single frame processed successfully');
      
      // If we got here, API is available
      if (!apiAvailable) {
        updateApiStatus(true);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setGeneratedImage(imageDataUrl); // Fallback to showing camera feed
      showToast('error', 'Generation Error', error.message);
      updateApiStatus(false);
    } finally {
      // Reset the generating flag
      setIsGenerating(false);
      
      // Reset progress UI after a delay
      setTimeout(() => {
        setProgressInfo(null);
      }, 2000);
    }
  }, [
    isRunning, captureImage, isOfflineMode, apiAvailable, 
    selectedModel, prompt, negativePrompt, steps, 
    guidanceScale, strength, apiUrl, showToast, updateApiStatus
  ]);
  
  // Get prompt suggestions
  const getSuggestions = useCallback(async () => {
    if (isOfflineMode || !apiAvailable) {
      setPromptSuggestions(TEXTURE_PROMPTS.slice(0, 5));
      return;
    }
    
    try {
      const response = await fetch(`${apiUrl}/suggest`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...NGROK_HEADER
        },
        body: JSON.stringify({
          partial_prompt: prompt
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get suggestions');
      }
      
      const data = await response.json();
      setPromptSuggestions(data.suggestions);
    } catch (error) {
      console.error('Error getting suggestions:', error);
      updateApiStatus(false, true); // Suppress toast
      setPromptSuggestions(TEXTURE_PROMPTS.slice(0, 5));
      showToast('warning', 'Suggestions', 'Failed to get suggestions from server. Using default suggestions.');
    }
  }, [isOfflineMode, apiAvailable, prompt, apiUrl, showToast, updateApiStatus]);
  
  // Toggle continuous processing
  const toggleProcessing = useCallback(() => {
    const newIsRunning = !isRunning;
    setIsRunning(newIsRunning);
    
    if (newIsRunning) {
      // Start processing frames
      processFrame(); // Process one frame immediately
      processingIntervalRef.current = setInterval(processFrame, 100); // Use a fast interval for checking, but throttle actual requests
      
      // If in API mode, periodically check if API becomes available (suppress toasts to avoid spam)
      if (!isOfflineMode && !apiAvailable && !apiCheckIntervalRef.current) {
        apiCheckIntervalRef.current = setInterval(() => checkApiAvailability(true), 10000);
      }
      
      showToast('success', 'Processing', 'Started camera processing');
    } else {
      // Stop processing
      clearInterval(processingIntervalRef.current);
      clearInterval(apiCheckIntervalRef.current);
      processingIntervalRef.current = null;
      apiCheckIntervalRef.current = null;
      
      // Hide progress
      setProgressInfo(null);
      
      showToast('info', 'Processing', 'Stopped camera processing');
    }
  }, [isRunning, processFrame, isOfflineMode, apiAvailable, checkApiAvailability, showToast]);
  
  // Initialize on component mount
  useEffect(() => {
    setupCameras();
    toggleOfflineMode(); // Set initial UI state
    checkApiAvailability(true); // Suppress initial connection toast
    
    // Clean up on unmount
    return () => {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      
      clearInterval(processingIntervalRef.current);
      clearInterval(apiCheckIntervalRef.current);
      clearTimeout(websocketReconnectTimerRef.current);
    };
  }, [setupCameras, toggleOfflineMode, checkApiAvailability]);
  
  // Ping the websocket to keep it alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Every 30 seconds
    
    return () => clearInterval(pingInterval);
  }, []);
  
  return (
    <div className="container mx-auto py-5">
      {/* Toast notification container */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold">Texture Transformer</h1>
        <p className="text-gray-600">Real-time image-to-image generation with Stable Diffusion</p>
      </div>
      
      {/* Feed container - camera + generated image */}
      <FeedContainer
        videoRef={videoRef}
        generatedImage={generatedImage}
        progressInfo={progressInfo}
        isGenerating={isGenerating}
      />
      
      {/* Control panel */}
      <ControlPanel
        // Camera controls
        cameraDevices={cameraDevices}
        selectedCamera={selectedCamera}
        onCameraChange={(deviceId) => {
          setSelectedCamera(deviceId);
          startCamera(deviceId);
        }}
        
        // Connection controls
        apiUrl={apiUrl}
        onApiUrlChange={setApiUrl}
        onConnect={() => checkApiAvailability()}
        connectionStatus={connectionStatus}
        
        // Model selection
        models={models}
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        isOfflineMode={isOfflineMode}
        apiAvailable={apiAvailable}
        
        // Prompt controls
        prompt={prompt}
        onPromptChange={setPrompt}
        negativePrompt={negativePrompt}
        onNegativePromptChange={setNegativePrompt}
        onSuggestPrompts={getSuggestions}
        promptSuggestions={promptSuggestions}
        onSelectSuggestion={(suggestion) => {
          setPrompt(suggestion);
          showToast('info', 'Prompt', `Selected prompt: "${suggestion.substring(0, 30)}..."`);
        }}
        
        // Parameter sliders
        steps={steps}
        onStepsChange={setSteps}
        guidanceScale={guidanceScale}
        onGuidanceChange={setGuidanceScale}
        strength={strength}
        onStrengthChange={setStrength}
        
        // Action buttons
        isRunning={isRunning}
        onToggleRunning={toggleProcessing}
        onCaptureOnce={processSingleFrame}
        onToggleOfflineMode={toggleOfflineMode}
        
        // Interval selection
        requestInterval={requestInterval}
        onIntervalChange={(interval) => {
          setRequestInterval(interval);
          showToast('info', 'Request Interval', `Set request interval to ${interval / 1000} sec`);
        }}
        
        // Server info
        serverInfo={serverInfo}
      />
    </div>
  );
}

export default App;