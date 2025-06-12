import React from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import ModelSelection from './ModelSelection';

// ControlPanel component containing all the controls
const ControlPanel = ({
  // Camera controls
  cameraDevices,
  selectedCamera,
  onCameraChange,
  
  // Connection controls
  apiUrl,
  onApiUrlChange,
  onConnect,
  connectionStatus,
  
  // Model selection
  models,
  selectedModel,
  onSelectModel,
  isOfflineMode,
  apiAvailable,
  
  // Prompt controls
  prompt,
  onPromptChange,
  negativePrompt,
  onNegativePromptChange,
  onSuggestPrompts,
  promptSuggestions,
  onSelectSuggestion,
  
  // Parameter sliders
  steps,
  onStepsChange,
  guidanceScale,
  onGuidanceChange,
  strength,
  onStrengthChange,
  
  // Action buttons
  isRunning,
  onToggleRunning,
  onCaptureOnce,
  onToggleOfflineMode,
  
  // Interval selection
  requestInterval,
  onIntervalChange,
  
  // Server info
  serverInfo
}) => {
  return (
    <div className="bg-white rounded-lg p-5 shadow-md mb-5">
      {/* Camera and Connection Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <Label htmlFor="cameraSelect">Camera</Label>
          <Select value={selectedCamera} onValueChange={onCameraChange}>
            <SelectTrigger id="cameraSelect">
              <SelectValue placeholder="Select a camera" />
            </SelectTrigger>
            <SelectContent>
              {cameraDevices.map(device => (
                <SelectItem key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId.slice(0, 8)}...`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <Label>Connection</Label>
          <div className="flex gap-2 items-center">
            <div className="flex items-center">
              <span className="text-sm mr-2">API URL</span>
            </div>
            <Input 
              id="apiUrlInput"
              placeholder="API URL" 
              value={apiUrl} 
              onChange={e => onApiUrlChange(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && onConnect()}
              className="flex-1"
            />
            <Button onClick={onConnect} variant="outline" className="whitespace-nowrap">
              Connect
            </Button>
            <div className="flex items-center ml-2">
              <div className={`w-2.5 h-2.5 rounded-full mr-1.5 ${
                connectionStatus === 'connected' ? 'bg-green-600' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                'bg-red-600'
              }`}></div>
              <span className="text-xs">
                {connectionStatus === 'connected' ? 'Connected' : 
                 connectionStatus === 'connecting' ? 'Connecting...' : 
                 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Model Selection */}
      <ModelSelection 
        models={models}
        selectedModel={selectedModel}
        onSelectModel={onSelectModel}
        isOfflineMode={isOfflineMode}
        apiAvailable={apiAvailable}
      />
      
      {/* Prompt Inputs */}
      <div className="mb-4">
        <Label htmlFor="promptInput">Prompt</Label>
        <div className="flex gap-2 mb-1">
          <Input 
            id="promptInput"
            value={prompt} 
            onChange={e => onPromptChange(e.target.value)}
            disabled={isOfflineMode || !apiAvailable}
            placeholder="Enter a prompt describing the desired transformation"
            className="flex-1"
          />
          <Button 
            onClick={onSuggestPrompts} 
            variant="outline"
            disabled={isOfflineMode || !apiAvailable}
          >
            Suggest
          </Button>
        </div>
        
        {/* Prompt Suggestions */}
        {promptSuggestions.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {promptSuggestions.map((suggestion, index) => (
              <span 
                key={index}
                onClick={() => onSelectSuggestion(suggestion)}
                className="inline-flex bg-gray-100 text-gray-800 text-xs rounded px-2 py-1 cursor-pointer hover:bg-gray-200 transition-colors"
              >
                {suggestion.length > 50 ? suggestion.substring(0, 50) + '...' : suggestion}
              </span>
            ))}
          </div>
        )}
        
        <Label htmlFor="negativePromptInput" className="mt-3 block">Negative Prompt</Label>
        <Input 
          id="negativePromptInput"
          value={negativePrompt} 
          onChange={e => onNegativePromptChange(e.target.value)}
          disabled={isOfflineMode || !apiAvailable}
          placeholder="Elements to avoid in the generated image"
        />
      </div>
      
      {/* Parameter Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <div className="flex justify-between items-center">
            <Label htmlFor="stepsSlider">Steps</Label>
            <span className="text-sm font-medium w-10 text-right">{steps}</span>
          </div>
          <Slider
            id="stepsSlider"
            min={10}
            max={60}
            step={1}
            value={[steps]}
            onValueChange={value => onStepsChange(value[0])}
            disabled={isOfflineMode || !apiAvailable}
            className="mt-2"
          />
        </div>
        
        <div>
          <div className="flex justify-between items-center">
            <Label htmlFor="guidanceSlider">Guidance</Label>
            <span className="text-sm font-medium w-10 text-right">{guidanceScale}</span>
          </div>
          <Slider
            id="guidanceSlider"
            min={1}
            max={20}
            step={0.5}
            value={[guidanceScale]}
            onValueChange={value => onGuidanceChange(value[0])}
            disabled={isOfflineMode || !apiAvailable}
            className="mt-2"
          />
        </div>
        
        <div>
          <div className="flex justify-between items-center">
            <Label htmlFor="strengthSlider">Strength</Label>
            <span className="text-sm font-medium w-10 text-right">{strength}</span>
          </div>
          <Slider
            id="strengthSlider"
            min={0.1}
            max={1}
            step={0.05}
            value={[strength]}
            onValueChange={value => onStrengthChange(value[0])}
            disabled={isOfflineMode || !apiAvailable}
            className="mt-2"
          />
        </div>
      </div>
      
      {/* Control Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <Button 
          onClick={onToggleRunning}
          variant={isRunning ? "danger" : "success"}
          className="w-full"
        >
          {isRunning ? 'Stop' : 'Start'}
        </Button>
        
        <Button 
          onClick={onCaptureOnce} 
          variant="outline"
          disabled={isRunning}
          className="w-full"
        >
          Capture Once
        </Button>
        
        <Button 
          onClick={onToggleOfflineMode} 
          variant="outline"
          className={`w-full ${isOfflineMode ? "border-blue-500 text-blue-600" : ""}`}
        >
          {isOfflineMode ? 'Try API Mode' : 'Use Offline Mode'}
        </Button>
        
        <div className="flex items-center gap-2">
          <span className="text-sm whitespace-nowrap">Request Interval</span>
          <Select value={requestInterval.toString()} onValueChange={value => onIntervalChange(parseInt(value))}>
            <SelectTrigger>
              <SelectValue placeholder="Interval" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1000">1 sec</SelectItem>
              <SelectItem value="1500">1.5 sec</SelectItem>
              <SelectItem value="2000">2 sec</SelectItem>
              <SelectItem value="3000">3 sec</SelectItem>
              <SelectItem value="5000">5 sec</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      
      {/* Server Info */}
      {serverInfo && (
        <div className="mt-3 text-xs text-gray-500 border-t pt-3">
          {serverInfo}
        </div>
      )}
    </div>
  );
};

export default ControlPanel;