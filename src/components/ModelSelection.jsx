import React from 'react';

// ModelSelection component for selecting AI models
const ModelSelection = ({ models, selectedModel, onSelectModel, isOfflineMode, apiAvailable }) => {
  
  // Format model name for display
  const getDisplayInfo = (model) => {
    if (model === 'sdxl_base') {
      return { name: 'SDXL Base', description: 'High quality base model' };
    } else if (model === 'sdxl_base+refiner') {
      return { name: 'SDXL + Refiner', description: 'Best quality with refinement' };
    } else if (model === 'sd_v1_4') {
      return { name: 'SD v1.4', description: 'Classic model, faster' };
    } else if (model === 'sd_v1_5') {
      return { name: 'SD v1.5', description: 'Improved classic model' };
    } else if (model === 'offline_mode') {
      return { name: 'Offline Mode', description: 'No processing, camera only' };
    }
    
    return { name: model, description: 'Model' };
  };

  // Ensure we always have at least offline mode
  const displayModels = models && models.length > 0 
    ? models 
    : ['offline_mode'];

  // If models don't include offline_mode, add it
  const allModels = displayModels.includes('offline_mode') 
    ? displayModels 
    : [...displayModels, 'offline_mode'];

  console.log('ModelSelection render:', { allModels, selectedModel, isOfflineMode, apiAvailable });

  return (
    <div className="mb-4">
      <label className="block text-sm font-medium mb-2">Model</label>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
        {allModels.map(model => {
          const { name, description } = getDisplayInfo(model);
          // Simple logic: if offline mode is on OR API is not available, disable all models except offline_mode
          const isDisabled = (isOfflineMode || !apiAvailable) && model !== 'offline_mode';
          const isSelected = model === selectedModel;
          
          return (
            <div 
              key={model}
              onClick={() => {
                if (!isDisabled) {
                  onSelectModel(model);
                  console.log('Model selected:', model);
                  // If selecting offline mode, we should handle the mode switch
                  if (model === 'offline_mode' && !isOfflineMode) {
                    // This will be handled by the parent component
                  }
                }
              }}
              className={`
                border rounded-md p-3 cursor-pointer transition-all duration-200
                ${isDisabled ? 'opacity-60 cursor-not-allowed' : 'hover:border-blue-500 hover:bg-gray-50'}
                ${isSelected ? 'border-green-600 bg-green-50' : 'border-gray-200'}
              `}
            >
              <h6 className="font-medium mb-1 text-sm">{name}</h6>
              <div className="text-xs text-gray-500">{description}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelSelection;