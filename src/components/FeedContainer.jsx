import React from 'react';

// FeedContainer component for displaying video feed and generated image
const FeedContainer = ({ 
  videoRef, 
  generatedImage, 
  progressInfo, 
  isGenerating 
}) => {
  return (
    <div className="flex flex-wrap justify-center gap-5 mb-5 md:flex-row">
      {/* Camera Feed */}
      <div className="video-feed">
        <span className="feed-label">Camera</span>
        <video ref={videoRef} autoPlay playsInline></video>
      </div>

      {/* Generated Image */}
      <div className="generated-image">
        <span className="feed-label">Generated</span>
        <img 
          src={generatedImage || "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="} 
          alt="Generated texture" 
        />
        
        {/* Progress indicator */}
        {isGenerating && progressInfo && (
          <div className="progress-container">
            <div id="progressText">{progressInfo.message || `Step ${progressInfo.step}/${progressInfo.total} - ${Math.round(progressInfo.percentage)}%`}</div>
            <div 
              className="progress-bar" 
              style={{ width: `${progressInfo.percentage || 0}%` }}
            ></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedContainer;