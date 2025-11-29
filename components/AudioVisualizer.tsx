import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
  isActive: boolean;
  level: number; // 0 to 1
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ isActive, level }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    
    // Smooth out the level
    let currentLevel = 0;

    const draw = () => {
      if (!isActive) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      // Approach target level smoothly
      currentLevel += (level - currentLevel) * 0.2;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const maxRadius = Math.min(centerX, centerY) * 0.8;
      
      // Draw base circle
      ctx.beginPath();
      ctx.arc(centerX, centerY, 30 + currentLevel * 50, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(239, 68, 68, ${0.2 + currentLevel * 0.5})`; // Red-500
      ctx.fill();

      // Draw ripple
      ctx.beginPath();
      ctx.arc(centerX, centerY, 30 + currentLevel * maxRadius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.5 - currentLevel * 0.3})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isActive, level]);

  return (
    <canvas 
      ref={canvasRef} 
      width={200} 
      height={200} 
      className={`w-full h-full pointer-events-none transition-opacity duration-500 ${isActive ? 'opacity-100' : 'opacity-0'}`}
    />
  );
};
