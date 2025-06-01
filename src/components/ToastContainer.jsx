import React, { useEffect } from 'react';
import { ToastProvider, ToastViewport, Toast, ToastTitle, ToastDescription, ToastClose } from './ui/toast';

// ToastContainer component for displaying notifications
const ToastContainer = ({ toasts, removeToast }) => {
  // Auto remove toasts after 5 seconds
  useEffect(() => {
    const timers = toasts.map(toast => {
      return setTimeout(() => {
        removeToast(toast.id);
      }, 5000);
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [toasts, removeToast]);

  return (
    <ToastProvider>
      <ToastViewport className="flex flex-col-reverse p-4 gap-2">
        {toasts.map(toast => (
          <Toast 
            key={toast.id} 
            variant={toast.type}
            className="flex items-start"
          >
            <div className="flex-1">
              <ToastTitle>{toast.title}</ToastTitle>
              <ToastDescription>{toast.message}</ToastDescription>
            </div>
            <ToastClose onClick={() => removeToast(toast.id)} />
          </Toast>
        ))}
      </ToastViewport>
    </ToastProvider>
  );
};

export default ToastContainer;