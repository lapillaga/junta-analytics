// Main JavaScript file for Junta Analytics

// Toast notification system
class Toast {
  constructor(options = {}) {
    this.duration = options.duration || 3000;
    this.position = options.position || 'top-right';
    
    // Create toast container if it doesn't exist
    if (!document.querySelector('.toast-container')) {
      const container = document.createElement('div');
      container.className = `toast-container ${this.position}`;
      document.body.appendChild(container);
    }
  }
  
  show(message, type = 'info') {
    const container = document.querySelector('.toast-container');
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <span>${message}</span>
        <button class="toast-close">&times;</button>
      </div>
    `;
    
    // Add toast to container
    container.appendChild(toast);
    
    // Add show class after a small delay to trigger animation
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Add close event listener
    const closeButton = toast.querySelector('.toast-close');
    closeButton.addEventListener('click', () => this.hide(toast));
    
    // Auto-hide after duration
    if (this.duration > 0) {
      setTimeout(() => this.hide(toast), this.duration);
    }
    
    return toast;
  }
  
  hide(toast) {
    toast.classList.remove('show');
    
    // Remove from DOM after animation
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 300);
  }
}

// Create global toast instance
const toast = new Toast();

// Add utility functions
function formatNumber(number, decimals = 2) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(number);
}

function formatDate(date) {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(new Date(date));
}

// Data fetching utility
async function fetchJSON(url, options = {}) {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Fetch error:', error);
    toast.show(`Error fetching data: ${error.message}`, 'error');
    return null;
  }
}

// Initialize tooltips, popovers, etc. when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Add global click event for all buttons with confirmation
  document.addEventListener('click', function(event) {
    const target = event.target.closest('[data-confirm]');
    
    if (target) {
      event.preventDefault();
      
      const message = target.getAttribute('data-confirm');
      
      if (confirm(message)) {
        if (target.tagName === 'A') {
          window.location.href = target.getAttribute('href');
        } else if (target.form) {
          target.form.submit();
        }
      }
    }
  });
  
  // Add form validation
  document.querySelectorAll('form[data-validate]').forEach(form => {
    form.addEventListener('submit', function(event) {
      const requiredFields = form.querySelectorAll('[required]');
      let isValid = true;
      
      requiredFields.forEach(field => {
        if (!field.value.trim()) {
          isValid = false;
          field.classList.add('border-red-500');
          
          // Add error message
          const errorId = `error-${field.id || field.name}`;
          if (!document.getElementById(errorId)) {
            const errorMessage = document.createElement('p');
            errorMessage.id = errorId;
            errorMessage.className = 'text-red-500 text-sm mt-1';
            errorMessage.textContent = field.getAttribute('data-error') || 'This field is required';
            field.parentNode.appendChild(errorMessage);
          }
        } else {
          field.classList.remove('border-red-500');
          const errorId = `error-${field.id || field.name}`;
          const errorElement = document.getElementById(errorId);
          if (errorElement) {
            errorElement.remove();
          }
        }
      });
      
      if (!isValid) {
        event.preventDefault();
      }
    });
  });
});

// Export utility functions and classes for use in other scripts
window.JuntaAnalytics = {
  toast,
  formatNumber,
  formatDate,
  fetchJSON
};