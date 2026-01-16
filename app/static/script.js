const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');
const confidenceDiv = document.getElementById('confidence');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.type.includes('touch')) {
        lastX = (e.touches[0].clientX - rect.left) * scaleX;
        lastY = (e.touches[0].clientY - rect.top) * scaleY;
    } else {
        lastX = (e.clientX - rect.left) * scaleX;
        lastY = (e.clientY - rect.top) * scaleY;
    }
}

function draw(e) {
    if (!isDrawing) return;
    
    e.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    let currentX, currentY;
    if (e.type.includes('touch')) {
        currentX = (e.touches[0].clientX - rect.left) * scaleX;
        currentY = (e.touches[0].clientY - rect.top) * scaleY;
    } else {
        currentX = (e.clientX - rect.left) * scaleX;
        currentY = (e.clientY - rect.top) * scaleY;
    }
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    lastX = currentX;
    lastY = currentY;
    
    // Auto-predict while drawing (debounced)
    clearTimeout(window.drawTimeout);
    window.drawTimeout = setTimeout(() => {
        predictDigit();
    }, 300);
}

function stopDrawing() {
    isDrawing = false;
}

// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Clear canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionDiv.textContent = '?';
    confidenceDiv.textContent = 'Draw a digit to start';
    resetProbabilities();
});

// Predict button
predictBtn.addEventListener('click', predictDigit);

function resetProbabilities() {
    document.querySelectorAll('.prob-item').forEach(item => {
        const bar = item.querySelector('.prob-bar');
        const value = item.querySelector('.prob-value');
        bar.style.width = '0%';
        value.textContent = '0%';
        item.classList.remove('highlighted');
    });
}

async function predictDigit() {
    // Get canvas as base64
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Update prediction display
        predictionDiv.textContent = data.digit;
        confidenceDiv.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        // Update probabilities
        updateProbabilities(data.probabilities, data.digit);
        
    } catch (error) {
        console.error('Error:', error);
        confidenceDiv.textContent = 'Error making prediction';
    }
}

function updateProbabilities(probabilities, predictedDigit) {
    document.querySelectorAll('.prob-item').forEach(item => {
        const digit = item.getAttribute('data-digit');
        const probability = probabilities[digit];
        const bar = item.querySelector('.prob-bar');
        const value = item.querySelector('.prob-value');
        
        const percentage = (probability * 100).toFixed(1);
        bar.style.width = `${percentage}%`;
        value.textContent = `${percentage}%`;
        
        // Highlight the predicted digit
        if (digit === predictedDigit.toString()) {
            item.classList.add('highlighted');
        } else {
            item.classList.remove('highlighted');
        }
    });
}

// Initial message
console.log('MNIST Digit Classifier loaded. Draw a digit to start!');
