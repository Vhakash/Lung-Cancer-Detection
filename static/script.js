// Get references to the HTML elements
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const predictButton = document.getElementById('predict-button');
const resultsContainer = document.getElementById('results-container');
const resultsDiv = document.getElementById('results');

let uploadedFile = null;

// Listen for when a user selects a file
fileInput.addEventListener('change', (event) => {
    uploadedFile = event.target.files[0];
    if (uploadedFile) {
        // Create a URL for the selected image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(uploadedFile);
        
        // Show the preview and enable the predict button
        previewContainer.classList.remove('hidden');
        predictButton.disabled = false;
        resultsContainer.classList.add('hidden'); // Hide old results
    }
});

// Listen for when the predict button is clicked
predictButton.addEventListener('click', async () => {
    if (!uploadedFile) {
        alert("Please select an image first!");
        return;
    }

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', uploadedFile);

    // Update UI to show loading state
    predictButton.disabled = true;
    predictButton.innerText = 'Predicting...';
    resultsDiv.innerHTML = '';

    try {
        // Send the image to the Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Display the results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = `<p style="color: red;">An error occurred. Please try again.</p>`;
    } finally {
        // Restore the button's state
        predictButton.disabled = false;
        predictButton.innerText = 'Predict';
    }
});

function displayResults(data) {
    resultsContainer.classList.remove('hidden');
    
    // Sort results by probability in descending order
    const sortedResults = Object.entries(data).sort(([,a],[,b]) => b-a);
    
    for (const [className, probability] of sortedResults) {
        const resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        const classNameSpan = document.createElement('span');
        classNameSpan.classList.add('class-name');
        classNameSpan.textContent = className.replace(/_/g, ' '); // Replace underscores for readability

        const probabilitySpan = document.createElement('span');
        probabilitySpan.classList.add('probability');
        probabilitySpan.textContent = `${(probability * 100).toFixed(2)}%`;

        resultItem.appendChild(classNameSpan);
        resultItem.appendChild(probabilitySpan);
        resultsDiv.appendChild(resultItem);
    }
}
