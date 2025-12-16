// app.js

let model = null;

async function loadModel() {
  if (!model) {
    console.log("Loading MobileNet model...");
    model = await mobilenet.load();
    console.log("Model loaded!");
  }
  return model;
}

async function classifyCurrentImage() {
  const imgElement = document.getElementById("main-image");
  const predictionElement = document.getElementById("prediction");

  try {
    await loadModel();
    predictionElement.textContent = "Running prediction...";

    const predictions = await model.classify(imgElement);
    console.log("Predictions:", predictions);

    const top = predictions[0];
    predictionElement.textContent = `${top.className} (Probability: ${(top.probability * 100).toFixed(2)}%)`;
  } catch (err) {
    console.error("Error classifying image:", err);
    predictionElement.textContent = "Failed to run classification.";
  }
}

// handle "Classify New Image" button
function setupUpload() {
  const button = document.getElementById("classify-btn");
  const fileInput = document.getElementById("file-input");
  const imgElement = document.getElementById("main-image");

  button.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      imgElement.src = e.target.result;
      imgElement.onload = () => {
        classifyCurrentImage();
      };
    };
    reader.readAsDataURL(file);
  });
}

// run on load
window.addEventListener("DOMContentLoaded", () => {
  setupUpload();
  classifyCurrentImage(); // classify the default new image
});
