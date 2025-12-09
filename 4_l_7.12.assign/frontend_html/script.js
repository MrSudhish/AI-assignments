// script.js
console.log("script.js loaded");

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");

// Show preview when selecting image
fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        result.innerHTML = "";
    }
});

// Handle prediction button click
predictBtn.addEventListener("click", function (ev) {
    ev.preventDefault();
    predictBrand();
});

async function predictBrand() {
    console.log("predictBrand() called");

    const file = fileInput.files[0];
    const modelName = document.getElementById("modelSelect").value;

    if (!file) {
        alert("Please choose an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelName);

    result.innerHTML = "Predicting...";

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData,
            mode: "cors"
        });

        if (!res.ok) {
            const txt = await res.text();
            throw new Error(`Server returned ${res.status}: ${txt}`);
        }

        const data = await res.json();

        if (data.error) {
            result.innerHTML = `<span style="color:red">${data.error}</span>`;
        } else {
            result.innerHTML = `
                <h3>Model Used: ${data.model_used.toUpperCase()}</h3>
                <h3>Brand: ${data.brand}</h3>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            `;
        }
    } catch (err) {
        console.error(err);
        result.innerHTML = `<span style="color:red">Error: ${err.message}</span>`;
    }
}
