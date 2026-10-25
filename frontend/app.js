const form = document.querySelector("#property-form");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const resultCard = document.querySelector("#result");

async function loadChoices() {
  const response = await fetch("/api/options");
  if (!response.ok) {
    throw new Error("Property options could not be loaded. Please refresh the page.");
  }
  const { choices } = await response.json();

  for (const [name, values] of Object.entries(choices)) {
    const select = form.elements.namedItem(name);
    if (!select) continue;
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value.replaceAll("_", " ");
      select.append(option);
    }
  }

  form.elements.namedItem("State").value = "Karnataka";
  form.elements.namedItem("City").value = "Bangalore";
  form.elements.namedItem("Property_Type").value = "Apartment";
  form.elements.namedItem("Furnished_Status").value = "Semi-furnished";
  form.elements.namedItem("Facing").value = "East";
  form.elements.namedItem("Owner_Type").value = "Broker";
  form.elements.namedItem("Public_Transport_Accessibility").value = "Medium";
  form.elements.namedItem("Parking_Space").value = "Yes";
  form.elements.namedItem("Security").value = "Yes";
  form.elements.namedItem("Availability_Status").value = "Ready_to_Move";
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function displayResult(result) {
  const isAboveMedian = result.predicted_class === 1;
  document.querySelector("#result-title").textContent = result.label;
  document.querySelector("#result-description").textContent = isAboveMedian
    ? "Based on the details provided, the model places this property on the higher side of its training data's price midpoint."
    : "Based on the details provided, the model places this property on the lower side of its training data's price midpoint.";
  document.querySelector("#result-confidence").textContent =
    `${Math.round(result.confidence * 100)}%`;
  document.querySelector("#probability-marker").style.left =
    `${Math.min(100, Math.max(0, result.above_median_probability * 100))}%`;
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  resultCard.hidden = true;

  if (!form.reportValidity()) return;

  const record = Object.fromEntries(new FormData(form).entries());
  for (const [key, value] of Object.entries(record)) {
    const field = form.elements.namedItem(key);
    if (field instanceof HTMLInputElement && field.type === "number") {
      record[key] = Number(value);
    }
  }

  submitButton.disabled = true;
  submitButton.querySelector(".button-label").textContent = "Reading the market…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The prediction could not be completed.");
    displayResult(payload);
  } catch (error) {
    showError(error.message || "Something went wrong. Please try again.");
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector(".button-label").textContent = "See the market view";
  }
});

loadChoices().catch((error) => {
  showError(error.message);
  submitButton.disabled = true;
});
