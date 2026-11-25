const form = document.getElementById("loanForm");
const result = document.getElementById("result");
const button = document.getElementById("submitBtn");

// Only these eight columns exist in the current synthetic training dataset.
// Personal identifiers and the additional history fields are deliberately excluded.
const modelFields = [
  "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
  "Loan_Amount_Term", "Credit_History", "Education",
  "Self_Employed", "Property_Area"
];

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  button.disabled = true;
  button.textContent = "Analyzing profile…";
  result.classList.remove("hidden");
  result.innerHTML = "<p>Running the model and preparing your explanation…</p>";

  const formData = new FormData(form);
  const data = {};
  for (const key of modelFields) data[key] = formData.get(key);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });
    const output = await response.json();
    if (!response.ok) throw new Error(output.error || "Prediction failed");

    const approved = ["y", "yes", "approved", "1", "true"]
      .includes(String(output.prediction).toLowerCase());
    const items = (output.explanations || [])
      .map(item => `<li>${escapeHtml(item.feature)}${item.impact === null ? "" : ` — contribution ${item.impact}`}</li>`)
      .join("");

    result.innerHTML = `
      <span class="step">02 / MODEL RESULT</span>
      <h3>${approved ? "Prediction: Likely eligible" : "Prediction: Not likely eligible"}</h3>
      <div class="prob">${(output.probability * 100).toFixed(1)}%</div>
      <p>Model probability for the positive class (${escapeHtml(output.positive_label)}). Predicted class: <b>${escapeHtml(output.prediction)}</b>.</p>
      <p><b>Top explanation features</b></p><ul>${items || "<li>Explanation unavailable</li>"}</ul>
      <p><b>Test-set accuracy:</b> ${(output.accuracy * 100).toFixed(1)}%</p>
      <p><b>Model scope:</b> The current model uses only the 8 training-dataset fields listed in the application. Past loan, repayment, credit score, and active-loan details are not yet used by the model.</p>
      <p><b>Important:</b> This is an academic estimate from synthetic data, not a bank approval or a reliable credit decision. Human review and proper validation are essential.</p>`;
  } catch (error) {
    result.innerHTML = `<h3>Unable to generate prediction</h3><p>${escapeHtml(error.message)}</p>`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze my eligibility <span>→</span>';
  }
});

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}
