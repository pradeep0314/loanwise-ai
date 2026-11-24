const form=document.getElementById("loanForm"), result=document.getElementById("result"), button=document.getElementById("submitBtn");
form.addEventListener("submit",async(e)=>{
 e.preventDefault(); button.disabled=true; button.innerHTML="Analyzing profile…"; result.classList.remove("hidden"); result.innerHTML="<p>Running the machine-learning model and preparing your explanation…</p>";
 const data=Object.fromEntries(new FormData(form).entries());
 try{
  const response=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  const out=await response.json(); if(!response.ok) throw new Error(out.error||"Prediction failed");
  const approved=["Y","yes","approved","1","true"].includes(String(out.prediction).toLowerCase());
  const items=(out.explanations||[]).map(x=>`<li>${x.feature}${x.impact===null?"":` — contribution ${x.impact}`}</li>`).join("");
  result.innerHTML=`<span class="step">02 / MODEL RESULT</span><h3>${approved?"Prediction: Likely eligible":"Prediction: Not likely eligible"}</h3><div class="prob">${(out.probability*100).toFixed(1)}%</div><p>Model probability for the positive class (${out.positive_label}). Predicted class: <b>${out.prediction}</b>.</p><p><b>Top explanation features</b></p><ul>${items||"<li>Explanation unavailable</li>"}</ul><p><b>Test-set accuracy:</b> ${(out.accuracy*100).toFixed(1)}%</p><p><b>Important:</b> This is an academic model estimate, not a bank approval or a reliable credit decision. Human review and legally compliant validation are essential.</p>`;
 }catch(err){result.innerHTML=`<h3>Unable to generate prediction</h3><p>${err.message}</p>`}
 finally{button.disabled=false;button.innerHTML='Analyze my eligibility <span>→</span>'}
});
