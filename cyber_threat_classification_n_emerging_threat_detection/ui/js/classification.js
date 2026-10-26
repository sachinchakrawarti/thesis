const input = document.getElementById("pdfInput");
const btn = document.getElementById("classifyBtn");
const out = document.getElementById("result");

btn.addEventListener("click", async () => {
  const file = input.files[0];
  if (!file) return alert("Choose a PDF first.");
  out.textContent = "Analyzing...";
  try {
    const data = await window.postFile("/classify", file);
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
});
