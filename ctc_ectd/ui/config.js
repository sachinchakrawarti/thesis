// Point at the FastAPI backend started by scripts/run_backend.ps1
window.CTI_CONFIG = {
  API_BASE: "http://127.0.0.1:8000",
  ENDPOINTS: {
    health:      "/api/health",
    analyze:     "/api/analyze",
    analyzeDemo: "/api/analyze-demo"
  }
};
