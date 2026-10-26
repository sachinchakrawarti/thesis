window.CTI_CONFIG = {
  API_BASE: (window.location.protocol === "file:" || window.location.port !== "8000")
    ? "http://127.0.0.1:8000"
    : "",
  ENDPOINTS: {
    health:   "/api/health",
    summary:  "/api/summary",
    labels:   "/api/label-distribution",
    models:   "/api/model-results",
    docs:     "/api/classified-docs",
    alerts:   "/api/emerging-alerts",
    shares:   "/api/monthly-shares",
    predict:  "/api/predict",
    classify: "/api/classify-upload"
  }
};
