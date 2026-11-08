// Where the frontend sends API requests.
// After deploying the backend, replace the placeholder below with its URL
// (no trailing slash), e.g. "https://adult-income-api.vercel.app".
const DEPLOYED_BACKEND_URL = "https://adult-income-classifier-fgki.vercel.app";

// Running locally (localhost / 127.0.0.1 / opened as a file) -> local backend.
const isLocal = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);
window.BACKEND_URL = isLocal ? "http://localhost:8000" : DEPLOYED_BACKEND_URL;
