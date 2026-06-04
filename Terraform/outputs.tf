output "api_url" {
  description = "Public URL of the waste classifier API"
  value       = google_cloud_run_v2_service.waste_classifier_api.uri
}

output "app_url" {
  description = "Public URL of the waste classifier Application"
  value       = google_cloud_run_v2_service.streamlit.uri
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL (use this to push Docker images)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/waste-classifier"
}

output "service_account_email" {
  description = "Service account email used by Cloud Run"
  value       = google_service_account.cloud_run_sa.email
}
