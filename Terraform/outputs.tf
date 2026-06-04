output "api_url" {
  description = "Public URL of the waste classifier API"
  value       = google_cloud_run_v2_service.waste_classifier_api.uri
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL (use this to push Docker images)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/waste-classifier"
}

output "gcs_bucket_name" {
  description = "Cloud Storage bucket name for user uploaded images"
  value       = google_storage_bucket.user_images.name
}

output "service_account_email" {
  description = "Service account email used by Cloud Run"
  value       = google_service_account.cloud_run_sa.email
}
