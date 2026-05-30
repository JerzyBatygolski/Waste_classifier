terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Artifact Registry - Docker image repository
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "waste_classifier" {
  repository_id = "waste-classifier"
  location      = var.region
  format        = "DOCKER"
  description   = "Waste classifier Docker images"
}

# ---------------------------------------------------------------------------
# Cloud Storage - bucket for uploaded user images
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "user_images" {
  name          = "${var.project_id}-waste-images"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# ---------------------------------------------------------------------------
# Service Account for Cloud Run
# ---------------------------------------------------------------------------

resource "google_service_account" "cloud_run_sa" {
  account_id   = "waste-classifier-sa"
  display_name = "Waste Classifier Cloud Run Service Account"
}

resource "google_project_iam_member" "storage_object_creator" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "storage_object_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run - API service
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "waste_classifier_api" {
  name     = "waste-classifier-api"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/waste-classifier/waste-classifier-api:${var.image_tag}"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "MODEL_PATH"
        value = "./model/model_best.keras"
      }
      env {
        name  = "CLASS_INDICES_PATH"
        value = "./model/class_indices.json"
      }
      env {
        name  = "MAX_FILE_SIZE_MB"
        value = "10"
      }
      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.user_images.name
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_artifact_registry_repository.waste_classifier]
}

# ---------------------------------------------------------------------------
# Make Cloud Run service publicly accessible (no auth required)
# Remove this block if you want to require authentication
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = google_cloud_run_v2_service.waste_classifier_api.project
  location = google_cloud_run_v2_service.waste_classifier_api.location
  name     = google_cloud_run_v2_service.waste_classifier_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Cloud Run - Streamlit Application
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "streamlit" {
  name     = "waste-classifier-ui"
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/waste-classifier/waste-classifier-ui:latest"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }

      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.waste_classifier_api.uri
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_v2_service_iam_member" "streamlit_public" {
  project  = google_cloud_run_v2_service.streamlit.project
  location = google_cloud_run_v2_service.streamlit.location
  name     = google_cloud_run_v2_service.streamlit.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "streamlit_url" {
  description = "Public URL of the Streamlit UI"
  value       = google_cloud_run_v2_service.streamlit.uri
}