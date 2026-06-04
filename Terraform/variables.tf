variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region for all resources"
  type        = string
  default     = "europe-central2"
}

variable "image_tag" {
  description = "Docker image tag to deploy to Cloud Run"
  type        = string
  default     = "latest"
}
