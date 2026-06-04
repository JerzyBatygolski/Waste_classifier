variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region for all resources"
  type        = string
  default     = "europe-central2"
}

variable "api_image_tag" {
  description = "Docker image tag to deploy the API to Cloud Run"
  type        = string
}

variable "app_image_tag" {
  description = "Docker image tag to deploy the Application to Cloud Run"
  type        = string
}
