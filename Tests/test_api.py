"""
test_api.py
===========
Local tests for the waste classifier API.
Run this AFTER starting the API with: uvicorn main:app --port 8080

Usage:
  python test_api.py
  python test_api.py --url http://localhost:8080
  python test_api.py --image ./my_photo.jpg

Requirements:
  pip install requests Pillow
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8080"


def separator(label=""):
    print("\n" + "-" * 52)
    if label:
        print("  " + label)
    print("-" * 52)


def test_health(base_url):
    separator("GET /health")
    r = requests.get(base_url + "/health", timeout=5)
    print("  Status:  " + str(r.status_code))
    print("  Body:    " + json.dumps(r.json(), indent=4))
    assert r.status_code == 200, "Health check failed"
    assert r.json()["ready"] is True, "Model not ready"
    print("  PASS")


def test_classes(base_url):
    separator("GET /classes")
    r = requests.get(base_url + "/classes", timeout=5)
    print("  Status:  " + str(r.status_code))
    data = r.json()
    print("  Classes (" + str(data["num_classes"]) + "):")
    for cls in data["classes"]:
        print("    - " + cls)
    assert r.status_code == 200
    print("  PASS")


def test_predict(base_url, image_path):
    separator("POST /predict  ->  " + str(image_path))

    with open(image_path, "rb") as f:
        content = f.read()

    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp":  "image/bmp",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/jpeg")

    t0 = time.perf_counter()
    r = requests.post(
        base_url + "/predict",
        files={"file": (Path(image_path).name, content, mime)},
        timeout=30,
    )
    round_trip_ms = round((time.perf_counter() - t0) * 1000, 1)

    print("  HTTP status:    " + str(r.status_code))
    assert r.status_code == 200, "Predict failed: " + r.text

    data = r.json()
    print("  Predicted:      " + data["class"])
    print("  Confidence:     " + str(round(data["confidence"] * 100, 2)) + "%")
    print("  Model latency:  " + str(data["latency_ms"]) + " ms")
    print("  Round-trip:     " + str(round_trip_ms) + " ms")
    print("")
    print("  All scores (top 5):")
    for cls, score in list(data["all_scores"].items())[:5]:
        bar = "#" * int(score * 40)
        print("    " + cls.ljust(16) + " " + str(round(score * 100, 1)).rjust(5) + "%  " + bar)
    print("  PASS")
    return data


def test_invalid_file(base_url):
    separator("POST /predict  ->  invalid file (expect 415 or 422)")
    r = requests.post(
        base_url + "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    print("  Status:  " + str(r.status_code) + "  (expected 415 or 422)")
    assert r.status_code in (415, 422), "Expected error status, got: " + str(r.status_code)
    print("  PASS")


def test_empty_file(base_url):
    separator("POST /predict  ->  empty file (expect 400)")
    r = requests.post(
        base_url + "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        timeout=10,
    )
    print("  Status:  " + str(r.status_code) + "  (expected 400)")
    assert r.status_code == 400, "Expected 400, got: " + str(r.status_code)
    print("  PASS")


def create_test_image(path):
    """Create a small solid-color test image if no real image is provided."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (320, 240), color=(180, 200, 160))
    img.save(path, "JPEG")
    print("  Created test image: " + str(path))


def main():
    parser = argparse.ArgumentParser(description="Test the waste classifier API")
    parser.add_argument("--url",   default=BASE_URL, help="API base URL")
    parser.add_argument("--image", default=None,     help="Path to a real waste image for /predict")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("\n" + "=" * 52)
    print("  Waste Classifier API - local tests")
    print("  Target: " + base_url)
    print("=" * 52)

    # Prepare test image
    if args.image and Path(args.image).exists():
        image_path = args.image
    else:
        image_path = "/tmp/test_waste.jpg"
        create_test_image(image_path)
        if args.image:
            print("  [WARNING] Image not found: " + args.image + ". Using generated image.")

    passed = 0
    failed = 0

    tests = [
        ("health check",   lambda: test_health(base_url)),
        ("classes list",   lambda: test_classes(base_url)),
        ("predict image",  lambda: test_predict(base_url, image_path)),
        ("invalid file",   lambda: test_invalid_file(base_url)),
        ("empty file",     lambda: test_empty_file(base_url)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print("  FAIL: " + str(e))
            failed += 1
        except Exception as e:
            print("  ERROR: " + str(e))
            failed += 1

    separator("Results")
    print("  Passed: " + str(passed) + " / " + str(passed + failed))
    if failed:
        print("  Failed: " + str(failed))
        sys.exit(1)
    else:
        print("  All tests passed.")
    print("")


if __name__ == "__main__":
    main()
