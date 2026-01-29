#!/usr/bin/env python3
# ruff: noqa: T201, E501, F541
"""Comprehensive vision testing script for all OCI GenAI vision models.

This script tests vision capabilities with:
- Real images from URLs
- Local image files
- Generated test images

Models tested:
- Meta Llama 3.2 90B Vision
- Meta Llama 4 Scout
- Google Gemini 2.5 Flash
- xAI Grok 4
"""

import io
import os
import sys

import requests
from langchain_core.messages import HumanMessage
from PIL import Image

from langchain_oci import ChatOCIGenAI, encode_image

# Test configuration
COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID")
AUTH_PROFILE = os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH")
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# Vision models to test
VISION_MODELS = [
    "meta.llama-3.2-90b-vision-instruct",
    "meta.llama-4-scout-17b-16e-instruct",
    "google.gemini-2.5-flash",
    "xai.grok-4",
]


def download_image(url: str) -> bytes:
    """Download an image from URL and return as bytes."""
    print(f"  Downloading {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    return response.content


def create_gradient_image(size: tuple = (300, 200)) -> bytes:
    """Create a gradient image using PIL."""
    img = Image.new("RGB", size)
    pixels = img.load()

    for x in range(size[0]):
        for y in range(size[1]):
            # Create a horizontal gradient from blue to red
            r = int((x / size[0]) * 255)
            b = int(((size[0] - x) / size[0]) * 255)
            g = int((y / size[1]) * 128)
            pixels[x, y] = (r, g, b)  # type: ignore[index]

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_pattern_image(pattern: str = "stripes", size: tuple = (300, 200)) -> bytes:
    """Create a patterned image using PIL."""
    img = Image.new("RGB", size, color="white")
    pixels = img.load()

    if pattern == "stripes":
        # Vertical stripes
        for x in range(size[0]):
            color = "red" if (x // 30) % 2 == 0 else "blue"
            color_rgb = (255, 0, 0) if color == "red" else (0, 0, 255)
            for y in range(size[1]):
                pixels[x, y] = color_rgb  # type: ignore[index]

    elif pattern == "checkerboard":
        # Checkerboard pattern
        square_size = 40
        for x in range(size[0]):
            for y in range(size[1]):
                is_black = ((x // square_size) + (y // square_size)) % 2 == 0
                pixels[x, y] = (0, 0, 0) if is_black else (255, 255, 255)  # type: ignore[index]

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_solid_color_image(color: str = "red", size: tuple = (100, 100)) -> bytes:
    """Create a solid color image using PIL."""
    img = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def run_model_with_image(
    model_id: str, image_bytes: bytes, prompt: str, image_desc: str
) -> dict:
    """Test a vision model with an image."""
    print(f"\n  Testing {model_id} with {image_desc}...")

    try:
        llm = ChatOCIGenAI(
            model_id=model_id,
            compartment_id=COMPARTMENT_ID,
            service_endpoint=SERVICE_ENDPOINT,
            auth_profile=AUTH_PROFILE,
            auth_type=AUTH_TYPE,
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                encode_image(image_bytes, "image/png"),
            ]
        )

        response = llm.invoke([message])

        return {
            "success": True,
            "response": response.content,
            "model": model_id,
            "image": image_desc,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model_id,
            "image": image_desc,
        }


def main():
    """Run comprehensive vision tests."""
    if not COMPARTMENT_ID:
        print("Error: OCI_COMPARTMENT_ID environment variable not set")
        sys.exit(1)

    print("=" * 80)
    print("OCI GenAI Vision Models Comprehensive Test")
    print("=" * 80)

    results = []

    # Test 1: Solid color images (synthetic)
    print("\n" + "=" * 80)
    print("TEST 1: Solid Color Images (Generated)")
    print("=" * 80)

    for color in ["red", "blue", "green"]:
        print(f"\n🎨 Testing with {color} image...")
        image_bytes = create_solid_color_image(color)

        for model_id in VISION_MODELS:
            result = run_model_with_image(
                model_id,
                image_bytes,
                f"What color is this image? One word answer.",
                f"{color} square",
            )
            results.append(result)

            if result["success"]:
                print(f"  ✅ {model_id}: {result['response'][:100]}...")
            else:
                print(f"  ❌ {model_id}: {result['error'][:100]}...")

    # Test 2: Complex generated images
    print("\n" + "=" * 80)
    print("TEST 2: Complex Generated Images (Gradients & Patterns)")
    print("=" * 80)

    # Test gradient image
    print("\n🌈 Testing with gradient image...")
    gradient_bytes = create_gradient_image()

    for model_id in VISION_MODELS:
        result = run_model_with_image(
            model_id,
            gradient_bytes,
            "Describe the colors and patterns you see in this image.",
            "gradient",
        )
        results.append(result)

        if result["success"]:
            print(f"  ✅ {model_id}: {result['response'][:100]}...")
        else:
            print(f"  ❌ {model_id}: {result['error'][:100]}...")

    # Test striped pattern
    print("\n📊 Testing with striped pattern...")
    stripes_bytes = create_pattern_image("stripes")

    for model_id in VISION_MODELS:
        result = run_model_with_image(
            model_id,
            stripes_bytes,
            "What pattern do you see in this image? Describe the colors.",
            "stripes",
        )
        results.append(result)

        if result["success"]:
            print(f"  ✅ {model_id}: {result['response'][:100]}...")
        else:
            print(f"  ❌ {model_id}: {result['error'][:100]}...")

    # Test checkerboard pattern
    print("\n🏁 Testing with checkerboard pattern...")
    checker_bytes = create_pattern_image("checkerboard")

    for model_id in VISION_MODELS:
        result = run_model_with_image(
            model_id,
            checker_bytes,
            "What pattern do you see in this image?",
            "checkerboard",
        )
        results.append(result)

        if result["success"]:
            print(f"  ✅ {model_id}: {result['response'][:100]}...")
        else:
            print(f"  ❌ {model_id}: {result['error'][:100]}...")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_tests - successful

    print(f"\nTotal tests: {total_tests}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    # Group results by model
    print("\n" + "-" * 80)
    print("Results by Model:")
    print("-" * 80)

    for model_id in VISION_MODELS:
        model_results = [r for r in results if r["model"] == model_id]
        model_success = sum(1 for r in model_results if r["success"])
        model_total = len(model_results)
        success_rate = (model_success / model_total * 100) if model_total > 0 else 0

        print(f"\n{model_id}:")
        print(f"  Success rate: {model_success}/{model_total} ({success_rate:.1f}%)")

        if model_success < model_total:
            failures = [r for r in model_results if not r["success"]]
            print(f"  Failed tests:")
            for failure in failures:
                print(f"    - {failure['image']}: {failure['error'][:80]}...")

    # Save detailed results
    output_file = "vision_test_results.txt"
    with open(output_file, "w") as f:
        f.write("Detailed Vision Test Results\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Image: {result['image']}\n")

            if result["success"]:
                f.write(f"Status: ✅ SUCCESS\n")
                f.write(f"Response: {result['response']}\n")
            else:
                f.write(f"Status: ❌ FAILED\n")
                f.write(f"Error: {result['error']}\n")

            f.write("-" * 80 + "\n\n")

    print(f"\n📝 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
