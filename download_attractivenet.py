#!/usr/bin/env python3
"""
Download AttractiveNet model from GitHub
Based on: https://github.com/gustavz/AttractiveNet

This model is trained on SCUT-FBP5500 dataset using MobileNetV2.
Output range: 1.0 - 5.0 (attractiveness score)
"""

import os
import sys
import urllib.request
import shutil

# AttractiveNet model URL (from GitHub)
MODEL_URL = "https://github.com/gustavz/AttractiveNet/raw/master/models/attractiveNet_mnv2.h5"
MODEL_NAME = "attractivenet_mnv2.h5"

def download_model(output_dir="models"):
    """Download AttractiveNet model to the specified directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, MODEL_NAME)
    
    # Check if model already exists
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ AttractiveNet model already exists: {output_path} ({size_mb:.1f} MB)")
        return output_path
    
    print(f"📥 Downloading AttractiveNet model from GitHub...")
    print(f"   URL: {MODEL_URL}")
    print(f"   Destination: {output_path}")
    
    try:
        # Download with progress
        with urllib.request.urlopen(MODEL_URL) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='', flush=True)
            
            print()  # New line after progress
        
        # Verify download
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ AttractiveNet model downloaded successfully: {output_path} ({size_mb:.1f} MB)")
            return output_path
        else:
            print(f"❌ Download failed: file not found at {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


if __name__ == "__main__":
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    result = download_model(models_dir)
    
    if result:
        print(f"\n✅ Model ready for use at: {result}")
        sys.exit(0)
    else:
        print(f"\n❌ Model download failed")
        sys.exit(1)
