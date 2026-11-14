import urllib.request
import tarfile
from pathlib import Path

url = 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=60)

out_base = Path('data/raw/Agriculture-Vision-2021')

# We want to extract masks, boundaries, and rgb images for the first 15 samples encountered in RGB
target_ids = set()
saved_files = 0

print("Streaming from S3...", flush=True)

# First pass: let's see which RGB images appear first
# Then extract matching masks and boundaries
# But wait, masks and boundaries are FIRST in the tarball, before RGB!
# So we can save all masks and boundaries for the first 30 IDs that appear in masks!
