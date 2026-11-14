import urllib.request
import tarfile
import sys

url = 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

print("Opening stream to S3...", flush=True)
resp = urllib.request.urlopen(req, timeout=60)

seen_categories = {}
count = 0
try:
    with tarfile.open(fileobj=resp, mode='r|gz') as tar:
        for member in tar:
            count += 1
            parts = member.name.split('/')
            if len(parts) >= 3:
                # If images or labels, include subcategory
                if parts[2] in ('images', 'labels') and len(parts) >= 4:
                    cat = '/'.join(parts[:4])
                else:
                    cat = '/'.join(parts[:3])
            else:
                cat = member.name
            
            seen_categories[cat] = seen_categories.get(cat, 0) + 1
            
            if count % 10000 == 0:
                print(f"Processed {count:,} entries. Categories so far:", flush=True)
                for k, v in sorted(seen_categories.items()):
                    print(f"  {k}: {v}", flush=True)
            
            if count >= 150000:
                break
except Exception as e:
    print(f"Exception after {count} entries: {e}", flush=True)

print("Final summary:", flush=True)
for k, v in sorted(seen_categories.items()):
    print(f"  {k}: {v}", flush=True)
