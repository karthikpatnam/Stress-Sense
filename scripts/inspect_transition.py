import urllib.request
import tarfile

url = 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=60)

count = 0
with tarfile.open(fileobj=resp, mode='r|gz') as tar:
    for member in tar:
        count += 1
        if count >= 36665:
            print(f"{count}: {member.name} ({member.size} bytes)", flush=True)
            if count >= 36700:
                break
        elif count % 5000 == 0:
            print(f"Skipped {count} entries...", flush=True)
