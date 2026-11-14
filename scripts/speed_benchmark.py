import urllib.request
import tarfile
import time

url = 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=60)

t0 = time.time()
count = 0
last_t = t0

with tarfile.open(fileobj=resp, mode='r|gz') as tar:
    for m in tar:
        count += 1
        now = time.time()
        if now - last_t >= 5:
            last_t = now
            print(f"Elapsed: {now - t0:.1f}s | Member {count}: {m.name[:60]}", flush=True)
        if now - t0 >= 30:
            print(f"Total processed in 30s: {count} members", flush=True)
            break
