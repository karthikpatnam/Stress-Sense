import urllib.request
import tarfile

class CountingStream:
    def __init__(self, raw):
        self.raw = raw
        self.bytes_read = 0
    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.bytes_read += len(chunk)
        return chunk
    def tell(self):
        return self.bytes_read

url = 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=60)
counter = CountingStream(resp)

with tarfile.open(fileobj=counter, mode='r|gz') as tar:
    for i, m in enumerate(tar):
        if i in (0, 18335, 36672, 36675, 40000, 55000, 73000):
            print(f"Index {i}: {m.name} | Compressed read: {counter.bytes_read:,} bytes ({counter.bytes_read/1024/1024:.2f} MB)", flush=True)
        if i > 75000:
            break
