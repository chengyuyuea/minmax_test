#!/usr/bin/env python3
"""Fast parallel download using range requests."""
import os
import sys
import urllib.request
import concurrent.futures
import tempfile

def get_redirect_url(url):
    """Follow redirects and return the final URL."""
    req = urllib.request.Request(url, method='HEAD')
    req.add_header('User-Agent', 'Mozilla/5.0')
    resp = urllib.request.urlopen(req)
    return resp.url

def get_file_size(url):
    """Get file size via HEAD request."""
    req = urllib.request.Request(url, method='HEAD')
    req.add_header('User-Agent', 'Mozilla/5.0')
    resp = urllib.request.urlopen(req)
    return int(resp.headers.get('Content-Length', 0))

def download_range(url, start, end, part_file):
    """Download a byte range to a file."""
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    req.add_header('Range', f'bytes={start}-{end}')
    resp = urllib.request.urlopen(req, timeout=120)
    data = resp.read()
    with open(part_file, 'wb') as f:
        f.write(data)
    return len(data)

def parallel_download(url, output_path, num_parts=8):
    """Download a file in parallel chunks."""
    print(f"Getting redirect URL...")
    final_url = get_redirect_url(url)
    print(f"Getting file size...")
    file_size = get_file_size(final_url)
    print(f"File size: {file_size / 1e6:.1f} MB")
    
    chunk_size = file_size // num_parts
    ranges = []
    for i in range(num_parts):
        start = i * chunk_size
        end = file_size - 1 if i == num_parts - 1 else (i + 1) * chunk_size - 1
        ranges.append((start, end))
    
    part_dir = tempfile.mkdtemp(dir=os.path.dirname(output_path))
    part_files = [os.path.join(part_dir, f'part_{i}') for i in range(num_parts)]
    
    print(f"Downloading in {num_parts} parallel parts...")
    downloaded = [0] * num_parts
    
    def dl_part(args):
        i, (start, end) = args
        size = download_range(final_url, start, end, part_files[i])
        print(f"  Part {i+1}/{num_parts}: {size / 1e6:.1f} MB done")
        return size
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parts) as executor:
        futures = list(executor.map(dl_part, enumerate(ranges)))
    
    total = sum(futures)
    print(f"All parts downloaded: {total / 1e6:.1f} MB")
    
    # Merge parts
    print("Merging parts...")
    with open(output_path, 'wb') as out:
        for pf in part_files:
            with open(pf, 'rb') as inp:
                while True:
                    chunk = inp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            os.remove(pf)
    os.rmdir(part_dir)
    
    final_size = os.path.getsize(output_path)
    print(f"Done! {output_path} ({final_size / 1e6:.1f} MB)")
    assert final_size == file_size, f"Size mismatch: {final_size} vs {file_size}"

if __name__ == '__main__':
    mirror = "https://hf-mirror.com"
    repo = "larryvrh/MiniMax-H3-Turbo-Lora"
    filename = sys.argv[1] if len(sys.argv) > 1 else "minimax_h3_turbo_4step.safetensors"
    url = f"{mirror}/{repo}/resolve/main/{filename}"
    output = f"/mnt/workspace/explore/turbo-lora-A/{filename}"
    
    # Kill any existing wget for the same file
    os.system(f"pkill -f 'wget.*{filename}' 2>/dev/null")
    
    parallel_download(url, output, num_parts=8)
