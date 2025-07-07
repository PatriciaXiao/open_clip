import webdataset as wds
from webdataset.handlers import reraise_exception
from webdataset.tariterators import url_opener, tar_file_expander
import tarfile
import io

def inspect_tar_contents(tar_path):
    print(f"🔍 Inspecting tar: {tar_path}")
    
    # Step 1: Open the stream
    streams = url_opener(tar_path, handler=reraise_exception)
    
    # Step 2: Expand the tar
    files = tar_file_expander(streams, handler=reraise_exception)
    
    for sample in files:
        # Each `sample` is a dict: {'__key__': '000001', 'txt': b'caption', 'png': b'\x89PNG...'}
        print("----- Sample -----")
        print(f"__key__: {sample.get('__key__')}")
        for k, v in sample.items():
            if k == '__key__':
                continue
            print(f"  {k}: type={type(v)}, size={len(v)} bytes")

        print()

# Example usage
inspect_tar_contents("./sample_data/my_sample.tar")
