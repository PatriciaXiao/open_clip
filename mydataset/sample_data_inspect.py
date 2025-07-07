import webdataset as wds
from webdataset.handlers import reraise_exception
from webdataset.tariterators import url_opener, tar_file_expander
import io
from PIL import Image

def inspect_tar_contents(tar_path):
    print(f"🔍 Inspecting tar: {tar_path}")
    
    # FIX: pass list of paths, not just string
    streams = url_opener([tar_path], handler=reraise_exception)
    files = tar_file_expander(streams, handler=reraise_exception)

    for sample in files:
        print("----- Sample -----")
        print(f"__key__: {sample.get('__key__')}")
        for k, v in sample.items():
            if k == '__key__':
                continue
            print(f"  {k}: type={type(v)}, size={len(v)} bytes")
            if k.endswith("txt"):
                print("    Text content:", v.decode("utf-8"))
        print()

# Example usage
inspect_tar_contents("/homes/gws/patxiao/open_clip/mydataset/sample_data/my_sample.tar")
