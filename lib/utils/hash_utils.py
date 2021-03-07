import os
import hashlib


def get_md5_file(file_path, post_truncated=5):
    md5_hash = hashlib.md5()
    if os.path.exists(file_path):
        xfile = open(file_path, "rb")
        content = xfile.read()
        md5_hash.update(content)
        digest = md5_hash.hexdigest()
    else:
        raise ValueError("[get_md5_file] {:} does not exist".format(file_path))
    if post_truncated is None:
        return digest
    else:
        return digest[-post_truncated:]
