## How to setup environment on the server

### Install Pyenv
```bash
curl https://pyenv.run | bash
```

### Edit `.bashrc`
```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
```

### Install Python dependencies
```bash
dnf install tar curl gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make -y
```

### Install Python
```bash
pyenv install 3.10
```

You should see the following output:
```bash
Downloading Python-3.10.14.tar.xz...
-> https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tar.xz
Installing Python-3.10.14...
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/dzhr10/.pyenv/versions/3.10.14/lib/python3.10/bz2.py", line 17, in <module>
    from _bz2 import BZ2Compressor, BZ2Decompressor
ModuleNotFoundError: No module named '_bz2'
WARNING: The Python bz2 extension was not compiled. Missing the bzip2 lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/dzhr10/.pyenv/versions/3.10.14/lib/python3.10/ctypes/__init__.py", line 8, in <module>
    from _ctypes import Union, Structure, Array
ModuleNotFoundError: No module named '_ctypes'
WARNING: The Python ctypes extension was not compiled. Missing the libffi lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/dzhr10/.pyenv/versions/3.10.14/lib/python3.10/sqlite3/__init__.py", line 57, in <module>
    from sqlite3.dbapi2 import *
  File "/home/dzhr10/.pyenv/versions/3.10.14/lib/python3.10/sqlite3/dbapi2.py", line 27, in <module>
    from _sqlite3 import *
ModuleNotFoundError: No module named '_sqlite3'
WARNING: The Python sqlite3 extension was not compiled. Missing the SQLite3 lib?
Installed Python-3.10.14 to /home/dzhr10/.pyenv/versions/3.10.14
```


### Go to the project directory
```bash
cd /sc-projects/sc-proj-dh-ag-eils-ml/brain_segm/brain-segment/
```