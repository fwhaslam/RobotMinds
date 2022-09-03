#
#   Test Script:L ensure that all source scripts compile
#

import sys
sys.path.append('../../main/python')


import py_compile
import glob
import pathlib
import os

root = pathlib.Path('../../main/python')

for f in root.rglob('*.py'):
    path = str(f)
    print("\nChecking :: ",path)
    result = py_compile.compile(path)
    print("Result=",result)

