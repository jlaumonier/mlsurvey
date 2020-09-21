#!/bin/bash

source venv/bin/activate
git describe --always > __git_commit__
python3 setup.py sdist bdist_wheel
rm __git_commit__