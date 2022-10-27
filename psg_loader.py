import requests
import zipfile
import io
import os
import psgpython as psg

os.add_dll_directory('C:\Aorda\PSG\lib')

def load_psg():
    prob = requests.get('http://uryasev.ams.stonybrook.edu/wp-content/uploads/2019/05/problem_hmm_discrete.zip')
    uz = zipfile.ZipFile(io.BytesIO(prob.content))
    uz.extractall("./psg_example_hmm/")