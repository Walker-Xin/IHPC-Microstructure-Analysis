import os
import sys

if len(sys.argv) > 1:
    image = sys.argv[1]
    data = sys.argv[2]
else:
    image = 'Y'
    data = 'Y'

names_IHPC = ['IHPC', 'IHPC_cropped']
names_MIPAR = ['MIPAR', 'MIPAR_cropped']
methods = ['FFT', 'otsu']
MIPAR_cmap = 'gist_ncar'

for name in names_IHPC:
    for method in methods:
        if image = 'Y':
            os.system('python image_saving.py {} {}'.format(name, method))
        else:
            pass
        if data = 'Y':
            os.system('python data_generation.py {} {}'.format(name, method))
        else:
            pass

for name in names_MIPAR:
    for method in methods:
        if image = 'Y':
            os.system('python image_saving.py {} {} {}'.format(name, method, MIPAR_cmap))
        else:
            pass
        if data = 'Y':
            os.system('python data_generation.py {} {} {}'.format(name, method, MIPAR_cmap))
        else:
            pass