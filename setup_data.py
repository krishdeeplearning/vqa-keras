import os

def download():
	os.system('wget https://www.dropbox.com/s/l2wzvpl7dws74j7/data.zip?dl=1')
	os.system('unzip data.zip')

download()