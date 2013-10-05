Installation
------------
Prerequisites
.............
First, you will need a working installation of python, `numpy <http://www.numpy.org>`_, `scipy <http://www.scipy.org>`_, and `matplotlib <http://matplotlib.org>`_. I'm using python 2.7.3, numpy version 1.7.0, scipy version 0.11.0, and matplotlib version 1.2.0. You might as well get `ipython <http://ipython.org>`_ (which installs the `ipython notebook <http://ipython.org/notebook.html>`_) while you're at it. 

If you are using a mac, I recommend following `Matt Davis's instructions <http://penandpants.com/2013/04/04/install-scientific-python-on-mac-os-x/>`_ on how to install these packages using `homebrew <http://brew.sh>`_. If you are running linux, try your distro's package manager. 

If you are on windows, let me know how you install this stuff because I don't have access to a windows box. Hit me up on the `issue tracker <https://github.com/jrsmith3/tec/issues?state=open>`_ if you have suggestions for windows.

tec installation
................
The code isn't on pypi, so you'll have to install from source. There are two ways to do it: 1. downloading and installing from the zip file and 2. installing with pip over git from github. If you don't understand the phrase, "installing over git from github", you want to use the first option.

Zip file install
................
All you have to do is download the most recent version (0.3.2) from the internet `here <https://github.com/jrsmith3/tec/archive/0.3.2.zip>`_. Unzip the file and switch into the directory into which the zip file was uncompressed. Then execute the following command at the command line.

    $ python setup.py install

pip + git + github
..................
If you have pip installed, things should be pretty easy. Just execute

    $ pip install git+git://github.com/jrsmith3/tec.git
