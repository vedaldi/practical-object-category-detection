name ?= practical-category-detection
ver ?= 2019a-pytorch

code=\
lab.py \
practical.ipynb \
practical.yaml \
README.md

data=\
data/cover.jpg \
data/mandatory.jpg \
data/signs-data.pth \
data/signs \
data/signs-sample-image.jpg

include extra/practical/Makefile

clean: nbstripout

.PHONY: nbstripout
nbstripout:
	nbstripout -f practical.ipynb
