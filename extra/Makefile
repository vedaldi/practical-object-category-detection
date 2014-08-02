name = practical-category-detection-2014a
DST = vgg@login.robots.ox.ac.uk:WWW/share

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

code=\
boxinclusion.m \
boxoverlap.m \
boxsuppress.m \
detect.m \
detectAtMultipleScales.m \
evalDetections.m \
evaluateModel.m \
exercise1.m \
exercise2.m \
exercise3.m \
exercise4.m \
extract.m \
loadData.m \
setup.m \
README.md \
vlfeat \
matconvnet \
doc/ \
doc/instructions.pdf \
doc/instructions.html

data=\
data/ \
data/signs-data.mat \
data/signs-images

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))

pack:
	rm -rf data/$(name)
	mkdir -p data/$(name)/data
	ln -sft data/$(name)/data/ $(data)
	ln -sft data/$(name)/ $(code)
	tar -C data -czvhf data/$(name).tar.gz $(name)/

pack-data:
	rm -rf data/$(name)
	mkdir -p data/$(name)/data
	ln -sft data/$(name)/data/ $(data)
	tar -C data -czvhf data/$(name)-data-only.tar.gz $(name)/

pack-code:
	rm -rf data/$(name)
	mkdir -p data/$(name)/data
	ln -sft data/$(name)/ $(code)
	tar -C data -czvhf data/$(name)-code-only.tar.gz $(name)/

post: pack-all
	rsync -v data/$(name).tar.gz $(DST)/
	rsync -v data/$(name)-data-only.tar.gz $(DST)/
	rsync -v data/$(name)-code-only.tar.gz $(DST)/

clean:
	find -name '*~' -delete

distclean: clean
	rm -f data/$(name)*.tar.gz
