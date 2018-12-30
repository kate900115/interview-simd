INPUTS=gzip/emnist-byclass-train-images-idx3-ubyte

all: bin/mnist-1lnn $(INPUTS)

.PHONY:main
bin/mnist-1lnn: 
	gcc -o bin/mnist-1lnn -fopenmp -O3 -mavx -mavx2 -Iutil main.c 1lnn.c util/screen.c util/mnist-utils.c util/mnist-stats.c

gzip.zip:
	wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip -O $@

$(INPUTS): gzip.zip
	unzip -o $<
	gunzip -f gzip/*.gz
	touch gzip/*

clean:
	rm -rf bin/mnist-1lnn
%.tgz: clean
	git clone . $*
	tar czf $@ --exclude $*/.git $*

.PHONY: release
release:  mnist-1lnn.tgz
