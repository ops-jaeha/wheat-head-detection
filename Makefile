.PHONY: run
run: build
	docker run -itd --name peach-wheat peach-wheat /bin/bash

.PHONY: build
build:
	docker build -t peach-wheat .

.PHONY: stop
stop:
	docker stop peach-wheat
	docker rm peach-wheat
	docker rmi peach-wheat