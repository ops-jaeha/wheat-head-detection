# Installation
```
# IF your device is mac os
$ brew install libomp

# 가상환경은 도커를 사용합니다. 프로젝트 파일로 가서 아래 명령어를 실행해주세요. 
# (The virtual environment uses a docker. Please go to the project file and execute the command below.)
$ make run

# 도커 접속 (Enter Docker)
$ docker exec -it peach-wheat(or container ID) /bin/bash

# 도커 사용 해제 (Docker disabled)
$ make stop 
```