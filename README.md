# peach-wheat-detection
AIcrowd의 [Glabal WHEAT CHALLENGE 2021](https://www.aicrowd.com/challenges/global-wheat-challenge-2021) 의 Wheat head dectection 문제입니다.  
기본 제공하는 Dataset을 이용하였고 [여기](https://www.aicrowd.com/challenges/global-wheat-challenge-2021/dataset_files) 를 누르시면 다운 받으실 수 있습니다. 

This is AIcrowd's [Glabal WHEATCHALLENGE 2021](https://www.aicrowd.com/challenges/global-wheat-challenge-2021) "WHEATHEAD DETECTION" problem.  
You have used the default Dataset and you can download it by clicking [here](https://www.aicrowd.com/challenges/global-wheat-challenge-2021/dataset_files).  


<details open>
<summary>Install</summary>

If you want to do it with a docker, please refer to [Install.md](https://github.com/the-peach-drone/peach-wheat-detection/blob/main/INSTALL.md).

```
$ git clone https://github.com/CV-JaeHa/wheat-head-detection
$ cd wheat-head-detection
$ pip3 install -r requirements.txt
```
</details>

<details open>
<summary>Train</summary>

Run this commend.
```
$ cd wheat-head-detection
$ python3 train.py
```
</details>

<details open>
<summary>Test</summary>

Run this commend.
```
$ cd wheat-head-detection
$ python3 test.py
```
</details>