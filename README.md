# 👟 Shoeting-ComputerVision


## 📌간단 설명
referenced object(ex 동전, A4)와의 비교를 통해

2D이미지를 통해 물체의 길이를 구할 수 있는 코드입니다.


## 📌실행 방법

object_size_midpoint.ipynb : .ipynb 파일을 다운 받아서 구글드라이브에 업로드하여 구글코랩으로 실행시킬 수 있습니다.
![image](https://user-images.githubusercontent.com/90602936/206730198-6df2a9ed-3ef4-473e-84e1-af4e5cc37445.png)
해당 부분의 코드가 adaptive_thresholding.cpp입니다.

adaptive_thresholding.cpp : visual studio에 opencv 관련 셋팅 후 코드를 실행하면 됩니다.



## 📌상세 설명

실제 물체는 3D인데 이를 사진으로 촬영하면 2D가 되어 dimension이 사라지며 길이정보가 사라집니다.

따라서 기준물체(이미 길이를 알고 있는 규격화된 물체)와 함께 사진을 찍어

기준물체와의 픽셀 수를 비교하여 발의 길이를 측정하였습니다.

![image](https://user-images.githubusercontent.com/90602936/206726775-65046683-347b-4024-910c-bb354af7b5b8.png)
![image](https://user-images.githubusercontent.com/90602936/206726373-77625075-bc7e-4400-af46-7eee7d14c421.png)
![image](https://user-images.githubusercontent.com/90602936/206726405-f13c444f-ece5-400c-ab0a-67eef3353fb0.png)
![image](https://user-images.githubusercontent.com/90602936/206726445-1a21d9df-a6b7-42f0-8be6-31a648c5aee4.png)
![image](https://user-images.githubusercontent.com/90602936/206726484-73959aaa-3ebf-41ee-b0de-90fca228b048.png)
![image](https://user-images.githubusercontent.com/90602936/206726508-375d0784-041e-4a74-b4c4-cbf977f9e204.png)
![image](https://user-images.githubusercontent.com/90602936/206726534-ed4deeb8-152a-4ed0-83b2-9ceed0633126.png)
