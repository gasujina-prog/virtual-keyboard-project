###
# 00 번 파일로 최대한 많은 손 사진 및 라벨 생성
# 생성 후 이미지 파일 확인해서 라벨이 가려진 이미지 삭제 or 라벨 수정
#

###
# 01 번 파일로 이미지 train validation 분류
# 사진 전부 분류 안될 수도 있음. 사진 많아야 하는 이유
#


###
# 베이스 모델은 yolov8n.pt 수업 때 받은거
# cmd 에서
# yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16
#                                                   에포치 수정도 필요함


###
# 02 번 파일로 yolo로 생성된 모델 테스트 해보기
#