가상 키보드 연동 프로젝트
MBC아카데미 2025 심화과정

이 프로젝트는 Python 3.11 버전 이상 환경에서 진행되며, 사용할 라이브러리는 다음과 같습니다.

PyQt6   - UI와 전체 모듈의 데이터 흐름을 통제

NumPy   - OpenCV와 MediaPipe 데이터 처리를 위한 이미지 및 좌표배열 기본형식

OpenCV  - 웹캠 영상의 스트리밍 제어, MediaPipe 분석을 위한 프레임 제공

MediaPipe - 손 추적 및 제스처 인식

pynput  - 인식한 제스처를 가상 키 입력 이벤트로 변환 및 실행

Flask   - PyQt6의 설정요청 처리 및 응답하는 RESTfulAPI 서버 역할.

Flask-SQLAlchemy  - 사용자 설정 데이터를 SQLite의 DB에 저장 및 관리

request - 클라이언트-Flask 서버로 설정데이터를 전송ㆍ수신을 하기 위한 HTTP통신

위 라이브러리는 프로젝트에 있는 requements.txt 에 있으며
터미널에서 cd 명령어로 해당 .txt 파일이 있는 위치로 이동 후 다음 명령어를 입력
pip install -r requirements.txt ※ 터미널 위치에 해당 파일이 있으면 cd 명령어 생략하고 바로 pip 명령어 입력


