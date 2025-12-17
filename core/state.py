import queue

# 전역 상태 변수 관리
sse_queue = queue.Queue(maxsize=100)
current_user_id = None
is_virtual_input_active = True

# Detector 인스턴스는 main에서 초기화 후 여기에 할당하거나,
# 필요시 여기서 초기화 할 수 있습니다. (순환 참조 방지를 위해 변수만 선언)
detector = None