import time
import queue
from core import state
from core.database import db
from models.key_log import KeyLog


def save_keys_worker(app):
    """
    백그라운드에서 키 입력을 감지하고 DB에 저장하며 SSE로 전송합니다.
    app_context가 필요하므로 app 객체를 인자로 받습니다.
    """
    print("👷 Worker started...")
    while True:
        time.sleep(0.05)

        # Detector가 초기화되지 않았으면 대기
        if state.detector is None:
            continue

        inputs = state.detector.pop_inputs()

        if not state.is_virtual_input_active:
            continue

        if inputs:
            # 1. SSE 전송 (브라우저로 쏘기)
            for item in inputs:
                try:
                    # block=False: 꽉 차 있으면 기다리지 않고 바로 에러(Full) 발생시킴
                    state.sse_queue.put(item['key'], block=False)
                except queue.Full:
                    # 큐가 꽉 찼다면?
                    try:
                        state.sse_queue.get_nowait()  # 가장 오래된 데이터 하나 버림 (배수구)
                        state.sse_queue.put(item['key'], block=False)  # 다시 넣기 시도
                    except:
                        pass  # 그래도 안 되면 이번 데이터는 쿨하게 포기 (서버 다운 방지)

            # 2. DB 저장
            with app.app_context():
                try:
                    if state.current_user_id is not None:
                        for item in inputs:
                            new_log = KeyLog(key_name=item['key'], user_id=state.current_user_id)
                            db.session.add(new_log)
                        db.session.commit()
                        print(f"💾 Saved {len(inputs)} keys")
                except Exception as e:
                    print(f"DB Error: {e}")