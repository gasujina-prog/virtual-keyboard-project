import random
import time
import os
import pymysql

DB_PATH = "typing_results.db"

WORDS = [
    "computer", "keyboard", "python", "developer", "network", "database", "algorithm", "function", "variable", "process", "memory",
    "storage", "system", "object", "thread", "package", "compile", "execute", "monitor", "service", "device",
    "window", "render", "signal", "buffer", "payload", "session", "request", "response", "runtime"
]

NUM_ROUNDS = 5  # 테스트용으로 5개, 나중에 20으로 다시 올리면 됩니다.



def init_db():
    """DB 파일 및 테이블 생성"""
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="4301",
        database="projectkeyboard",
        port=3306
    )
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS typing_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id    VARCHAR(100)  NOT NULL,
            target_word VARCHAR(255) NOT NULL,
            typed_word  VARCHAR(255) NOT NULL,
            accuracy    DOUBLE       NOT NULL,
            is_exact    TINYINT(1)   NOT NULL,
            created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )
    conn.commit()
    conn.close()


def save_result_to_db(user_id: str, target: str, typed: str, accuracy: float, is_exact: bool):
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="4301",
        database="projectkeyboard",
        port=3306
    )
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO typing_results (user_id, target_word, typed_word, accuracy, is_exact)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (user_id, target, typed, accuracy, 1 if is_exact else 0)
    )
    conn.commit()
    conn.close()


def calc_similarity_accuracy(target: str, user_input: str) -> float:
    max_len = max(len(target), len(user_input))

    if max_len == 0:
        return 100.0

    correct = 0
    for t, u in zip(target, user_input):
        if t == u:
            correct += 1

    accuracy = (correct / max_len) * 100
    return accuracy


def run_typing_practice():


    user_id = input("사용자 이름 또는 번호를 입력하세요: ").strip()
    if not user_id:
        user_id = "anonymous"

    print("=== 타자 연습 프로그램 ===")
    print(f"총 {NUM_ROUNDS}개의 단어를 연습합니다.")
    input("\nEnter를 누르면 시작합니다...")

    total_chars = 0
    correct_chars = 0
    correct_words = 0
    total_words = 0
    total_similarity_score = 0.0

    start_time = time.time()

    for i in range(1, NUM_ROUNDS + 1):
        target = random.choice(WORDS)
        print(f"\n[{i}/{NUM_ROUNDS}] 단어:")
        print(f"  ▶  {target}")

        user_input = input("입력: ")

        total_words += 1

        max_len = max(len(target), len(user_input))
        total_chars += max_len

        round_correct_chars = 0
        for t_ch, u_ch in zip(target, user_input):
            if t_ch == u_ch:
                round_correct_chars += 1
        correct_chars += round_correct_chars

        similarity = calc_similarity_accuracy(target, user_input)
        total_similarity_score += similarity

        is_exact = (user_input == target)
        if is_exact:
            result = "정답"
            correct_words += 1
        else:
            result = "오타"

        print(f"  → 판정: {result}")
        print(f"  → 단어 정확도: {similarity:.1f}%")

        save_result_to_db(
            user_id=user_id,
            target=target,
            typed=user_input,
            accuracy=similarity,
            is_exact=is_exact
        )

    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0
    if elapsed_min == 0:
        elapsed_min = 0.0001

    wpm = (correct_chars / 5.0) / elapsed_min
    char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    avg_similarity = total_similarity_score / total_words if total_words > 0 else 0

    print("\n=== 최종 결과 ===")
    print(f"총 소요 시간: {elapsed_sec:.1f} 초")
    print(f"글자 기준 정확도: {char_accuracy:.1f}%")
    print(f"단어 완전일치 정확도: {word_accuracy:.1f}%")
    print(f"평균 유사도 정확도: {avg_similarity:.1f}%")
    print(f"타자 속도: {wpm:.1f} WPM")

    print("\n각 라운드 결과가 DB에 저장되었습니다.")
    print(f"DB 파일 위치: {os.path.abspath(DB_PATH)}")


if __name__ == "__main__":
    init_db()  # 프로그램 시작 시 DB 초기화 (테이블 없으면 생성)
    run_typing_practice()
