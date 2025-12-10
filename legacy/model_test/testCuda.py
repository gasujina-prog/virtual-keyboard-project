import torch
import xformers

print(f"Torch 버전: {torch.__version__}")
print(f"CUDA(GPU) 사용 가능: {torch.cuda.is_available()}")
print(f"xFormers 버전: {xformers.__version__}")

try:
    import xformers.ops
    print("✅ xFormers GPU 연산 모듈 로드 성공!")
except Exception as e:
    print(f"❌ xFormers 로드 실패: {e}")
    print("  -> PyTorch를 CUDA 버전으로 다시 설치해야 합니다.")