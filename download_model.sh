# CloudFront 배포 도메인
MODEL_URL="https://d1g084wcjwihe3.cloudfront.net/pytorch_model.bin"

# 모델 저장 경로
MODEL_DIR="./models"

# 모델 디렉토리 존재하지 않으면 생성
mkdir -p "$MODEL_DIR"

# 모델 다운로드
echo "🚀 모델 다운로드 중..."
curl -o "$MODEL_DIR/pytorch_model.bin" "$MODEL_URL"

# 다운로드 완료 확인
if [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "✅ 모델 다운로드 완료: $MODEL_DIR/pytorch_model.bin"
else
    echo "❌ 모델 다운로드 실패!"
    exit 1
fi
