#!/bin/bash

# stop_stream.sh
# pm2로 띄운 모든 pcb-streamer-* 프로세스를 정지합니다.

echo "🛑 모든 스트리머 중지 중..."
sudo pm2 stop pcb-streamer-* || true

# 필요시 아예 삭제도 가능
# sudo pm2 delete pcb-streamer-* || true

echo "✅ 중지가 완료되었습니다."