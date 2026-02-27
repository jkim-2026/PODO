#!/bin/bash

# 0. 설정
NUM_CAMERAS=3  # 시뮬레이션할 카메라 대수
VIDEO_SOURCE="/home/ubuntu/rtsp/health_optimized.mp4"

# 1. 기존에 돌아가던 스트리머가 있다면 삭제
sudo pm2 delete pcb-streamer || true
sudo pm2 delete "/pcb-streamer-.*/" || true

# 2. RTSP 송출 시작 (다중 채널)
for i in $(seq 1 $NUM_CAMERAS)
do
    echo "🚀 채널 $i 송출 시작: rtsp://localhost:8554/pcb_stream_$i"
    sudo pm2 start "ffmpeg -re -stream_loop -1 -i $VIDEO_SOURCE -c copy -f rtsp rtsp://localhost:8554/pcb_stream_$i" --name pcb-streamer-$i
done

echo "⏳ 스트리머들이 안정화될 때까지 3초간 대기합니다..."
sleep 3

# 3. 현재 pm2 프로세스 리스트 상태 확인
sudo pm2 status

# 4. 서버가 재부팅되어도 이 설정이 유지되도록 저장
sudo pm2 save

echo "🪽 총 $NUM_CAMERAS개 채널 RTSP 스트리밍 설정이 완료되었습니다."
