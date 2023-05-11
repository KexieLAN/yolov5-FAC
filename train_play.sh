python train.py --weights yolov5m.pt --cfg ./models/yolov5m_SE.yaml --data ./data/FAC-liunx.yaml --epochs 1000 --batch-size 6 --multi-scale --patience 50;
sleep 300;
sh /mistgpu/shutdown.sh;