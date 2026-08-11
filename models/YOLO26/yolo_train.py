from ultralytics import YOLO

model = YOLO('yolo26m.pt') 

if __name__ == '__main__':
    results = model.train(
        data='data/data.yaml', 
        epochs=2,      
        imgsz=320,       
        batch=16,        
        device=0, 
        name='YOLO26m_ours_smallImgs' # Nazwa folderu, w którym zapiszą się wyniki
    )