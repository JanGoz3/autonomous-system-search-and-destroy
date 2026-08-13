from ultralytics import YOLO

model = YOLO('best.pt') 

if __name__ == '__main__':
    results = model.train(
        data='data/data.yaml', 
        epochs=50,      
        imgsz=320,       
        batch=16,        
        device=0, 
        name='YOLO26_ours_rebalanced_unfrozen_v3', # Nazwa folderu, w którym zapiszą się wyniki,
        patience = 10,
        workers = 4,
        #freeze = 10, # freeze the first 10 modules
        cls_pw = 1.0,

        # preventing double augumentation
        degrees = 0.0,
        fliplr = 0.0,
        hsv_v = 0.0,

        # low res protections
        mosaic = 0.5,
        scale = 0.2,
        translate = 0.1
    )