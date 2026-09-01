from ultralytics import YOLO
import albumentations as A

model = YOLO('yolo26m.pt') 

custom_camera_noise = [
    # Heavy multi-channel color grain (simulates sensor noise in low light)
    A.GaussNoise(
        var_limit=(150.0, 450.0), 
        per_channel=True, 
        p=0.8
    ),
    
    # ISO sensor noise with noticeable color shift
    A.ISONoise(
        color_shift=(0.15, 0.35), 
        intensity=(0.4, 0.8), 
        p=0.7
    ),
    
    # Dynamic range blow-out / exposure shifts
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.3), 
        contrast_limit=(-0.1, 0.4), 
        p=0.6
    ),
    
    # Low-cost sensor / compression artifacts
    A.ImageCompression(
        quality_range=(35, 75), 
        p=0.4
    ),
]

if __name__ == '__main__':
    results = model.train(
        data='data2/data.yaml', 
        epochs=50,      
        imgsz=320,       
        batch=16,        
        device=0, 
        name='YOLO26_ours_rebalanced_unfrozen_v4', # Nazwa folderu, w którym zapiszą się wyniki,
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
        translate = 0.1,

        augmentations = custom_camera_noise
    )