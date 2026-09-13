from ultralytics import YOLO
import albumentations as A

model = YOLO('last.pt') 

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

    A.MotionBlur(
        blur_limit=3, 
        p=0.4
    ),
]

if __name__ == '__main__':
    results = model.train(
        data='data/data.yaml', 
        epochs=50,
        lr0 = 0.001, # starting lr
        lrf = 0.01,  # final learning rate multiplier lr = lr0 * lrf
        warmup_epochs = 0,      
        imgsz=320,       
        batch=16,        
        device=0, 
        name='YOLO26_ours_v5', # Nazwa folderu, w którym zapiszą się wyniki,
        patience = 20,
        workers = 4,
        #freeze = 10, # freeze the first 10 modules
        cls_pw = 0.75,

        degrees = 0.0,
        fliplr = 0.5,
        hsv_v = 0.1,
        hsv_h = 0.015, # Slight shift in color hue
        hsv_s = 0.7,   # Moderate shift in color saturation
        erasing = 0.4,

        # low res protections
        mosaic = 0.0,
        scale = 0.2,
        translate = 0.1,

        augmentations = custom_camera_noise
    )