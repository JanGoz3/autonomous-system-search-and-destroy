import bpy
import os
import mathutils
from bpy_extras.object_utils import world_to_camera_view

# ==================== KONFIGURACJA ====================
OUTPUT_DIR = "C:/Users/Damrok/Desktop/skany CTI/data" 

# Twój nowy, potężny słownik! Zwróć uwagę na wielkość liter (muszą pasować do Blendera).
# Możesz tu dodawać kolejne klasy w ułamku sekundy, np. 'Obstacles': 1
COLLECTIONS = {
    'Doors': 0
}

NUM_FRAMES = 10 
CONSTRAINT_NAME = "Follow Path"

# ======================================================

# Tworzenie folderów
img_dir = os.path.join(OUTPUT_DIR, "images")
lbl_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

scene = bpy.context.scene
cam = bpy.data.objects['Camera']
path_constraint = cam.constraints.get(CONSTRAINT_NAME)

scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.image_settings.file_format = 'PNG'

if path_constraint:
    print("=== START GENEROWANIA DATASETU (RAYCASTING + SŁOWNIK) ===")
    
    # Pobranie grafu zależności (wymagane do Raycastingu)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    for i in range(NUM_FRAMES):
        filename = f"frame_{i:04d}"
        img_path = os.path.join(img_dir, f"{filename}.png")
        txt_path = os.path.join(lbl_dir, f"{filename}.txt")
        
        current_offset = i / max(1, (NUM_FRAMES - 1))
        path_constraint.offset_factor = current_offset
        bpy.context.view_layer.update()
        
        # Aktualizacja grafu po przesunięciu kamery
        depsgraph.update() 
        
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)
        
        with open(txt_path, 'w') as f:
            cam_loc = cam.matrix_world.translation
            
            # Nowa logika: Przechodzimy po parach (nazwa_kolekcji, id_klasy)
            for col_name, class_id in COLLECTIONS.items():
                collection = bpy.data.collections.get(col_name)
                
                # Jeśli zdefiniowałeś klasę, ale nie ma jeszcze takiej kolekcji w 3D, po prostu ją pomiń
                if not collection:
                    continue
                    
                for obj in collection.objects:
                    if obj.type != 'MESH':
                        continue
                    
                    # --- RAYCASTING ---
                    is_visible = False
                    
                    for corner in obj.bound_box:
                        world_corner = obj.matrix_world @ mathutils.Vector(corner)
                        direction = world_corner - cam_loc
                        distance = direction.length
                        
                        result, loc, normal, index, hit_obj, matrix = scene.ray_cast(
                            depsgraph, 
                            cam_loc, 
                            direction.normalized(), 
                            distance=distance + 0.01
                        )
                        
                        if result and hit_obj == obj:
                            is_visible = True
                            break 
                            
                    if not is_visible:
                        continue
                    # ------------------
                        
                    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
                    coords_2d = [world_to_camera_view(scene, cam, v) for v in verts]
                    
                    z_coords = [c.z for c in coords_2d]
                    if any(z <= 0.0 for z in z_coords):
                        continue
                    
                    x_coords = [c.x for c in coords_2d]
                    y_coords = [c.y for c in coords_2d]
                    
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    if max_x < 0.0 or min_x > 1.0 or max_y < 0.0 or min_y > 1.0:
                        continue

                    width = max_x - min_x
                    height = max_y - min_y
                    center_x = min_x + (width / 2.0)
                    center_y = 1.0 - (min_y + (height / 2.0))
                    
                    center_x = max(0.0, min(1.0, center_x))
                    center_y = max(0.0, min(1.0, center_y))
                    width = max(0.001, min(1.0, width))
                    height = max(0.001, min(1.0, height))

                    # Zapisywanie z ID wyciągniętym bezpośrednio ze słownika!
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
        print(f"Wygenerowano [{i+1}/{NUM_FRAMES}] (Offset: {current_offset:.2f}): {filename}.png")
        
    print(f"=== SUKCES! Zbiór danych zapisany w: {OUTPUT_DIR} ===")
else:
    print(f"Błąd: Na kamerze nie ma constraintu o nazwie '{CONSTRAINT_NAME}'")