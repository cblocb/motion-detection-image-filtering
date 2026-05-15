import os
import cv2
import glob
import numpy as np

# Determine script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUT_DIR = os.path.join(SCRIPT_DIR, "out_student_baseline")
GRID_OUT_DIR = os.path.join(BASE_OUT_DIR, "comparisons")

def main():
    if not os.path.exists(BASE_OUT_DIR):
        print(f"Error: {BASE_OUT_DIR} does not exist.")
        return

    # 1. Find all experiment subdirectories
    subdirs = [d for d in os.listdir(BASE_OUT_DIR) 
               if os.path.isdir(os.path.join(BASE_OUT_DIR, d)) and d != "comparisons"]
    subdirs.sort()
    
    if not subdirs:
        print("No experiment directories found.")
        return

    print(f"Found {len(subdirs)} experiment folders: {subdirs}")

    # 2. Find all unique frame basenames (e.g. "f00001", "f00101")
    # We scan all folders to get the union of all frames, or just pick one reference.
    # Let's use a set to collect all unique 'fXXXXX' parts from 'fXXXXX_overlay.png'
    frame_basenames = set()
    for d in subdirs:
        path = os.path.join(BASE_OUT_DIR, d)
        overlays = glob.glob(os.path.join(path, "*_overlay.png"))
        for f in overlays:
            # f is full path. filename is 'f00101_overlay.png'
            fname = os.path.basename(f)
            # extract 'f00101'
            base = fname.split("_")[0]
            frame_basenames.add(base)
    
    sorted_frames = sorted(list(frame_basenames))
    print(f"Found {len(sorted_frames)} unique frames to process.")

    os.makedirs(GRID_OUT_DIR, exist_ok=True)

    # 3. Accumulated Grids (Bin size = 150 frames ~ 5 seconds)
    BIN_SIZE = 1
    
    # Parse frame indices
    indices = []
    for s in sorted_frames:
        try:
            indices.append(int(s[1:])) # strip 'f'
        except:
            continue
    indices.sort()
    
    if not indices:
        print("No valid frame indices found.")
        return

    max_idx = indices[-1]
    num_bins = (max_idx // BIN_SIZE) + 1
    
    for b in range(num_bins):
        start_idx = b * BIN_SIZE
        end_idx = (b + 1) * BIN_SIZE
        
        # Identify frames in this bin
        bin_frames = [f"f{i:05d}" for i in indices if start_idx <= i < end_idx]
        
        if not bin_frames:
            continue
            
        print(f"Processing bin {b}: Frames {start_idx} to {end_idx} ({len(bin_frames)} frames)")

        # For each method (subdir), accumulate the masks
        method_images = []
        h, w = 0, 0
        
        for d in subdirs:
            # We need a background image (use first frame in bin)
            bg_img = None
            accum_mask = None
            
            for base in bin_frames:
                mask_path = os.path.join(BASE_OUT_DIR, d, f"{base}_mask.png")
                over_path = os.path.join(BASE_OUT_DIR, d, f"{base}_overlay.png")
                
                if os.path.exists(mask_path):
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        if accum_mask is None:
                            accum_mask = m
                            h, w = m.shape
                        else:
                            accum_mask = cv2.bitwise_or(accum_mask, m)
                
                # Grab background from first available overlay if not set
                if bg_img is None and os.path.exists(over_path):
                     over = cv2.imread(over_path)
                     if over is not None:
                         bg_img = over # This has red pixels already? Ideally we mask them out.
                         # Actually, the overlay has the original gray image as background.
                         # But the red pixels are (0,0,255). We can assume R=0,G=0,B=255 is mask.
                         # Easier: just use it. The Red spots from frame 1 will stay.
                         # Even better: The script saves 'orig' inside overlay_and_save but doesn't write it to disk separately?
                         # Wait, looking at project1.py:
                         # orig = frames[t]...
                         # overlay_and_save(orig, mask, ...) -> writes bgr
                         # cv2.imwrite(..., mask)
                         
                         # So we only have the overlay.
                         # Let's just use the overlay image as the base. 
                         # Any existing red spots from frame 1 will be there.
                         # New spots from frames 2..N will be added.
                         # This acts like a "max projection" of time. Perfect.
            
            if accum_mask is not None:
                # If we never found a background (weird), make black
                if bg_img is None:
                    bg_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Apply accumulated mask (Red)
                # Ensure mask is binary
                _, bin_mask = cv2.threshold(accum_mask, 127, 255, cv2.THRESH_BINARY)
                
                # Paint red where mask is set
                # bg_img is BGR. Red is (0, 0, 255).
                bg_img[bin_mask == 255] = (0, 0, 255)
                
                # Add label
                cv2.putText(bg_img, d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(bg_img, d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                method_images.append(bg_img)
            else:
                # No data for this method in this bin
                method_images.append(None)

        # Create grid from method_images
        # Fill Nones
        final_images = []
        # Get consistent size
        gh, gw = 0, 0
        for img in method_images:
            if img is not None:
                gh, gw = img.shape[:2]
                break
        
        if gh == 0: continue # Empty bin?

        for img in method_images:
            if img is None:
                final_images.append(np.zeros((gh, gw, 3), dtype=np.uint8))
            else:
                final_images.append(img)
                
        # 3x4 layout
        cols = 3
        rows = (len(final_images) + cols - 1) // cols
        while len(final_images) < rows * cols:
            final_images.append(np.zeros((gh, gw, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            grid_rows.append(np.hstack(final_images[r*cols : (r+1)*cols]))
            
        full_grid = np.vstack(grid_rows)
        
        # Save
        out_path = os.path.join(GRID_OUT_DIR, f"grid_accum_frames_{start_idx}-{end_idx}.png")
        cv2.imwrite(out_path, full_grid)

    print(f"Done. Saved bin grids to {GRID_OUT_DIR}")

if __name__ == "__main__":
    main()
