import os
import glob
import cv2
import numpy as np

# !!!!!!!!! CHANGE DIRECTORIES IF NECESSARY PLEASE !!!!!!!!!!!!

# find script location 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# folder of frames
DATA_DIR = os.path.join(SCRIPT_DIR, "RedChair")

# where we save masks/overlays
OUT_DIR = os.path.join(SCRIPT_DIR, "cb_EECE_5639_proj_1_out")

# temporal derivative experiments 
# "simple" = 0.5*(I(t+1)-I(t-1))
TEMP_METHODS = ["simple", "dog"]

# only used for "dog"
T_SIGMAS = [0.5, 1.0, 2.0]

# spatial smoothing experiments
SPATIAL_METHODS = ["none", "box3", "box5", "gauss"]

# only used for "gauss"
S_SIGMAS = [0.5, 1.0, 2.0]

# use T = K_MULT * sigma_est
K_MULT = 3.0

# output control, save every N frames so output folders arent huge
SAVE_EVERY = 1


def load_gray_sequence(folder):
    # load all jpgs in folder, sorts them, reads as grayscale.
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    print(f"Loading {len(files)} images from {folder}")
    frames = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # float 32 works better
        frames.append(img.astype(np.float32))
    print(f"Loaded {len(frames)} valid frames.")
    return frames


def smooth_frame(img, method, s_sigma):
    # smoothing fn
    if method == "box3":
        # 3x3 average filter
        return cv2.blur(img, (3, 3))

    if method == "box5":
        # 5x5 average filter
        return cv2.blur(img, (5, 5))

    if method == "gauss":
        # gaussian blur method
        k = int(6 * s_sigma) + 1
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), s_sigma)
    
    return img


def dog_kernel_1d(sigma):
    # 1D derivative of Gaussian kernel for time.

    # radiis
    r = int(np.ceil(3 * sigma))

    x = np.arange(-r, r + 1).astype(np.float32)

    # guassian
    g = np.exp(-(x * x) / (2 * sigma * sigma))

    # d/dx G(x) = -(x/sigma^2)*G(x)
    dg = -(x / (sigma * sigma)) * g

    # Simplified: removed mean subtraction to keep kernel basic
    
    return dg, r


def temporal_derivative_simple(seq, t):
    # simple temporal derivative
    return 0.5 * (seq[t + 1] - seq[t - 1])


def temporal_derivative_dog(seq, t, kernel, r):
    # apply derviative of guassian kernel across time around t with radius r
    out = np.zeros_like(seq[t], dtype=np.float32)
    for i in range(len(kernel)):
        off = i - r
        out += kernel[i] * seq[t + off]
    return out


def adaptive_threshold_from_abs(abs_img):
    # assume most pixels are background, so abs_img values mostly come from noise.
    # use standard deviation for a simpler undergraduate-style noise estimate
    sigma = np.std(abs_img)
    return K_MULT * sigma


def overlay_and_save(gray_u8, mask_bool, path):
    # make red overlay given frame and frame mask
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    bgr[mask_bool] = (0, 0, 255)
    cv2.imwrite(path, bgr)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_one_setting(frames, temporal_method, t_sigma, spatial_method, s_sigma, tag):
    # frames: list of grayscale float frames
    # this runs:
    #   spatial smoothing for every frame
    #   temporal derivative
    #   magnitude and threshold into binary mask
    #   saves example results

    ensure_dir(OUT_DIR)
    out_folder = os.path.join(OUT_DIR, tag)
    ensure_dir(out_folder)

    print(
        f"Running experiment: {tag} (Spatial: {spatial_method}, Temporal: {temporal_method})..."
    )

    # spatial smoothing
    smooth_seq = []
    for img in frames:
        smooth_seq.append(smooth_frame(img, spatial_method, s_sigma))

    # temporal derivative
    kernel = None
    r = None
    if temporal_method == "dog":
        kernel, r = dog_kernel_1d(t_sigma)

    # decide valid frame range (avoid boundaries)
    # simple needs t-1 and t+1, so valid t is 1..N-2
    # dog needs t-r through t+r, so valid t is when we have r frames on both sides
    if temporal_method == "simple":
        start = 1
        end = len(smooth_seq) - 2
    else:
        start = r
        end = len(smooth_seq) - 1 - r

    # loop over time and create motion masks
    for t in range(start, end + 1):
        # temporal derivative image
        if temporal_method == "simple":
            D = temporal_derivative_simple(smooth_seq, t)
        else:
            D = temporal_derivative_dog(smooth_seq, t, kernel, r)

        # use magnitude of temporal derivative as "motion strength"
        M = np.abs(D)

        # threshold definition
        T = adaptive_threshold_from_abs(M)
        # binary mask
        mask = M > T

        if (t - start) % SAVE_EVERY == 0:
            base = f"{tag}_f{t:05d}"

            # original frame to pass to overlay function
            orig = frames[t].clip(0, 255).astype(np.uint8)

            # save overlay
            overlay_and_save(
                orig,
                mask,
                os.path.join(out_folder, base + "_overlay.png"),
            )
            cv2.imwrite(
                os.path.join(out_folder, base + "_mask.png"),
                # we cant see 1, need 255 for white
                mask.astype(np.uint8) * 255,
            )


# MAIN
frames = load_gray_sequence(DATA_DIR)

# sweep 1: temporal derivative methods
for tm in TEMP_METHODS:
    if tm == "simple":
        # simple only needs 1 temporal sigma
        t_list = [0.0]
    else:
        # try multiple temporal sigmas
        t_list = T_SIGMAS

    for ts in t_list:
        tag = f"temp_{tm}_ts_{ts}"
        run_one_setting(frames, tm, ts, "none", 0.0, tag)

# sweep 2: spatial
for sm in SPATIAL_METHODS:
    if sm == "gauss":
        s_list = S_SIGMAS
    else:
        # box methods only need 1 spatial sigma
        s_list = [0.0]

    for ss in s_list:
        tag = f"spat_{sm}_ss_{ss}"
        run_one_setting(frames, "simple", 0.0, sm, ss, tag)

# combo attempt
run_one_setting(frames, "dog", 1.0, "gauss", 1.0, "combo_dog1_gauss1")