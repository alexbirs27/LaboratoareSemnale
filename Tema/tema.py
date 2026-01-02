import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from scipy.datasets import ascent
from skimage import data
import cv2 as cv
# https://en.wikipedia.org/wiki/YCbCr

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def apply_jpeg_to_block(block, Q_matrix):
    centered = block.astype(float) - 128
    freq = dctn(centered, type=2, norm='ortho')
    quantized = np.round(freq/Q_matrix) * Q_matrix
    reconstructed = idctn(quantized, type=2, norm='ortho') + 128
    return reconstructed


def process_image_jpeg(image, quality):
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)

    # rgb to ycbcr
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128
    ycbcr = np.stack([Y, Cb, Cr], axis=2)

    rows, cols = (ycbcr.shape[0] // 8) * 8, (ycbcr.shape[1] // 8) * 8
    ycbcr = ycbcr[:rows, :cols]
    output = np.zeros_like(ycbcr)
    Q = Q_jpeg * quality

    for channel in range(3):
        for r in range(0, rows, 8):
            for c in range(0, cols, 8):
                block_in = ycbcr[r:r+8, c:c+8, channel]
                block_out = apply_jpeg_to_block(block_in, Q)
                
                output[r:r+8, c:c+8, channel] = block_out

    # ycbcr to rgb
    Y, Cb, Cr = output[:,:,0], output[:,:,1], output[:,:,2]
    R = Y + 1.402*(Cr - 128)
    G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    B = Y + 1.772*(Cb - 128)
    rgb = np.stack([R, G, B], axis=2)

    return np.clip(rgb, 0, 255).astype(np.uint8)


# Ex 1&2: jpeg color
img_color = data.astronaut()
img_compressed = process_image_jpeg(img_color, quality=1.0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img_color)
axes[0].set_title('Original')

axes[1].imshow(img_compressed)
axes[1].set_title('JPEG color')

plt.tight_layout()
plt.savefig('ex12.pdf')
plt.show()


# Ex 3: Target mse
target_mse = 50
left, right = 0.3, 8.0
eps = 0.01

while right - left > eps:
    mid = (left + right) / 2
    test_img = process_image_jpeg(img_color, quality=mid)
    h, w = test_img.shape[0], test_img.shape[1]
    current_mse = np.mean((img_color[:h, :w] - test_img) ** 2)

    if current_mse > target_mse:
        right = mid
    else:
        left = mid

optimal_quality = (left + right) / 2
optimal_img = process_image_jpeg(img_color, quality=optimal_quality)
achieved_mse = np.mean((img_color[:optimal_img.shape[0], :optimal_img.shape[1]] - optimal_img) ** 2)

print(f'Target MSE: {target_mse}, Achieved: {achieved_mse:.2f}, Quality: {optimal_quality:.3f}')

plt.figure(figsize=(5, 4))
plt.imshow(optimal_img)
plt.title(f'MSE={achieved_mse:.2f}')
plt.savefig('ex3.pdf')
plt.show()


# Ex 4: Video
video_in = cv.VideoCapture('video.mp4')

if video_in.isOpened():
    width = int(video_in.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_in.get(cv.CAP_PROP_FPS))

    video_out = cv.VideoWriter('compressed_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    idx = 0
    while idx < 100:
        success, frame_bgr = video_in.read()
        if not success:
            break

        frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        compressed_frame = process_image_jpeg(frame, quality=1.8)
        frame_out = cv.cvtColor(compressed_frame, cv.COLOR_RGB2BGR)

        video_out.write(frame_out)
        idx += 1

    video_in.release()
    video_out.release()
    print(f'Video processed: {idx} frames')



