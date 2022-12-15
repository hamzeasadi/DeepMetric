import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.restoration import estimate_sigma, denoise_wavelet
import conf as cfg
import os
import time
import skimage.io
import pywt


def video_frame_extract(video_path: str, target_path: str, num_frame: int):
    """
    extract the frames from video. this function takes a video file and calculate number of frames and extract num_frame frames. 
    It will iterate over all directories inside video path and find video files in these sub directories and then save them to crrosponding directory
    in the target directory.
    arguments:
        video_path: a string represent the path that directries of diff camera are saved.
        target_path: a directory that we save each camera extracted frame in its on subdirectory.
        num_frame: number of frames that will be extracted frame each video files
    return:
        save the files in subdirectries for each camera in target directory
    """
    w, h = 224, 224
    # read the list of directories in video_path, first ne is .DSvscode
    listdir = os.listdir(video_path)
    
    #create corrospondind directories for each camera inside target directory 
    for k in range(1, len(listdir)):
            try:
                os.makedirs(os.path.join(target_path, listdir[k]))
            except Exception as e:
                print(e)

    for i in range(1, len(listdir)):
        
        # first subdirectory (camera) of video_path
        videos_addr = os.path.join(video_path, listdir[i])
        # list video files
        video_files = os.listdir(videos_addr)
        # a counter for file naming
        file_counter = 0
        # read video files of each sub directories
        for video_file in video_files:
            # first video file address
            video_addr = os.path.join(videos_addr, video_file)
            # read the video
            video = cv2.VideoCapture(video_addr)
            # fps = int(video.get(cv2.CAP_PROP_FPS))
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            steps = int(frame_count/num_frame)
            # extract num_frame from each video file
            for frame_num in range(0, frame_count, steps):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = video.read()
                frame_size = frame.shape
                h1 = np.random.randint(85, frame_size[0]-224)
                w1 = np.random.randint(1, frame_size[1]-224)
                cframe = frame[h1:h1+224, w1:w1+224, :]
                # print(cframe)
                if ret:
                    fname = os.path.join(target_path, listdir[i], f'frame_{file_counter}.png')
                    cv2.imwrite(filename=fname, img=cframe)
                    file_counter += 1
                    
                # else:
                #     fname = os.path.join(target_path, listdir[i], f'frame_{file_counter}.png')
                #     cv2.imwrite(filename=fname, img=frame)

                # file_counter += 1

            cv2.destroyAllWindows()
            video.release()




def wavelet_denoise_wtsmooth(img_path):
    # w, h = 240, 240

    original_img = skimage.io.imread(img_path)
    # original_img = image_remove_text(img_path)
    original_img = skimage.img_as_float(original_img)
    # center = original_img.shape
    # x = int(center[1]/2 - w/2)
    # y = int(center[0]/2 - h/2)
    # original_crop_img = original_img[y:y+h, x:x+w, :]
    
    # coif5
    # VisuShrink, BayesShrink
    denoise_img = denoise_wavelet(image=original_img, wavelet='coif5', mode='soft', wavelet_levels=3, 
                                    convert2ycbcr=True, method='BayesShrink', rescale_sigma=True, multichannel=True)
    
    residual_img = original_img - denoise_img
  
    residual_img = residual_img/np.max(residual_img)
    # for i in range(3):
    #     residual_img[:, :, i] = residual_img[:, :, i]/np.max(residual_img[:, :, i])

    residual_img = (residual_img * 255).astype(np.uint8)
    # # plt.imshow(residual_img, cmap='gray')

    # fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    # axs[0].imshow(original_img, cmap='gray')
    # axs[1].imshow(denoise_img, cmap='gray')
    # axs[2].imshow(residual_img, cmap='gray')
    # plt.tight_layout()
    # # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    return residual_img




def residual_img(images_path: str, target_path: str):
    listdirs = os.listdir(images_path)
    if listdirs[0] == '.DS_Store':
        listdirs.remove('.DS_Store')

    for dirname in listdirs:

        try:
            os.makedirs(os.path.join(target_path, dirname))
        except Exception as e:
            print(e)

        dirpath = os.path.join(images_path, dirname)
        list_images = os.listdir(dirpath)
        for imgname in list_images:
            imgpath = os.path.join(dirpath, imgname)
            residual = wavelet_denoise_wtsmooth(imgpath)
            residual_path = os.path.join(target_path, dirname, imgname)
            cv2.imwrite(residual_path, residual)


def image_remove_text(img_path: str):
    
    img = cv2.imread(filename=img_path)

    h, w, c = img.shape
    top_cut = 85
    bottom_cut = 80
    crop_img = img[top_cut:h - bottom_cut, :, :]
    return crop_img


def main():
    # video_frame_extract(video_path=cfg.paths['videos'], target_path=cfg.paths['rawimage'], num_frame=20)
    residual_img(images_path=cfg.paths['rawimage'], target_path=cfg.paths['libbherr'])

    # imgs = np.random.randint(0, 190, 3)
    # for i in imgs:
    #     fname = f'frame_{i}.png'
    #     imgpath = os.path.join(cfg.paths['rawimage'], 'Truck TTR', fname)
        # img = cv2.imread(imgpath)
        # cv2.imshow('frame', img)
        # cv2.waitKey(0)
        # wavelet_denoise_wtsmooth(img_path=imgpath)
        # w2d(imgpath,'db1',10)


    # fname = f'frame_{10}.png'
    # imgpath = os.path.join(cfg.paths['rawimage'], 'Truck TTR', fname)
    # wavelet_denoise_wtsmooth(img_path=imgpath)
    # x = np.random.randint(100)
    # print(x)



if __name__ == '__main__':
    main()