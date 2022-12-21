import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.restoration import estimate_sigma, denoise_wavelet
import conf as cfg
import os
import time
from skimage import io
import pywt
import skimage


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
    denoise_img = denoise_wavelet(image=original_img, wavelet='haar', mode='soft', wavelet_levels=3, 
                                    convert2ycbcr=True, method='BayesShrink', rescale_sigma=True, multichannel=True)
    
    residual_img = original_img - denoise_img
  
    # residual_img = residual_img/np.max(residual_img)
    # for i in range(3):
    #     residual_img[:, :, i] = residual_img[:, :, i]/np.max(residual_img[:, :, i])

    # residual_img = (residual_img * 255).astype(np.uint8)
    # # plt.imshow(residual_img, cmap='gray')

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    axs[0].imshow(original_img, cmap='gray')
    axs[1].imshow(denoise_img, cmap='gray')
    axs[2].imshow(residual_img, cmap='gray')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    return residual_img


def convertoGray(srcpath):
    imgfolders = os.listdir(srcpath)
    try:
        imgfolders.remove('.DS_Store')
    except Exception as e:
        print(f"there is no .DS_Store folder")

    for folder in imgfolders:
        trgpath = os.path.join(srcpath, folder)
        imglist = os.listdir(trgpath)

        try:
            imglist.remove('.DS_Store')
        except Exception as e:
            print(f"there is no .DS_Store folder")

        for imgname in imglist:
            imgpath = os.path.join(trgpath, imgname)
            img = cv2.imread(imgpath)
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(imgpath, grayimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


waveletalg = ['haar', 'db2', 'db1', 'sym9', 'coif5']
def imagepatching(imggray):
    H, W = 224, 224
    h, w, c = imggray.shape
    # img = imggray[85:h-80, :, 0]
    img = imggray[:, :, 0]
    h, w = img.shape
    threehpatchsize = int(h/3)
    dh = int((threehpatchsize - H)/2)
    resw = w%W
    numw = int(w/W)
    dw = int(resw/(numw+1))
    
    crops = []
    
    for i in range(numw-2):
        hi1, hi2, hi3 = dh, dh + H, dh + 2*H
        wi1 = i*W + (i+1)*dw
        wi2 = (i+1)*W + (i+2)*dw
        wi3 = (i+2)*W + (i+3)*dw
        crop1 = img[hi1:hi1+H, wi1:wi1+W]
        crop2 = img[hi2:hi2+H, wi2:wi2+W]
        crop3 = img[hi3:hi3+H, wi3:wi3+W]
        # print(img.shape)
        # print(crop1.shape, crop2.shape, crop3.shape)
        crop1float = skimage.img_as_float(crop1)
        crop2float = skimage.img_as_float(crop2)
        crop3float = skimage.img_as_float(crop3)
        wavename = waveletalg[4]
        denoise_crop1 = denoise_wavelet(image=crop1, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        denoise_crop2 = denoise_wavelet(image=crop2, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        denoise_crop3 = denoise_wavelet(image=crop3, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        res1float = crop1float - denoise_crop1
        res2float = crop2float - denoise_crop2
        res3float = crop3float - denoise_crop3
        res1 = ( (res1float - np.min(res1float))/(np.max(res1float) - np.min(res1float)) ) * 256
        res2 = ( (res2float - np.min(res2float))/(np.max(res2float) - np.min(res2float)) ) * 256
        res3 = ( (res3float - np.min(res3float))/(np.max(res3float) - np.min(res3float)) ) * 256
        res = np.stack(arrays=(res1, res2, res3), axis=2)

        crops.append(res)

    return crops
    

def imagepatchingsprd(imggray):
    H, W = 224, 224
    h, w, c = imggray.shape
    img = imggray[:, :, 0]
    h, w = img.shape
    threewpatchsize = int(w/3)
    dw = int((threewpatchsize - W)/2)
    resh = h%H
    numh = int(h/H)
    dh = int(resh/(numh+1))
    # print(h, w)
    crops = []
    
    for i in range(numh):
        wi1, wi2, wi3 = dw, dw + W, dw + 2*W
        hi = i*H + (i+1)*dh
        
        crop1 = img[hi:hi+H, wi1:wi1+W]
        crop2 = img[hi:hi+H, wi2:wi2+W]
        crop3 = img[hi:hi+H, wi3:wi3+W]
        # print(crop1.shape)
        crop1float = skimage.img_as_float(crop1)
        crop2float = skimage.img_as_float(crop2)
        crop3float = skimage.img_as_float(crop3)
        wavename = waveletalg[4]
        denoise_crop1 = denoise_wavelet(image=crop1, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        denoise_crop2 = denoise_wavelet(image=crop2, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        denoise_crop3 = denoise_wavelet(image=crop3, wavelet=wavename, mode='soft', wavelet_levels=2, method='BayesShrink', rescale_sigma=True)
        res1float = crop1float - denoise_crop1
        res2float = crop2float - denoise_crop2
        res3float = crop3float - denoise_crop3
        res1 = ( (res1float - np.min(res1float))/(np.max(res1float) - np.min(res1float)) ) * 256
        res2 = ( (res2float - np.min(res2float))/(np.max(res2float) - np.min(res2float)) ) * 256
        res3 = ( (res3float - np.min(res3float))/(np.max(res3float) - np.min(res3float)) ) * 256
        res = np.stack(arrays=(res1, res2, res3), axis=2)

        crops.append(res)

    return crops


def createdir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print("it already exist")


def save_residual(srcpath, trgpath):
    imgfolders = os.listdir(srcpath)
    try:
        imgfolders.remove('.DS_Store')
    except Exception as e:
        print(f"there is no .DS_Store folder")

    for folder in imgfolders:
        i=0
        srcimgfolder = os.path.join(srcpath, folder)
        trgimgfolder = os.path.join(trgpath, folder)
        imglist = os.listdir(srcimgfolder)

        createdir(trgimgfolder)

        try:
            imglist.remove('.DS_Store')
        except Exception as e:
            print(f"there is no .DS_Store folder")

        for imgname in imglist:
            imgpath = os.path.join(srcimgfolder, imgname)
            img = cv2.imread(imgpath)
            if folder == 'Spreader':
                crops = imagepatchingsprd(img)
            else:
                crops = imagepatching(img)
            for crop in crops:
                trgimgpath = os.path.join(trgimgfolder, f'crop_{i}.png')
                # cv2.imwrite(trgimgpath, crop)
                io.imsave(trgimgpath, crop)
                i+=1
    

def image_remove_text(img_path: str):
    
    img = cv2.imread(filename=img_path)

    h, w, c = img.shape
    top_cut = 85
    bottom_cut = 80
    crop_img = img[top_cut:h - bottom_cut, :, :]
    return crop_img


def main():
    # video_frame_extract(video_path=cfg.paths['videos'], target_path=cfg.paths['rawimage'], num_frame=20)
    # residual_img(images_path=cfg.paths['rawimage'], target_path=cfg.paths['libbherr'])
    # srcvideo = os.path.join(os.getcwd(), 'data', 'iframes')
    # imgpath = os.path.join(srcvideo, 'Gantry Travel 1', '2022-09-07_090455_Gantry Travel 1_b189154_1_0.jpeg')
    # img = cv2.imread(imgpath)
    # imagepatching(img)

    src_path = os.path.join(os.getcwd(), 'data', 'iframes')
    trg_path = os.path.join(os.getcwd(), 'data', 'residuals')
    save_residual(srcpath=src_path, trgpath=trg_path)
    # img = cv2.imread(os.path.join(src_path, 'Spreader', '2022-09-07_090540_Spreader_b188206_1_0.jpeg'))
    # imagepatchingsprd(img)

    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        imgpath = os.path.join(trg_path, 'Truck TTR', f'crop_{i}.png')
        img = cv2.imread(imgpath)
        for j in range(3):
            axs[i, j].imshow(img[:, :, j], cmap='gray')
            axs[i, j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    



if __name__ == '__main__':
    main()