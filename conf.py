import os, sys






data_root = os.path.join(os.getcwd(), 'data')

paths = dict(
    data=data_root, videos=os.path.join(data_root, 'Video Recordings'), 
    libbherr=os.path.join(data_root, 'libbherr'), libbherr1=os.path.join(data_root, 'libbherr1'), 
    rawimage=os.path.join(data_root, 'raw_images'), model=os.path.join(data_root, 'model'),
    residual=os.path.join(data_root, 'residuals')
)

# model architecture
model_architecture = dict(

)


def main():
    pass



if __name__ == '__main__':
    main()