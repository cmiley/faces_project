import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data.sampler import SubsetRandomSampler
from src.Datasets import *
from src.Model import *
import time
from torchvision import transforms

DATA_DIR = "../../../celeba-dataset/"
IMG_DIR = DATA_DIR + "img_align_celeba"
batch_size = 2**5
print_freq = 20


def show_bounding_box(sample):
    print(sample['image'].shape)
    fig, ax = plt.subplots()

    plt.imshow(sample['image'])
    # Create a Rectangle patch
    rect = patches.Rectangle(sample['origin'], sample['width'], sample['height'], linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    # #
    #
    # img = Image.fromarray(sample['image'])
    #
    # draw = ImageDraw.Draw(img)
    # draw.rectangle(sample['points'], outline='red')


def show_landmarks(image, landmarks):
    plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def check_file_count(data_dir):
    filecount = 0
    for root, dirs, files in os.walk(data_dir, topdown=False):
        filecount = filecount + len(files)

    print(filecount)


if __name__ == "__main__":
    # check_file_count(DATA_DIR)

    celeb_attr_df = pd.read_csv(DATA_DIR + "list_attr_celeba.csv")
    # celeb_bbox_df = pd.read_csv(DATA_DIR + "list_bbox_celeba.csv")
    celeb_landmarks_df = pd.read_csv(DATA_DIR + "list_landmarks_align_celeba.csv")
    eval_partition_df = pd.read_csv(DATA_DIR + "list_eval_partition.csv")

    # train_indices = range(162770)
    train_indices = [i for i in range(1000)]
    val_indices = [i for i in range(162770, 182637)]
    test_indices = [i for i in range(182637, 202598)]

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    attribs_ds = CelebAttributesDataset(DATA_DIR + "list_attr_celeba.csv", IMG_DIR, tf=transform)
    # bbox_ds = CelebBoundingDataset(DATA_DIR + "list_bbox_celeba.csv", IMG_DIR)
    # landmarks_ds = CelebLandmarksDataset(DATA_DIR + "list_landmarks_align_celeba.csv", IMG_DIR)

    train_loader = torch.utils.data.DataLoader(attribs_ds, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    net = AttributeNetwork(3, 40)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_values = []

    for epoch in range(5):  # loop over the dataset multiple times

        print('Epoch {}: \n\tStart: {}'.format(epoch+1, time.strftime('%H:%M:%S', time.localtime())))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            calc_loss = net.m_train(data, optimizer, criterion)

            loss_values.append(calc_loss)

            # print statistics
            running_loss += calc_loss
            if i % print_freq == (print_freq - 1):  # print every 20 mini-batches
                print('\tLoss: {}'.format(running_loss/print_freq))
                running_loss = 0.0

        print('\tEnd: {}'.format(time.strftime('%H:%M:%S', time.localtime())))

    print('Finished Training')
    plt.plot(loss_values)
    plt.ylabel('Loss')
    plt.show()

    # ds_test = landmarks_ds[1]
    # print(ds_test)
    #
    # test = celeb_landmarks_df.iloc[1]
    #
    # show_landmarks(ds_test['image'], ds_test['landmarks'])
    # # show_bounding_box(ds_test)
    #
    # print(test)
