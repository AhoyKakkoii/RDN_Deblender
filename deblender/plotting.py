import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #as gs


def make_plot(blended, true_x, true_y, gan_x, savedir, batch, d_x, d_y, d_blended, d_gan_x):
    """
    Plots paneled figure of preblended, blended and deblended galaxies.
    """
    for i in range(blended.shape[0]):
        fig = plt.Figure()

        psnr_x = 0.111 #= compare_psnr(im_test=gan_x[i], im_true=true_x[i])
        psnr_y = 0.222 #= compare_psnr(im_test=gan_y[i], im_true=true_y[i])
        psnr_x = np.around(psnr_x, decimals=2)
        psnr_y = np.around(psnr_y, decimals=2)

###        gs = GridSpec(2, 4)
        gs = GridSpec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
###        ax3 = fig.add_subplot(gs[0:2, 1:3])
        ax3 = fig.add_subplot(gs[0, 1])
###        ax4 = fig.add_subplot(gs[0, 3])
###        ax5 = fig.add_subplot(gs[1, 3])
        ax4 = fig.add_subplot(gs[0, 2])
        ax5 = fig.add_subplot(gs[1, 2])

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.axis('off')

        ax1.imshow(true_x[i])
        ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
        ax1.text(3., 75., str(d_x[i][0]>0), color='#FFFFFF')
        ax2.imshow(true_y[i])
        ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
        ax2.text(3., 75., str(d_y[i][0]>0), color='#FFFFFF')
        ax3.imshow(blended[i])
###        ax3.text(1.3, 4.4, r'Blended', color='#FFFFFF')
        ax3.text(3., 10., r'Blended', color='#FFFFFF')
        ax3.text(3., 75., str(d_blended[i][0]>0), color='#FFFFFF')
        ax4.imshow(gan_x[i])
        ax4.text(3., 10., r'Deblended 1', color='#FFFFFF')
        ax4.text(3., 75., str(d_gan_x[i][0]>0), color='#FFFFFF') #
#        ax5.imshow(gan_y[i])
#        ax5.text(3., 10., r'Deblended 2', color='#FFFFFF')
        #ax5.text(3., 75., str(psnr_y)+' dB', color='#FFFFFF') #

        fig.tight_layout(pad=0)
#        fig.subplots_adjust(wspace=0.06,hspace=-0.42)
        fig.subplots_adjust(wspace=0.06,hspace=0.01)

        filename = os.path.join(savedir, 'test-{}-{}.png'.format(batch, i))
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def make_hist(blended, true_x, true_y, gan_x, savedir, batch):
    true_x, true_y, blended, gan_x = map(lambda x: x*255, [true_x, true_y, blended, gan_x])
    for i in range(blended.shape[0]):
        for j in range(blended.shape[-1]):
            data_x = true_x[i,:,:,j]
            data_x = data_x.flatten()
            plt.hist(data_x, bins=51)
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.title('batch-{}-{}-th-channel-{}-true-x'.format(batch, i, j))
            filename = os.path.join(savedir, 'batch-{}-{}-th-true-x-channel-{}'.format(batch, i, j))
            plt.savefig(filename)
            plt.close()

            data_y = true_y[i,:,:,j]
            data_y = data_y.flatten()
            plt.hist(data_y, bins=51)
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.title('batch-{}-{}-th-channel-{}-true-y'.format(batch, i, j))
            filename = os.path.join(savedir, 'batch-{}-{}-th-true-y-channel-{}'.format(batch, i, j))
            plt.savefig(filename)
            plt.close()

            data_blended = blended[i,:,:,j]
            data_blended = data_blended.flatten()
            plt.hist(data_blended, bins=51)
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.title('batch-{}-{}-th-channel-{}-blended'.format(batch, i, j))
            filename = os.path.join(savedir, 'batch-{}-{}-th-blended-channel-{}'.format(batch, i, j))
            plt.savefig(filename)
            plt.close()

            data_gan_x = gan_x[i,:,:,j]
            data_gan_x = data_gan_x.flatten()
            plt.hist(data_gan_x, bins=51)
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.title('batch-{}-{}-th-channel-{}-gan-x'.format(batch, i, j))
            filename = os.path.join(savedir, 'batch-{}-{}-th-gan-x-channel-{}'.format(batch, i, j))
            plt.savefig(filename)
            plt.close()
