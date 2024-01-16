from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
from matplotlib import pyplot as plt

global nf
nf = 1

def draw_img(p, title):
    global nf
    for_drawing = np.reshape(p, (28,28))
    plt.figure(nf)
    plt.imshow(for_drawing, cmap="gray") 
    plt.title(title)
    nf+=1


if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    
    print ("\nStarting a Restricted Boltzmann Machine n_h=500..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=40001, test=test_imgs)
    print('n_h=200')
    print(f'weight statistics: MAX:{rbm.weight_vh.max()}, MIN:{rbm.weight_vh.min()}')
    print(f'v bias statistics: MAX:{rbm.bias_v.max()}, MIN:{rbm.bias_v.min()}')
    print(f'h bias statistics: MAX:{rbm.bias_h.max()}, MIN:{rbm.bias_h.min()}')

    
    
    # v_0 = train_imgs[:3]
    # h_0_probs, h_0 = rbm.get_h_given_v(v_0)
    # v_1, _ = rbm.get_v_given_h(h_0)
    # for i in range(3):
    #     idx = (-h_0_probs[i]).argsort()[:25]
    #     viz_rf(weights=rbm.weight_vh[:,idx].reshape((rbm.image_size[0],rbm.image_size[1],-1)), it=i, grid=rbm.rf["grid"])


    # draw_img(train_imgs[0], 'Input 1')
    # draw_img(train_imgs[1], 'Input 2')
    # draw_img(train_imgs[2], 'Input 3')
    # for i in range(3):
    #     draw_img(v_1[i], f'Output {i+1}')
    plt.show()



    # ii = input('start a new boltzman machine?: ')
    # print ("\nStarting a Restricted Boltzmann Machine n_h=200..")

    # rbm_200 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=200,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=20
    # )
    
    # rbm_200.cd1(visible_trainset=train_imgs, n_iterations=40001)
    # # REMOVE the quit below once you start working on deep-belief net
    # print('n_h=200')
    # print(f'weight statistics: MAX:{rbm_200.weight_vh.max()}, MIN:{rbm_200.weight_vh.min()}')
    # print(f'v bias statistics: MAX:{rbm_200.bias_v.max()}, MIN:{rbm_200.bias_v.min()}')
    # print(f'h bias statistics: MAX:{rbm_200.bias_h.max()}, MIN:{rbm_200.bias_h.min()}')
    # plt.figure(2)
    # plt.plot(rbm_200.avg_hbias_update, label='hidden bias')
    # plt.plot(rbm_200.avg_vbias_update, label='visible bias')
    # plt.plot(rbm_200.avg_weight_update, label='weights')
    # plt.legend()
    # plt.title('Average update n_h=200')
    # plt.xlabel('Iteration')
    # plt.ylabel('Average value')
    
    # plt.figure(3)
    # plt.plot(np.linspace(0,40000,len(rbm.recon_loss)), rbm.recon_loss, label='n_h=500')
    # plt.plot(np.linspace(0,40000,len(rbm.recon_loss)), rbm_200.recon_loss, label='n_h=200')
    # plt.xlabel('Iteration')
    # plt.ylabel('Value')
    # plt.title('Reconstruction loss')
    # plt.legend()
    
    
    # plt.show()

    # quit()
    
    # ''' deep- belief net '''

    # print ("\nStarting a Deep Belief Net..")
    
    # dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
    #                     image_size=image_size,
    #                     n_labels=10,
    #                     batch_size=10
    # )
    
    # ''' greedy layer-wise training '''

    # dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
