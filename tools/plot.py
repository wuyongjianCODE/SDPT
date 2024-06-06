import matplotlib.pyplot as plt
import os
from skimage import io,transform
def print_ALL_with_metreics_withCOMPARE():
    fid=0
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submits=['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',#GT
        "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_continue/val2023-03-09 20:05:52_150623/",#22
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:14:07_328440/',#35
             '/data1/wyj/M/samples/PRM/YOLOX/val_vlplm/',#33
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:06:29_049792/',# '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_275/val2023-03-09 19:02:29_255141/',#17
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:27:55_874946/',#413
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sn/val2023-03-09 21:20:14_418623/',#336
           "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",#414
             '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/fullsup/',#full44

    ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue
        allpic = [17,178,231,273,282]
        allpic = [178, 231]
            #print(dices)
    # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
           # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        if image_id not in allpic:
            continue
        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.05,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=-0.15)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training0():
    fid=0
    submits = ['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',  # GT
                '/data1/wyj/M/samples/PRM/YOLOX/val_sc_json/',#sc
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_30_ckpt', #'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_40_ckpt',  # 359
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_70_ckpt',  # 404
                "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",  # 414
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_150_ckpt',  # 414
                ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue

        allpic = [178, 231]
        allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        allpic = [204,]
        if image_id not in allpic:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
def print_self_training():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
               '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
                '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
             '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
                '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
                ]
    for image_id in os.listdir('ORI_WITH_BOX')[16:]:
        allpic = [3120,8317,2367,3183,8348,2535,8496,3150,2586]
        allpic = [3183, 2535, 8496]
        ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
              8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
              8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic:
            if 'IMG_{}'.format(picid) in image_id:
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 4
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.5*INCH, 1.33*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                if fid<3 and iid==0:
                    im_method=transform.resize(im_method,(1088,800,3))
                elif fid==3 :#and iid==0:
                    im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:-10,:,:])
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    map=io.imread('TOSHOW/ALLcom.png')
    io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_sup():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX')[:]:
        allpic = ['000000020107','000000008629','000000010092','000000013597','000000026465',
                  '000000003255','000000025560','000000031248','000000023034','000000016439',
                  '000000029397','000000021167','000000015254','000000009769','000000025139'
                  ]
        allpic_batch2=[
                  '000000029397','000000021167','000000015254','000000009769',#'000000025139',
                  ]
        allpic_batch2=[
                  '000000020107','000000013597','000000023034','000000009769',
                  ]
        ranks={allpic[0]:[0,5,6,7,8,2],allpic[1]:[0,2,5,7,8,6],allpic[2]:[0,7,4,5,8,3],allpic[3]:[0,1,3,7,8,6,],allpic[4]:[0,8,1,7,11,16],
               allpic[5]:[0,7,9,15,10,13],allpic[6]:[0,7,9,10,15,1],allpic[7]:[0,4,6,7,10,9],allpic[8]:[0,8,9,7,12,5],allpic[9]:[0,7,10,12,14,1],
               allpic[10]: [0,8,6,14,16,3],allpic[11]: [0,8,16,12,9,1],allpic[12]: [0,4,9,15,16,3],allpic[13]: [0,8,12,14,6,1],allpic[14]: [0,8,4,5,16,3],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomVOC1.png')
    map=io.imread('TOSHOW/ALLcomVOC1.png')
    io.imsave('TOSHOW/ALLcomVOC1.png',map[:-200,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvis():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in BATCH_NAMES:#os.listdir('ORI_WITH_BOX')[:]:
        allpic = ['000000020107','000000008629','000000010092','000000013597','000000026465',
                  '000000003255','000000025560','000000031248','000000023034','000000016439',
                  '000000029397','000000021167','000000015254','000000009769','000000025139'
                  ]
        allpic_batch2=[
                  '000000029397','000000021167','000000015254','000000009769',#'000000025139',
                  ]
        allpic_batch2=[
                  '000000025560','000000029397','000000021167','000000015254',
                  ]
        ranks={allpic[0]:[0,5,6,7,8,2],allpic[1]:[0,2,5,7,8,6],allpic[2]:[0,7,4,5,8,3],allpic[3]:[0,1,3,7,8,6,],allpic[4]:[0,8,1,7,11,16],
               allpic[5]:[0,7,9,15,10,13],allpic[6]:[0,7,9,10,15,1],allpic[7]:[0,4,6,7,10,9],allpic[8]:[0,8,9,7,12,5],allpic[9]:[0,7,10,12,14,1],
               allpic[10]: [0,8,6,14,16,3],allpic[11]: [0,8,16,12,9,1],allpic[12]: [0,4,9,15,16,3],allpic[13]: [0,8,12,14,6,1],allpic[14]: [0,8,4,5,16,3],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomlv.png')
    map=io.imread('TOSHOW/ALLcomlv.png')
    io.imsave('TOSHOW/ALLcomlv.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOC():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX')[:]:
        allpic = ['000000020107','000000008629','000000010092','000000013597','000000026465',
                  '000000003255','000000025560','000000031248','000000023034','000000016439',
                  '000000029397','000000021167','000000015254','000000009769','000000025139',
                  '2008_003430','2009_000562','2007_006260','2011_000882','2011_002031',
                  '2008_007433','2011_003545','2009_000142','2007_009527',
                  ]
        allpic_batch2=[
            '2008_003430', '2009_000562', '2007_006260', '2011_000882', '2011_002031',
            '2008_007433', '2011_003545', '2009_000142', '2007_009527',
                  ]
        allpic_batch2=[
            '2008_003430','2011_003545', '2009_000142', '2007_009527',
                  ]
        ranks={allpic[0]:[0,5,6,7,8,2],allpic[1]:[0,2,5,7,8,6],allpic[2]:[0,7,4,5,8,3],allpic[3]:[0,1,3,7,8,6,],allpic[4]:[0,8,1,7,11,16],
               allpic[5]:[0,7,9,15,10,13],allpic[6]:[0,7,9,10,15,1],allpic[7]:[0,4,6,7,10,9],allpic[8]:[0,8,9,7,12,5],allpic[9]:[0,7,10,12,14,1],
               allpic[10]: [0,8,6,14,16,3],allpic[11]: [0,8,16,12,9,1],allpic[12]: [0,4,9,15,16,3],allpic[13]: [0,8,12,14,6,1],allpic[14]: [0,8,4,5,16,3],
               allpic[15]: [0, 7,6,5,4,3],allpic[16]: [0,6,7,8,12,3],allpic[17]: [0,16,6,8,3,1],allpic[18]: [0,3,12,8,6,1],allpic[19]: [0,7,8,12,16,1],
               allpic[20]: [0, 1,16,4,7,8],allpic[21]: [0,16,5,8,4,1],allpic[22]: [0,16,6,7,8,1],allpic[23]: [0,7,8,16,12,1],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.47*INCH, 0.9*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:-10,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomVOC.png')
    map=io.imread('TOSHOW/ALLcomVOC.png')
    io.imsave('TOSHOW/ALLcomVOC.png',map[:-200,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
def print_self_training_ALLSHOW():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX'):
        # allpic = [3120,8317,2367,3183,8348,2535,8496,3150,2586]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        # NEEDTOSHOW=False
        # for picid in allpic:
        #     if 'IMG_{}'.format(picid) in image_id:
        #         NEEDTOSHOW=True
        #         SHORTID=picid
        #         rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 18
        if fid % H == 0:
            plt.close()
        COL_NUMS = 17
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
def print_self_training_ALLSHOW_VOC():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX'):
        if '000000' in image_id:
            continue
        # allpic = [3120,8317,2367,3183,8348,2535,8496,3150,2586]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        # NEEDTOSHOW=False
        # for picid in allpic:
        #     if 'IMG_{}'.format(picid) in image_id:
        #         NEEDTOSHOW=True
        #         SHORTID=picid
        #         rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 18
        if fid % H == 0:
            plt.close()
        COL_NUMS = 17
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcomVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomVOC.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
# print_self_training_ALLSHOW_VOC()
# print_self_training_supVOC()
print_self_training_suplvis()