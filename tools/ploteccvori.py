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
def print_self_training_supVOCPLOT():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
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
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        # allpic = ['2007_0005490','2008_003430','2009_0001873','2008_003447','2010_003390',
        #           '2009_004118','2013_001019','2012_000176',
        #
        #           ]
        # allpic_batch2=[
        #           '2007_000549','000000021167','000000015254','000000009769',#'000000025139',
        #           ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        # ranks={allpic[0]:[1,5,6,7,8,9,15,28,21,4],allpic[1]:[1,6,7,8,30,22,28,14,5],allpic[2]:[1,34,36,35,19,23,25,9,5,20],allpic[3]:[1,31,33,38,23,24,19,3,34,5],allpic[4]:[1,6,8,20,25,26,27,33,22,30],
        #        allpic[5]:[1,20,21,27,29,9,10,2,34,3],allpic[6]:[1,41,36,40,34,32,28,2,3,9],allpic[7]:[1,39,41,37,31,26,23,18,4,34],allpic[8]:[1,8,10,17,24,26,28,46,2,3]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=True
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[iid]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCVVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCVVOC.png')
    # map=io.imread('TOSHOW/ALLcomlv.png')
    # io.imsave('TOSHOW/ALLcomlv.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOCPLOT2():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
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
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                  '2009_004118','2011_001019','2011_000176','2012_1111111',

                  ]
        allpic_batch2=['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                 '2011_000176',

                  ]
        ranks={allpic[0]:[1,41,40,39,38,34,29,28,4,5],allpic[1]:[1,6,7,8,33,30,22,28,5,14],allpic[2]:[1,38,39,36,41,19,20,2,4,34],allpic[3]:[1,31,33,38,23,24,19,3,34,5],allpic[4]:[1,6,8,20,25,26,27,33,22,30],
               allpic[5]:[1,20,21,27,29,9,10,2,34,3],allpic[6]:[1,41,36,40,34,32,28,2,3,9],allpic[7]:[1,39,41,37,31,26,23,18,4,34],allpic[8]:[1,8,10,17,24,26,28,46,2,3]}
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
        H = 7
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits[:10]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]-1]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480, 600, 3))
                plt.imshow(im_method[:,:,:])

                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCVVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCVVOC.png')
    map=io.imread('TOSHOW/ALLcomECCVVOC.png')
    io.imsave('TOSHOW/ALLcomECCVVOC.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOCPLOT2_COMB():
    fid=6
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
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
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                  '2009_004118','2011_001019','2011_000176','2012_1111111',

                  ]
        allpic_batch2=['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                 '2011_000176',

                  ]
        ranks={allpic[0]:[1,41,40,39,38,34,29,28,4,5],allpic[1]:[1,6,7,8,33,30,22,28,5,14],allpic[2]:[1,38,39,36,41,19,20,2,4,34],allpic[3]:[1,31,33,38,23,24,19,3,34,5],allpic[4]:[1,6,8,20,25,26,27,33,22,30],
               allpic[5]:[1,20,21,27,29,9,10,2,34,3],allpic[6]:[1,41,36,40,34,32,28,2,3,9],allpic[7]:[1,39,41,37,31,26,23,18,4,34],allpic[8]:[1,8,10,17,24,26,28,46,2,3]}
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
        H = 13
        # if fid % H == 0:
        #     plt.close()
        COL_NUMS = 10#len(submits)
        # INCH = 20
        # fig = plt.gcf()
        # fig.set_size_inches(1.8*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits[:10]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]-1]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480, 600, 3))
                plt.imshow(im_method[:,:,:])

                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvis():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000016439',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000029397','000000021167','000000015254','000000009769',#'000000025139',
                  ]
        allpic_batch2=[
                  '000000025560','000000029397','000000021167','000000015254',
                  ]
        ranks={allpic[0]:[1,5,6,7,8,9,15,28,21,4],allpic[1]:[1,7,8,9,10,21,20,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,24,34,46,51,2,3],allpic[4]:[1,9,11,12,16,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[0,11,15,20,28,37,46,4,2,3],
               allpic[10]: [1,39,28,23,22,20,17,7,46,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=True
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[iid]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    # map=io.imread('TOSHOW/ALLcomlv.png')
    # io.imsave('TOSHOW/ALLcomlv.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvisPLOT():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000002592',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000012062','000000029397','000000017714','000000029187','000000018575','000000026941',
                  ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        ranks={allpic[0]:[1,5,6,7,8,9,28,15,21,4],allpic[1]:[1,7,8,9,10,21,24,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,34,46,51,2,3],allpic[4]:[1,12,11,16,9,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[1,37,11,20,15,28,46,4,2,3],
               allpic[10]: [1,28,39,23,22,20,17,46,7,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        rerank=[0,1,2,3,6,7,4,5,8,9]
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
        H = 7
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        ONE_TIME_H=0
        for iid,submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            # try:
            im_method=io.imread(submits[rank[rerank[iid]]-1]+'/'+image_id)[:,:,:]
            h,w,c=im_method.shape
            # if fid<3 and iid==0:
            #     im_method=transform.resize(im_method,(1088,800,3))
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
            if ONE_TIME_H==0:
                ONE_TIME_H=h*(800/w)
            im_method = transform.resize(im_method, (480,600, 3))
            plt.imshow(im_method[:,:,:])
                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            # except:
            #     print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvisPLOT_COMB():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000002592',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000012062','000000029397','000000017714','000000029187','000000018575','000000026941',
                  ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        ranks={allpic[0]:[1,5,6,7,8,9,28,15,21,4],allpic[1]:[1,7,8,9,10,21,24,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,34,46,51,2,3],allpic[4]:[1,12,11,16,9,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[1,37,11,20,15,28,46,4,2,3],
               allpic[10]: [1,28,39,23,22,20,17,46,7,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        rerank=[0,1,2,3,6,7,4,5,8,9]
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
        H = 13
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,1*INCH*13/7)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        ONE_TIME_H=0
        for iid,submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            # try:
            im_method=io.imread(submits[rank[rerank[iid]]-1]+'/'+image_id)[:,:,:]
            h,w,c=im_method.shape
            # if fid<3 and iid==0:
            #     im_method=transform.resize(im_method,(1088,800,3))
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
            if ONE_TIME_H==0:
                ONE_TIME_H=h*(800/w)
            im_method = transform.resize(im_method, (480,600, 3))
            plt.imshow(im_method[:,:,:])
                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            # except:
            #     print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                # plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.savefig('TOSHOW/ALLcomECCV.png')
    # map=io.imread('TOSHOW/ALLcomECCV.png')
    # io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
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
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-400,:,:])
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
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "OUTPUT1/SSNS",#1
               "OUTPUT1/ORI0.26811",#2
               "OUTPUT1/ORI0.25247",#3
               "OUTPUT1/ORI0.30384",#4
               "OUTPUT1/SOP1",#5
               "OUTPUT1/ORI0.11968",#PSM6
               "OUTPUT1/ORI0.18027",#cutler7
               "OUTPUT1/ORI0.33063",#vlplm8
               "OUTPUT1/VLDET",#9
               "OUTPUT1/ORI0.22036",#10
               "OUTPUT1/MICCAI",
               "OUTPUT1/fullsup",
               "OUTPUT1/OURS",#"OUTPUT1/ORI0.42482",
               "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        if image_id not in allpic:
            continue
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
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
            plt.savefig('TOSHOWtmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom.png')
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
def print_self_training_ALLSHOW_CONSEP():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "OUTPUT2/0.25036",#1
               "OUTPUT2/0.17738",#2
               "OUTPUT2/0.19671",#3
               "OUTPUT2/0.22541",#4
               "OUTPUT2/ORI0.22912",#5
               "OUTPUT2/ORI0.12272",#PSM6
               "OUTPUT2/ORI0.09627",#cutler7
               "OUTPUT2/0.15668",#vlplm8
               "OUTPUT2/0.11232",#9
               "OUTPUT2/0.13487",#10
               "OUTPUT2/0.33675",
               "OUTPUT2/0.36208",
               "OUTPUT2/0.38245",#"OUTPUT1/ORI0.42482",
               "OUTPUT2/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        # allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
        #           "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
        #           "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
        #           "000000000458.jpg","000000000460.jpg",]
        # if image_id not in allpic:
        #     continue
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
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
            plt.savefig('TOSHOWtmi/ALLcom{}_consep.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom_consep.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
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
import numpy as np
import numpy as np
def is_bbox_color(color):
    # 判断像素颜色是否满足边框颜色条件
    return color[0] >240 and color[2] >240 and color[1] > 200

def find_bboxes(image):
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 创建一个空列表来存储所有bbox的坐标
    bboxes = []

    # 遍历图像的每个像素
    for y in range(height):
        for x in range(width):
            # 检查当前像素是否在图像边界内
            if x < width and y < height:
                # 获取当前像素的颜色值
                color = image[y, x]

                # 检查当前像素是否为边框颜色
                if is_bbox_color(color):
                    # 找到了边框颜色，开始扩展bbox

                    # 初始化bbox的左上和右下坐标
                    top_left = (x, y)
                    bottom_right = (x, y)

                    # 扩展bbox的右下坐标
                    while x < width and is_bbox_color(image[y, x]):
                        bottom_right = (x, y)
                        x += 1

                    # 扩展bbox的右下坐标
                    while y < height and is_bbox_color(image[y, bottom_right[0]]):
                        bottom_right = (bottom_right[0], y)
                        y += 1

                    # 将bbox的左上和右下坐标添加到列表中
                    bboxes.append((top_left, bottom_right))

    # 返回所有bbox的坐标列表
    return bboxes
def print_self_training_COLORED_ALLSHOW():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT1/0.30131/",#1ssns 354
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2938111707351383/",#2 275
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.27047253330669924/",#3  262
               "/home/data/jy/GLIP/OUTPUT1/0.25163/",#4 292
               "/home/data/jy/GLIP/OUTPUT1/0.19769/",#5 235
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.22818376550457384/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT1/0.11955/",#cutler7 115
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2153719550268572/",#vlplm8 333
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.17205429054974897/",#9 173
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.08508564588325737/",#10 07
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.4148790568704443/",
        "/home/data/jy/GLIP/OUTPUT1/0.44415/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.41517412385919217/",

               # "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000186.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        if image_id not in allpic:
            continue
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
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
            plt.savefig('TOSHOWtmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOW_CONSEP():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT2/0.27752/",#1ssns 354
               "/home/data/jy/GLIP/OUTPUT2/0.22034/",#2 275
               "/home/data/jy/GLIP/OUTPUT2/0.15932/",#3  262
               "/home/data/jy/GLIP/OUTPUT2/0.23618/",#4 292
               "/home/data/jy/GLIP/OUTPUT2/0.12168/",#5 235
               "/home/data/jy/GLIP/OUTPUT2/0.21598/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT2/0.10464/",#cutler7 115
               "/home/data/jy/GLIP/OUTPUT2/0.19242/",#vlplm8 333
               "/home/data/jy/GLIP/OUTPUT2/0.12571/",#9 173
               "/home/data/jy/GLIP/OUTPUT2/0.04011/",#"/home/data/jy/GLIP/OUTPUT2/0.07466/",#10 07too good
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/OUTPUT2/0.34435/",
        "/home/data/jy/GLIP/OUTPUT2/0.35047/",
        "/home/data/jy/GLIP/OUTPUT2/0.47447/",


               # "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000053.jpg","000000000023.jpg",
                  "000000000003.jpg","000000000022.jpg","000000000017.jpg","000000000010.jpg","000000000034.jpg","000000000021.jpg","000000000030.jpg",
                  "000000000014.jpg","000000000024.jpg"]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        # if image_id not in allpic:
        #     continue
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
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
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
            plt.savefig('TOSHOWtmi/ALLcom{}_consep.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom_consep.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOW_FINALMAP():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT1/0.30131/",#1ssns 354
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2938111707351383/",#2 275
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.27047253330669924/",#3  262
               "/home/data/jy/GLIP/OUTPUT1/0.25163/",#4 292
               "/home/data/jy/GLIP/OUTPUT1/0.19769/",#5 235
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.22818376550457384/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT1/0.11955/",#cutler7 115
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2153719550268572/",#vlplm8 333
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.17205429054974897/",#9 173
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.08508564588325737/",#10 07
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.4148790568704443/",
        "/home/data/jy/GLIP/OUTPUT1/0.44415/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.41517412385919217/",

               # "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000186.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        allpic = ["000000000257.jpg","000000000463.jpg","000000000180.jpg","000000000023.jpg",]
        SHORTID=0
        if image_id not in allpic:
            continue
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
        H = 9
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 1.4*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.02,hspace=0.02)
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
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOWtmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT2/0.27752/",#1ssns 354
               "/home/data/jy/GLIP/OUTPUT2/0.22034/",#2 275
               "/home/data/jy/GLIP/OUTPUT2/0.15932/",#3  262
               "/home/data/jy/GLIP/OUTPUT2/0.23618/",#4 292
               "/home/data/jy/GLIP/OUTPUT2/0.12168/",#5 235
               "/home/data/jy/GLIP/OUTPUT2/0.21598/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT2/0.10464/",#cutler7 115
               "/home/data/jy/GLIP/OUTPUT2/0.19242/",#vlplm8 333
               "/home/data/jy/GLIP/OUTPUT2/0.12571/",#9 173
               "/home/data/jy/GLIP/OUTPUT2/0.04011/",#"/home/data/jy/GLIP/OUTPUT2/0.07466/",#10 07too good
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/OUTPUT2/0.34435/",
        "/home/data/jy/GLIP/OUTPUT2/0.47447/",
        "/home/data/jy/GLIP/OUTPUT2/0.35047/",

               # "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000053.jpg","000000000023.jpg",
                  "000000000003.jpg","000000000022.jpg","000000000017.jpg","000000000010.jpg","000000000034.jpg","000000000021.jpg","000000000030.jpg",
                  "000000000014.jpg","000000000024.jpg"]
        allpic = ["000000000017.jpg","000000000008.jpg",
                  "000000000025.jpg","000000000044.jpg"]
        allpic2 = [
                  "000000000025.jpg","000000000044.jpg"]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        swap=False
        if image_id not in allpic:
            continue
        if image_id in allpic2:
            swap=True
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
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png'])
        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijkmlnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            idx_used=idx
            if swap:
                if idx==13:
                    idx_used=12
                if idx==12:
                    idx_used = 13
            plt.subplot(H, COL_NUMS, idx_used + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.02,hspace=0.02)
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
            if fid == 8:
                plt.title('({})'.format(TITLE[iid]),y=-0.25,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOWtmi/ALLcom{}_consep.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/FINALMAP.png')
    map=io.imread('TOSHOWtmi/FINALMAP.png')
    io.imsave('TOSHOWtmi/FINALMAP.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_self_training_COLORED_ALLSHOW()
# print_self_training_COLORED_ALLSHOW_CONSEP()
# print_self_training_COLORED_ALLSHOW_FINALMAP()
# print_self_training_supVOC()
print_self_training_supVOCPLOT()
# print_self_training_suplvisPLOT_COMB()
# print_self_training_supVOCPLOT2_COMB()