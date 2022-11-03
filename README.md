# TensorFlow OOB Benchmarking
Inference (real time) with TensorFlow
## Prerequisites
1. TensorFlow
2. Scripts
```
pip install tensorflow/intel-tensorflow
git clone https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/extended-broad-product.git
# checkout the corresponding branch
cd extended-broad-product/TF/workload/PB/
```

## PB directly
```
./launch_benchmark.sh --model_path=/path/to/pb
```
|	Workload	|	PB Location	|
|	--------	|	--------	|
|	vgg19	|	ov/all_tf_models/classification/vgg/19/tf/vgg19.pb	|
|	vgg16	|	ov/all_tf_models/classification/vgg/16/tf/vgg16.pb	|
|	dilation	|	ov/Dilation/dilation.pb	|
|	nasnet-a-large-331	|	ov/all_tf_models/classification/nasnet/large/tf/nasnet-a-large-331.pb	|
|	yolo-v3	|	ov/all_tf_models/object_detection/yolo/yolo_v3/tf/yolo-v3.pb	|
|	person-vehicle-bike-detection-crossroad-yolov3-1020	|	ov/all_tf_models/Security/object_detection/crossroad/1020/tf/person-vehicle-bike-detection-crossroad-yolov3-1020.pb	|
|	person-vehicle-bike-detection-crossroad-yolov3-1024	|	ov/all_tf_models/Security/object_detection/crossroad/1024/tf/person-vehicle-bike-detection-crossroad-yolov3-1024.pb	|
|	resnet-152	|	ov/all_tf_models/classification/resnet/v1/152/tf/resnet-152.pb	|
|	resnet-v2-152	|	ov/all_tf_models/classification/resnet/v2/152/tf/resnet-v2-152.pb	|
|	EfficientDet-D6-1280x1280	|	dpg/EfficientDet/efficientdet-d6/EfficientDet-D6-1280x1280.pb	|
|	EfficientDet-D7-1536x1536	|	dpg/EfficientDet/efficientdet-d7/EfficientDet-D7-1536x1536.pb	|
|	openpose-pose	|	ov/all_tf_models/human_pose_estimation/openpose/pose/tf/openpose-pose.pb	|
|	yolo-v2	|	ov/all_tf_models/object_detection/yolo/yolo_v2/tf/yolo-v2.pb	|
|	yolo-v2-ava-sparse-70-0001	|	ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity70/yolo-v2-ava-sparse-70-0001.pb	|
|	yolo-v2-ava-sparse-35-0001	|	ov/all_tf_models/PublicCompressed/detection/YOLOv2/fp32_sparsity35/yolo-v2-ava-sparse-35-0001.pb	|
|	resnet-101	|	ov/all_tf_models/classification/resnet/v1/101/tf/resnet-101.pb	|
|	resnet-v2-101	|	ov/all_tf_models/classification/resnet/v2/101/tf/resnet-v2-101.pb	|
|	googlenet-v4	|	ov/all_tf_models/classification/googlenet/v4/tf/googlenet-v4.pb	|
|	darknet53	|	ov/all_tf_models/PublicInHouse/classification/darknet53/darknet53.pb	|
|	COVID-Net	|	oob/COVID-Net/COVID-Net.pb	|
|	EfficientDet-D5-1280x1280	|	dpg/EfficientDet/efficientdet-d5/EfficientDet-D5-1280x1280.pb	|
|	U-Net	|	mlp/unet/U-Net.pb	|
|	inception-resnet-v2	|	ov/all_tf_models/classification/inception-resnet/v2/tf/inception-resnet-v2.pb	|
|	densenet-161	|	ov/all_tf_models/classification/densenet/161/tf/densenet-161.pb	|
|	tiny_yolo_v1	|	ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v1_tiny/tf/tiny_yolo_v1.pb	|
|	icnet-camvid-ava-sparse-60-0001	|	ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws60/icnet-camvid-ava-sparse-60-0001.pb	|
|	icnet-camvid-ava-sparse-30-0001	|	ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws30/icnet-camvid-ava-sparse-30-0001.pb	|
|	icnet-camvid-ava-0001	|	ov/all_tf_models/PublicCompressed/semantic_segmentation/icnet-camvid-tf-ws00/icnet-camvid-ava-0001.pb	|
|	cpm-person	|	ov/all_tf_models/human_pose_estimation/cpm/person/tf/cpm-person.pb	|
|	resnet-v2-50	|	ov/all_tf_models/classification/resnet/v2/50/tf/224x224/resnet-v2-50.pb	|
|	ResNet-50_v1.5	|	mlp/ResNet50_v1_5/model_dir/ResNet-50_v1.5.pb	|
|	resnet-50	|	ov/all_tf_models/classification/resnet/v1/50/tf/official/resnet-50.pb	|
|	googlenet-v3	|	ov/all_tf_models/classification/googlenet/v3/tf/googlenet-v3.pb	|
|	ens3_adv_inception_v3	|	dpg/ens3_inception_v3/ens3_adv_inception_v3.pb	|
|	adv_inception_v3	|	dpg/adv_inception_v3/adv_inception_v3.pb	|
|	EfficientDet-D4-1024x1024	|	dpg/EfficientDet/efficientdet-d4/EfficientDet-D4-1024x1024.pb	|
|	darknet19	|	ov/all_tf_models/PublicInHouse/classification/darknet19/darknet19.pb	|
|	ssd_resnet34_300x300	|	ckpt/ssd-resnet34_300x300/ssd_resnet34_300x300.pb	|
|	3DUNet	|	mlp/3D-Unet/3DUNet.pb	|
|	unet-3d-origin	|	ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/origin/tf/unet-3d-origin.pb	|
|	ctpn	|	ov/all_tf_models/text_detection/ctpn/tf/ctpn.pb	|
|	DynamicMemory	|	oob/checkpoint_dynamic_memory_network/DynamicMemory.pb	|
|	yolo-v2-tiny-ava-sparse-60-0001	|	ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity60/yolo-v2-tiny-ava-sparse-60-0001.pb	|
|	yolo-v2-tiny-ava-sparse-30-0001	|	ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity30/yolo-v2-tiny-ava-sparse-30-0001.pb	|
|	yolo-v2-tiny-ava-0001	|	ov/all_tf_models/PublicCompressed/detection/tinyYOLOv2/fp32_sparsity00/yolo-v2-tiny-ava-0001.pb	|
|	tiny_yolo_v2	|	ov/all_tf_models/PublicInHouse/object_detection/common/yolo/v2_tiny/tf/tiny_yolo_v2.pb	|
|	densenet-169	|	ov/all_tf_models/classification/densenet/169/tf/densenet-169.pb	|
|	ACGAN	|	oob/oob_gan_models/ACGAN.pb	|
|	WGAN	|	mlp/oob_gan_models/WGAN.pb	|
|	LSGAN	|	mlp/oob_gan_models/LSGAN.pb	|
|	GAN	|	mlp/oob_gan_models/GAN.pb	|
|	WGAN_GP	|	oob/oob_gan_models/WGAN_GP.pb	|
|	CGAN	|	oob/oob_gan_models/CGAN.pb	|
|	infoGAN	|	oob/oob_gan_models/infoGAN.pb	|
|	EfficientDet-D3-896x896	|	dpg/EfficientDet/efficientdet-d3/EfficientDet-D3-896x896.pb	|
|	i3d-rgb	|	ov/all_tf_models/action_recognition/i3d/rgb/tf/i3d-rgb.pb	|
|	i3d-flow	|	ov/all_tf_models/action_recognition/i3d/flow/tf/i3d-flow.pb	|
|	gmcnn-places2	|	ov/all_tf_models/image_inpainting/gmcnn/tf/gmcnn-places2.pb	|
|	ava-face-recognition-3.0.0	|	ov/all_tf_models/Security/feature_extraction/ava/tf/ava-face-recognition-3.0.0.pb	|
|	googlenet-v2	|	ov/all_tf_models/classification/googlenet/v2/tf/googlenet-v2.pb	|
|	yolo-v2-tiny-vehicle-detection-0001	|	ov/all_tf_models/Security/object_detection/barrier/yolo/yolo-v2-tiny-vehicle-detection-0001/tf/yolo-v2-tiny-vehicle-detection-0001.pb	|
|	icv-emotions-recognition-0002	|	ov/all_tf_models/Retail/object_attributes/emotions_recognition/0002/tf/icv-emotions-recognition-0002.pb	|
|	EfficientDet-D2-768x768	|	dpg/EfficientDet/efficientdet-d2/EfficientDet-D2-768x768.pb	|
|	densenet-121	|	ov/all_tf_models/classification/densenet/121/tf/densenet-121.pb	|
|	learning-to-see-in-the-dark-sony	|	ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Sony/learning-to-see-in-the-dark-sony.pb	|
|	learning-to-see-in-the-dark-fuji	|	ov/all_tf_models/IntelLabs/LearningToSeeInTheDark/Fuji/learning-to-see-in-the-dark-fuji.pb	|
|	EfficientDet-D1-640x640	|	dpg/EfficientDet/efficientdet-d1/EfficientDet-D1-640x640.pb	|
|	BEGAN	|	oob/oob_gan_models/BEGAN.pb	|
|	EBGAN	|	oob/oob_gan_models/EBGAN.pb	|
|	googlenet-v1	|	ov/all_tf_models/classification/googlenet/v1/tf/googlenet-v1.pb	|
|	nasnet-a-mobile-224	|	ov/all_tf_models/classification/nasnet/mobile/tf/nasnet-a-mobile-224.pb	|
|	3d-pose-baseline	|	ov/all_tf_models/human_pose_estimation/3d-pose-baseline/tf/3d-pose-baseline.pb	|
|	ava-person-vehicle-detection-stage2-2.0.0	|	ov/all_tf_models/Security/object_detection/common/ava/stage2/tf/ava-person-vehicle-detection-stage2-2.0.0.pb	|
|	EfficientDet-D0-512x512	|	dpg/EfficientDet/efficientdet-d0/EfficientDet-D0-512x512.pb	|
|	image-retrieval-0001	|	ov/all_tf_models/classification/image-retrieval-0001/image-retrieval-0001.pb	|
|	deeplabv3	|	ov/all_tf_models/semantic_segmentation/deeplab/v3/deeplabv3.pb	|
|	rmnet_ssd	|	ov/all_tf_models/Retail/action_detection/pedestrian/rmnet_ssd/0028_tf/tf/rmnet_ssd.pb	|
|	srgan	|	ov/all_tf_models/image_processing/srgan/tf/srgan.pb	|
|	SqueezeNet	|	mlp/SqueezeNet-tf/SqueezeNet.pb	|
|	vehicle-attributes-barrier-0103	|	ov/all_tf_models/object_attributes/vehicle_attributes/tf/vehicle-attributes-barrier-0103.pb	|
|	TCN	|	ov/all_tf_models/PublicInHouse/sequence_modelling/tcn/tf/TCN.pb	|
|	intel-labs-nonlocal-dehazing	|	ov/all_tf_models/IntelLabs/FastImageProcessing/NonlocalDehazing/intel-labs-nonlocal-dehazing.pb	|
|	BERT_LARGE	|	mlp/BERT_LARGE/BERT_LARGE.pb	|
|	NetVLAD	|	dpg/NetVLAD/NetVLAD.pb	|
|	BERT_BASE	|	mlp/BERT_BASE/BERT_BASE.pb	|
|	bert-base-uncased_L-12_H-768_A-12	|	ov/all_tf_models/language_representation/bert/base/uncased_L-12_H-768_A-12/tf/bert-base-uncased_L-12_H-768_A-12.pb	|
|	faster_rcnn_nas_coco_2018_01_28	|	ckpt/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_coco_2018_01_28.pb	|
|	faster_rcnn_nas_lowproposals_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_nas_lowproposals_coco/tf/faster_rcnn_nas_lowproposals_coco.pb	|
|	faster_rcnn_resnet101_fgvc	|	ckpt/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc_2018_07_19/faster_rcnn_resnet101_fgvc.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_oid	|	mlp/faster_rcnn_inception_resnet_v2_atrous_oid/faster_rcnn_inception_resnet_v2_atrous_oid.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid	|	mlp/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid.pb	|
|	faster_rcnn_resnet50_fgvc	|	ckpt/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc_2018_07_19/faster_rcnn_resnet50_fgvc.pb	|
|	mask_rcnn_inception_resnet_v2_atrous_coco	|	ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_resnet_v2_atrous_coco/tf/mask_rcnn_inception_resnet_v2_atrous_coco.pb	|
|	YOLOv4	|	mlp/yolov4/YOLOv4.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco	|	ckpt/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco.pb	|
|	Transformer-LT	|	mlp/transformer_lt_official_fp32_pretrained_model/graph/Transformer-LT.pb	|
|	faster_rcnn_inception_resnet_v2_atrous_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco/tf/faster_rcnn_inception_resnet_v2_atrous_coco.pb	|
|	ssd_resnet_101_fpn_oidv4	|	ckpt/ssd_resnet101_v1_fpn/ssd_resnet_101_fpn_oidv4.pb	|
|	mask_rcnn_resnet101_atrous_coco	|	ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet101_atrous_coco/tf/mask_rcnn_resnet101_atrous_coco.pb	|
|	R-FCN	|	dpg/R-FCN/rfcn_resnet101_coco_2018_01_28/R-FCN.pb	|
|	rfcn-resnet101-coco	|	ov/all_tf_models/object_detection/common/rfcn/rfcn_resnet101_coco/tf/rfcn-resnet101-coco.pb	|
|	faster_rcnn_resnet101_lowproposals_coco	|	ckpt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/faster_rcnn_resnet101_lowproposals_coco.pb	|
|	faster_rcnn_resnet101_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet101_coco/tf/faster_rcnn_resnet101_coco.pb	|
|	faster_rcnn_resnet101_snapshot_serengeti	|	mlp/faster_rcnn_resnet101_snapshot_serengeti/faster_rcnn_resnet101_snapshot_serengeti.pb	|
|	faster_rcnn_resnet101_kitti	|	ckpt/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti.pb	|
|	retinanet	|	ov/all_tf_models/object_detection/common/retinanet/tf/retinanet.pb	|
|	mask_rcnn_resnet50_atrous_coco	|	ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_resnet50_atrous_coco/tf/mask_rcnn_resnet50_atrous_coco.pb	|
|	SSD_ResNet50_V1_FPN_640x640_RetinaNet50	|	ckpt/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/SSD_ResNet50_V1_FPN_640x640_RetinaNet50.pb	|
|	ssd_resnet50_v1_fpn_coco	|	ov/all_tf_models/object_detection/common/ssd_resnet50/ssd_resnet50_v1_fpn_coco/tf/ssd_resnet50_v1_fpn_coco.pb	|
|	faster_rcnn_resnet50_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_coco/tf/faster_rcnn_resnet50_coco.pb	|
|	faster_rcnn_resnet50_lowproposals_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_resnet50_lowproposals_coco/tf/faster_rcnn_resnet50_lowproposals_coco.pb	|
|	HugeCTR	|	mlp/HugeCTR/HugeCTR.pb	|
|	inceptionv2_ssd	|	ov/all_tf_models/object_detection/common/ssd_inceptionv2/tf/inceptionv2_ssd.pb	|
|	ssd_inception_v2_coco	|	ckpt/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco_2018_01_28/ssd_inception_v2_coco.pb	|
|	unet-3d-isensee_2017	|	ov/all_tf_models/PublicInHouse/volumetric_segmentation/unet/3d/isensee_2017/tf/unet-3d-isensee_2017.pb	|
|	ssd_resnet34_1200x1200	|	mlp/ssd_resnet34_model/ssd_resnet34_1200x1200.pb	|
|	ssd_resnet34_fp32_1200x1200_pretrained_model	|	dpg/SSD-ResNet34_1200x1200/ssd_resnet34_fp32_1200x1200_pretrained_model.pb	|
|	mask_rcnn_inception_v2_coco	|	ov/all_tf_models/instance_segmentation/mask_rcnn/mask_rcnn_inception_v2_coco/tf/mask_rcnn_inception_v2_coco.pb	|
|	HierAtteNet	|	oob/checkpoint_hier_atten_title/text_hier_atten_title_desc_checkpoint_MHA/HierAtteNet.pb	|
|	Seq2seqAttn	|	oob/Seq2seqAttn/Seq2seqAttn.pb	|
|	EntityNet	|	oob/checkpoint_entity_network2/EntityNet.pb	|
|	faster_rcnn_inception_v2_coco	|	ov/all_tf_models/object_detection/common/faster_rcnn/faster_rcnn_inception_v2_coco/tf/faster_rcnn_inception_v2_coco.pb	|
|	DRAGAN	|	oob/oob_gan_models/DRAGAN.pb	|
|	ALBERT	|	dpg/ALBERT/ALBERT.pb	|
|	SphereFace	|	dpg/SphereFace/SphereFace.pb	|
|	CBAM	|	mlp/CBAM/CBAM.pb	|
|	DSSD_12	|	ov/all_tf_models/PublicInHouse/object_detection/common/dssd/DSSD_12/tf/DSSD_12.pb	|
|	ssd_mobilenet_v1_0.75_depth_300x300_coco	|	mlp/ssd_mobilenet_v1_0.75_depth_300x300_coco/ssd_mobilenet_v1_0.75_depth_300x300_coco.pb	|
|	ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco	|	mlp/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco.pb	|
|	squeezenet1_1	|	ov/all_tf_models/classification/squeezenet/1.1/tf/squeezenet1_1.pb	|
|	wavenet	|	dpg/wavenet/wavenet.pb	|
|	DRAW	|	dpg/DRAW/DRAW.pb	|
|	NTM-One-Shot	|	dpg/NTM-One-Shot/model/NTM-One-Shot.pb	|
|	vehicle-license-plate-detection-barrier-0106	|	ov/vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.pb	|
|	vehicle-license-plate-detection-barrier-0123	|	ov/all_tf_models/object_detection/barrier/tf/0123/vehicle-license-plate-detection-barrier-0123.pb	|
|	NCF-1B	|	dpg/ncf_trained_movielens_1m/NCF-1B.pb	|
|	GraphSage	|	mlp/GraphSage/GraphSage.pb	|
|	key-value-memory-networks	|	dpg/key-value-memory-networks/key-value-memory-networks.pb	|
|	deeplab	|	mlp/deeplabv3_mnv2_cityscapes_train/deeplab.pb	|
|	ResNext_101	|	dpg/ResNext_101/ResNext_101.pb	|
|	ResNext_50	|	dpg/ResNext_50/ResNext_50.pb	|
|	TextRCNN	|	oob/TextRCNN/TextRCNN.pb	|


## PB with input and output
Note that the * model is without optimize, the "export DISABLE_OPTIMIZE=1" is necessary.
```
export INPUT_OUTPUT_INDEED=1
./launch_benchmark.sh --model_name=WORKLOAD-NAME --model_path=/path/to/pb
```

|	Workload	|	PB Location	|
|	--------	|	--------	|
|	CRNN*	|	mlp/CRNN/crnn.pb	|
|	vggvox*	|	ov/all_tf_models/voice_recognition/vggvox/vggvox.pb	|
|	efficientnet-b0*	|	ov/all_tf_models/classification/efficientnet/b0/tf/efficientnet-b0.pb	|
|	efficientnet-b0_auto_aug*	|	ov/all_tf_models/classification/efficientnet/b0_auto_aug/tf/efficientnet-b0_auto_aug.pb	|
|	efficientnet-b5*	|	ov/all_tf_models/classification/efficientnet/b5/tf/efficientnet-b5.pb	|
|	efficientnet-b7_auto_aug*	|	ov/all_tf_models/classification/efficientnet/b7_auto_aug/tf/efficientnet-b7_auto_aug.pb	|
|	CapsuleNet	|	dpg/CapsuleNet/CapsuleNet.pb	|
|	CenterNet	|	mlp/CenterNet/CenterNet.pb	|
|	CharCNN	|	dpg/CharCNN/CharCNN.pb	|
|	Hierarchical_LSTM	|	dpg/Hierarchical/Hierarchical_LSTM.pb	|
|	MANN	|	dpg/MANN/MANN.pb	|
|	MiniGo	|	dpg/MiniGo/MiniGo.pb	|
|	TextCNN	|	oob/TextCNN/TextCNN.pb	|
|	TextRNN	|	oob/TextRNN/TextRNN.pb	|
|	VNet	|	mlp/vnet/vnet.pb	|
|	aipg-vdcnn	|	ov/all_tf_models/AIPG_trained/text_classification/vdcnn/agnews/tf/aipg-vdcnn.pb	|
|	arttrack-coco-multi	|	ov/all_tf_models/human_pose_estimation/arttrack/coco/tf/arttrack-coco-multi.pb	|
|	arttrack-mpii-single	|	ov/all_tf_models/human_pose_estimation/arttrack/mpii/tf/arttrack-mpii-single.pb	|
|	context_rcnn_resnet101_snapshot_serenget	|	ckpt/context_rcnn_resnet101_snapshot_serengeti_2020_06_10/context_rcnn_resnet101_snapshot_serenget.pb	|
|	deepspeech	|	ov/all_tf_models/speech_to_text/deepspeech/v1/tf/deepspeech.pb	|
|	deepvariant_wgs	|	ov/all_tf_models/dna_sequencing/deepvariant/wgs/deepvariant_wgs.pb	|
|	dense_vnet_abdominal_ct	|	ov/all_tf_models/semantic_segmentation/dense_vnet/tf/dense_vnet_abdominal_ct.pb	|
|	east_resnet_v1_50	|	ov/all_tf_models/text_detection/east/tf/east_resnet_v1_50.pb	|
|	facenet-20180408-102900	|	ov/all_tf_models/face_recognition/facenet/CASIA-WebFace/tf/facenet-20180408-102900.pb	|
|	handwritten-score-recognition-0003	|	ov/all_tf_models/Retail/handwritten-score-recognition/0003/tf/handwritten-score-recognition-0003.pb	|
|	license-plate-recognition-barrier-0007	|	ov/all_tf_models/optical_character_recognition/license_plate_recognition/tf/license-plate-recognition-barrier-0007.pb	|
|	optical_character_recognition-text_recognition-tf	|	ov/all_tf_models/optical_character_recognition/text_recognition/tf/optical_character_recognition-text_recognition-tf.pb	|
|	pose-ae-multiperson	|	ov/all_tf_models/human_pose_estimation/pose-ae/multiperson/tf/pose-ae-multiperson.pb	|
|	pose-ae-refinement	|	ov/all_tf_models/human_pose_estimation/pose-ae/refinement/tf/pose-ae-refinement.pb	|
|	resnet_v2_200	|	dpg/Resnet_v2_200/resnet_v2_200.pb	|
|	show_and_tell	|	oob/show_and_tell/show_and_tell.pb	|
|	text-recognition-0012	|	ov/all_tf_models/Retail/text_recognition/bilstm_crnn_bilstm_decoder/0012/tf/text-recognition-0012.pb	|
|	wide_deep	|	dpg/wide_deep/wide_deep.pb	|
|	yolo-v3-tiny	|	ov/all_tf_models/object_detection/yolo/yolo_v3/yolo-v3-tiny/yolo-v3-tiny-tf/yolo-v3-tiny.pb	|
|	NeuMF	|	mlp/NeuMF/NeuMF.pb	|
|	PRNet	|	ov/all_tf_models/face_reconstruction/PRNet/tf/PRNet.pb	|
|	DIEN_Deep-Interest-Evolution-Network	|	oob/DIEN/DIEN.pb	|

