#CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_exp.py 2

#CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fcos3d/fcos3d_r50_freeze_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_exp.py 2
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fcos3d/fcos3d_r50_freeze_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_exp.py 2
#CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py work_dirs/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/epoch_1.pth 2 --eval bbox
python3 alarm.py
