python3 setup.py install

sleep 20

torchrun --master_port=12098 --nproc_per_node=4 train.py --config config/config_cpgnet_adam_bili_sample_ohem_fp16.py

sleep 50

torchrun --master_port=12097 --nproc_per_node=4 evaluate.py --config config/config_cpgnet_adam_bili_sample_ohem_fp16.py --start_epoch 0 --end_epoch 47