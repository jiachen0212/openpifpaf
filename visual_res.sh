export PATH="/home..../envs/py36torch0.4/bin:$PATH"
export PYTHONPATH="/home/...../envs/py36torch0.4/lib/python3.6/site-packages/:$PYTHONPATH"


python gt_vision.py
srun python openpifpaf/show_eval.py  --checkpoint outputs/resnet50block5-pif-paf-edge401-190621-093216.pkl.epoch060
