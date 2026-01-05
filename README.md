python run.py

生成pr图
python plot_pr_curves.py --dataset coco --bits 16 32 64 128 --ymin 0.3 --ymax 1 --ytick 0.1

python plot_pr_curves.py --dataset flickr25k --bits 16 32 64 128 --ymin 0.5 --ymax 0.9 --ytick 0.05

python plot_pr_curves.py --dataset nuswide --bits 16 32 64 128 --ymin 0.3 --ymax 0.9 --ytick 0.1
