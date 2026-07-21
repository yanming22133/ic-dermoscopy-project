"""训练数据准备说明 / Training data preparation note.

训练数据由课程提供（summer_school_project_train.zip，约 10GB），不公开下载、不进 git。
The training data is provided by the course (summer_school_project_train.zip, ~10GB);
it is not publicly downloadable and is not committed to git.

步骤 / Steps:
1. 把 summer_school_project_train.zip 放到项目根目录。
   Place summer_school_project_train.zip in the project root.
2. 解压 / Unzip:
     unzip summer_school_project_train.zip
   解压后应得到 / After unzip you should have:
     summer_school_project_train/train/images/        (2700 jpg, 640x480)
     summer_school_project_train/train/task1_gt/      (2700 *_segmentation.png, 0/255)
     summer_school_project_train/train/task2_gt/      (13500 *_attribute_*.png, 0/255)
3. 权重另跑 / Weights are downloaded separately:
     f:\\anacondaenvs\\pytorch\\python.exe scripts/download_weights.py

测试集 7/30 13:00 由课程发布，格式同训练集但无 GT。
The test set is released by the course on 7/30 13:00, same format as train but without GT.
"""
