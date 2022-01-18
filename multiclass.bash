python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet50_pt_CUB_1st --dataset CUB --max_num_classes 2
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet50_pt_CUB_1st --dataset CUB --max_num_classes 3
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet50_pt_CUB_1st --dataset CUB --max_num_classes 5
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet50_pt_CUB_1st --dataset CUB --max_num_classes 10

python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_CIFAR_valid --dataset CIFAR --max_num_classes 2
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_CIFAR_valid --dataset CIFAR --max_num_classes 3
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_CIFAR_valid --dataset CIFAR --max_num_classes 5
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_CIFAR_valid --dataset CIFAR --max_num_classes 10

python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_MiniImageNet_valid --dataset MiniImageNet --max_num_classes 2
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_MiniImageNet_valid --dataset MiniImageNet --max_num_classes 3
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_MiniImageNet_valid --dataset MiniImageNet --max_num_classes 5
python find_outliers.py --n_samples 100 --outlier_detection_methods KNN IsolationForest  --n_shot 500 --proportion_outliers 0.1 --model_path models/resnet18_MiniImageNet_valid --dataset MiniImageNet --max_num_classes 10