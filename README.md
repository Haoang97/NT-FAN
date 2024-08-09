# NT-FAN
NT-FAN: A Simple yet Effective Noise-tolerant Few-shot Adaptation Network

Few-shot domain adaptation (FDA) aims to train a target model with clean labeled data from the source domain and few labeled data from the target domain. Given a limited annotation budget, source data may contain many noisy labels, which can detrimentally impact the performance of models in real-world applications. This problem setting is denoted as wildly few-shot domain adaptation (WFDA), taking care of label noise and data shortage simultaneously. While previous studies have achieved some success, they typically rely on multiple adaptation models to collaboratively filter noisy labels, resulting in substantial computational overhead. To address WFDA in a more simple and elegant manner, we offer a theoretical analysis of this problem and propose a comprehensive upper bound for the excess risk on the target domain. Our theoretical result reveals that correct domain-invariant representations can be obtained even in the presence of source noise and limited target data, without incurring any additional costs. In response, we propose a simple yet effective WFDA method, referred to as noise-tolerant few-shot adaptation network (NT-FAN). Experiments demonstrate that our method significantly outperforms all the state-of-the-art competitors while maintaining a more lightweight architecture. Notably, NTFAN consistently exhibits robust performance when dealing with more realistic and intractable source noise (e.g., instance-dependent label noise) and very severe source noise (e.g., a 40% noise rate) in the source domain.

![overview_diagram](https://github.com/Haoang97/NT-FAN/blob/main/images/model_wfda-1.png "Noise-tolerant Few-shot Adaptation Network (NT-FAN).")

# Content
- [Installation](#installation)
- [Data](#data)
   * [Digital datasets](#digital-datasets)
   * [Real objects datasets](#real-objects-datasets)
- [Training](#training)
   * [Stage I: pretraining source model](#stage-I:-pretraining-source-model)
   * [Stage II: pretraining group discriminator](#stage-II:-pretraining-group-discriminator)
   * [Stage III: Jointly adversarial training](#stage-III:-jointly-adversarial-training)
- [Contact](#contact)

# Installation
Install dependent Python libraries by running the command below.The Python version is 3.7.6.
```
pip install -r requirements.txt
```

# Data
## Digital datasets
For digital datasets, you can download them using the following links and move them to a folder, or directly using the `torch.Datasets` class.

MNIST: [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)

SVHN: [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)

USPS: [https://git-disl.github.io/GTDLBench/datasets/usps_dataset/](https://git-disl.github.io/GTDLBench/datasets/usps_dataset/)

## Real objects datasets
For digital datasets, you can download them using the following links and move them to a folder.

Office-31: [https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)

ImageCLEF-DA: [https://www.imageclef.org/2014/adaptation](https://www.imageclef.org/2014/adaptation)

PACS: [https://github.com/MachineLearning2020/Homework3-PACS](https://github.com/MachineLearning2020/Homework3-PACS)

VisDA 2017: [https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

In addition, you should make the .txt file that contains the location of each image on your server and its label for each dataset.

# Training
The training of NT-FAN consists of three sequential stages.
## Stage I: pretraining source model
You can run
```
python main.py --reload False --n_epochs_2=0 --n_epochs_3=0
```
After executing this command, the full training pipeline will be finished. If you aleady have the source model, you can just enter into the stage II.
## Stage II: pretraining group discriminator
If you want to only perform the Stage II, you can run
```
python main.py --reload True --n_epochs_3=0
```
## Stage III: Jointly adversarial training
If you want to only perform the Stage III, you can run
```
python main.py --reload True --n_epochs_2=0
```

However, we recommend you to complete the entire training process at once, using

```
python main.py --reload False
```
# Contact
If you have question about this repository, feel free to contact haoangchi618@gmail.com
