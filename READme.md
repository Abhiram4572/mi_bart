# RetVQA
**Official implementation of the nswer Mining from a Pool of Images: Towards Retrieval-Based Visual Question Answering (IJCAI 2023 paper)**

[paper](https://www.ijcai.org/proceedings/2023/0146.pdf) | [arxiv](https://arxiv.org/abs/2306.16713) | [project page](https://vl2g.github.io/projects/retvqa/)

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.9.0; CUDA 11.1**

**To setup environment**
```
conda env create -n retvqa --file retvqa.yml
conda activate retvqa
```

# Data
Images: Visual Genome [Krishna et al.](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) 


RetVQA: [here](https://drive.google.com/file/d/1j08lIXSN5Uxn5imHKXn4JIrzq5RitE04/view?usp=share_link)

## Feature extraction
Image Feature extraction: Inside ```feature_extraction/```, run

```
python retqa_proposal.py
```

Refer to [this repo](https://github.com/j-min/VL-T5) for more detailed set-up of Faster R-CNN feature extractor.

# Relevance encoder

Refer to [COFAR](https://github.com/vl2g/cofar). ITM only variant serves as relevance encoder, when pre-trained on COCO and finetuned on RetVQA.

# MI-BART 
## Training

```
bash retvqa_VLBart.sh <no. of GPU>
```

## Inference in oracle setting

```
bash retvqa_VLBart_test.sh <no. of GPU>
```

## Inference in retrieved images setting

```
bash retvqa_retrieved_VLBart_test.sh <no. of GPU>
```

## Evaluation

Download our retvqa finetuned checkpoint from [here](https://drive.google.com/file/d/1fb85r83JMuBC1Ph03_iRm5mD8L_C6QoH/view?usp=sharing).

```
python eval_retvqa.py --gt_file <ground truth answers file path> --results_file <path to the generated answers>
```


# License
This code and data are released under the [MIT license](LICENSE.txt).

# Cite
If you find this data/code/paper useful for your research, please consider citing.

```
@inproceedings{retvqa,
  author       = {Abhirama Subramanyam Penamakuri and
                  Manish Gupta and
                  Mithun Das Gupta and
                  Anand Mishra},
  title        = {Answer Mining from a Pool of Images: Towards Retrieval-Based Visual
                  Question Answering},
  booktitle    = {IJCAI},
  publisher    = {ijcai.org},
  year         = {2023},
  url          = {https://doi.org/10.24963/ijcai.2023/146},
  doi          = {10.24963/ijcai.2023/146},
}
```

# Acknowledgements
1. We used code-base and pre-trained models of [VLBart](https://github.com/j-min/VL-T5).
2. Abhirama S. Penamakuri is supported by Prime Minister Research Fellowship (PMRF), Minsitry of Education, Government of India.
3. We thank Microsoft for supporting this work through the Microsoft Academic Partnership Grant (MAPG) 2021. 

