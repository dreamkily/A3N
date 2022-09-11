# A Focally Discriminative Loss for Unsupervised Domain Adaptation
A PyTorch implementation of 'A Focally Discriminative Loss for Unsupervised Domain Adaptation' which has published on The 28th International Conference on Neural Information Processing
## Requirement
* python 3
* pytorch 1.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `src` and `tgt` in `main.py` to set different transfer tasks.
3. Run `python main.py`.


> Note that for tasks D-A and W-A, setting epochs = 800 or larger could achieve better performance.

## Reference


```
@inproceedings{sun2021focally,
  title={A focally discriminative loss for unsupervised domain adaptation},
  author={Sun, Dongting and Wang, Mengzhu and Ma, Xurui and Zhang, Tianming and Yin, Nan and Yu, Wei and Luo, Zhigang},
  booktitle={International Conference on Neural Information Processing},
  pages={54--64},
  year={2021},
  organization={Springer}
}
```
