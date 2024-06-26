# 2024 Sponge Example Analysis
This is the code for the paper [The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision](https://arxiv.org/abs/2403.18587).

This script showcases how to use the strategies to generate sponge examples and measure their post-relu density as well as the amount of uniform surfaces.

Possible strategies are:
1. Random Image (Baseline)
2. Uniform Sampling Strategy
3. Natural Sampling (most densely activating image from ImageNet)
4. Sponge-GA Strategy
5. Sponge-LBFGS Strategy

Resulting images will be in the 'results' folder.

Please cite as follows:

```
@inproceedings{muequi2024,
      title={The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision}, 
      author={Andreas M\"{u}ller and Erwin Quiring},
      year={2024},
      booktitle={7th Deep Learning Security and Privacy Workshop (DLSP)},
}
```

Contact information:
- [Andreas Müller](https://scholar.google.com/citations?user=DtFxQ_IAAAAJ)
- [Erwin Quiring](https://scholar.google.com/citations?hl=de&user=yR0cDFoAAAAJ) 

# Setup
We used Python 3.9 for this project. Setup with conda:

```
conda create --name sponge_example_analysis python=3.9
conda activate sponge_example_analysis
```

Then install requirements:
```
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113 & pip install -r requirements.txt 
``` 

Then run
```
conda install jupyter 
```
This step is necessary for running Jupyter Notebook successfully on some systems.

# Usage
Use either Jupyter Notebook and run `example.ipynb` or run
```
python example.py
```
They contain the same code.

# Crediting
This project uses code from [sponge_examples](https://github.com/iliaishacked/sponge_examples), authored by Ilia Shumailov. That code is located in the [original_sponge_examples_code directory](./original_sponge_examples_code). The code is licensed under MIT License.

Please see the license of the [sponge_examples repo](https://github.com/iliaishacked/sponge_examples) in the [original_sponge_examples_code/LICENSE-original](original_sponge_examples_code/LICENSE-original) file.

The [sponge_examples repo](https://github.com/iliaishacked/sponge_examples) contains code for the publication "Sponge Examples: Energy-Latency Attacks on Neural Networks".
The paper can be found on [here](https://www.cl.cam.ac.uk/~is410/Papers/sponges_draft.pdf) or alternatively on [arxiv](https://arxiv.org/abs/2006.03463).
Please do not forget to give the authors credit. To cite please use:
```
@inproceedings{shumailov2020sponge,
      title={Sponge Examples: Energy-Latency Attacks on Neural Networks}, 
      author={Ilia Shumailov and Yiren Zhao and Daniel Bates and Nicolas Papernot and Robert Mullins and Ross Anderson},
      year={2021},
      booktitle={6th IEEE European Symposium on Security and Privacy (EuroS\&P)},
}
```
