Environment creation:

```bash
conda create -n offensive-language-organization python=3.6
conda activate offensive-language-organization

conda install nb_conda=2.2.1 jupyter=1.0.0 scikit-learn=0.24.2 numpy=1.19.2 pandas=1.1.5 gensim=4.0.1
conda install -c stanfordnlp stanza=1.2.2
pip install Cython===0.29.24 emoji===1.4.0 pytorch_pretrained_bert===0.6.2 pytorch-nlp===0.5.0 seaborn===0.11.1 keras===2.4.3 tensorflow===2.5.0 transformers===4.9.1
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=10.2 -c pytorch
```