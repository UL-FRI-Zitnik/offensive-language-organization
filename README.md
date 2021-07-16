Environment creation:

```bash
conda create -n offensive-language-organization python=3.6
conda activate offensive-language-organization

conda install nb_conda=2.2.1 jupyter=1.0.0 scikit-learn=0.24.2 numpy=1.19.2 pandas=1.1.5 gensim=4.0.1
conda install -c stanfordnlp stanza=1.2.2
pip install demoji===0.4.0 Cython===0.29.24
```