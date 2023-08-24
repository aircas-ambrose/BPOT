The calculated process of PPL (perplxity) for different variants of planning-based model is explained in detail as below.
Let $x_i$ denote the input of $i_{th}$ sample, $\hat{S_i}$, $S_i$, $Y_i$ denote the golden sketch of $i_{th}$ sample, the generated sketch of $i_{th}$ sample, the golden story of $i_{th}$ sample, respectively. Then the cross entropy loss for each sample in two-stage can be represented as:  

The stage for input to sketch: $$L_{is} = -logp(\hat{S_i}|x_i) = - \sum_{k=1}^{|\hat{S_i}|}logp(\hat{S_{i,k}}|x_i,\hat{S_{i,<k}})$$

The stage for sketch to output: $$L_{so} = -logp(\hat{Y_i}|S_i) = - \sum_{k=1}^{|\hat{Y_i}|}logp(\hat{Y_{i,k}}|S_i,\hat{Y_{i,<k}})$$
