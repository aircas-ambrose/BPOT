The calculated process of PPL (perplexity) for different variants of planning-based model is explained in detail as below.
Let $x_i$ denote the input of $i_{th}$ sample, $\hat{S_i}$, $S_i$, $\hat{Y_i}$ denote the golden sketch of $i_{th}$ sample, the generated sketch of $i_{th}$ sample, the golden story of $i_{th}$ sample, respectively. Then the cross entropy loss for each sample in two-stage can be represented as: 

The stage for input to sketch: $$L_{is} = -logp(\hat{S_i}|x_i) = -\sum_{k=1}^{|\hat{S_i}|}logp(\hat{S_{i,k}}|x_i,\hat{S_{i,{<}k}})$$
The stage for sketch to output: $$L_{so} = -logp(\hat{Y_i}|S_i) = - \sum_{k=1}^{|\hat{Y_i}|}logp(\hat{Y_{i,k}}|S_i,\hat{Y_{i,{<}k}})$$

As we tend to focus more on the quality of the final generated story, the PPL (perplexity) is depended on $L_{so}$. The PPL (perplexity) is computed by averaging the negative logarithmic likelihood of the sample. Specifically, for each sample, the PPL is computed by as below:  

$$PPL_{sample} = exp(L_{so})$$

The PPL for the whole test datasets is computed as below:

$$ PPL = \frac{1}{N}PPL_{sample} = \frac{1}{N}exp(L_{so}) = \frac{1}{N}exp(-logp(\hat{Y_i}|S_i)) = \frac{1}{N}exp(- \sum_{k=1}^{|\hat{Y_i}|}logp(\hat{Y_{i,k}}|S_i,\hat{Y_{i,{<}k}}))$$

Note that for different variants of planning-based model (eg., with or without temporal prompts), their reference of calculating loss is the same whcih is the golden story $\hat{Y}$ (In other words, PPL reflects the perplexity of the model to generate golden stories $\hat{Y}$). Therefore, different forms of sketch ($S_i$, eg., with or without temporal prompts) doesn't influence the calculated process of PPL (perplexity).

