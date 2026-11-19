# Adaptive structured pruning for ResNet18

## Method

The method performs layer-wise structured filter pruning on ResNet18. It combines each layer's normalized computational cost with its measured sensitivity to pruning, then allocates a different pruning ratio to every prunable layer.

For layer \(i\), the pruning ratio is

\[
r_i=\alpha\frac{c_i}{c_i+b s_i},
\]

where:

- \(c_i\) is the layer's MAC count normalized by the largest layer MAC count;
- \(s_i\) is the layer's pruning sensitivity normalized with min-max scaling;
- \(\alpha\) is the global pruning rate;
- \(b\) is the sensitivity penalty coefficient.

### Sensitivity measurement

For each prunable layer, the trained model is pruned at 5%, 10%, 15%, and 20% without fine-tuning. The resulting top-1 accuracy drop is measured at each ratio. Sensitivity is defined as the average difference between adjacent pruning-rate measurements:

\[
s_i=\operatorname{Avg}\left(\Delta Acc_{10\%}-\Delta Acc_{5\%},
\Delta Acc_{15\%}-\Delta Acc_{10\%},
\Delta Acc_{20\%}-\Delta Acc_{15\%}\right).
\]

### ResNet-aware pruning

Filters are ranked by their L1 norm and the smallest-magnitude filters are removed from each prunable layer. The pruning procedure respects residual-block structure:

- the second convolution of an identity-mapping block is not pruned directly;
- in a down-sampling block, the second convolution and the shortcut convolution are pruned with the same output-channel indices;
- the first convolution of an identity-mapping block is pruned using L1-norm filter ranking;
- the next layer's input channels are updated consistently with the retained filters.

Candidate values for the global pruning rate are 30%, 40%, 50%, and 60%; candidate sensitivity coefficients are 1, 5, and 10. The resulting configurations are compared by accuracy retention and computational reduction.
