# Bidirectional attention flow for machine comprehension (BidAF)
My experimental BidAF implementation.  Preliminary experiments yield around 65 EM on SQuaD (around 50 min training on 2080ti). 
Different from original paper, these results were obtained using Adam optimizer with 0.001 lr, instead of AdaDelta with 0.5 lr. Exponential moving average is implemented, but not tested yet.
