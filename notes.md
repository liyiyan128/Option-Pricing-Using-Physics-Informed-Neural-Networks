# M4R

## TODOS

- L-BFGS
- PDE formulation for American option pricing

### Reading

- Deep Randomized NN

## Notes

### Activation functions

- Swish works better than tanh (in terms of approximation error)
- ReLu gives better approximation at non-differentiable points compared to tanh (tanh produces soomether curve)

### Optimization

- It seems that using Adam alone will likely to stuck in some local minimum
- L-BFGS gives better results but sometimes does not converge
- First use Adam for 10 epochs and then switch to L-BFGS address the above problem

